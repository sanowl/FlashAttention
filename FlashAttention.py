import torch
import triton
import triton.language as tl
import math
import torch.nn.functional as F
from typing import Optional, Tuple


@triton.jit
def _fwd_kernel(
    Q, K, V,
    sm_scale,
    Output,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    seqlen_q, seqlen_k, head_dim,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // tl.num_programs(2)
    head_idx = pid_bh % tl.num_programs(2)

    block_start_m = pid_m * BLOCK_M
    block_start_n = 0

    q_ptr = Q + batch_idx * stride_qb + head_idx * stride_qh + block_start_m * stride_qm
    k_ptr = K + batch_idx * stride_kb + head_idx * stride_kh
    v_ptr = V + batch_idx * stride_vb + head_idx * stride_vh
    o_ptr = Output + batch_idx * stride_ob + head_idx * stride_oh + block_start_m * stride_om

    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    m_prev = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_prev = tl.zeros((BLOCK_M,), dtype=tl.float32)

    q = tl.load(q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
               mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < head_dim),
               other=0.0).to(tl.float32)

    for start_n in range(0, seqlen_k, BLOCK_N):
        k_ptr_n = k_ptr + start_n * stride_kn
        v_ptr_n = v_ptr + start_n * stride_vn
        offs_n_curr = start_n + offs_n

        k = tl.load(k_ptr_n + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk,
                   mask=(offs_n_curr[None, :] < seqlen_k) & (offs_d[:, None] < head_dim),
                   other=0.0).to(tl.float32)

        qk = tl.dot(q, k)
        qk *= sm_scale

        if causal:
            mask = (offs_m[:, None] >= offs_n_curr[None, :])
            qk = tl.where(mask, qk, float('-inf'))

        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        qk_exp = tl.exp(qk - m_curr[:, None])

        l_curr = tl.sum(qk_exp, 1)
        l_prev = l_prev * tl.exp(m_prev - m_curr) + l_curr
        m_prev = m_curr

        v = tl.load(v_ptr_n + offs_n[None, :] * stride_vn + offs_d[:, None] * stride_vk,
                   mask=(offs_n_curr[None, :] < seqlen_k) & (offs_d[:, None] < head_dim),
                   other=0.0).to(tl.float32)

        p = qk_exp / l_prev[:, None]
        acc += tl.dot(p, v)

    tl.store(o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
             acc,
             mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < head_dim))


class FlashAttention(torch.nn.Module):
    def __init__(
        self,
        softmax_scale: Optional[float] = None,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, num_heads, seqlen_q, head_dim = q.shape
        _, _, seqlen_k, _ = k.shape

        if self.softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)
        else:
            softmax_scale = self.softmax_scale

        output = torch.empty_like(q)

        q_ = q.reshape(batch_size * num_heads, seqlen_q, head_dim)
        k_ = k.reshape(batch_size * num_heads, seqlen_k, head_dim)
        v_ = v.reshape(batch_size * num_heads, seqlen_k, head_dim)
        output_ = output.reshape(batch_size * num_heads, seqlen_q, head_dim)

        stride_qb, stride_qm, stride_qk = q_.stride()
        stride_kb, stride_kn, stride_kk = k_.stride()
        stride_vb, stride_vn, stride_vk = v_.stride()
        stride_ob, stride_om, stride_ok = output_.stride()

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_DMODEL = head_dim

        grid = (triton.cdiv(seqlen_q, BLOCK_M), batch_size * num_heads)

        _fwd_kernel[grid](
            q_, k_, v_,
            softmax_scale,
            output_,
            stride_qb, 0, stride_qm, stride_qk,
            stride_kb, 0, stride_kn, stride_kk,
            stride_vb, 0, stride_vn, stride_vk,
            stride_ob, 0, stride_om, stride_ok,
            seqlen_q, seqlen_k, head_dim,
            self.causal,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
            num_warps=4,
            num_stages=3
        )

        output = output_.reshape(batch_size, num_heads, seqlen_q, head_dim)

        if self.dropout > 0.0 and self.training:
            output = F.dropout(output, p=self.dropout)

        if need_weights:
            return output, None

        return output, None


def test_flash_attention():
    torch.manual_seed(0)

    batch_size = 2
    num_heads = 4
    seqlen_q = 128
    seqlen_k = 128
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seqlen_q, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seqlen_k, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seqlen_k, head_dim, device='cuda', dtype=torch.float16)

    flash_attn = FlashAttention(causal=False).cuda()

    output, _ = flash_attn(q, k, v)

    assert output.shape == (batch_size, num_heads, seqlen_q, head_dim)

    q_pytorch = q.clone().detach().requires_grad_(True)
    k_pytorch = k.clone().detach()
    v_pytorch = v.clone().detach()

    q_pytorch_ = q_pytorch.permute(0, 2, 1, 3).reshape(batch_size * seqlen_q, num_heads * head_dim)
    k_pytorch_ = k_pytorch.permute(0, 2, 1, 3).reshape(batch_size * seqlen_k, num_heads * head_dim)
    v_pytorch_ = v_pytorch.permute(0, 2, 1, 3).reshape(batch_size * seqlen_k, num_heads * head_dim)

    attn_weights = torch.matmul(q_pytorch_, k_pytorch_.transpose(0, 1)) / math.sqrt(head_dim)
    attn_probs = F.softmax(attn_weights, dim=-1)
    expected_output = torch.matmul(attn_probs, v_pytorch_)

    expected_output = expected_output.reshape(batch_size, seqlen_q, num_heads, head_dim).permute(0, 2, 1, 3)

    torch.testing.assert_close(output.float(), expected_output.float(), atol=1e-2, rtol=1e-2)

    print("FlashAttention test passed!")


if __name__ == "__main__":
    test_flash_attention()
