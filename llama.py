import dataclasses
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

import torch
import math
import triton
import triton.language as tl



# Triton线性层kernel
@triton.jit
def _linear_forward_kernel(
    X,  # [M, K] 输入
    W,  # [N, K] 权重 (转置存储)
    Y,  # [M, N] 输出
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 程序ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算偏移量
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 分块计算
    for start_k in range(0, K, BLOCK_K):
        start_k = tl.multiple_of(start_k, BLOCK_K)
        offs_k_curr = start_k + offs_k
        
        # 计算指针偏移
        x_mask = (offs_m[:, None] < M) & (offs_k_curr[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (offs_k_curr[None, :] < K)
        
        # 加载数据块
        x = tl.load(X + offs_m[:, None] * stride_xm + offs_k_curr[None, :] * stride_xk, 
                   mask=x_mask, other=0.0)
        w = tl.load(W + offs_n[:, None] * stride_wn + offs_k_curr[None, :] * stride_wk,
                   mask=w_mask, other=0.0)
        
        # 计算矩阵乘法: x @ w^T
        acc += tl.dot(x, w.T)
    
    # 计算存储指针和掩码
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    
    # 存储结果
    tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask=y_mask)

def apply_triton_linear(x, weight):
    """
    Triton优化的线性层实现
    
    参数:
        x: 输入张量 [M, K] 或 [..., M, K]
        weight: 权重张量 [N, K]
    
    返回:
        output: 输出张量 [M, N] 或 [..., M, N]
    """
    original_shape = x.shape
    
    # 展平除了最后两维之外的所有维度
    if x.dim() > 2:
        batch_dims = x.shape[:-2]
        x_flat = x.view(-1, x.shape[-2], x.shape[-1])
        batch_size = x_flat.shape[0]
        M, K = x_flat.shape[-2:]
    else:
        batch_size = 1
        M, K = x.shape
        x_flat = x.view(-1, K)
    
    N = weight.shape[0]
    
    # 输出张量
    if x_flat.dim() == 2:
        out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    else:
        out = torch.empty(batch_size, M, N, dtype=x.dtype, device=x.device)
    
    # 设置分块大小
    BLOCK_M = 64
    BLOCK_N = 64  
    BLOCK_K = 64
    
    # 计算grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # 处理不同维度情况
    if x_flat.dim() == 2:
        # 2D case: [M, K]
        _linear_forward_kernel[grid](
            x_flat, weight, out,
            M, N, K,
            x_flat.stride(0), x_flat.stride(1),
            weight.stride(0), weight.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
    else:
        # 3D+ case: [B, M, K]
        for b in range(batch_size):
            _linear_forward_kernel[grid](
                x_flat[b], weight, out[b],
                M, N, K,
                x_flat.stride(1), x_flat.stride(2),
                weight.stride(0), weight.stride(1),
                out.stride(1), out.stride(2),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
            )
    
    # 恢复原始形状
    if len(original_shape) > 2:
        out = out.view(original_shape[:-1] + (N,))
    else:
        out = out.view(original_shape[:-1] + (N,))
    
    return out


@dataclasses.dataclass
class ModelConfig:
    head_dim: int

    hidden_size: int

    intermediate_size: int

    num_attention_heads: int

    num_hidden_layers: int

    num_key_value_heads: int

    rms_norm_eps: float

    rope_theta: float

    torch_dtype: str

    vocab_size: int


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))

        self.eps = eps

    def forward(self, input):
        return (
            input
            * torch.rsqrt(input.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            * self.weight
        )


class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 初始化权重
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # 使用Triton优化的线性函数
        out = apply_triton_linear(x, self.weight)
        if self.bias is not None:
            out += self.bias
        return out

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        # 使用Triton优化的线性层
        self.gate_proj = TritonLinear(hidden_size, intermediate_size, bias=False)

        self.up_proj = TritonLinear(hidden_size, intermediate_size, bias=False)

        self.down_proj = TritonLinear(intermediate_size, hidden_size, bias=False)

        self.silu = nn.SiLU()

    def forward(self, input):
        return self.down_proj(self.silu(self.gate_proj(input)) * self.up_proj(input))


def apply_rotary_position_embedding(input, sin_table, cos_table):
    sin_table = sin_table[None, :, None, :]
    cos_table = cos_table[None, :, None, :]

    input_0 = input[..., : input.shape[-1] // 2]
    input_1 = input[..., input.shape[-1] // 2 :]
    input_0_rotated = input_0 * cos_table - input_1 * sin_table
    input_1_rotated = input_0 * sin_table + input_1 * cos_table

    return torch.cat((input_0_rotated, input_1_rotated), dim=-1)


def apply_scaled_dot_product_attention(query, key, value):
    _, num_heads_q, seq_len_q, emb_dim = query.shape
    _, num_heads_k, seq_len_k, _ = key.shape
    _, num_heads_v, _, _ = value.shape

    key = key.repeat_interleave(num_heads_q // num_heads_k, 1)
    value = value.repeat_interleave(num_heads_q // num_heads_v, 1)

    scale = 1 / math.sqrt(emb_dim)
    attn_mask = torch.tril(
        torch.full((seq_len_q, seq_len_k), True, device=query.device)
    )

    attn_output = torch.matmul(query, key.permute(0, 1, 3, 2)) * scale
    attn_output = torch.where(attn_mask, attn_output, float("-inf"))
    attn_output = torch.softmax(attn_output, dim=-1)
    attn_output = torch.matmul(attn_output, value)

    return attn_output


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.head_dim

        self.hidden_size = config.hidden_size

        self.num_attention_heads = config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads

        # 使用Triton优化的线性层
        self.q_proj = TritonLinear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )

        self.k_proj = TritonLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.v_proj = TritonLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.o_proj = TritonLinear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(self, hidden_states, sin_table, cos_table):
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape).permute(0, 2, 1, 3)

        query_states = apply_rotary_position_embedding(
            query_states, sin_table, cos_table
        ).permute(0, 2, 1, 3)
        key_states = apply_rotary_position_embedding(
            key_states, sin_table, cos_table
        ).permute(0, 2, 1, 3)

        attn_output = apply_scaled_dot_product_attention(
            query_states, key_states, value_states
        )

        return self.o_proj(
            attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        )


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.self_attn = Attention(config)

        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states, sin_table, cos_table):
        hidden_states += self.self_attn(
            self.input_layernorm(hidden_states), sin_table, cos_table
        )

        hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states


def generate_sin_and_cos_tables(seq_len, emb_dim, base, dtype, device):
    theta = base ** (
        -2 * (torch.arange(emb_dim // 2, dtype=dtype, device=device) / emb_dim)
    )

    positions = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(1)
    sin_table = torch.sin(positions * theta)
    cos_table = torch.cos(positions * theta)

    return sin_table, cos_table


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.head_dim

        self.hidden_size = config.hidden_size

        self.num_hidden_layers = config.num_hidden_layers

        self.rms_norm_eps = config.rms_norm_eps

        self.rope_theta = config.rope_theta

        self.torch_dtype = config.torch_dtype

        self.vocab_size = config.vocab_size

        self.embed_tokens = torch.nn.Embedding(self.vocab_size, self.hidden_size)

        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(self.num_hidden_layers)
        )

        self.norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)

        seq_len = hidden_states.shape[1]

        sin_table, cos_table = generate_sin_and_cos_tables(
            seq_len,
            self.head_dim,
            base=self.rope_theta,
            dtype=getattr(torch, self.torch_dtype),
            device=input_ids.device,
        )

        for i in range(self.num_hidden_layers):
            hidden_states = self.layers[i](hidden_states, sin_table, cos_table)

        return self.norm(hidden_states)


class ModelForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = Model(config)

        # 使用Triton优化的线性层
        self.lm_head = TritonLinear(config.hidden_size, config.vocab_size, bias=False)

    def generate(self, input_ids, max_new_tokens=20):
        for _ in range(max_new_tokens):
            hidden_states = self.model(input_ids)

            logits = self.lm_head(hidden_states[:, -1, :])

            next = torch.argmax(logits, dim=-1).unsqueeze(-1)

            input_ids = torch.cat((input_ids, next), dim=-1)

        return input_ids

    @staticmethod
    def from_pretrained(model_path):
        model_path = Path(model_path)

        with open(model_path / "config.json") as f:
            config = json.load(f)

        if "head_dim" not in config:
            config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]

        config = ModelConfig(
            **{
                key: value
                for key, value in config.items()
                if key in ModelConfig.__annotations__
            }
        )

        model = ModelForCausalLM(config).to(getattr(torch, config.torch_dtype))

        state_dict = load_file(model_path / "model.safetensors")

        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        model.load_state_dict(state_dict)

        return model
