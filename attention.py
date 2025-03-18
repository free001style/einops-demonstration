import torch
from torch import nn
from einops import rearrange


class MHSA(nn.Module):
    def __init__(self, dim=512, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, 3 * heads * dim_head)
        self.proj_out = nn.Linear(heads * dim_head, dim)

    def forward(self, x):
        b, l, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(b, l, self.heads, self.dim_head).permute(2, 0, 1, 3).contiguous().view(-1, l, self.dim_head)
        k = k.view(b, l, self.heads, self.dim_head).permute(2, 0, 1, 3).contiguous().view(-1, l, self.dim_head)
        v = v.view(b, l, self.heads, self.dim_head).permute(2, 0, 1, 3).contiguous().view(-1, l, self.dim_head)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = nn.functional.softmax(attn, -1)
        output = torch.matmul(attn, v)

        output = output.view(self.heads, b, l, self.dim_head).permute(1, 2, 0, 3).contiguous().view(b, l, -1)
        return self.proj_out(output)


class MHSAEinops(nn.Module):
    def __init__(self, dim=512, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, 3 * heads * dim_head)
        self.proj_out = nn.Linear(heads * dim_head, dim)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b l (h d) -> (b h) l d", h=self.heads)
        k = rearrange(k, "b l (h d) -> (b h) l d", h=self.heads)
        v = rearrange(v, "b l (h d) -> (b h) l d", h=self.heads)

        attn = torch.matmul(q, rearrange(k, "... l d -> ... d l")) * self.scale
        attn = nn.functional.softmax(attn, -1)
        output = torch.matmul(attn, v)

        output = rearrange(output, "(b h) l d -> b l (h d)", h=self.heads)
        return self.proj_out(output)
