"""PPU block-sparse SageAttention plans and references."""

from .plan import KV_BLOCK, SparseAttentionPlan
from .planners import make_h3_topk_plan, make_sol_plan
from .reference import sparse_attention_reference
from .api import (
    sageattn_block_sparse_ppu,
    sageattn_h3_topk_ppu,
    sageattn_sol_ppu,
)

__all__ = [
    "KV_BLOCK",
    "SparseAttentionPlan",
    "make_h3_topk_plan",
    "make_sol_plan",
    "sparse_attention_reference",
    "sageattn_block_sparse_ppu",
    "sageattn_h3_topk_ppu",
    "sageattn_sol_ppu",
]
