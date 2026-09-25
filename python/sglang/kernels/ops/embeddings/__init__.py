"""Embedding kernels."""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import KernelBackend, KernelSpec

register_kernel(
    KernelSpec(
        op="embeddings.vocab_parallel_embedding",
        backend=KernelBackend.TRITON,
        target=(
            "sglang.kernels.ops.embeddings.vocab_parallel_embedding:"
            "vocab_parallel_embedding"
        ),
    )
)

register_kernel(
    KernelSpec(
        op="embeddings.engram_gather",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.embeddings.engram_gather:engram_gather",
    )
)

register_kernel(
    KernelSpec(
        op="embeddings.engram_hash_ids",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.embeddings.engram_hash:engram_hash_ids",
    )
)

register_kernel(
    KernelSpec(
        op="embeddings.engram_hash_ids_and_commit",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.embeddings.engram_hash:engram_hash_ids_and_commit",
    )
)

register_kernel(
    KernelSpec(
        op="embeddings.engram_commit_history",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.embeddings.engram_hash:engram_commit_history",
    )
)

register_kernel(
    KernelSpec(
        op="embeddings.fused_engram_gate",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.embeddings.engram_gate:fused_engram_gate",
    )
)

register_kernel(
    KernelSpec(
        op="embeddings.engram_ring_post",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.embeddings.engram_ring:engram_ring_post",
    )
)

register_kernel(
    KernelSpec(
        op="embeddings.engram_ring_wait",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.embeddings.engram_ring:engram_ring_wait",
    )
)

__all__ = []
