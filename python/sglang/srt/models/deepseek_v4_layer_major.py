"""DeepSeek-V4.1 adapter for the layer-major prefill strategy (sglang.srt.layer_major): chunk plan, window-KV ring,
per-chunk metadata, one layer on one chunk, and the late-layer tail. Knows nothing about the expert quant format."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any

import msgspec
import torch

from sglang.srt.layer_major.state_store import FieldSpec, StateStore

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

DSV4_WINDOW = 128


class ChunkSpan(msgspec.Struct, frozen=True):
    index: int
    start: int
    end: int


def chunk_spans(*, prefix_len: int, seq_len: int, chunk: int) -> list[ChunkSpan]:
    return [
        ChunkSpan(index=i, start=start, end=min(seq_len, start + chunk))
        for i, start in enumerate(range(prefix_len, seq_len, chunk))
    ]


def ring_len_ok(*, chunk: int, page: int, window: int) -> bool:
    """True iff a chunk+page ring's no-branch keep floor can never sit below the ring's oldest intact page
    (swept exhaustively against _insert_keep_from's arithmetic)."""
    return page >= window and chunk >= 2 * page


def engram_history(ids: list[int], start: int, n: int) -> list[int]:
    window = list(ids[max(0, start - n) : start])
    return [0] * (n - len(window)) + window


class _Pass(msgspec.Struct):
    schedule_batch: Any
    spans: list
    forward_batches: list
    hash_ids: list
    extend_full_locs: torch.Tensor
    ring: torch.Tensor
    final_tail_metadata: Any
    copy_stream: Any
    finalized: bool = False


class DeepseekV4LayerMajorAdapter:
    """Strategy-side adapter for DSV4.1: chunk plan, window-KV ring, per-chunk metadata handoff, one layer on one
    chunk, and the late-layer tail. Implements sglang.srt.layer_major's LayerMajorModelAdapter protocol."""

    def __init__(self, model_runner):
        self.runner = model_runner
        self.causal_lm = model_runner.model
        self.model = model_runner.model.model
        self.page = model_runner.page_size
        self.chunk = model_runner.server_args.chunked_prefill_size

    @property
    def backend(self):
        # Not cached in __init__: TpModelWorker builds this adapter before attn_backend exists.
        return self.runner.attn_backend

    @property
    def allocator(self):
        # Not cached in __init__: TpModelWorker builds this adapter before init_memory_pools() runs.
        return self.runner.token_to_kv_pool_allocator

    def _check_ring_len(self) -> None:
        if not ring_len_ok(chunk=self.chunk, page=self.page, window=DSV4_WINDOW):
            if self.page < DSV4_WINDOW:
                raise ValueError(f"page {self.page} is below window {DSV4_WINDOW}: the predecessor window does "
                                 f"not fit in one page")
            raise ValueError(f"chunk {self.chunk} holds fewer than 2 pages ({2 * self.page}): the ring cannot "
                             f"cover the no-branch keep floor")

    def field_specs(self) -> list[FieldSpec]:
        return [
            FieldSpec(name="hidden", per_token_shape=(self.model.hc_mult, self.model.hidden_size), dtype="bfloat16"),
            FieldSpec(name="prev_pre", per_token_shape=(self.model.hc_mult,), dtype="float32"),
        ]

    # --- pass setup -------------------------------------------------------------------------------------------------

    def begin_pass(self, forward_batch, schedule_batch, store: StateStore) -> _Pass:
        if len(schedule_batch.reqs) != 1:
            raise ValueError("layer-major prefill runs one request")
        req = schedule_batch.reqs[0]
        if req.multimodal_inputs is not None:
            raise ValueError("layer-major prefill does not support multimodal requests")
        if self.runner.server_args.enable_dp_attention:
            raise ValueError("layer-major prefill does not support DP attention")
        if self.allocator.swa_req_ring:
            # The per-request SWA ring (unified-KV) allocator gives the extend no window mapping at all
            # (alloc_extend_swa_tail pages full KV only): ring_slots would read back zeros. DSV4.1 launches
            # already refuse the unified KV layout (deepseek_v4_hook.validate_deepseek_v41_features), so this
            # should be unreachable; kept as a hard guard rather than trusting that gate transitively.
            raise ValueError("layer-major prefill does not support the per-request SWA ring allocator")
        slot = int(schedule_batch.req_pool_indices_cpu[0])
        prefix_len = int(schedule_batch.prefix_lens[0])
        seq_len = int(schedule_batch.seq_lens_cpu[0])
        full = self.runner.req_to_token_pool.req_to_token[slot, prefix_len:seq_len].to(torch.int64)
        ring_len = self.chunk + self.page
        self._check_ring_len()
        ring = self.allocator.ring_slots(full[-ring_len:])
        if ring.numel() != ring_len:
            raise ValueError(f"window ring has {ring.numel()} slots, expected {ring_len}")
        spans = chunk_spans(prefix_len=prefix_len, seq_len=seq_len, chunk=self.chunk)
        # No candidate consumer may run inside the layer-major range: earlier chunks skip building candidate masks.
        candidate_source_layer_id = self.runner.model_config.hf_text_config.candidate_source_layer_id
        if not (candidate_source_layer_id < 0 or candidate_source_layer_id >= self.model.late_layer_start - 1):
            raise ValueError(
                f"candidate_source_layer_id={candidate_source_layer_id} has a consumer inside the "
                f"layer-major range (< {self.model.late_layer_start - 1})"
            )
        handle = _Pass(schedule_batch=schedule_batch, spans=spans, forward_batches=[], hash_ids=[],
                       extend_full_locs=full, ring=ring, final_tail_metadata=None,
                       copy_stream=torch.cuda.Stream())
        try:
            # Map the whole suffix once, before the per-chunk loop: mapping per span left a begin_pass failure
            # mid-loop with the kept window mapped two ways (ring slots for processed spans, allocation-time
            # slots for the rest), so two kept positions could share one ring slot and double-free later.
            # Chunk 0's own suffix positions are mapped like every other chunk; only its PREDECESSOR window
            # (positions before prefix_len, already resident in the prefix's own slots) is never touched here.
            positions = torch.arange(prefix_len, seq_len, device=full.device)
            self.allocator.map_ring_positions(full, positions, ring)
            for span in spans:
                fb = self._chunk_forward_batch(schedule_batch, req, slot, span, last=span is spans[-1])
                # Mapping before init_forward_metadata: the tail's swa_out_cache_loc is translated through the
                # mapping when metadata is built.
                self.backend.init_forward_metadata(fb)
                meta = self.backend.forward_metadata
                meta.layer_major_skip_candidates = span is not spans[-1]
                if span is spans[-1]:
                    handle.final_tail_metadata = self.backend.tail_forward_metadata
                store.park(span.index, meta)
                hash_ids = None
                if self.model.engram_hasher is not None:
                    hash_ids = self.model.engram_hasher(fb.input_ids, fb)
                handle.hash_ids.append(hash_ids)
                embedded = self.model.embed_tokens(fb.input_ids).unsqueeze(1).repeat(1, self.model.hc_mult, 1)
                store.write("hidden", span.start - prefix_len, embedded.to("cpu"))
                handle.forward_batches.append(fb)
        except Exception:
            store.clear_parked()
            self._finalize_ring(handle)
            raise
        return handle

    def _chunk_forward_batch(self, batch, req, slot, span, *, last):
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch

        device = self.runner.device
        sub = copy(batch)
        sub.reqs = [req]
        sub.input_ids = torch.tensor(list(req.full_untruncated_fill_ids[span.start : span.end]), dtype=torch.int64,
                                     device=device)
        sub.prefill_input_ids_cpu = None
        sub.prefix_lens = [span.start]
        sub.extend_lens = [span.end - span.start]
        sub.extend_num_tokens = span.end - span.start
        sub.seq_lens = torch.tensor([span.end], dtype=torch.int64, device=device)
        sub.seq_lens_cpu = torch.tensor([span.end], dtype=torch.int64)
        sub.seq_lens_sum = span.end
        sub.orig_seq_lens = torch.tensor([len(req.origin_input_ids)], dtype=torch.int32, device=device)
        sub.out_cache_loc = self.runner.req_to_token_pool.req_to_token[slot, span.start : span.end].long()
        sub.extend_logprob_start_lens = [span.end - span.start]
        if not last:
            sub.return_logprob = False
            sub.sampling_info = None
            sub.is_prefill_only = True
        sub.engram_history = None
        if self.model.engram_hasher is not None:
            n = self.model.engram_hasher.max_ngram_size - 1
            sub.engram_history = torch.tensor([engram_history(req.full_untruncated_fill_ids, span.start, n)],
                                              dtype=torch.int32, device=device)
        return ForwardBatch.init_new(sub, self.runner, capture_hidden_mode=CaptureHiddenMode.NULL,
                                     return_hidden_states_before_norm=False)

    # --- the strategy's loop ----------------------------------------------------------------------------------------

    def layer_ids(self, handle: _Pass) -> range:
        return range(self.model.start_layer, self.model.late_layer_start)

    def num_chunks(self, handle: _Pass) -> int:
        return len(handle.spans)

    def run_layer(self, handle: _Pass, layer_id: int, chunk: int, store: StateStore) -> None:
        from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder

        span = handle.spans[chunk]
        fb = handle.forward_batches[chunk]
        offset = span.start - int(handle.schedule_batch.prefix_lens[0])
        rows = span.end - span.start
        device = self.runner.device
        hidden = torch.empty((rows, self.model.hc_mult, self.model.hidden_size), dtype=torch.bfloat16, device=device)
        store.read_into("hidden", offset, hidden, stream=None)
        prev_pre = None
        if layer_id > self.model.start_layer:
            prev_pre = torch.empty((rows, self.model.hc_mult), dtype=torch.float32, device=device)
            store.read_into("prev_pre", offset, prev_pre, stream=None)
        meta = store.unpark(span.index, torch.device(device))
        # Only the final chunk has a tail: the layer-20 source can then publish tail-only
        # masks; earlier chunks have none and rely on layer_major_skip_candidates instead.
        tail_metadata = handle.final_tail_metadata if span is handle.spans[-1] else None
        self.backend.install_forward_metadata(meta, tail_metadata=tail_metadata)
        layer = self.model.layers[layer_id]
        if layer.engram is not None:
            before_engram = hidden
            hidden = layer.engram(hidden, handle.hash_ids[chunk][:, layer.engram.layer_hash_index], fb,
                                  cp_all_tokens=False)
            if self.model.config.model_type == "deepseek_v41" and self.model.config.vision_n_layers > 0:
                hidden = torch.where(
                    (fb.input_ids == self.model.config.image_token_id)[:, None, None],
                    before_engram,
                    hidden,
                )
        with get_global_expert_distribution_recorder().with_current_layer(layer_id):
            # The chunked path's own arguments under bounded replay: its tail makes next_combined None (4549).
            hidden, prev_pre = layer.forward_hc_pre_from_prev(
                positions=fb.positions, hidden_states=hidden, input_ids=fb.input_ids, forward_batch=fb,
                input_ids_global=fb.input_ids, prev_pre=prev_pre, precomputed_attn=None, next_norm=None,
                next_input=[], combined_attn=None, normalized_attn=None, next_combined=None,
            )
        store.write_from("hidden", offset, hidden, stream=None)
        store.write_from("prev_pre", offset, prev_pre, stream=None)
        # Top-k written in place by index-source layers travels with the chunk's metadata.
        store.park(span.index, self.backend.forward_metadata)

    def finish_pass(self, handle: _Pass, store: StateStore) -> Any:
        span = handle.spans[-1]
        fb = handle.forward_batches[-1]
        offset = span.start - int(handle.schedule_batch.prefix_lens[0])
        rows = span.end - span.start
        device = self.runner.device
        hidden = torch.empty((rows, self.model.hc_mult, self.model.hidden_size), dtype=torch.bfloat16, device=device)
        prev_pre = torch.empty((rows, self.model.hc_mult), dtype=torch.float32, device=device)
        store.read_into("hidden", offset, hidden, stream=None)
        store.read_into("prev_pre", offset, prev_pre, stream=None)
        self.backend.install_forward_metadata(store.unpark(span.index, torch.device(device)),
                                              tail_metadata=handle.final_tail_metadata)
        output = self.causal_lm.forward_late_tail(forward_batch=fb, hidden_states=hidden, prev_pre=prev_pre,
                                                  hash_ids=handle.hash_ids[-1])
        self._finalize_ring(handle)
        return output

    def _finalize_ring(self, handle: _Pass) -> None:
        if handle.finalized:
            # finish_pass already released the ring; release_pass(failed=True) must not double-free it.
            return
        seq_len = handle.spans[-1].end
        prefix_len = int(handle.schedule_batch.prefix_lens[0])
        req = handle.schedule_batch.reqs[0]
        keep_from, drop_branch = self._insert_keep_from(req, handle, prefix_len=prefix_len, seq_len=seq_len)
        if drop_branch:
            # The ring has overwritten the window below the branch point, so the insert runs to the end instead.
            req.swa_branching_seqlen = None
        self.allocator.finalize_ring(handle.extend_full_locs, extend_start=prefix_len, keep_from=keep_from,
                                     ring=handle.ring)
        req.kv.swa_evicted_seqlen = max(req.kv.swa_evicted_seqlen, keep_from)
        handle.finalized = True

    def _insert_keep_from(self, req: Req, handle: _Pass, *, prefix_len: int, seq_len: int) -> tuple[int, bool]:
        """Pure: the floor cache_unfinished_req's next insert must keep window KV from, and whether the branch
        point should be dropped. Ends at the SWA branch point if admission set one and the ring still covers it,
        else page_floor(seq_len) (one page lower when seq_len is unaligned); keeping less ends that key on an
        unmatchable tombstone."""
        margin = max(DSV4_WINDOW, self.page)
        # The ring still holds a position iff its page is among the last ring-pages pages of the extend.
        oldest_intact = ((seq_len - 1) // self.page - handle.ring.numel() // self.page + 1) * self.page
        no_branch_floor = max(prefix_len, (seq_len // self.page * self.page - 1 - margin) // self.page * self.page)
        # Backstop: begin_pass's _check_ring_len should make this unreachable; keep it as the last line of defense.
        assert no_branch_floor >= oldest_intact, f"ring of {handle.ring.numel()} slots cannot keep from {no_branch_floor}"
        branch = req.swa_branching_seqlen
        if branch is not None and prefix_len < branch <= seq_len:
            keep_from = max(prefix_len, (branch - 1 - margin) // self.page * self.page)
            if keep_from >= oldest_intact:
                return keep_from, False
            return no_branch_floor, True
        return no_branch_floor, False

    def release_pass(self, handle: _Pass, store: StateStore, *, failed: bool) -> None:
        store.clear_parked()
        if failed:
            # Leave no stale ring mapping for the request's release to follow, matching a normal finished extend.
            self._finalize_ring(handle)
