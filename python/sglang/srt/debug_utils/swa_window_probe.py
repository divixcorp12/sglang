"""Debug probe: after an extend forward, dump the SWA window rows a request's first decode will read.

For a bs=1 extend ending at seq_len it saves, for every layer of the paged SWA pool, the 576 data bytes
(448 fp8 nope + 64 bf16 rope) of positions [seq_len - window, seq_len), in FlashMLA page order. Written
for the decoder bounded-replay prefix-hit study; enabled by SGLANG_DEBUG_SWA_WINDOW_DUMP_DIR.
"""

from __future__ import annotations

import itertools
import os

import torch

_DATA_BYTES = 576
_counter = itertools.count()


def dump_swa_window(*, model_runner, forward_batch, out_dir: str, window: int = 128) -> None:
    if forward_batch.batch_size != 1:
        return
    kv_pool = model_runner.token_to_kv_pool
    swa_pool = kv_pool.swa_kv_pool
    seq_len = int(forward_batch.seq_lens_cpu[0])
    extend_len = int(forward_batch.extend_seq_lens_cpu[0])
    start = max(0, seq_len - window)
    req_pool_idx = int(forward_batch.req_pool_indices[0])
    full_locs = model_runner.req_to_token_pool.req_to_token[req_pool_idx, start:seq_len].long()
    swa_locs = kv_pool.translate_loc_from_full_to_swa(full_locs).long()
    page = swa_locs // swa_pool.page_size
    offset = (swa_locs % swa_pool.page_size) * _DATA_BYTES
    cols = offset[:, None] + torch.arange(_DATA_BYTES, device=offset.device)[None, :]
    rows = torch.stack([buf[page[:, None], cols] for buf in swa_pool.kv_buffer]).cpu()
    os.makedirs(out_dir, exist_ok=True)
    n = next(_counter)
    torch.save(
        {"seq_len": seq_len, "extend_len": extend_len, "start": start, "swa_locs": swa_locs.cpu(), "rows": rows},
        os.path.join(out_dir, f"{n:04d}-seq{seq_len}-ext{extend_len}.pt"),
    )
