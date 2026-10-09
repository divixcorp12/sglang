// DriveLoad: the RAM tier's per-drive in-flight accounting, shared by every reader of the tier (design
// docs/superpowers/specs/2026-10-09-dsv41-drive-aware-reads-design.md, section 2). Observe-only: nothing reads it to
// decide a read yet.
//
//   DriveSlot    one mirror root's counts, one cache line: both NUMA groups' threads write it
//   DriveLoad    a slot per root, the tier's clock origin and its clock-read count
//
// Key: the root index, `file % parts`, never st_dev. Every row-image file of root q is file row * parts + q
// (exl3_ram_miss.py), and the test fixtures put every root under one tmp_path, where an st_dev key would fold them into
// one drive.
//
// Counting (ReaderCore): a sub-read counts +1 read when its first leg goes in flight in refill(), and every leg adds
// its remaining bytes; a reaped leg takes its bytes back, and the sub-read's last reaped leg its read. A read that
// fails drains its ring, and drained completions never reach process(), so each reader keeps a local mirror of what
// it added and drain()/reset_pipeline() take that back. Every read() return therefore leaves the reader's share at 0,
// and the shared counts are the sum of the readers' shares.
//
// Time. A slot's reads word packs demand reads (low 32 bits) and speculative reads (high 32), so one fetch_add orders
// every change of the pair and tells its caller the pair before and after. Each indicator (demand > 0, spec > 0, both
// > 0) is integrated as a running sum: a rise at t subtracts t, a fall adds t, so the sum plus `now` while the
// indicator is up is the time it was up. The sum is exact once every reader is idle (the shutdown line); a live read
// can land between a change and its sum update and is clamped. Only a change of an indicator reads the clock, at
// most once per refill() or reap() turn (the caller's `now`, 0 until read).
//
// Relaxed atomics only: no syscall but the vDSO clock read, no allocation, no lock.
#pragma once

#include "reader_base.h"

namespace sglang {
namespace expert_stream {

// The two kinds of read a reader makes: a demand read (a request's miss, or a prefill fill) or a speculative read (the
// RAM prefetch's pool rows).
enum ReadKind : int { kDemandRead = 0, kSpecRead = 1 };

// One root's counts. 64 bytes: both groups' service and speculative threads write it.
struct alignas(64) DriveSlot {
  std::atomic<int64_t> reads{0};        // in flight: demand reads in bits 0-31, speculative reads in bits 32-63
  std::atomic<int64_t> inflight[2]{};   // bytes in flight, by kind
  std::atomic<int64_t> done[2]{};       // bytes landed (reaped), by kind, over the tier's life
  std::atomic<int64_t> busy[3]{};       // the running sums of demand > 0, spec > 0 and both > 0 (see the file comment)
};
static_assert(sizeof(DriveSlot) == 64, "a drive slot is one cache line");

// The words of DriveLoad::snapshot: a header, then kDriveFields per root for kMaxDrives roots (roots past `roots` are
// zero). Python's decode_drive_load (expert_stream_transport.py) reads the same layout.
constexpr int kDriveHeader = 3;  // elapsed ns since the origin, clock reads, roots
constexpr int kDriveFields = 9;  // demand reads, spec reads, demand/spec bytes in flight, demand/spec bytes landed,
                                 // demand/spec/overlap busy ns
constexpr int kDriveLoadWords = kDriveHeader + kMaxDrives * kDriveFields;

class DriveLoad {
 public:
  DriveLoad() : origin_(now_ns()) {}
  DriveLoad(const DriveLoad&) = delete;
  DriveLoad& operator=(const DriveLoad&) = delete;

  // The root of `file` in a table of `parts` mirror parts; roots past the last slot share it.
  static int root_of(int64_t file, int64_t parts) {
    return static_cast<int>(std::min<int64_t>(file % std::max<int64_t>(1, parts), kMaxDrives - 1));
  }

  // Adds `reads` sub-reads and `bytes` bytes of kind `kind` to root `root` (negative: takes them back). `now` is the
  // caller's clock for this turn, ns since the origin, 0 until read: a change of an indicator reads it once.
  void change(int root, int kind, int reads, int64_t bytes, int64_t& now) {
    DriveSlot& s = slots_[root];
    if (bytes != 0) s.inflight[kind].fetch_add(bytes, std::memory_order_relaxed);
    if (reads == 0) return;
    const int64_t delta = kind == kSpecRead ? static_cast<int64_t>(reads) * (int64_t{1} << 32) : reads;
    const int64_t before = s.reads.fetch_add(delta, std::memory_order_relaxed);
    const int64_t after = before + delta;
    const bool d0 = demand_of(before) > 0, d1 = demand_of(after) > 0;
    const bool s0 = spec_of(before) > 0, s1 = spec_of(after) > 0;
    if (d0 == d1 && s0 == s1) return;
    if (now == 0) {
      now = std::max<int64_t>(1, now_ns() - origin_);
      clock_reads_.fetch_add(1, std::memory_order_relaxed);
    }
    edge(s.busy[0], d0, d1, now);
    edge(s.busy[1], s0, s1, now);
    edge(s.busy[2], d0 && s0, d1 && s1, now);
  }

  // `bytes` of kind `kind` landed on root `root`.
  void landed(int root, int kind, int64_t bytes) {
    if (bytes > 0) slots_[root].done[kind].fetch_add(bytes, std::memory_order_relaxed);
  }

  // Writes kDriveLoadWords words: the header, then each of the first `roots` roots' fields (relaxed reads; exact once
  // every reader is idle).
  void snapshot(int64_t* out, int64_t roots) const {
    std::fill(out, out + kDriveLoadWords, 0);
    roots = std::clamp<int64_t>(roots, 0, kMaxDrives);
    const int64_t now = std::max<int64_t>(1, now_ns() - origin_);
    out[0] = now;
    out[1] = clock_reads_.load(std::memory_order_relaxed);
    out[2] = roots;
    for (int64_t q = 0; q < roots; ++q) {
      const DriveSlot& s = slots_[q];
      int64_t* f = out + kDriveHeader + q * kDriveFields;
      const int64_t reads = s.reads.load(std::memory_order_relaxed);
      const bool d = demand_of(reads) > 0, p = spec_of(reads) > 0;
      f[0] = demand_of(reads);
      f[1] = spec_of(reads);
      for (int k = 0; k < 2; ++k) {
        f[2 + k] = s.inflight[k].load(std::memory_order_relaxed);
        f[4 + k] = s.done[k].load(std::memory_order_relaxed);
      }
      const bool up[3] = {d, p, d && p};
      for (int k = 0; k < 3; ++k)
        f[6 + k] = std::clamp<int64_t>(s.busy[k].load(std::memory_order_relaxed) + (up[k] ? now : 0), 0, now);
    }
  }

 private:
  static int64_t demand_of(int64_t reads) {
    return static_cast<int32_t>(static_cast<uint32_t>(reads));
  }
  static int64_t spec_of(int64_t reads) {
    return (reads - demand_of(reads)) >> 32;
  }
  static void edge(std::atomic<int64_t>& sum, bool was, bool is, int64_t now) {
    if (was != is) sum.fetch_add(is ? -now : now, std::memory_order_relaxed);
  }

  DriveSlot slots_[kMaxDrives];
  const int64_t origin_;
  alignas(64) std::atomic<int64_t> clock_reads_{0};
};

}  // namespace expert_stream
}  // namespace sglang
