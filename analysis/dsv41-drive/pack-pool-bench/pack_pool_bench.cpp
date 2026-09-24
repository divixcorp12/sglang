// Packing-pool microbenchmark: the real PackPool (exl3_ram_miss_pack_pool.h) driven the way a piece-streaming read
// drives it, with per-chunk stamps, so a piece's pack time splits into worker wake-up and copy.
//
// Why: in serving, the last piece of a single-row demand (1.66 MB, 8 chunks of ~208 KB on 8 workers) takes a median
// ~200 us from its landing to its last chunk's end, which is ~1.3 GB/s per worker against ~7 GB/s for one core's
// memcpy. The stage trace stamps only a row's earliest chunk start, so it cannot say which part is the wake of the
// slowest worker and which is the copy itself.
//
// Build (divix01, in a pulled worktree):
//   g++ -O2 -std=c++17 -pthread -I python/sglang/kernels/jit/csrc/moe
//     analysis/dsv41-drive/pack-pool-bench/pack_pool_bench.cpp -o /tmp/pack_pool_bench
// Run under the server's cores, since the pool inherits the creating thread's affinity:
//   taskset -c 0-7,16-17,36-53 /tmp/pack_pool_bench --gap-us 250
//
// Model of one read: `--pieces` pieces posted `--gap-us` apart (the sub-read landing cadence), then `--read-gap-us`
// of idle before the next read (the decode step between read layers). The owner sleeps between posts, as the real
// owner blocks in io_uring_submit_and_wait when nothing is packing, and polls done() while a piece packs. Sources
// and destinations walk rings far larger than the LLC, so every copy is cold at both ends, as a freshly DMA'd bounce
// slot and a slab row are.

#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "exl3_ram_miss_pack_pool.h"

using sglang::exl3_ram_miss::ChunkStamp;
using sglang::exl3_ram_miss::CopyRun;
using sglang::exl3_ram_miss::PackJob;
using sglang::exl3_ram_miss::PackPool;

namespace {

int64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

int64_t clock_fn(const void*) { return now_ns(); }

void sleep_until(int64_t t) {
  timespec ts{static_cast<time_t>(t / 1000000000), static_cast<long>(t % 1000000000)};
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) != 0) {
  }
}

struct Options {
  unsigned workers = 8;
  unsigned split = 8;
  int64_t piece_bytes = 1665024;  // 13,320,192 / 8: one piece of a DSV4.1 row
  int runs = 2;                   // a piece is a contiguous file range; it usually meets one or two segments
  int pieces = 8;                 // per read: one single-row demand
  int reads = 300;
  int64_t gap_us = 250;
  int64_t read_gap_us = 30000;
  int64_t src_mb = 1024;
  int64_t dst_mb = 4096;
  int dst_node = -1;
  const char* csv = nullptr;
};

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    const char* v = argv[i + 1];
    if (k == "--workers") o.workers = static_cast<unsigned>(atoi(v));
    else if (k == "--split") o.split = static_cast<unsigned>(atoi(v));
    else if (k == "--piece-bytes") o.piece_bytes = atoll(v);
    else if (k == "--runs") o.runs = atoi(v);
    else if (k == "--pieces") o.pieces = atoi(v);
    else if (k == "--reads") o.reads = atoi(v);
    else if (k == "--gap-us") o.gap_us = atoll(v);
    else if (k == "--read-gap-us") o.read_gap_us = atoll(v);
    else if (k == "--src-mb") o.src_mb = atoll(v);
    else if (k == "--dst-mb") o.dst_mb = atoll(v);
    else if (k == "--dst-node") o.dst_node = atoi(v);
    else if (k == "--csv") o.csv = v;
    else {
      fprintf(stderr, "unknown option %s\n", k.c_str());
      exit(2);
    }
  }
  return o;
}

// Anonymous memory, huge-page advised, optionally bound to one NUMA node, then touched so no fault is timed.
uint8_t* ring(int64_t bytes, int node) {
  void* p = mmap(nullptr, static_cast<size_t>(bytes), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    perror("mmap");
    exit(1);
  }
  madvise(p, static_cast<size_t>(bytes), MADV_HUGEPAGE);
  if (node >= 0) {
    unsigned long mask = 1ul << node;
    if (syscall(SYS_mbind, p, static_cast<unsigned long>(bytes), 2 /* MPOL_BIND */, &mask, 64 + 1, 0) != 0) {
      perror("mbind");
      exit(1);
    }
  }
  memset(p, 0x5a, static_cast<size_t>(bytes));
  return static_cast<uint8_t*>(p);
}

double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(p / 100.0 * static_cast<double>(v.size())))];
}

void report(const char* name, const std::vector<double>& v) {
  printf("  %-44s n=%-6zu p10 %8.1f  p50 %8.1f  p90 %8.1f  max %8.1f\n", name, v.size(), pct(v, 10), pct(v, 50),
         pct(v, 90), v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()));
}

// One core's cold memcpy of `bytes`, repeated over the rings: the per-worker ceiling.
double memcpy_gbps(uint8_t* src, int64_t src_bytes, uint8_t* dst, int64_t dst_bytes, int64_t bytes, int reps) {
  int64_t s = 0, d = 0, total = 0;
  const int64_t t0 = now_ns();
  for (int i = 0; i < reps; ++i) {
    memcpy(dst + d, src + s, static_cast<size_t>(bytes));
    s = (s + bytes) % (src_bytes - bytes);
    d = (d + bytes) % (dst_bytes - bytes);
    total += bytes;
  }
  return static_cast<double>(total) / static_cast<double>(now_ns() - t0);
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  const int64_t src_bytes = o.src_mb << 20, dst_bytes = o.dst_mb << 20;
  uint8_t* src = ring(src_bytes, -1);
  uint8_t* dst = ring(dst_bytes, o.dst_node);

  cpu_set_t inherited;
  CPU_ZERO(&inherited);
  pthread_getaffinity_np(pthread_self(), sizeof(inherited), &inherited);
  const int64_t chunk = o.piece_bytes / o.split;
  printf("workers %u split %u piece %lld B (chunk ~%lld B) runs %d | %d pieces per read, gap %lld us, read gap %lld us,"
         " %d reads | dst node %d | %d allowed cpus\n",
         o.workers, o.split, static_cast<long long>(o.piece_bytes), static_cast<long long>(chunk), o.runs, o.pieces,
         static_cast<long long>(o.gap_us), static_cast<long long>(o.read_gap_us), o.reads, o.dst_node,
         CPU_COUNT(&inherited));
  printf("one-core cold memcpy: chunk %.2f GB/s, piece %.2f GB/s\n",
         memcpy_gbps(src, src_bytes, dst, dst_bytes, chunk, 2000),
         memcpy_gbps(src, src_bytes, dst, dst_bytes, o.piece_bytes, 300));

  PackPool pool(o.workers, inherited, 64);
  std::vector<CopyRun> runs(static_cast<size_t>(o.runs));
  std::vector<ChunkStamp> stamps(o.split);
  PackJob job;
  FILE* csv = o.csv ? fopen(o.csv, "w") : nullptr;
  if (csv) fprintf(csv, "read,piece,chunk,worker,cpu,post_ns,start_ns,end_ns,done_ns,bytes\n");

  // Per piece: when the first and the last chunk started, and when the last ended, all from the post; the owner's
  // view of done; and per chunk its copy rate. "First" pieces follow the read gap, the others the piece gap.
  std::vector<double> first_start[2], last_start[2], tail[2], done_seen[2], chunk_gbps, chunk_us;
  int64_t s_at = 0, d_at = 0;
  int64_t next = now_ns() + 1000000;
  for (int r = 0; r < o.reads; ++r) {
    for (int p = 0; p < o.pieces; ++p) {
      int64_t left = o.piece_bytes, at = 0;
      for (int i = 0; i < o.runs; ++i) {
        const int64_t bytes = i + 1 == o.runs ? left : o.piece_bytes / o.runs / 64 * 64;
        runs[static_cast<size_t>(i)] = CopyRun{dst + d_at + at, src + s_at + at, bytes};
        at += bytes;
        left -= bytes;
      }
      s_at = (s_at + o.piece_bytes + 4096) % (src_bytes - 2 * o.piece_bytes);
      d_at = (d_at + o.piece_bytes + 4096) % (dst_bytes - 2 * o.piece_bytes);
      std::fill(stamps.begin(), stamps.end(), ChunkStamp{});
      sleep_until(next);
      job.arm(runs.data(), runs.size(), o.split, 0, &clock_fn, nullptr, stamps.data());
      const int64_t post = now_ns();
      pool.post(&job);
      while (!job.done()) _mm_pause();
      const int64_t done = now_ns();
      const int kind = p == 0 ? 0 : 1;
      int64_t fs = INT64_MAX, ls = 0, le = 0;
      for (unsigned c = 0; c < o.split; ++c) {
        const ChunkStamp& s = stamps[c];
        fs = std::min(fs, s.start);
        ls = std::max(ls, s.start);
        le = std::max(le, s.end);
        // Chunks are equal to within 64 B (PackPool's edges), so `chunk` stands for each.
        chunk_us.push_back(static_cast<double>(s.end - s.start) / 1e3);
        chunk_gbps.push_back(static_cast<double>(chunk) / static_cast<double>(std::max<int64_t>(1, s.end - s.start)));
        if (csv) {
          fprintf(csv, "%d,%d,%u,%d,%d,%lld,%lld,%lld,%lld,%lld\n", r, p, c, s.worker, s.cpu, static_cast<long long>(post),
                  static_cast<long long>(s.start), static_cast<long long>(s.end), static_cast<long long>(done),
                  static_cast<long long>(chunk));
        }
      }
      first_start[kind].push_back(static_cast<double>(fs - post) / 1e3);
      last_start[kind].push_back(static_cast<double>(ls - post) / 1e3);
      tail[kind].push_back(static_cast<double>(le - post) / 1e3);
      done_seen[kind].push_back(static_cast<double>(done - post) / 1e3);
      next += (p + 1 == o.pieces ? o.read_gap_us : o.gap_us) * 1000;
      next = std::max(next, now_ns());
    }
  }
  if (csv) fclose(csv);
  const char* kinds[2] = {"first piece of a read (after the read gap)", "later pieces (after the piece gap)"};
  for (int k = 0; k < 2; ++k) {
    printf("%s, us from post:\n", kinds[k]);
    report("first chunk starts", first_start[k]);
    report("last chunk starts (slowest worker's claim)", last_start[k]);
    report("last chunk ends (the pack tail)", tail[k]);
    report("owner sees done", done_seen[k]);
  }
  printf("per chunk:\n");
  report("copy time, us", chunk_us);
  report("copy rate, GB/s", chunk_gbps);
  return 0;
}
