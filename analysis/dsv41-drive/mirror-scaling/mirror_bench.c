// mirror_bench: do expert row reads scale with the number of NVMe mirror roots, and which io_uring settings read them
// fastest?
//
// Disk only, no GPU. Reads whole EXL3 row images the way the production direct-mode reader does with piece streaming
// on (SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM=1):
//   - the row's page-rounded row_stride is split across the chosen roots by weight exactly as
//     exl3_read_split.ReadSplit does (pages floored per weight, the leftover pages to the first largest weight), and
//     each part is clipped at image_bytes (exl3_ram_miss.py, row-image tables); a zero-length part reads nothing;
//   - each reading part is cut into sub_reads_per_part(reading) = min(4, 8 / reading) sub-reads by split_part
//     (piece_geometry.h): len_k = round_up(ceil(len / per_part), 4096), the last takes the rest;
//   - each sub-read is one O_DIRECT READV whose iovecs are the destination slab rows (6 slabs, one per streamed name,
//     SLOTS rows each), as RowReader::image_iovecs.
// The io_uring options mirror master's UringOptions / UringReader / ReaderCore (ba01695c35):
//   --mode default|iopoll|sqpoll|sqpoll_iopoll, --sq-cpu N (SQ_AFF), --sq-idle-ms (default 10000);
//   --wait block|spin: block = io_uring_submit_and_wait(1); spin = submit then poll the CQ; IOPOLL without SQPOLL
//     always takes master's reap loop (submit, then io_uring_get_events passes, min_complete=0) whatever --wait says;
//   --cuts: each sub-read is cut into legs its device takes whole (read_cuts.h cut_legs: at cut_bytes =
//     min(max_sectors_kb, (max_segments - 1) pages) and at every iovec join off virt_boundary_mask), one SQE per leg;
//   --read-mode normal|readv_fixed|fixed with --arena 0|1: registered buffers. arena 0 registers each of the 6 slabs
//     as its own buffer (production SLAB_ARENA=0: a READV_FIXED leg is the run of iovecs in one buffer, so a sub-read
//     fans out per slab, as UringReader::fixed_legs); arena 1 places the 6 slabs in one allocation registered as one
//     buffer (SLAB_ARENA=1). --fixed-files registers the 40 x roots files;
//   --ring N: ring entries; 0 = master's default queue_depth(): 16 * parts * (cuts ? leg_stride : 1). The ring depth
//     is also the SQE credit: a row starts only when its SQEs fit beside those in flight.
// Rows are drawn uniformly at random over (layer 0..39, expert 0..383) with a fixed seed, so every setting reads the
// same rows. QD = rows in flight; every SQE of a row is submitted together.
//
// Per cell it prints one JSON line: row latency percentiles, GB/s, CPU (submitting thread, SQ thread, io-wq workers,
// from /proc/self/task/*/schedstat), and per root: bytes, SQE latency (row submit to CQE) p50/p99, part completion
// (the part's last CQE) p50/p99, disk utilization (io_ticks delta / wall), disk bytes (which exposes foreign I/O),
// interrupts, and how often that root's part finished last in a row (the straggler share). --raw appends one CSV line
// per row.
//
//   gcc -O2 -pthread -o mirror_bench mirror_bench.c -luring
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <liburing.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#define PAGE 4096ULL
#define MAX_ROOTS 3
#define LAYERS 40
#define EXPERTS 384
#define NSEG 6
#define SLOTS 16
#define K_SUBREADS 4
#define K_PIECES 8
#define MAX_SQE_PER_ROW 512
#define MAX_IOV_PER_ROW 2048
#define MAX_QD 16

// layer-000.rows.digests.json (identical layout on every layer and root)
static const uint64_t seg_bytes[NSEG] = {8847360, 20480, 9216, 4423680, 4608, 10240};
static const uint64_t seg_off[NSEG] = {0, 8847360, 8867840, 8877056, 13300736, 13305344};
static const uint64_t image_bytes = 13315584, row_stride = 13316096;

enum { RM_NORMAL, RM_READV_FIXED, RM_FIXED };
static int nroots = 0, qd = 1, spin_wait = 0, iopoll = 0, sqpoll = 0, sq_cpu = -1, cuts = 0, read_mode = RM_NORMAL,
           arena = 0, fixed_files = 0;
static unsigned sq_idle_ms = 10000, ring_entries = 0;
static const char* mode_name = "default";
static const char* roots[MAX_ROOTS];
static double weights[MAX_ROOTS];
static int fds[MAX_ROOTS][LAYERS];
static char disk[MAX_ROOTS][64], ctrl[MAX_ROOTS][64];
static uint64_t part_lo[MAX_ROOTS], part_hi[MAX_ROOTS];
static uint64_t cut_bytes[MAX_ROOTS], virt_mask[MAX_ROOTS];
static int per_part = K_SUBREADS;
static char* slab[NSEG];
static char* buf_base[NSEG];
static uint64_t buf_len[NSEG];
static int nbufs = 0;

static double now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}
static uint64_t rng(uint64_t* s) {
  *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
  return *s;
}
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return x < y ? -1 : x > y; }
static double pct(double* v, size_t n, double p) { return n ? v[(size_t)(p * (n - 1))] : 0.0; }
static long long read_num(const char* path) {
  FILE* f = fopen(path, "r"); long long v = -1;
  if (f) { if (fscanf(f, "%lld", &v) != 1) v = -1; fclose(f); }
  return v;
}

// ---- the production split (exl3_read_split.ReadSplit + exl3_ram_miss row-image clip + sub_reads_per_part) ----
static void plan_parts(void) {
  uint64_t total_pages = row_stride / PAGE, pages[MAX_ROOTS], sum = 0;
  double tw = 0; int largest = 0;
  for (int r = 0; r < nroots; ++r) { tw += weights[r]; if (weights[r] > weights[largest]) largest = r; }
  for (int r = 0; r < nroots; ++r) { pages[r] = (uint64_t)((double)total_pages * weights[r] / tw); sum += pages[r]; }
  pages[largest] += total_pages - sum;
  uint64_t at = 0; int reading = 0;
  for (int r = 0; r < nroots; ++r) {
    uint64_t lo = at, hi = at + pages[r] * PAGE; at = hi;
    part_lo[r] = lo < image_bytes ? lo : image_bytes;
    part_hi[r] = hi < image_bytes ? hi : image_bytes;
    reading += part_hi[r] > part_lo[r];
  }
  per_part = K_PIECES / reading < K_SUBREADS ? K_PIECES / reading : K_SUBREADS;
}

typedef struct { int root, buf; uint64_t off, len; int iovcnt; struct iovec* iov; } Sqe;

static int buffer_of(const void* p, size_t len) {
  for (int b = 0; b < nbufs; ++b)
    if ((const char*)p >= buf_base[b] && (const char*)p + len <= buf_base[b] + buf_len[b]) return b;
  fprintf(stderr, "iovec in no registered buffer\n"); exit(1);
}

// read_cuts.h cut_legs: a new leg at every join off the virt boundary, and wherever a leg reaches cut_bytes (splitting
// that iovec). Writes legs' iovecs to out, leg starts/counts to leg_first/leg_count; returns the leg count.
static int cut_legs(const struct iovec* in, int count, uint64_t cb, uint64_t mask, struct iovec* out, int* leg_first,
                    int* leg_count, uint64_t* leg_bytes) {
  int n = 0, k = 0;
#define ON_B(a) (mask == 0 || ((a) & mask) == 0)
  for (int i = 0; i < count; ++i) {
    char* at = in[i].iov_base; size_t left = in[i].iov_len;
    uintptr_t prev_end = i > 0 ? (uintptr_t)in[i - 1].iov_base + in[i - 1].iov_len : 0;
    int gap = i > 0 && prev_end != (uintptr_t)at && !(ON_B(prev_end) && ON_B((uintptr_t)at));
    int open_leg = n == 0 || gap;
    while (left > 0) {
      if (!open_leg && leg_bytes[n - 1] >= cb) open_leg = 1;
      if (open_leg) { leg_first[n] = k; leg_count[n] = 0; leg_bytes[n] = 0; ++n; open_leg = 0; }
      size_t take = left < cb - leg_bytes[n - 1] ? left : cb - leg_bytes[n - 1];
      out[k].iov_base = at; out[k].iov_len = take; ++k;
      ++leg_count[n - 1]; leg_bytes[n - 1] += take; at += take; left -= take;
    }
  }
#undef ON_B
  return n;
}

// The row's SQEs: split_part per reading part; per sub-read, legs (cuts or one), then per leg the fixed-buffer runs
// (readv_fixed: consecutive iovecs in one buffer; fixed: one per iovec). Returns the SQE count; *max_leg_out gets the
// widest sub-read's SQE count (leg_stride).
static int build_row(int expert, int slot, Sqe* out, struct iovec* iv, int* max_leg_out) {
  int nsq = 0, niv = 0, max_leg = 0;
  for (int r = 0; r < nroots; ++r) {
    uint64_t len = part_hi[r] - part_lo[r];
    if (!len) continue;
    uint64_t lk = ((len + per_part - 1) / per_part + PAGE - 1) / PAGE * PAGE;
    for (uint64_t a = part_lo[r]; a < part_hi[r]; a += lk) {
      uint64_t b = a + lk < part_hi[r] ? a + lk : part_hi[r];
      struct iovec first[NSEG]; int cnt = 0;
      for (int s = 0; s < NSEG; ++s) {
        uint64_t lo = a > seg_off[s] ? a : seg_off[s], hi = b < seg_off[s] + seg_bytes[s] ? b : seg_off[s] + seg_bytes[s];
        if (lo >= hi) continue;
        first[cnt].iov_base = slab[s] + slot * seg_bytes[s] + (lo - seg_off[s]);
        first[cnt].iov_len = hi - lo; ++cnt;
      }
      struct iovec legiv[256]; int lf[128], lc[128]; uint64_t lb[128]; int nl;
      if (cuts) nl = cut_legs(first, cnt, cut_bytes[r], virt_mask[r], legiv, lf, lc, lb);
      else { memcpy(legiv, first, cnt * sizeof(struct iovec)); lf[0] = 0; lc[0] = cnt; nl = 1; }
      uint64_t off = (uint64_t)expert * row_stride + a;
      int before = nsq;
      for (int l = 0; l < nl; ++l) {
        int i = lf[l], end = lf[l] + lc[l];
        while (i < end) {
          Sqe* q = &out[nsq++];
          q->root = r; q->off = off; q->iov = &iv[niv]; q->iovcnt = 0; q->len = 0; q->buf = -1;
          if (read_mode != RM_NORMAL) q->buf = buffer_of(legiv[i].iov_base, legiv[i].iov_len);
          do {
            iv[niv++] = legiv[i]; q->iovcnt++; q->len += legiv[i].iov_len; ++i;
          } while (i < end && (read_mode == RM_NORMAL ||
                               (read_mode == RM_READV_FIXED && buffer_of(legiv[i].iov_base, legiv[i].iov_len) == q->buf)));
          off += q->len;
          if (nsq >= MAX_SQE_PER_ROW || niv >= MAX_IOV_PER_ROW - 64) { fprintf(stderr, "row too wide\n"); exit(1); }
        }
      }
      if (nsq - before > max_leg) max_leg = nsq - before;
    }
  }
  if (max_leg_out) *max_leg_out = max_leg;
  return nsq;
}

// ---- /sys/block/<disk>/stat, queue limits, /proc/interrupts ----
typedef struct { uint64_t rd_ios, rd_sec, ticks, inflight_ms; uint64_t irq, irq_hi; } Dstat;
static void disk_of(int r) {
  struct stat st; fstat(fds[r][0], &st);
  char p[PATH_MAX], rp[PATH_MAX];
  snprintf(p, sizeof p, "/sys/dev/block/%u:%u", major(st.st_dev), minor(st.st_dev));
  if (!realpath(p, rp)) { perror(p); exit(1); }
  char part[PATH_MAX + 16]; snprintf(part, sizeof part, "%s/partition", rp);
  if (access(part, F_OK) == 0) { char* d = dirname(rp); memmove(rp, d, strlen(d) + 1); }
  snprintf(disk[r], sizeof disk[r], "%s", basename(rp));
  char dev[PATH_MAX], cp[PATH_MAX];
  snprintf(dev, sizeof dev, "/sys/block/%s/device", disk[r]);
  if (!realpath(dev, cp)) { perror(dev); exit(1); }
  snprintf(ctrl[r], sizeof ctrl[r], "%s", basename(cp));
  // read_cuts.h cut_bytes_for / limits_from_queue_dir
  char q[256];
  snprintf(q, sizeof q, "/sys/block/%s/queue/max_sectors_kb", disk[r]); long long kb = read_num(q);
  snprintf(q, sizeof q, "/sys/block/%s/queue/max_segments", disk[r]); long long segs = read_num(q);
  snprintf(q, sizeof q, "/sys/block/%s/queue/virt_boundary_mask", disk[r]); long long mask = read_num(q);
  if (kb <= 0 || segs <= 0) { fprintf(stderr, "%s: queue limits unreadable\n", disk[r]); exit(1); }
  uint64_t bytes = (uint64_t)kb * 1024;
  if (segs > 1 && (uint64_t)(segs - 1) * PAGE < bytes) bytes = (uint64_t)(segs - 1) * PAGE;
  bytes = bytes / PAGE * PAGE; if (bytes < PAGE) bytes = PAGE;
  cut_bytes[r] = bytes; virt_mask[r] = mask < 0 ? 4095 : (uint64_t)mask;
}
static void dstat(Dstat* d) {
  for (int r = 0; r < nroots; ++r) {
    char p[256]; snprintf(p, sizeof p, "/sys/block/%s/stat", disk[r]);
    FILE* f = fopen(p, "r");
    unsigned long long v[11] = {0};
    if (f) { if (fscanf(f, "%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8], &v[9], &v[10]) != 11) {} fclose(f); }
    d[r].rd_ios = v[0]; d[r].rd_sec = v[2]; d[r].ticks = v[9]; d[r].inflight_ms = v[10]; d[r].irq = d[r].irq_hi = 0;
  }
  FILE* f = fopen("/proc/interrupts", "r");
  static char line[8192];
  if (!fgets(line, sizeof line, f)) { fclose(f); return; }
  int ncpu = 0; for (char* t = strtok(line, " \t\n"); t; t = strtok(NULL, " \t\n")) ++ncpu;
  while (fgets(line, sizeof line, f)) {
    char* last = strrchr(line, ' '); if (!last) continue;
    char name[64]; if (sscanf(last + 1, "%63s", name) != 1) continue;
    for (int r = 0; r < nroots; ++r) {
      size_t L = strlen(ctrl[r]);
      if (strncmp(name, ctrl[r], L) || name[L] != 'q') continue;
      char* s = strchr(line, ':'); if (!s) break; ++s;
      for (int c = 0; c < ncpu; ++c) {
        char* e; unsigned long long x = strtoull(s, &e, 10); if (e == s) break; s = e;
        d[r].irq += x; if (c >= 64) d[r].irq_hi += x;
      }
    }
  }
  fclose(f);
}

// ---- io_uring kernel threads of this process: SQ thread (iou-sqp-*) and io-wq workers (iou-wrk-*) ----
#define MAXT 1024
static int t_tid[MAXT], t_kind[MAXT], nt = 0;  // kind 1 sqp, 2 wrk
static double t_first[MAXT], t_last[MAXT];
static atomic_int stop_sampler;
static double thread_ns(const char* tid) {
  char p[128]; snprintf(p, sizeof p, "/proc/self/task/%s/schedstat", tid);
  FILE* f = fopen(p, "r"); unsigned long long ns = 0;
  if (f) { if (fscanf(f, "%llu", &ns) != 1) ns = 0; fclose(f); }
  return (double)ns;
}
static void sample_threads(void) {
  DIR* d = opendir("/proc/self/task"); struct dirent* e;
  while ((e = readdir(d))) {
    if (e->d_name[0] == '.') continue;
    char p[128], comm[64] = {0}; snprintf(p, sizeof p, "/proc/self/task/%s/comm", e->d_name);
    FILE* f = fopen(p, "r"); if (!f) continue;
    if (!fgets(comm, sizeof comm, f)) comm[0] = 0;
    fclose(f);
    int kind = !strncmp(comm, "iou-sqp", 7) ? 1 : !strncmp(comm, "iou-wrk", 7) ? 2 : 0;
    if (!kind) continue;
    int tid = atoi(e->d_name); double ns = thread_ns(e->d_name);
    int k; for (k = 0; k < nt && t_tid[k] != tid; ++k) {}
    if (k == nt && nt < MAXT) { t_tid[nt] = tid; t_kind[nt] = kind; t_first[nt] = ns; t_last[nt] = ns; ++nt; }
    else if (k < nt && ns > t_last[k]) t_last[k] = ns;
  }
  closedir(d);
}
static void* sampler(void* arg) {
  (void)arg;
  while (!atomic_load(&stop_sampler)) { sample_threads(); usleep(20000); }
  return NULL;
}

int main(int argc, char** argv) {
  const char* label = ""; const char* raw = NULL; const char* wstr = NULL;
  long rows = 2000; uint64_t seed = 1;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i]; const char* v = i + 1 < argc ? argv[i + 1] : "";
    if (!strcmp(a, "--root")) { if (nroots < MAX_ROOTS) roots[nroots++] = v; ++i; }
    else if (!strcmp(a, "--weights")) { wstr = v; ++i; }
    else if (!strcmp(a, "--qd")) { qd = atoi(v); ++i; }
    else if (!strcmp(a, "--rows")) { rows = atol(v); ++i; }
    else if (!strcmp(a, "--seed")) { seed = strtoull(v, 0, 0); ++i; }
    else if (!strcmp(a, "--wait")) { spin_wait = !strcmp(v, "spin"); ++i; }
    else if (!strcmp(a, "--mode")) {
      mode_name = v; ++i;
      if (!strcmp(v, "default")) {} else if (!strcmp(v, "iopoll")) iopoll = 1; else if (!strcmp(v, "sqpoll")) sqpoll = 1;
      else if (!strcmp(v, "sqpoll_iopoll")) iopoll = sqpoll = 1; else { fprintf(stderr, "bad --mode\n"); return 2; }
    }
    else if (!strcmp(a, "--sq-cpu")) { sq_cpu = atoi(v); ++i; }
    else if (!strcmp(a, "--sq-idle-ms")) { sq_idle_ms = (unsigned)atoi(v); ++i; }
    else if (!strcmp(a, "--cuts")) { cuts = 1; }
    else if (!strcmp(a, "--read-mode")) {
      ++i; if (!strcmp(v, "normal")) read_mode = RM_NORMAL; else if (!strcmp(v, "readv_fixed")) read_mode = RM_READV_FIXED;
      else if (!strcmp(v, "fixed")) read_mode = RM_FIXED; else { fprintf(stderr, "bad --read-mode\n"); return 2; }
    }
    else if (!strcmp(a, "--arena")) { arena = atoi(v); ++i; }
    else if (!strcmp(a, "--fixed-files")) { fixed_files = 1; }
    else if (!strcmp(a, "--ring")) { ring_entries = (unsigned)atoi(v); ++i; }
    else if (!strcmp(a, "--label")) { label = v; ++i; }
    else if (!strcmp(a, "--raw")) { raw = v; ++i; }
    else { fprintf(stderr, "unknown arg %s\n", a); return 2; }
  }
  if (!nroots || qd < 1 || qd > MAX_QD || rows < 1 || !seed || (sq_cpu >= 0 && !sqpoll)) { fprintf(stderr, "bad args\n"); return 2; }
  for (int r = 0; r < nroots; ++r) weights[r] = 1.0;
  if (wstr) {
    char buf[256]; snprintf(buf, sizeof buf, "%s", wstr); int n = 0;
    for (char* t = strtok(buf, ":"); t && n < MAX_ROOTS; t = strtok(NULL, ":")) weights[n++] = atof(t);
    if (n != nroots) { fprintf(stderr, "--weights needs %d values\n", nroots); return 2; }
  }
  for (int r = 0; r < nroots; ++r)
    for (int l = 0; l < LAYERS; ++l) {
      char p[PATH_MAX]; snprintf(p, sizeof p, "%s/exl3_row_images/layer-%03d.rows", roots[r], l);
      fds[r][l] = open(p, O_RDONLY | O_DIRECT | O_CLOEXEC);
      if (fds[r][l] < 0) { perror(p); return 1; }
      struct stat st; fstat(fds[r][l], &st);
      if ((uint64_t)st.st_size < EXPERTS * row_stride - (row_stride - image_bytes)) { fprintf(stderr, "%s short\n", p); return 1; }
    }
  for (int r = 0; r < nroots; ++r) disk_of(r);
  plan_parts();
  // Slabs: separate 2 MiB-aligned THP allocations (arena 0), or one allocation holding all six (arena 1).
  size_t slab_sz[NSEG], total = 0;
  for (int s = 0; s < NSEG; ++s) { slab_sz[s] = (SLOTS * seg_bytes[s] + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1); total += slab_sz[s]; }
  if (arena) {
    char* base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    madvise(base, total, MADV_HUGEPAGE); memset(base, 0, total);
    size_t at = 0; for (int s = 0; s < NSEG; ++s) { slab[s] = base + at; at += slab_sz[s]; }
    buf_base[0] = base; buf_len[0] = total; nbufs = 1;
  } else {
    for (int s = 0; s < NSEG; ++s) {
      slab[s] = mmap(NULL, slab_sz[s], PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (slab[s] == MAP_FAILED) { perror("mmap"); return 1; }
      madvise(slab[s], slab_sz[s], MADV_HUGEPAGE); memset(slab[s], 0, slab_sz[s]);
      buf_base[s] = slab[s]; buf_len[s] = slab_sz[s];
    }
    nbufs = NSEG;
  }
  // leg_stride: the widest sub-read's SQE count over every slot (the slab offsets change the gap joins).
  static Sqe probe_sq[MAX_SQE_PER_ROW]; static struct iovec probe_iv[MAX_IOV_PER_ROW];
  int leg_stride = 1, row_sqes_max = 0;
  for (int slot = 0; slot < SLOTS; ++slot) {
    int ml; int n = build_row(0, slot, probe_sq, probe_iv, &ml);
    if (ml > leg_stride) leg_stride = ml;
    if (n > row_sqes_max) row_sqes_max = n;
  }
  int reading = 0; for (int r = 0; r < nroots; ++r) reading += part_hi[r] > part_lo[r];
  unsigned depth = ring_entries ? ring_entries : (unsigned)(16 * reading * (cuts ? leg_stride : 1));
  struct io_uring ring;
  struct io_uring_params prm; memset(&prm, 0, sizeof prm);
  if (sqpoll) {
    prm.flags |= IORING_SETUP_SQPOLL; prm.sq_thread_idle = sq_idle_ms;
    if (sq_cpu >= 0) { prm.flags |= IORING_SETUP_SQ_AFF; prm.sq_thread_cpu = (unsigned)sq_cpu; }
  }
  if (iopoll) prm.flags |= IORING_SETUP_IOPOLL;
  int rc = io_uring_queue_init_params(depth, &ring, &prm);
  if (rc < 0) { fprintf(stderr, "queue_init(%u): %s\n", depth, strerror(-rc)); return 1; }
  unsigned sq_entries = prm.sq_entries;
  if (fixed_files) {
    int all[MAX_ROOTS * LAYERS]; for (int r = 0; r < nroots; ++r) for (int l = 0; l < LAYERS; ++l) all[r * LAYERS + l] = fds[r][l];
    rc = io_uring_register_files(&ring, all, nroots * LAYERS);
    if (rc < 0) { fprintf(stderr, "register_files: %s\n", strerror(-rc)); return 1; }
  }
  if (read_mode != RM_NORMAL) {
    struct iovec b[NSEG]; for (int i = 0; i < nbufs; ++i) { b[i].iov_base = buf_base[i]; b[i].iov_len = buf_len[i]; }
    rc = io_uring_register_buffers(&ring, b, nbufs);
    if (rc < 0) { fprintf(stderr, "register_buffers: %s\n", strerror(-rc)); return 1; }
  }
  int polls_in_wait = iopoll && !sqpoll, blocking = !spin_wait && !polls_in_wait;
  const char* eff_wait = blocking ? "block" : polls_in_wait ? "reap" : "spin";

  typedef struct { int busy, left, layer, expert, nsq; long idx; double t0, part_end[MAX_ROOTS]; Sqe sq[MAX_SQE_PER_ROW]; struct iovec iv[MAX_IOV_PER_ROW]; } Op;
  static Op ops[MAX_QD];
  double* lat = calloc(rows, sizeof(double));
  double* pend[MAX_ROOTS]; double* sqe_lat[MAX_ROOTS]; size_t nsqe[MAX_ROOTS] = {0};
  uint64_t bytes_root[MAX_ROOTS] = {0};
  long last_cnt[MAX_ROOTS] = {0}; double* lead[MAX_ROOTS]; size_t nlead[MAX_ROOTS] = {0};
  for (int r = 0; r < nroots; ++r) {
    pend[r] = calloc(rows, sizeof(double)); sqe_lat[r] = calloc(rows * (size_t)row_sqes_max, sizeof(double)); lead[r] = calloc(rows, sizeof(double));
  }
  FILE* rawf = raw ? fopen(raw, "a") : NULL;
  uint64_t s = seed; long issued = 0, done = 0, errors = 0, shorts = 0, total_sqes = 0, eagain = 0;
  unsigned outstanding = 0;
  Dstat d0[MAX_ROOTS], d1[MAX_ROOTS];
  sample_threads();  // baseline for the SQ thread (it exists from ring creation)
  struct timespec ct0, ct1; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ct0);
  struct rusage ru0, ru1; getrusage(RUSAGE_SELF, &ru0);
  pthread_t sp; pthread_create(&sp, 0, sampler, 0);
  dstat(d0);
  double t_start = now_us();
  while (done < rows) {
    for (int i = 0; i < qd && issued < rows; ++i) {
      Op* o = &ops[i];
      if (o->busy) continue;
      // peek the next row without consuming the rng if it does not fit the credit
      uint64_t s2 = s; int layer = (int)(rng(&s2) % LAYERS), expert = (int)(rng(&s2) % EXPERTS);
      int slot = (int)(issued % SLOTS);
      int n = build_row(expert, slot, o->sq, o->iv, NULL);
      if (outstanding > 0 && outstanding + (unsigned)n > sq_entries) break;
      s = s2; o->layer = layer; o->expert = expert; o->idx = issued++; o->nsq = n;
      o->busy = 1; o->left = n;
      for (int r = 0; r < nroots; ++r) o->part_end[r] = 0;
      o->t0 = now_us();
      for (int k = 0; k < n; ++k) {
        Sqe* q = &o->sq[k];
        struct io_uring_sqe* e = io_uring_get_sqe(&ring);
        if (!e) { io_uring_submit(&ring); e = io_uring_get_sqe(&ring); }
        if (!e) { fprintf(stderr, "no sqe\n"); return 1; }
        int fd = fixed_files ? q->root * LAYERS + o->layer : fds[q->root][o->layer];
        if (read_mode == RM_NORMAL) io_uring_prep_readv(e, fd, q->iov, q->iovcnt, q->off);
        else if (read_mode == RM_READV_FIXED) io_uring_prep_readv_fixed(e, fd, q->iov, q->iovcnt, q->off, 0, q->buf);
        else io_uring_prep_read_fixed(e, fd, q->iov[0].iov_base, (unsigned)q->iov[0].iov_len, q->off, q->buf);
        if (fixed_files) e->flags |= IOSQE_FIXED_FILE;
        io_uring_sqe_set_data64(e, ((uint64_t)i << 40) | ((uint64_t)q->root << 32) | q->len);
      }
      total_sqes += n; outstanding += n;
    }
    if (blocking) rc = io_uring_submit_and_wait(&ring, 1);
    else {
      rc = io_uring_submit(&ring);
      while (rc >= 0 && io_uring_cq_ready(&ring) == 0) {
        if (io_uring_sq_ready(&ring) != 0) { int r2 = io_uring_submit(&ring); if (r2 < 0) rc = r2; }
        if (polls_in_wait) { int g = io_uring_get_events(&ring); if (g < 0) rc = g; }
        __builtin_ia32_pause();
      }
    }
    if (rc < 0 && rc != -EINTR && rc != -EAGAIN && rc != -EBUSY) { fprintf(stderr, "submit: %s\n", strerror(-rc)); return 1; }
    struct io_uring_cqe* c; unsigned head, seen = 0;
    double tn = now_us();
    io_uring_for_each_cqe(&ring, head, c) {
      uint64_t dd = io_uring_cqe_get_data64(c);
      Op* o = &ops[dd >> 40]; int r = (int)((dd >> 32) & 0xff); uint64_t want = dd & 0xffffffffULL;
      if (c->res < 0) { ++errors; if (c->res == -EAGAIN) ++eagain; if (errors == 1) fprintf(stderr, "cqe error %s\n", strerror(-c->res)); }
      else if ((uint64_t)c->res != want) ++shorts;
      double el = tn - o->t0;
      sqe_lat[r][nsqe[r]++] = el; bytes_root[r] += want;
      if (el > o->part_end[r]) o->part_end[r] = el;
      --outstanding;
      if (--o->left == 0) {
        o->busy = 0;
        lat[done] = el;
        int last = -1, readers = 0; double best = -1, second = -1;
        for (int q = 0; q < nroots; ++q) {
          if (part_hi[q] == part_lo[q]) continue;
          ++readers; pend[q][done] = o->part_end[q];
          if (o->part_end[q] > best) { second = best; best = o->part_end[q]; last = q; }
          else if (o->part_end[q] > second) second = o->part_end[q];
        }
        if (readers > 1) { ++last_cnt[last]; lead[last][nlead[last]++] = best - second; }
        if (rawf) {
          fprintf(rawf, "%s,%ld,%d,%d,%.1f", label, o->idx, o->layer, o->expert, el);
          for (int q = 0; q < MAX_ROOTS; ++q) fprintf(rawf, ",%.1f", q < nroots ? o->part_end[q] : 0.0);
          fputc('\n', rawf);
        }
        ++done;
      }
      ++seen;
    }
    io_uring_cq_advance(&ring, seen);
  }
  double wall = (now_us() - t_start) / 1e6;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ct1);
  dstat(d1);
  atomic_store(&stop_sampler, 1); pthread_join(sp, 0);
  sample_threads();
  getrusage(RUSAGE_SELF, &ru1);
  if (rawf) fclose(rawf);
  double sub_cpu = (ct1.tv_sec - ct0.tv_sec) + (ct1.tv_nsec - ct0.tv_nsec) / 1e9;
  double sqp_cpu = 0, wrk_cpu = 0; int nwrk = 0;
  for (int k = 0; k < nt; ++k) {
    if (t_kind[k] == 1) sqp_cpu += (t_last[k] - t_first[k]) / 1e9;
    else { wrk_cpu += t_last[k] / 1e9; ++nwrk; }  // workers are born during the run: count all their time
  }
  double proc_cpu = (ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec) + (ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec) / 1e6 +
                    (ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec) + (ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec) / 1e6;
  uint64_t tb = 0; for (int r = 0; r < nroots; ++r) tb += bytes_root[r];
  double mean = 0; for (long i = 0; i < rows; ++i) mean += lat[i] / rows;
  qsort(lat, rows, sizeof(double), cmpd);
  double gb = tb / 1e9;
  printf("{\"label\":\"%s\",\"roots\":\"", label);
  for (int r = 0; r < nroots; ++r) printf("%s%s", r ? "," : "", disk[r]);
  printf("\",\"weights\":\"");
  for (int r = 0; r < nroots; ++r) printf("%s%g", r ? ":" : "", weights[r]);
  printf("\",\"mode\":\"%s\",\"sq_cpu\":%d,\"sq_idle_ms\":%u,\"wait\":\"%s\",\"cuts\":%d,\"read_mode\":\"%s\",\"arena\":%d,\"fixed_files\":%d,"
         "\"ring\":%u,\"leg_stride\":%d,\"qd\":%d,\"seed\":%llu,\"rows\":%ld,\"per_part\":%d,\"sqes_per_row\":%.2f,\"wall_s\":%.3f,"
         "\"GBps\":%.3f,\"p50_us\":%.1f,\"p90_us\":%.1f,\"p99_us\":%.1f,\"max_us\":%.1f,\"mean_us\":%.1f,\"errors\":%ld,\"eagain\":%ld,\"short\":%ld,"
         "\"cpu_s\":%.3f,\"submitter_cpu_s\":%.3f,\"sqthread_cpu_s\":%.3f,\"iowq_cpu_s\":%.3f,\"iowq_workers\":%d,"
         "\"submitter_cpu_per_GB\":%.4f,\"sqthread_cpu_per_GB\":%.4f,\"iowq_cpu_per_GB\":%.4f,\"total_cpu_per_GB\":%.4f,\"drives\":[",
         mode_name, sq_cpu, sq_idle_ms, eff_wait, cuts, read_mode == RM_NORMAL ? "normal" : read_mode == RM_READV_FIXED ? "readv_fixed" : "fixed",
         arena, fixed_files, sq_entries, leg_stride, qd, (unsigned long long)seed, rows, per_part, (double)total_sqes / rows, wall, tb / wall / 1e9,
         pct(lat, rows, .5), pct(lat, rows, .9), pct(lat, rows, .99), lat[rows - 1], mean, errors, eagain, shorts,
         proc_cpu, sub_cpu, sqp_cpu, wrk_cpu, nwrk, sub_cpu / gb, sqp_cpu / gb, wrk_cpu / gb, (sub_cpu + sqp_cpu + wrk_cpu) / gb);
  for (int r = 0; r < nroots; ++r) {
    qsort(sqe_lat[r], nsqe[r], sizeof(double), cmpd);
    size_t np = part_hi[r] > part_lo[r] ? (size_t)rows : 0;
    qsort(pend[r], np, sizeof(double), cmpd);
    qsort(lead[r], nlead[r], sizeof(double), cmpd);
    double dsec = (double)(d1[r].rd_sec - d0[r].rd_sec) * 512;
    printf("%s{\"disk\":\"%s\",\"root\":\"%s\",\"weight\":%g,\"cut_bytes\":%llu,\"part_bytes\":%llu,\"bytes\":%llu,\"GBps\":%.3f,\"sqes\":%zu,"
           "\"sqe_p50_us\":%.1f,\"sqe_p99_us\":%.1f,\"part_p50_us\":%.1f,\"part_p90_us\":%.1f,\"part_p99_us\":%.1f,"
           "\"util\":%.3f,\"disk_MB\":%.1f,\"foreign_MB\":%.1f,\"disk_reqs\":%llu,\"inflight_ms\":%llu,\"irqs\":%llu,\"irqs_cpu64plus\":%llu,"
           "\"last_share\":%.3f,\"last_lead_p50_us\":%.1f,\"last_lead_p90_us\":%.1f}",
           r ? "," : "", disk[r], roots[r], weights[r], (unsigned long long)cut_bytes[r], (unsigned long long)(part_hi[r] - part_lo[r]),
           (unsigned long long)bytes_root[r], bytes_root[r] / wall / 1e9, nsqe[r], pct(sqe_lat[r], nsqe[r], .5), pct(sqe_lat[r], nsqe[r], .99),
           pct(pend[r], np, .5), pct(pend[r], np, .9), pct(pend[r], np, .99),
           (double)(d1[r].ticks - d0[r].ticks) / (wall * 1e3), dsec / 1e6, (dsec - (double)bytes_root[r]) / 1e6,
           (unsigned long long)(d1[r].rd_ios - d0[r].rd_ios), (unsigned long long)(d1[r].inflight_ms - d0[r].inflight_ms),
           (unsigned long long)(d1[r].irq - d0[r].irq), (unsigned long long)(d1[r].irq_hi - d0[r].irq_hi),
           (double)last_cnt[r] / rows, pct(lead[r], nlead[r], .5), pct(lead[r], nlead[r], .9));
  }
  printf("]}\n");
  io_uring_queue_exit(&ring);
  return errors || shorts ? 3 : 0;
}
