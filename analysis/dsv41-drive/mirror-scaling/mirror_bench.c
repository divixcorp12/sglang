// mirror_bench: do expert row reads scale with the number of NVMe mirror roots?
//
// Disk only, no GPU. Reads whole EXL3 row images the way the production direct-mode reader does with piece streaming
// on (SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM=1, MODE=default, WAIT_MODE=block, READ_CUTS=0):
//   - the row's page-rounded row_stride is split across the chosen roots by weight exactly as
//     exl3_read_split.ReadSplit does (pages floored per weight, the leftover pages to the first largest weight), and
//     each part is clipped at image_bytes (exl3_ram_miss.py, row-image tables); a zero-length part reads nothing;
//   - each reading part is cut into sub_reads_per_part(reading) = min(4, 8 / reading) sub-reads by split_part
//     (piece_geometry.h): len_k = round_up(ceil(len / per_part), 4096), the last takes the rest;
//   - each sub-read is ONE O_DIRECT IORING_OP_READV whose iovecs are the destination slab rows (6 slabs, one per
//     streamed name, SLOTS rows each), as RowReader::image_iovecs.
// Rows are drawn uniformly at random over (layer 0..39, expert 0..383) with a fixed seed, so every root set reads
// the same rows. QD = rows in flight; every SQE of a row is submitted together.
//
// Per cell it prints one JSON line: row latency percentiles, GB/s, and per root: bytes, SQE latency (row submit to
// CQE) p50/p99, part completion (the part's last CQE) p50/p99, disk utilization (io_ticks delta / wall), disk bytes
// (which exposes foreign I/O), interrupts, and how often that root's part finished last in a row (the straggler
// share) with the median lead it had over the next-to-last part. --raw appends one CSV line per row.
//
//   gcc -O2 -o mirror_bench mirror_bench.c -luring
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <liburing.h>
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
#define MAX_SQE_PER_ROW 32
#define MAX_QD 16

// layer-000.rows.digests.json (identical layout on every layer and root)
static const uint64_t seg_bytes[NSEG] = {8847360, 20480, 9216, 4423680, 4608, 10240};
static const uint64_t seg_off[NSEG] = {0, 8847360, 8867840, 8877056, 13300736, 13305344};
static const uint64_t image_bytes = 13315584, row_stride = 13316096;

static int nroots = 0, qd = 1, spin_wait = 0;
static const char* roots[MAX_ROOTS];
static double weights[MAX_ROOTS];
static int fds[MAX_ROOTS][LAYERS];
static char disk[MAX_ROOTS][64], ctrl[MAX_ROOTS][64];
static uint64_t part_lo[MAX_ROOTS], part_hi[MAX_ROOTS];
static int per_part = K_SUBREADS;
static char* slab[NSEG];

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

// ---- the production split (exl3_read_split.ReadSplit + exl3_ram_miss row-image clip + sub_reads_per_part) ----
static void plan_parts(void) {
  uint64_t total_pages = row_stride / PAGE, pages[MAX_ROOTS], sum = 0;
  double tw = 0; int largest = 0;
  for (int r = 0; r < nroots; ++r) { tw += weights[r]; if (weights[r] > weights[largest]) largest = r; }
  for (int r = 0; r < nroots; ++r) { pages[r] = (uint64_t)((double)total_pages * weights[r] / tw); sum += pages[r]; }
  // Python: int(total_pages * w // total_weight); floor of a float product, same as above for these magnitudes.
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

typedef struct { int root; uint64_t off, len; int iovcnt; struct iovec* iov; } Sqe;

// The row's SQEs: split_part per reading part, one READV per sub-read over the slab rows at `slot`.
static int build_row(int layer, int expert, int slot, Sqe* out, struct iovec* iv) {
  int nsq = 0, niv = 0;
  for (int r = 0; r < nroots; ++r) {
    uint64_t len = part_hi[r] - part_lo[r];
    if (!len) continue;
    uint64_t lk = ((len + per_part - 1) / per_part + PAGE - 1) / PAGE * PAGE;
    for (uint64_t a = part_lo[r]; a < part_hi[r]; a += lk) {
      uint64_t b = a + lk < part_hi[r] ? a + lk : part_hi[r];
      Sqe* q = &out[nsq++];
      q->root = r; q->off = (uint64_t)expert * row_stride + a; q->len = b - a; q->iov = &iv[niv]; q->iovcnt = 0;
      for (int s = 0; s < NSEG; ++s) {
        uint64_t lo = a > seg_off[s] ? a : seg_off[s], hi = b < seg_off[s] + seg_bytes[s] ? b : seg_off[s] + seg_bytes[s];
        if (lo >= hi) continue;
        iv[niv].iov_base = slab[s] + slot * seg_bytes[s] + (lo - seg_off[s]);
        iv[niv].iov_len = hi - lo;
        ++niv; ++q->iovcnt;
      }
      (void)layer;
    }
  }
  return nsq;
}

// ---- /sys/block/<disk>/stat and /proc/interrupts ----
typedef struct { uint64_t rd_ios, rd_sec, ticks, inflight_ms; uint64_t irq, irq_hi; } Dstat;
static void disk_of(int r) {
  struct stat st; fstat(fds[r][0], &st);
  char p[PATH_MAX], rp[PATH_MAX];
  snprintf(p, sizeof p, "/sys/dev/block/%u:%u", major(st.st_dev), minor(st.st_dev));
  if (!realpath(p, rp)) { perror(p); exit(1); }
  char part[PATH_MAX]; snprintf(part, sizeof part, "%s/partition", rp);
  if (access(part, F_OK) == 0) { char* d = dirname(rp); memmove(rp, d, strlen(d) + 1); }
  snprintf(disk[r], sizeof disk[r], "%s", basename(rp));
  char dev[PATH_MAX], cp[PATH_MAX];
  snprintf(dev, sizeof dev, "/sys/block/%s/device", disk[r]);
  if (!realpath(dev, cp)) { perror(dev); exit(1); }
  snprintf(ctrl[r], sizeof ctrl[r], "%s", basename(cp));
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
    else if (!strcmp(a, "--label")) { label = v; ++i; }
    else if (!strcmp(a, "--raw")) { raw = v; ++i; }
    else { fprintf(stderr, "unknown arg %s\n", a); return 2; }
  }
  if (!nroots || qd < 1 || qd > MAX_QD || rows < 1 || !seed) { fprintf(stderr, "bad args\n"); return 2; }
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
  for (int s = 0; s < NSEG; ++s) {
    size_t n = (SLOTS * seg_bytes[s] + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
    slab[s] = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (slab[s] == MAP_FAILED) { perror("mmap"); return 1; }
    madvise(slab[s], n, MADV_HUGEPAGE); memset(slab[s], 0, n);
  }
  struct io_uring ring;
  int rc = io_uring_queue_init(256, &ring, 0);
  if (rc < 0) { fprintf(stderr, "queue_init: %s\n", strerror(-rc)); return 1; }

  typedef struct { int busy, left, layer, expert; long idx; double t0, part_end[MAX_ROOTS]; Sqe sq[MAX_SQE_PER_ROW]; struct iovec iv[MAX_SQE_PER_ROW * NSEG]; } Op;
  static Op ops[MAX_QD];
  double* lat = calloc(rows, sizeof(double));
  double* pend[MAX_ROOTS]; double* sqe_lat[MAX_ROOTS]; size_t nsqe[MAX_ROOTS] = {0};
  uint64_t bytes_root[MAX_ROOTS] = {0};
  long last_cnt[MAX_ROOTS] = {0}; double* lead[MAX_ROOTS]; size_t nlead[MAX_ROOTS] = {0};
  for (int r = 0; r < nroots; ++r) {
    pend[r] = calloc(rows, sizeof(double)); sqe_lat[r] = calloc(rows * K_SUBREADS, sizeof(double)); lead[r] = calloc(rows, sizeof(double));
  }
  FILE* rawf = raw ? fopen(raw, "a") : NULL;
  uint64_t s = seed; long issued = 0, done = 0, errors = 0, shorts = 0, total_sqes = 0;
  Dstat d0[MAX_ROOTS], d1[MAX_ROOTS];
  struct rusage ru0, ru1; getrusage(RUSAGE_SELF, &ru0);
  dstat(d0);
  double t_start = now_us();
  int inflight = 0;
  while (done < rows) {
    for (int i = 0; i < qd && issued < rows; ++i) {
      Op* o = &ops[i];
      if (o->busy) continue;
      o->layer = (int)(rng(&s) % LAYERS); o->expert = (int)(rng(&s) % EXPERTS); o->idx = issued++;
      int slot = (int)(o->idx % SLOTS);
      int n = build_row(o->layer, o->expert, slot, o->sq, o->iv);
      o->busy = 1; o->left = n;
      for (int r = 0; r < nroots; ++r) o->part_end[r] = 0;
      o->t0 = now_us();
      for (int k = 0; k < n; ++k) {
        Sqe* q = &o->sq[k];
        struct io_uring_sqe* e = io_uring_get_sqe(&ring);
        io_uring_prep_readv(e, fds[q->root][o->layer], q->iov, q->iovcnt, q->off);
        io_uring_sqe_set_data64(e, ((uint64_t)i << 40) | ((uint64_t)q->root << 32) | q->len);
      }
      total_sqes += n; ++inflight;
    }
    if (!spin_wait) rc = io_uring_submit_and_wait(&ring, 1);
    else { rc = io_uring_submit(&ring); while (rc >= 0 && io_uring_cq_ready(&ring) == 0) __builtin_ia32_pause(); }
    if (rc < 0 && rc != -EINTR && rc != -EAGAIN && rc != -EBUSY) { fprintf(stderr, "submit: %s\n", strerror(-rc)); return 1; }
    struct io_uring_cqe* c; unsigned head, seen = 0;
    double tn = now_us();
    io_uring_for_each_cqe(&ring, head, c) {
      uint64_t dd = io_uring_cqe_get_data64(c);
      Op* o = &ops[dd >> 40]; int r = (int)((dd >> 32) & 0xff); uint64_t want = dd & 0xffffffffULL;
      if (c->res < 0) ++errors; else if ((uint64_t)c->res != want) ++shorts;
      double el = tn - o->t0;
      sqe_lat[r][nsqe[r]++] = el; bytes_root[r] += want;
      if (el > o->part_end[r]) o->part_end[r] = el;
      if (--o->left == 0) {
        o->busy = 0; --inflight;
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
  dstat(d1);
  getrusage(RUSAGE_SELF, &ru1);
  if (rawf) fclose(rawf);
  double cpu = (ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec) + (ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec) / 1e6 +
               (ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec) + (ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec) / 1e6;
  uint64_t tb = 0; for (int r = 0; r < nroots; ++r) tb += bytes_root[r];
  double mean = 0; for (long i = 0; i < rows; ++i) mean += lat[i] / rows;
  qsort(lat, rows, sizeof(double), cmpd);
  printf("{\"label\":\"%s\",\"roots\":\"", label);
  for (int r = 0; r < nroots; ++r) printf("%s%s", r ? "," : "", disk[r]);
  printf("\",\"weights\":\"");
  for (int r = 0; r < nroots; ++r) printf("%s%g", r ? ":" : "", weights[r]);
  printf("\",\"qd\":%d,\"wait\":\"%s\",\"seed\":%llu,\"rows\":%ld,\"per_part\":%d,\"sqes_per_row\":%.2f,\"wall_s\":%.3f,"
         "\"GBps\":%.3f,\"p50_us\":%.1f,\"p90_us\":%.1f,\"p99_us\":%.1f,\"max_us\":%.1f,\"mean_us\":%.1f,\"errors\":%ld,\"short\":%ld,\"cpu_s\":%.3f,\"drives\":[",
         qd, spin_wait ? "spin" : "block", (unsigned long long)seed, rows, per_part, (double)total_sqes / rows, wall, tb / wall / 1e9,
         pct(lat, rows, .5), pct(lat, rows, .9), pct(lat, rows, .99), lat[rows - 1], mean, errors, shorts, cpu);
  for (int r = 0; r < nroots; ++r) {
    qsort(sqe_lat[r], nsqe[r], sizeof(double), cmpd);
    size_t np = part_hi[r] > part_lo[r] ? (size_t)rows : 0;
    qsort(pend[r], np, sizeof(double), cmpd);
    qsort(lead[r], nlead[r], sizeof(double), cmpd);
    double dsec = (double)(d1[r].rd_sec - d0[r].rd_sec) * 512;
    printf("%s{\"disk\":\"%s\",\"root\":\"%s\",\"weight\":%g,\"part_bytes\":%llu,\"bytes\":%llu,\"GBps\":%.3f,\"sqes\":%zu,"
           "\"sqe_p50_us\":%.1f,\"sqe_p99_us\":%.1f,\"part_p50_us\":%.1f,\"part_p90_us\":%.1f,\"part_p99_us\":%.1f,"
           "\"util\":%.3f,\"disk_MB\":%.1f,\"foreign_MB\":%.1f,\"disk_reqs\":%llu,\"inflight_ms\":%llu,\"irqs\":%llu,\"irqs_cpu64plus\":%llu,"
           "\"last_share\":%.3f,\"last_lead_p50_us\":%.1f,\"last_lead_p90_us\":%.1f}",
           r ? "," : "", disk[r], roots[r], weights[r], (unsigned long long)(part_hi[r] - part_lo[r]), (unsigned long long)bytes_root[r],
           bytes_root[r] / wall / 1e9, nsqe[r], pct(sqe_lat[r], nsqe[r], .5), pct(sqe_lat[r], nsqe[r], .99),
           pct(pend[r], np, .5), pct(pend[r], np, .9), pct(pend[r], np, .99),
           (double)(d1[r].ticks - d0[r].ticks) / (wall * 1e3), dsec / 1e6, (dsec - (double)bytes_root[r]) / 1e6,
           (unsigned long long)(d1[r].rd_ios - d0[r].rd_ios), (unsigned long long)(d1[r].inflight_ms - d0[r].inflight_ms),
           (unsigned long long)(d1[r].irq - d0[r].irq), (unsigned long long)(d1[r].irq_hi - d0[r].irq_hi),
           (double)last_cnt[r] / rows, pct(lead[r], nlead[r], .5), pct(lead[r], nlead[r], .9));
  }
  printf("]}\n");
  return errors || shorts ? 3 : 0;
}
