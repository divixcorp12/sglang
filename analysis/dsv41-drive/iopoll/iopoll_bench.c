// iopoll_bench: does IORING_SETUP_IOPOLL poll the expert-stream reader's O_DIRECT reads, or punt them to io-wq?
//
// Standalone (liburing only). Two workloads:
//   flat: SQEs of one size into one page-aligned buffer, QD SQEs in flight per thread, random offsets in one file.
//   row:  whole expert rows of an EXL3 row-image file, shaped like RowReader direct mode: each row is cut into
//         parts (one per file, i.e. mirror root), each part into sub-reads (split_part), each sub-read ONE readv
//         whose iovecs are the destination slab rows (6 separate slabs, like the per-name slabs). QD rows in flight.
//   --cut N (row only): cut each sub-read into SQEs of at most N bytes, and at every iovec boundary where the
//         previous iovec does not end, or the next does not start, on a 4 KiB page (NVMe virt_boundary 4095):
//         the shape a polled bio can take without a block-layer split. --op read: also cut at EVERY iovec (READ).
// Reports per-op latency percentiles, MB/s, CPU (process incl. io-wq workers, and the submitting threads), io-wq
// workers seen in /proc/self/task, and /proc/diskstats requests per SQE on each file's partition.
//
//   gcc -O2 -pthread -o iopoll_bench iopoll_bench.c -luring
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
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
#define MAX_FILES 4
#define MAX_THREADS 16
#define NSEG 6

// layer-000.rows.digests.json on /mnt/nvme0/dsv41_flash/exl3_row_images (names, row_bytes, name_offsets, row_stride)
static const char* seg_name[NSEG] = {"w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh"};
static const uint64_t seg_bytes[NSEG] = {8847360, 20480, 9216, 4423680, 4608, 10240};
static const uint64_t seg_off[NSEG] = {0, 8847360, 8867840, 8877056, 13300736, 13305344};
static const uint64_t image_bytes = 13315584, row_stride = 13316096;
#define SLOTS 16  // destination slots per slab: rows land at varied (non-page-aligned for 9216/4608/10240) offsets

static const char* files[MAX_FILES];
static int nfiles = 0, fds[MAX_FILES];
static int gap_cut = 1, iopoll = 0, fixed_files = 0, use_read = 0, row_mode = 0, threads = 1, qd = 1, per_part = 2;
static uint64_t sqe_bytes = 262144, cut = 0;
static double seconds = 3.0;
static atomic_int stop_sampler;
static uint64_t file_bytes[MAX_FILES];
static atomic_int first_error;

static double now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}
static uint64_t rng(uint64_t* s) {
  *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
  return *s;
}
static void* alloc_buf(size_t n) {
  n = (n + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
  void* p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) { perror("mmap"); exit(1); }
  madvise(p, n, MADV_HUGEPAGE);
  memset(p, 0, n);  // fault in (the reader's slabs are pinned, so resident)
  return p;
}

// ---------- one SQE's description ----------
typedef struct { int file; uint64_t off; int iovcnt; struct iovec* iov; uint64_t len; } Sqe;

typedef struct {
  int id;
  struct io_uring ring;
  uint64_t seed;
  // flat
  char* flat;
  // row
  char* slab[NSEG];
  // results
  double* lat; size_t nlat, caplat;
  double* flat_[MAX_FILES]; size_t nf[MAX_FILES], capf[MAX_FILES];
  uint64_t ops, bytes, sqes, errors, eagain, shorts;
  struct rusage ru;
} Thr;

static void file_push(Thr* t, int f, double v) {
  if (t->nf[f] == t->capf[f]) { t->capf[f] = t->capf[f] ? 2 * t->capf[f] : 4096; t->flat_[f] = realloc(t->flat_[f], t->capf[f] * sizeof(double)); }
  t->flat_[f][t->nf[f]++] = v;
}
static void lat_push(Thr* t, double v) {
  if (t->nlat == t->caplat) { t->caplat = t->caplat ? 2 * t->caplat : 4096; t->lat = realloc(t->lat, t->caplat * sizeof(double)); }
  t->lat[t->nlat++] = v;
}

// Build the SQEs of one row (row mode) into out[], iovec storage in iv[]. Returns SQE count.
static int build_row(Thr* t, uint64_t row, int slot, Sqe* out, struct iovec* iv) {
  // parts: page-aligned thirds (one per file), then split_part into per_part page-aligned sub-reads
  int parts = nfiles, nsq = 0, niv = 0;
  uint64_t plen = ((image_bytes + parts - 1) / parts + PAGE - 1) / PAGE * PAGE;
  for (int p = 0; p < parts; ++p) {
    uint64_t pa = p * plen, pb = pa + plen < image_bytes ? pa + plen : image_bytes;
    if (pa >= pb) break;
    // O_DIRECT: read lengths must be 512 multiples; image_bytes is.
    uint64_t sl = (((pb - pa) + per_part - 1) / per_part + PAGE - 1) / PAGE * PAGE;
    for (uint64_t a = pa; a < pb; a += sl) {
      uint64_t b = a + sl < pb ? a + sl : pb;
      // iovecs of image bytes [a,b)
      struct iovec first[NSEG];
      int cnt = 0;
      for (int s = 0; s < NSEG; ++s) {
        uint64_t lo = a > seg_off[s] ? a : seg_off[s], hi = b < seg_off[s] + seg_bytes[s] ? b : seg_off[s] + seg_bytes[s];
        if (lo >= hi) continue;
        first[cnt].iov_base = t->slab[s] + slot * seg_bytes[s] + (lo - seg_off[s]);
        first[cnt].iov_len = hi - lo;
        ++cnt;
      }
      // cut into SQEs
      uint64_t off = row * row_stride + a;
      int i = 0;
      while (i < cnt) {
        Sqe* q = &out[nsq++];
        q->file = p; q->off = off; q->iov = &iv[niv]; q->iovcnt = 0; q->len = 0;
        while (i < cnt) {
          struct iovec* v = &first[i];
          if (q->iovcnt > 0) {
            struct iovec* pv = &iv[niv - 1];
            uintptr_t pend = (uintptr_t)pv->iov_base + pv->iov_len;
            int gap = (pend % PAGE) != 0 || ((uintptr_t)v->iov_base % PAGE) != 0;
            if (use_read || (cut && gap_cut && gap)) break;
          }
          uint64_t room = cut ? cut - q->len : UINT64_MAX;
          if (cut && room == 0) break;
          uint64_t take = v->iov_len < room ? v->iov_len : room;
          iv[niv].iov_base = v->iov_base; iv[niv].iov_len = take; ++niv;
          q->iovcnt++; q->len += take;
          if (take < v->iov_len) { v->iov_base = (char*)v->iov_base + take; v->iov_len -= take; break; }
          ++i;
        }
        off += q->len;
      }
    }
  }
  return nsq;
}

static void* worker(void* arg) {
  Thr* t = arg;
  unsigned depth = row_mode ? 256 : (unsigned)qd;
  if (depth < 8) depth = 8;
  struct io_uring_params p = {0};
  if (iopoll) p.flags |= IORING_SETUP_IOPOLL;
  int rc = io_uring_queue_init_params(depth * 4, &t->ring, &p);
  if (rc < 0) { fprintf(stderr, "queue_init: %s\n", strerror(-rc)); exit(1); }
  if (fixed_files) {
    rc = io_uring_register_files(&t->ring, fds, nfiles);
    if (rc < 0) { fprintf(stderr, "register_files: %s\n", strerror(-rc)); exit(1); }
  }
  // op table
  int maxsq = row_mode ? 4096 : 1;
  int nops = row_mode ? qd : qd;
  typedef struct { double t0; int left; int busy; uint64_t bytes; Sqe* sq; struct iovec* iv; int nsq; } Op;
  Op* ops = calloc(nops, sizeof(Op));
  for (int i = 0; i < nops; ++i) {
    ops[i].sq = calloc(maxsq, sizeof(Sqe));
    ops[i].iv = calloc(maxsq + 64, sizeof(struct iovec));
    if (!row_mode) ops[i].iv[0].iov_base = t->flat + (size_t)i * ((sqe_bytes + PAGE - 1) / PAGE * PAGE);
  }
  double end = now_us() + seconds * 1e6;
  int inflight = 0;
  size_t pending_sqes = 0;  // prepared not yet in SQ (ring-full backpressure)
  (void)pending_sqes;
  for (;;) {
    int started = 0;
    if (now_us() < end) {
      for (int i = 0; i < nops; ++i) {
        Op* o = &ops[i];
        if (o->busy) continue;
        if (row_mode) {
          uint64_t rows = file_bytes[0] / row_stride;
          for (int f = 1; f < nfiles; ++f) if (file_bytes[f] / row_stride < rows) rows = file_bytes[f] / row_stride;
          o->nsq = build_row(t, rng(&t->seed) % rows, (int)(rng(&t->seed) % SLOTS), o->sq, o->iv);
        } else {
          uint64_t blocks = (file_bytes[0] - sqe_bytes) / PAGE;
          o->sq[0].file = 0; o->sq[0].off = (rng(&t->seed) % blocks) * PAGE;
          o->iv[0].iov_len = sqe_bytes; o->sq[0].iov = o->iv; o->sq[0].iovcnt = 1; o->sq[0].len = sqe_bytes;
          o->nsq = 1;
        }
        if (io_uring_sq_space_left(&t->ring) < (unsigned)o->nsq) break;
        o->busy = 1; o->left = o->nsq; o->bytes = 0; o->t0 = now_us();
        for (int k = 0; k < o->nsq; ++k) {
          Sqe* q = &o->sq[k];
          struct io_uring_sqe* s = io_uring_get_sqe(&t->ring);
          int fd = fixed_files ? q->file : fds[q->file];
          if (use_read)  // row mode cuts every iovec into its own SQE, so iovcnt is 1
            io_uring_prep_read(s, fd, q->iov[0].iov_base, (unsigned)q->iov[0].iov_len, q->off);
          else
            io_uring_prep_readv(s, fd, q->iov, q->iovcnt, q->off);
          if (fixed_files) s->flags |= IOSQE_FIXED_FILE;
          io_uring_sqe_set_data64(s, ((uint64_t)i << 40) | ((uint64_t)q->file << 32) | (uint64_t)q->len);
          o->bytes += q->len;
        }
        t->sqes += o->nsq;
        ++inflight; ++started;
      }
    }
    if (inflight == 0) break;
    rc = io_uring_submit_and_wait(&t->ring, 1);
    if (rc < 0 && rc != -EINTR && rc != -EAGAIN && rc != -EBUSY) { fprintf(stderr, "submit: %s\n", strerror(-rc)); exit(1); }
    struct io_uring_cqe* c; unsigned head, seen = 0;
    double tn = now_us();
    io_uring_for_each_cqe(&t->ring, head, c) {
      uint64_t d = io_uring_cqe_get_data64(c);
      Op* o = &ops[d >> 40];
      file_push(t, (int)((d >> 32) & 0xff), tn - o->t0);
      uint64_t want = d & 0xffffffffULL;
      if (c->res < 0) { int z = 0; atomic_compare_exchange_strong(&first_error, &z, c->res); t->errors++; if (c->res == -EAGAIN) t->eagain++; }
      else if ((uint64_t)c->res != want) t->shorts++;
      if (--o->left == 0) {
        o->busy = 0; --inflight; t->ops++; t->bytes += o->bytes;
        lat_push(t, tn - o->t0);
      }
      ++seen;
    }
    io_uring_cq_advance(&t->ring, seen);
  }
  getrusage(RUSAGE_THREAD, &t->ru);
  io_uring_queue_exit(&t->ring);
  return NULL;
}

// ---------- io-wq sampler ----------
#define MAXW 4096
#define MAXWC 32
static char wchan_name[MAXWC][64]; static long wchan_n[MAXWC]; static int nwchan = 0; static long wstate[128];
static int wtid[MAXW]; static double wcpu[MAXW]; static int nw = 0, wmax_live = 0;
static void* sampler(void* arg) {
  (void)arg;
  long hz = sysconf(_SC_CLK_TCK);
  while (!atomic_load(&stop_sampler)) {
    DIR* d = opendir("/proc/self/task");
    int live = 0;
    struct dirent* e;
    while ((e = readdir(d))) {
      if (e->d_name[0] == '.') continue;
      char path[300], comm[64] = {0};
      snprintf(path, sizeof path, "/proc/self/task/%s/comm", e->d_name);
      FILE* f = fopen(path, "r"); if (!f) continue;
      if (!fgets(comm, sizeof comm, f)) comm[0] = 0;
      fclose(f);
      if (strncmp(comm, "iou-wrk", 7) != 0) continue;
      ++live;
      int tid = atoi(e->d_name);
      snprintf(path, sizeof path, "/proc/self/task/%s/stat", e->d_name);
      f = fopen(path, "r"); if (!f) continue;
      char buf[1024]; size_t n = fread(buf, 1, sizeof buf - 1, f); fclose(f); buf[n] = 0;
      char* r = strrchr(buf, ')'); unsigned long ut = 0, st = 0;
      if (r && r[1] == ' ') wstate[(unsigned char)r[2] & 127]++;
      {  // where it sleeps (0 when running); readable for our own threads without root
        char wp[300], wc[64] = "?";
        snprintf(wp, sizeof wp, "/proc/self/task/%s/wchan", e->d_name);
        FILE* g = fopen(wp, "r");
        if (g) { size_t m = fread(wc, 1, sizeof wc - 1, g); wc[m] = 0; fclose(g); }
        int j; for (j = 0; j < nwchan && strcmp(wchan_name[j], wc); ++j) {}
        if (j == nwchan && nwchan < MAXWC) { strcpy(wchan_name[nwchan++], wc); }
        if (j < MAXWC) wchan_n[j]++;
      }
      if (r) sscanf(r + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu", &ut, &st);
      int k; for (k = 0; k < nw && wtid[k] != tid; ++k) {}
      if (k == nw && nw < MAXW) { wtid[nw] = tid; wcpu[nw] = 0; ++nw; }
      if (k < MAXW) { double c = (double)(ut + st) / hz; if (c > wcpu[k]) wcpu[k] = c; }
    }
    closedir(d);
    if (live > wmax_live) wmax_live = live;
    usleep(5000);
  }
  return NULL;
}

// ---------- diskstats ----------
static char part_name[MAX_FILES][64];
static void find_part(int i) {
  struct stat st; fstat(fds[i], &st);
  FILE* f = fopen("/proc/diskstats", "r"); char line[512];
  while (fgets(line, sizeof line, f)) {
    unsigned ma, mi; char nm[64];
    if (sscanf(line, "%u %u %63s", &ma, &mi, nm) == 3 && ma == major(st.st_dev) && mi == minor(st.st_dev)) strcpy(part_name[i], nm);
  }
  fclose(f);
}
static void diskstats(uint64_t reads[MAX_FILES], uint64_t merges[MAX_FILES], uint64_t sectors[MAX_FILES]) {
  FILE* f = fopen("/proc/diskstats", "r"); char line[512];
  while (fgets(line, sizeof line, f)) {
    unsigned ma, mi; char nm[64]; unsigned long long r, m, s;
    if (sscanf(line, "%u %u %63s %llu %llu %llu", &ma, &mi, nm, &r, &m, &s) != 6) continue;
    for (int i = 0; i < nfiles; ++i) if (!strcmp(nm, part_name[i])) { reads[i] = r; merges[i] = m; sectors[i] = s; }
  }
  fclose(f);
}

static int cmp(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return x < y ? -1 : x > y; }

int main(int argc, char** argv) {
  const char* label = "";
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    const char* v = i + 1 < argc ? argv[i + 1] : "";
    if (!strcmp(a, "--file")) { files[nfiles++] = v; ++i; }
    else if (!strcmp(a, "--mode")) { iopoll = !strcmp(v, "iopoll"); ++i; }
    else if (!strcmp(a, "--op")) { use_read = !strcmp(v, "read"); ++i; }
    else if (!strcmp(a, "--workload")) { row_mode = !strcmp(v, "row"); ++i; }
    else if (!strcmp(a, "--size")) { sqe_bytes = strtoull(v, 0, 0); ++i; }
    else if (!strcmp(a, "--cut")) { cut = strtoull(v, 0, 0); ++i; }
    else if (!strcmp(a, "--qd")) { qd = atoi(v); ++i; }
    else if (!strcmp(a, "--threads")) { threads = atoi(v); ++i; }
    else if (!strcmp(a, "--per-part")) { per_part = atoi(v); ++i; }
    else if (!strcmp(a, "--seconds")) { seconds = atof(v); ++i; }
    else if (!strcmp(a, "--fixed-files")) { fixed_files = 1; }
    else if (!strcmp(a, "--no-gap-cut")) { gap_cut = 0; }
    else if (!strcmp(a, "--label")) { label = v; ++i; }
    else { fprintf(stderr, "unknown arg %s\n", a); return 2; }
  }
  if (nfiles == 0 || threads < 1 || threads > MAX_THREADS || qd < 1) { fprintf(stderr, "bad args\n"); return 2; }
  if (row_mode && use_read && cut == 0) cut = UINT64_MAX / 2;  // READ = one iovec per SQE
  for (int i = 0; i < nfiles; ++i) {
    fds[i] = open(files[i], O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fds[i] < 0) { perror(files[i]); return 1; }
    struct stat st; fstat(fds[i], &st); file_bytes[i] = st.st_size;
    find_part(i);
  }
  Thr* th = calloc(threads, sizeof(Thr));
  for (int i = 0; i < threads; ++i) {
    th[i].id = i; th[i].seed = 0x9e3779b97f4a7c15ULL * (i + 1) ^ (uint64_t)getpid();
    if (row_mode) for (int s = 0; s < NSEG; ++s) th[i].slab[s] = alloc_buf(SLOTS * seg_bytes[s]);
    else th[i].flat = alloc_buf((size_t)qd * ((sqe_bytes + PAGE - 1) / PAGE * PAGE));
  }
  uint64_t r0[MAX_FILES] = {0}, m0[MAX_FILES] = {0}, s0[MAX_FILES] = {0}, r1[MAX_FILES] = {0}, m1[MAX_FILES] = {0}, s1[MAX_FILES] = {0};
  struct rusage ru0, ru1; getrusage(RUSAGE_SELF, &ru0);
  pthread_t sp; pthread_create(&sp, 0, sampler, 0);
  diskstats(r0, m0, s0);
  double t0 = now_us();
  pthread_t tid[MAX_THREADS];
  for (int i = 0; i < threads; ++i) pthread_create(&tid[i], 0, worker, &th[i]);
  for (int i = 0; i < threads; ++i) pthread_join(tid[i], 0);
  double wall = (now_us() - t0) / 1e6;
  diskstats(r1, m1, s1);
  atomic_store(&stop_sampler, 1); pthread_join(sp, 0);
  getrusage(RUSAGE_SELF, &ru1);
  // aggregate
  size_t n = 0; uint64_t ops = 0, bytes = 0, sqes = 0, errs = 0, eag = 0, shorts = 0; double sub_cpu = 0;
  for (int i = 0; i < threads; ++i) {
    n += th[i].nlat; ops += th[i].ops; bytes += th[i].bytes; sqes += th[i].sqes; errs += th[i].errors; eag += th[i].eagain; shorts += th[i].shorts;
    sub_cpu += th[i].ru.ru_utime.tv_sec + th[i].ru.ru_utime.tv_usec / 1e6 + th[i].ru.ru_stime.tv_sec + th[i].ru.ru_stime.tv_usec / 1e6;
  }
  double* all = malloc((n ? n : 1) * sizeof(double)); size_t k = 0;
  for (int i = 0; i < threads; ++i) { memcpy(all + k, th[i].lat, th[i].nlat * sizeof(double)); k += th[i].nlat; }
  qsort(all, n, sizeof(double), cmp);
#define PCT(p) (n ? all[(size_t)((p) * (n - 1))] : 0.0)
  double proc_cpu = (ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec) + (ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec) / 1e6 +
                    (ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec) + (ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec) / 1e6;
  double wq_cpu = 0; for (int i = 0; i < nw; ++i) wq_cpu += wcpu[i];
  uint64_t dreq = 0, dsec = 0;
  for (int i = 0; i < nfiles; ++i) { dreq += r1[i] - r0[i]; dsec += s1[i] - s0[i]; }
  printf("{\"label\":\"%s\",\"workload\":\"%s\",\"mode\":\"%s\",\"op\":\"%s\",\"size\":%llu,\"cut\":%llu,\"gap_cut\":%d,\"files\":%d,"
         "\"fixed_files\":%d,\"threads\":%d,\"qd\":%d,\"wall_s\":%.3f,\"ops\":%llu,\"sqes\":%llu,\"sqes_per_op\":%.2f,"
         "\"MBps\":%.1f,\"p50_us\":%.1f,\"p90_us\":%.1f,\"p99_us\":%.1f,\"max_us\":%.1f,\"errors\":%llu,\"first_error\":%d,\"eagain\":%llu,"
         "\"short\":%llu,\"proc_cpu_s\":%.3f,\"submitter_cpu_s\":%.3f,\"iowq_workers\":%d,\"iowq_max_live\":%d,"
         "\"iowq_cpu_s\":%.3f,\"disk_reqs\":%llu,\"disk_reqs_per_sqe\":%.2f,\"disk_MB\":%.1f}\n",
         label, row_mode ? "row" : "flat", iopoll ? "iopoll" : "default", use_read ? "read" : "readv",
         (unsigned long long)sqe_bytes, (unsigned long long)(cut > (UINT64_MAX / 4) ? 0 : cut), gap_cut, nfiles, fixed_files,
         threads, qd, wall, (unsigned long long)ops, (unsigned long long)sqes, ops ? (double)sqes / ops : 0,
         bytes / wall / 1e6, PCT(0.5), PCT(0.9), PCT(0.99), n ? all[n - 1] : 0, (unsigned long long)errs, atomic_load(&first_error),
         (unsigned long long)eag, (unsigned long long)shorts, proc_cpu, sub_cpu, nw, wmax_live, wq_cpu,
         (unsigned long long)dreq, sqes ? (double)dreq / sqes : 0, dsec * 512 / 1e6);
  printf("  sqe_p50_us_by_file:");
  for (int f = 0; f < nfiles; ++f) {
    size_t m = 0; for (int i = 0; i < threads; ++i) m += th[i].nf[f];
    double* v = malloc((m ? m : 1) * sizeof(double)); size_t j = 0;
    for (int i = 0; i < threads; ++i) { memcpy(v + j, th[i].flat_[f], th[i].nf[f] * sizeof(double)); j += th[i].nf[f]; }
    qsort(v, m, sizeof(double), cmp);
    printf(" %s=%.0f/%.0f", part_name[f], m ? v[m / 2] : 0.0, m ? v[(size_t)(0.99 * (m - 1))] : 0.0);
    free(v);
  }
  printf("\n  iowq_samples: state");
  for (int c = 0; c < 128; ++c) if (wstate[c]) printf(" %c=%ld", c, wstate[c]);
  printf(" wchan");
  for (int j = 0; j < nwchan; ++j) printf(" %s=%ld", wchan_name[j], wchan_n[j]);
  printf("\n");
  (void)seg_name;
  return errs || shorts ? 3 : 0;
}
