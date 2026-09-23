#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <liburing.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <vector>
#include <sys/mman.h>

namespace py = pybind11;
using CudaHostCallback = void (*)(void*);
using LaunchHostFunc = int (*)(void*, CudaHostCallback, void*);

static int wait_cqe_intr(io_uring* ring, io_uring_cqe** cqe) {
  int rc;
  do { rc = io_uring_wait_cqe(ring, cqe); } while (rc == -EINTR || rc == -EAGAIN);
  return rc;
}

static int submit_pending(io_uring* ring) {
  while (io_uring_sq_ready(ring) > 0) {
    int rc = io_uring_submit(ring);
    if (rc < 0) return rc;
    if (rc == 0) return -EIO;
  }
  return 0;
}

static LaunchHostFunc cuda_launch_host_func() {
  auto fn = reinterpret_cast<LaunchHostFunc>(dlsym(RTLD_DEFAULT, "cudaLaunchHostFunc"));
  if (!fn) throw std::runtime_error("cudaLaunchHostFunc is not loaded");
  return fn;
}

struct Table {
  std::string path;
  uint64_t weight_offset, scale_offset;
  int64_t rows, dim, scale_dim;
  int64_t tag;
};

struct Counters {
  uint64_t accesses=0, hits=0, unique_misses=0, sqes=0, cqes=0, direct_bytes=0, failures=0;
  uint64_t queue_wait_ns=0, worker_wait_ns=0;
  uint64_t evictions=0, filled_rows=0;
  uint64_t bounce_bytes_peak=0;
  uint64_t chunks=0;  // read passes; >1 means a request outgrew the bounce buffer
  int64_t pinned_staging_bytes=0, device_staging_bytes=0;
};

class Store {
 public:
  static constexpr size_t kBounceBytes = 16 << 20;
  Store(size_t budget, int64_t row_bytes) : row_bytes_(row_bytes) {
    if (row_bytes <= 0 || budget < static_cast<size_t>(row_bytes)) throw std::invalid_argument("invalid Engram cache size");
    ways_=8;
    size_t capacity=budget/static_cast<size_t>(row_bytes);
    nsets_=std::max<size_t>(1,capacity/ways_);
    tags_.assign(nsets_*ways_,-1); ages_.assign(nsets_*ways_,0);
    data_=static_cast<uint8_t*>(mmap(nullptr,nsets_*ways_*row_bytes_,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0));
    if(data_==MAP_FAILED) { data_=nullptr; throw std::runtime_error("Engram cache mmap failed"); }
    if(posix_memalign(reinterpret_cast<void**>(&bounce_),4096,kBounceBytes)!=0) {
      munmap(data_,nsets_*ways_*row_bytes_); data_=nullptr;
      throw std::runtime_error("Engram aligned bounce allocation failed");
    }
    q_.reserve(8);
    worker_=std::thread([this]{ worker_loop(); });
    std::unique_lock<std::mutex> l(mu_); ready_cv_.wait(l,[&]{return ready_;});
    if(!worker_error_.empty()) { l.unlock(); stop(); throw std::runtime_error(worker_error_); }
  }
  ~Store(){ stop(); std::free(bounce_); if(data_) munmap(data_,nsets_*ways_*row_bytes_); }

  void register_table(const Table& t) { std::lock_guard<std::mutex> l(mu_); tables_[t.tag]=t; }
  void adjust_staging(int64_t host_bytes, int64_t device_bytes) {
    std::lock_guard<std::mutex> l(stats_mu_);
    stats_.pinned_staging_bytes += host_bytes;
    stats_.device_staging_bytes += device_bytes;
  }
  void enqueue(uintptr_t ids, uintptr_t out, uintptr_t status, int64_t n, int64_t tag) {
    int rc=submit(Request{reinterpret_cast<const int64_t*>(ids),reinterpret_cast<uint8_t*>(out),reinterpret_cast<int32_t*>(status),n,tag});
    if(rc) { if(*reinterpret_cast<int32_t*>(status)!=2) *reinterpret_cast<int32_t*>(status)=3; std::memset(reinterpret_cast<void*>(out),0,n*row_bytes_); }
  }
  void lookup(const int64_t* ids, uint8_t* out, int64_t n, int64_t tag) { int rc=submit(Request{ids,out,nullptr,n,tag}); if(rc) throw std::runtime_error("Engram io_uring lookup failed: "+std::to_string(rc)); }
  py::dict stats() {
    std::lock_guard<std::mutex> l(stats_mu_); py::dict d;
    d["accesses"]=stats_.accesses; d["hits"]=stats_.hits; d["unique_misses"]=stats_.unique_misses;
    d["submitted_sqes"]=stats_.sqes; d["completed_cqes"]=stats_.cqes; d["direct_read_bytes"]=stats_.direct_bytes; d["failures"]=stats_.failures;
    d["queue_wait_ns"]=stats_.queue_wait_ns; d["worker_wait_ns"]=stats_.worker_wait_ns;
    d["evictions"]=stats_.evictions; d["filled_rows"]=stats_.filled_rows;
    d["bounce_bytes_peak"]=stats_.bounce_bytes_peak;
    d["bounce_bytes_allocated"]=kBounceBytes;
    d["read_chunks"]=stats_.chunks;
    d["pinned_staging_bytes"]=stats_.pinned_staging_bytes;
    d["device_staging_bytes"]=stats_.device_staging_bytes;
    d["lookups"]=clock_; d["hit_rate"]=stats_.accesses?double(stats_.hits)/stats_.accesses:0.0;
    d["misses"]=stats_.unique_misses; d["capacity_rows"]=nsets_*ways_;
    d["cache_bytes"]=nsets_*ways_*row_bytes_;
    d["cache_metadata_bytes"]=(tags_.size()+ages_.size())*sizeof(int64_t);
    d["row_bytes"]=row_bytes_; return d;
  }
 private:
  struct Request { const int64_t* ids=nullptr; uint8_t* out=nullptr; int32_t* status=nullptr; int64_t n=0,tag=0; bool done=false,in_use=false; std::condition_variable cv; int result=0; std::chrono::steady_clock::time_point queued; };
  struct PageRun { uint64_t page; size_t pages; void* buf; int result; };
  size_t row_bytes_, ways_, nsets_; uint8_t* data_=nullptr;
  std::vector<int64_t> tags_,ages_; uint64_t clock_=0; Counters stats_;
  std::mutex mu_; std::condition_variable wake_,ready_cv_,space_; bool stopping_=false,ready_=false; std::string worker_error_;
  std::mutex stats_mu_;
  std::array<Request,8> requests_{}; std::vector<Request*> q_; std::map<int64_t,Table> tables_; std::thread worker_; io_uring ring_{}; int ring_status_=0; std::map<std::string,int> fds_;

  void stop() { {std::lock_guard<std::mutex> l(mu_); stopping_=true;} wake_.notify_all(); space_.notify_all(); if(worker_.joinable()) worker_.join(); }
  int submit(Request r) {
    std::unique_lock<std::mutex> l(mu_);
    if(stopping_ || !worker_error_.empty()) throw std::runtime_error("Engram io_uring worker is stopped");
    space_.wait(l,[&]{return stopping_||std::any_of(requests_.begin(),requests_.end(),[](const Request& request){return !request.in_use;});});
    if(stopping_) throw std::runtime_error("Engram io_uring worker is stopped");
    Request* q=&*std::find_if(requests_.begin(),requests_.end(),[](const Request& request){return !request.in_use;});
    q->ids=r.ids;q->out=r.out;q->status=r.status;q->n=r.n;q->tag=r.tag;q->done=false;q->in_use=true;q->queued=std::chrono::steady_clock::now();
    q_.push_back(q); wake_.notify_one(); q->cv.wait(l,[&]{return q->done;});
    int result=q->result;q->in_use=false;space_.notify_one();return result;
  }
  void worker_loop() {
    int rc=io_uring_queue_init(128,&ring_,0);
    {std::lock_guard<std::mutex> l(mu_); ring_status_=rc; ready_=true; if(rc<0) worker_error_="io_uring_queue_init failed: "+std::to_string(-rc);} ready_cv_.notify_all();
    if(rc<0) return;
    for(;;) {
      Request* q;
      {std::unique_lock<std::mutex> l(mu_); wake_.wait(l,[&]{return stopping_||!q_.empty();}); if(stopping_&&q_.empty()) break; q=q_.front(); q_.erase(q_.begin());}
      space_.notify_one();
      auto worker_start=std::chrono::steady_clock::now();
      {std::lock_guard<std::mutex> s(stats_mu_); stats_.queue_wait_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(worker_start-q->queued).count();}
      int result=0;
      try {
        result=process(*q);
      } catch(const std::exception&) {
        // process builds its lookup metadata before issuing reads and does not
        // allocate after SQEs are submitted, so stack unwinding here cannot
        // invalidate any buffer still owned by the kernel. The callback will
        // clear its output after this failed request is released.
        result=-EIO;
      } catch(...) {
        result=-EIO;
      }
      auto worker_end=std::chrono::steady_clock::now();
      {std::lock_guard<std::mutex> s(stats_mu_); stats_.worker_wait_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(worker_end-worker_start).count();}
      {std::lock_guard<std::mutex> l(mu_); q->result=result; q->done=true;} if(result) {std::lock_guard<std::mutex> s(stats_mu_); stats_.failures++;} q->cv.notify_one();
    }
    for(auto& p:fds_) close(p.second); fds_.clear(); io_uring_queue_exit(&ring_);
  }
  int get_fd(const std::string& path) {
    auto it=fds_.find(path); if(it!=fds_.end()) return it->second;
    int fd=open(path.c_str(),O_RDONLY|O_CLOEXEC|O_DIRECT); if(fd<0) return -errno; fds_[path]=fd; return fd;
  }
  // Keep all read buffers alive until every in-kernel CQE has been reaped.
  // Unsubmitted SQEs are discarded by resetting the worker-owned ring.
  int drain(unsigned pending) {
    const unsigned unsubmitted=std::min(pending,io_uring_sq_ready(&ring_));
    unsigned in_kernel=pending-unsubmitted;
    while(in_kernel>0) {
      io_uring_cqe* cqe=nullptr; const int rc=wait_cqe_intr(&ring_,&cqe);
      if(rc<0) std::terminate();
      io_uring_cqe_seen(&ring_,cqe); --in_kernel;
    }
    if(unsubmitted>0) {
      io_uring_queue_exit(&ring_);
      const int rc=io_uring_queue_init(128,&ring_,0);
      if(rc<0) {
        std::lock_guard<std::mutex> l(mu_);
        worker_error_="io_uring ring reset after failed submission: "+std::to_string(-rc);
        return rc;
      }
    }
    return 0;
  }
  int process(Request& q) {
    std::lock_guard<std::mutex> cache_guard(cache_mu_);
    std::lock_guard<std::mutex> stats_guard(stats_mu_);
    Table t;
    {std::lock_guard<std::mutex> l(mu_); auto ti=tables_.find(q.tag); if(ti==tables_.end()) return -ENOENT; t=ti->second;}
    const size_t rb=row_bytes_; if(q.status) *q.status=1;
    std::vector<int64_t> misses; std::vector<std::vector<size_t>> miss_positions;
    std::unordered_map<int64_t,size_t> miss_index;
    ++clock_; stats_.accesses+=q.n;
    for(int64_t i=0;i<q.n;i++) {
      int64_t id=q.ids[i]; if(id<0||id>=t.rows) { if(q.status)*q.status=2; return -EINVAL; }
      int64_t key=id+t.tag; size_t set=static_cast<uint64_t>(key)%nsets_, base=set*ways_, way=ways_;
      for(size_t w=0;w<ways_;w++) if(tags_[base+w]==key){way=w;break;}
      if(way<ways_) { std::memcpy(q.out+i*rb,data_+(base+way)*rb,rb); ages_[base+way]=clock_; stats_.hits++; }
      else {
        auto found=miss_index.find(id);
        if(found==miss_index.end()) {miss_index[id]=misses.size();misses.push_back(id);miss_positions.push_back({static_cast<size_t>(i)});}
        else miss_positions[found->second].push_back(static_cast<size_t>(i));
      }
    }
    if(!misses.empty()) {
      stats_.unique_misses+=misses.size(); int fd=get_fd(t.path); if(fd<0) return fd;
      // The 4 KiB pages one miss row needs; O_DIRECT extents are aligned.
      auto for_each_page=[&](int64_t id,auto&& fn){
        auto add_range=[&](uint64_t off,uint64_t length){
          uint64_t first=off&~uint64_t(4095), last=(off+length-1)&~uint64_t(4095);
          for(uint64_t p=first;;p+=4096){fn(p);if(p==last)break;}
        };
        add_range(t.weight_offset+id*t.dim,t.dim);
        add_range(t.scale_offset+id*t.scale_dim,t.scale_dim);
      };
      // One chunk's reads must fit the fixed bounce buffer, so a request whose misses do
      // not all fit is split into several chunks rather than refused. Reading every miss
      // in one pass was a batch-1 decode assumption: eager layer 14 drives this same
      // store with prefill-sized batches, which overflowed the buffer, returned -E2BIG
      // and killed the scheduler (2026-09-22). A single row too large for the whole
      // buffer is still refused, because no chunking can make that one fit.
      for(size_t chunk_begin=0;chunk_begin<misses.size();) {
        std::map<uint64_t,PageRun> pages;
        uint64_t chunk_bytes=0;
        size_t chunk_end=chunk_begin;
        while(chunk_end<misses.size()) {
          std::vector<uint64_t> row_pages;
          for_each_page(misses[chunk_end],[&](uint64_t p){row_pages.push_back(p);});
          // A row's weight and scale ranges can share a page, so dedupe before costing it.
          std::sort(row_pages.begin(),row_pages.end());
          row_pages.erase(std::unique(row_pages.begin(),row_pages.end()),row_pages.end());
          uint64_t added=0; for(uint64_t p:row_pages) if(!pages.count(p)) added+=4096;
          if(chunk_bytes+added>kBounceBytes) { if(chunk_end==chunk_begin) return -E2BIG; break; }
          for(uint64_t p:row_pages) if(pages.emplace(p,PageRun{p,1,nullptr,0}).second) chunk_bytes+=4096;
          ++chunk_end;
        }
        stats_.bounce_bytes_peak=std::max(stats_.bounce_bytes_peak,chunk_bytes);
        stats_.chunks++;
        std::vector<uint64_t> pvec; for(auto& [p,v]:pages) pvec.push_back(p);
        std::vector<PageRun> runs;
        for(uint64_t p:pvec) { if(!runs.empty()&&runs.back().page+runs.back().pages*4096==p)runs.back().pages++; else runs.push_back(PageRun{p,1,nullptr,0}); }
        size_t bounce_used=0;
        for(auto& run:runs) {
          run.buf=bounce_+bounce_used;
          bounce_used+=run.pages*4096;
        }
        int err=0;
        constexpr size_t kBatchRuns=96;
        for(size_t begin=0;!err&&begin<runs.size();) {
          const size_t end=std::min(runs.size(),begin+kBatchRuns);
          size_t prepared=0;
          for(size_t i=begin;i<end;i++) {
            auto& run=runs[i]; io_uring_sqe* sqe=io_uring_get_sqe(&ring_);
            if(!sqe) { err=-ENOBUFS; break; }
            io_uring_prep_read(sqe,fd,run.buf,run.pages*4096,run.page);
            io_uring_sqe_set_data64(sqe,reinterpret_cast<uint64_t>(&run)); stats_.sqes++; ++prepared;
          }
          if(err) { drain(prepared); break; }
          int submitted=submit_pending(&ring_);
          if(submitted<0) { drain(end-begin); err=submitted; break; }
          for(size_t i=begin;i<end;i++) {
            io_uring_cqe* cqe=nullptr; int rc=wait_cqe_intr(&ring_,&cqe);
            if(rc<0) { drain(end-i); err=rc; break; }
            auto* run=reinterpret_cast<PageRun*>(io_uring_cqe_get_data64(cqe));
            run->result=cqe->res; stats_.cqes++;
            if(cqe->res>0) stats_.direct_bytes+=cqe->res;
            else if(cqe->res!=-EINTR&&cqe->res!=-EAGAIN) err=cqe->res;
            io_uring_cqe_seen(&ring_,cqe);
          }
          // A short aligned completion can be resumed. EINTR/EAGAIN retries the
          // unchanged aligned extent. A non-aligned tail is accepted only if all
          // requested row bytes lie inside the bytes returned by the kernel.
          for(size_t i=begin;!err&&i<end;i++) {
            auto& run=runs[i]; const size_t expected=run.pages*4096;
            size_t completed=run.result>0?static_cast<size_t>(run.result):0;
            unsigned attempts=0;
            while((run.result==-EINTR||run.result==-EAGAIN||
                   (completed>0&&completed<expected&&completed%4096==0))&&attempts++<8) {
              size_t remain=expected-completed;
              auto* buffer=static_cast<uint8_t*>(run.buf)+completed;
              uint64_t offset=run.page+completed;
              io_uring_sqe* sqe=io_uring_get_sqe(&ring_); if(!sqe) {err=-ENOBUFS;break;}
              io_uring_prep_read(sqe,fd,buffer,remain,offset);
              io_uring_sqe_set_data64(sqe,reinterpret_cast<uint64_t>(&run)); stats_.sqes++;
              int submitted_retry=submit_pending(&ring_);
              if(submitted_retry<0) {drain(1);err=submitted_retry;break;}
              io_uring_cqe* cqe=nullptr; int rc=wait_cqe_intr(&ring_,&cqe);
              if(rc<0) {drain(1);err=rc;break;}
              int result=cqe->res; stats_.cqes++; if(result>0) stats_.direct_bytes+=result;
              io_uring_cqe_seen(&ring_,cqe);
              if(result==-EINTR||result==-EAGAIN) {run.result=result;continue;}
              if(result<0) {err=result;break;}
              if(result==0) {run.result=static_cast<int>(completed);break;}
              completed+=static_cast<size_t>(result); run.result=static_cast<int>(completed);
              if(static_cast<size_t>(result)<remain && static_cast<size_t>(result)%4096!=0) break;
            }
            if(!err&&(run.result==-EINTR||run.result==-EAGAIN)) err=run.result;
          }
          begin=end;
        }
        if(err) return err;
        auto find_run=[&](uint64_t p)->PageRun* {
          auto it=std::upper_bound(runs.begin(),runs.end(),p,[](uint64_t page,const PageRun& run){return page<run.page;});
          if(it==runs.begin())return nullptr; --it;
          return p<it->page+it->pages*4096?&*it:nullptr;
        };
        for(size_t j=chunk_begin;j<chunk_end;j++) {
          int64_t id=misses[j]; uint8_t* dst=q.out+miss_positions[j][0]*rb;
          uint64_t wo=t.weight_offset+id*t.dim, so=t.scale_offset+id*t.scale_dim;
          auto scatter=[&](uint64_t off,size_t len,uint8_t* out)->bool {
            size_t done=0;
            while(done<len) {
              const uint64_t pos=off+done,page=pos&~uint64_t(4095);
              PageRun* run=find_run(page); if(!run)return false;
              const size_t within=pos-page,run_offset=page-run->page+within;
              if(run_offset>=static_cast<size_t>(std::max(run->result,0)))return false;
              const size_t available=static_cast<size_t>(run->result)-run_offset;
              const size_t take=std::min({len-done,size_t(4096-within),available});
              if(!take)return false;
              memcpy(out+done,static_cast<uint8_t*>(run->buf)+run_offset,take); done+=take;
            }
            return true;
          };
          if(!scatter(wo,t.dim,dst)||!scatter(so,t.scale_dim,dst+t.dim))return -EIO;
          int64_t key=id+t.tag; size_t set=static_cast<uint64_t>(key)%nsets_,base=set*ways_,way=ways_;
          for(size_t w=0;w<ways_;w++)if(tags_[base+w]==key){way=w;break;}
          if(way==ways_) way=std::min_element(ages_.begin()+base,ages_.begin()+base+ways_)-(ages_.begin()+base);
          if(tags_[base+way]<0)stats_.filled_rows++;else stats_.evictions++;
          tags_[base+way]=key; ages_[base+way]=clock_; memcpy(data_+(base+way)*rb,dst,rb);
          for(size_t k=1;k<miss_positions[j].size();k++) memcpy(q.out+miss_positions[j][k]*rb,dst,rb);
        }
        chunk_begin=chunk_end;
      }
    }
    if(q.status) *q.status=0; return 0;
  }
  std::mutex cache_mu_;
  uint8_t* bounce_=nullptr;
};

static std::mutex global_mu;
static std::shared_ptr<Store> global_store;
static std::shared_ptr<Store> get_store(size_t budget, int64_t row_bytes) {
  std::lock_guard<std::mutex> l(global_mu);
  if(!global_store) global_store=std::make_shared<Store>(budget,row_bytes);
  else if(global_store->stats()["row_bytes"].cast<int64_t>()!=row_bytes) throw std::runtime_error("Engram layers disagree on row bytes");
  return global_store;
}

class EngramHostLookup {
 public:
  EngramHostLookup(std::shared_ptr<Store> store, const std::string& path,
                   uint64_t weight_offset, uint64_t scale_offset,
                   int64_t num_embeddings, int64_t dim, int64_t scale_dim,
                   int64_t tag, int64_t n, uintptr_t ids, uintptr_t rows,
                   uintptr_t status, int64_t host_stage_bytes,
                   int64_t device_stage_bytes)
    : store_(std::move(store)),
      table_{path, weight_offset, scale_offset, num_embeddings, dim, scale_dim, tag},
      n_(n),ids_(ids),rows_(rows),status_(status),
      host_stage_bytes_(host_stage_bytes),device_stage_bytes_(device_stage_bytes) {
    store_->register_table(table_);
    store_->adjust_staging(host_stage_bytes_,device_stage_bytes_);
  }
  ~EngramHostLookup() { store_->adjust_staging(-host_stage_bytes_,-device_stage_bytes_); }
  void enqueue(uintptr_t stream){int rc=cuda_launch_host_func()(reinterpret_cast<void*>(stream),callback,this);if(rc)throw std::runtime_error("cudaLaunchHostFunc failed: "+std::to_string(rc));}
  void run() noexcept { try{store_->enqueue(ids_,rows_,status_,n_,table_.tag);}catch(...){*reinterpret_cast<int32_t*>(status_)=3;std::memset(reinterpret_cast<void*>(rows_),0,n_*(table_.dim+table_.scale_dim));} }
 private:
  static void callback(void* p) noexcept { static_cast<EngramHostLookup*>(p)->run(); }
  std::shared_ptr<Store> store_; Table table_; int64_t n_; uintptr_t ids_,rows_,status_;
  int64_t host_stage_bytes_,device_stage_bytes_;
};

PYBIND11_MODULE(engram_host_node_cpp,m) {
  py::class_<Store,std::shared_ptr<Store>>(m,"Store")
   .def("register_table",[](Store& s,const std::string& path,uint64_t wo,uint64_t so,int64_t n,int64_t d,int64_t sd,int64_t tag){s.register_table(Table{path,wo,so,n,d,sd,tag});})
   .def("lookup",[](Store& s,py::array_t<int64_t,py::array::c_style|py::array::forcecast> ids,int64_t tag){auto in=ids.unchecked<1>();py::array_t<uint8_t> out({in.shape(0),s.stats()["row_bytes"].cast<int64_t>()});auto* p=out.mutable_data();{py::gil_scoped_release release;s.lookup(in.data(0),p,in.shape(0),tag);}return out;})
   .def("stats",&Store::stats)
   .def("adjust_staging",&Store::adjust_staging);
  m.def("get_shared_store",&get_store);
  m.def("create_store",[](size_t budget,int64_t row_bytes){return std::make_shared<Store>(budget,row_bytes);});
  py::class_<EngramHostLookup>(m,"EngramHostLookup")
   .def(py::init<std::shared_ptr<Store>,const std::string&,uint64_t,uint64_t,int64_t,int64_t,int64_t,int64_t,int64_t,uintptr_t,uintptr_t,uintptr_t,int64_t,int64_t>())
   .def("enqueue",&EngramHostLookup::enqueue).def("run",&EngramHostLookup::run);
  py::class_<Table>(m,"Table");
}
