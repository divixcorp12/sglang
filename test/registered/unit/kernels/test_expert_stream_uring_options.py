"""Compile the real option parser; fake liburing exercises failure paths deterministically."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
INCLUDE = ROOT / "python/sglang/kernels/jit/csrc"
PREFIX = "SGLANG_EXPERT_STREAM_URING_"


def test_uring_environment_validation(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "options.cpp"
    source.write_text(r"""
#include "moe/expert_stream/host/uring_options.h"
#include <iostream>
int main() {
 try { auto o = sglang::expert_stream::UringOptions::from_env(); std::cout << o.queue_depth; }
 catch (const std::exception& e) { std::cerr << e.what(); return 2; }
}
""")
    binary = tmp_path / "options"
    subprocess.run(
        [compiler, "-std=c++20", "-I", str(INCLUDE), str(source), "-o", str(binary)],
        check=True,
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(PREFIX)}
    assert subprocess.check_output([binary], env=env, text=True) == "0"
    for key, value in [
        ("QUEUE_DEPTH", "64"),
        ("MODE", "sqpoll_iopoll"),
        ("READ_MODE", "readv_fixed"),
        ("WAIT_MODE", "spin"),
        ("FIXED_FILES", "1"),
        ("DIAGNOSTICS", "1"),
        ("SQ_THREAD_CPU", "2"),
        ("SQ_THREAD_IDLE_MS", "0"),
    ]:
        result = subprocess.run(
            [binary], env=env | {PREFIX + key: value}, capture_output=True, text=True
        )
        # Explicit affinity requires SQPOLL; parser rejects a meaningless option.
        if key == "SQ_THREAD_CPU":
            assert result.returncode == 2
        else:
            assert result.returncode == 0, result.stderr
    for key, value in [
        ("QUEUE_DEPTH", "-1"),
        ("QUEUE_DEPTH", "65536"),
        ("QUEUE_DEPTH", " 2"),
        ("QUEUE_DEPTH", "2x"),
        ("QUEUE_DEPTH", ""),
        ("MODE", "fast"),
        ("READ_MODE", "auto"),
        ("WAIT_MODE", "poll"),
        ("FIXED_FILES", "true"),
        ("SQ_THREAD_CPU", "-2"),
        ("SQ_THREAD_IDLE_MS", "-1"),
        ("DIAGNOSTICS", "yes"),
    ]:
        result = subprocess.run(
            [binary], env=env | {PREFIX + key: value}, capture_output=True, text=True
        )
        assert result.returncode == 2, (key, value, result.stdout, result.stderr)
        assert PREFIX + key in result.stderr


_FAKE_LIBURING = r"""
#pragma once
#include <sys/uio.h>
#include <cstdint>
#include <cerrno>
#include <deque>
#include <vector>
#include <algorithm>
#ifndef FAKE_OLD_HEADERS
#define IO_URING_VERSION_MAJOR 2
#define IO_URING_VERSION_MINOR 12
#endif
#define IORING_SETUP_SQPOLL 2u
#define IORING_SETUP_SQ_AFF 4u
#define IORING_SETUP_IOPOLL 1u
#define IOSQE_FIXED_FILE 1u
#ifndef FAKE_OLD_HEADERS
#define IORING_OP_READV_FIXED 60
#endif
struct io_uring_sqe { int fd = -1, opcode = -1, buf_index = -1; unsigned flags = 0, len = 0; uint64_t user_data = 0; const iovec* iov = nullptr; };
struct io_uring_cqe { uint64_t user_data; int res; };
struct io_uring { std::deque<io_uring_sqe> sq; std::deque<io_uring_cqe> cq; unsigned flags; };
struct io_uring_params { unsigned flags=0, sq_thread_idle=0, sq_thread_cpu=0, sq_entries=0, cq_entries=0, features=0; };
struct io_uring_probe {};
inline int setups=0, exits=0, register_files_calls=0, register_buffers_calls=0, unregister_files_calls=0;
inline int submit_calls=0, wait_calls=0, get_events_calls=0, registered_file_error=0, registered_buffer_error=0, setup_error=0;
inline int submit_error_once=0;
inline bool opcode_supported=true;
inline unsigned consume_limit=999;
inline io_uring* last_ring=nullptr;
inline std::vector<int> mapped_files;
inline std::vector<io_uring_sqe> submitted_log;
inline int io_uring_queue_init_params(unsigned n, io_uring* r, io_uring_params* p) {
 ++setups; if (setup_error) return setup_error; r->flags=p->flags; r->sq.clear(); r->cq.clear();
 p->sq_entries=n; p->cq_entries=2*n; p->features=17; last_ring=r; return 0;
}
inline void io_uring_queue_exit(io_uring* r) { ++exits; r->sq.clear(); r->cq.clear(); }
inline io_uring_sqe* io_uring_get_sqe(io_uring* r) { if(r->sq.size() == 8) return nullptr; return &r->sq.emplace_back(); }
inline void io_uring_prep_read(io_uring_sqe* s,int fd,void*,unsigned n,uint64_t) { s->fd=fd;s->len=n;s->opcode=0; }
inline void io_uring_prep_read_fixed(io_uring_sqe* s,int fd,void* p,unsigned n,uint64_t o,int index) { io_uring_prep_read(s,fd,p,n,o);s->opcode=1;s->buf_index=index; }
inline void io_uring_prep_readv(io_uring_sqe* s,int fd,const iovec* v,unsigned n,uint64_t) { s->fd=fd;s->iov=v;s->len=n;s->opcode=2; }
#ifndef FAKE_OLD_HEADERS
inline void io_uring_prep_readv_fixed(io_uring_sqe* s,int fd,const iovec* v,unsigned n,uint64_t o,int,int index) { io_uring_prep_readv(s,fd,v,n,o);s->opcode=3;s->buf_index=index; }
#endif
inline void io_uring_sqe_set_data64(io_uring_sqe* s,uint64_t d) { s->user_data=d; }
inline uint64_t io_uring_cqe_get_data64(const io_uring_cqe* c) { return c->user_data; }
inline int io_uring_submit(io_uring* r) {
 ++submit_calls; if(submit_error_once) { int e=submit_error_once;submit_error_once=0;return e; }
 unsigned n=std::min<unsigned>(r->sq.size(),consume_limit);
 for(unsigned i=0;i<n;++i) { auto s=r->sq.front(); r->sq.pop_front();
   int bytes=s.len; if(s.opcode>=2) { bytes=0;for(unsigned j=0;j<s.len;++j)bytes+=s.iov[j].iov_len; }
   submitted_log.push_back(s);r->cq.push_back({s.user_data,bytes}); }
 return n;
}
inline int io_uring_submit_and_wait(io_uring* r,unsigned) { ++wait_calls;return io_uring_submit(r); }
inline unsigned io_uring_cq_ready(io_uring* r) { return r->cq.size(); }
inline unsigned io_uring_sq_ready(io_uring* r) { return r->sq.size(); }
inline int io_uring_get_events(io_uring*) { ++get_events_calls;return 0; }
inline int io_uring_peek_cqe(io_uring* r,io_uring_cqe** c) { if(r->cq.empty())return -EAGAIN;*c=&r->cq.front();return 0; }
inline int io_uring_wait_cqe(io_uring* r,io_uring_cqe** c) { return io_uring_peek_cqe(r,c); }
inline void io_uring_cqe_seen(io_uring* r,io_uring_cqe*) { r->cq.pop_front(); }
inline void io_uring_cq_advance(io_uring* r,unsigned n) { while(n--)r->cq.pop_front(); }
#define io_uring_for_each_cqe(r,head,cqe) for(head=0;head<(r)->cq.size() && ((cqe)=&(r)->cq[head],true);++head)
inline int io_uring_register_files(io_uring*,const int* f,unsigned n) { ++register_files_calls; mapped_files.assign(f,f+n);return registered_file_error; }
inline int io_uring_register_buffers(io_uring*,const iovec*,unsigned) { ++register_buffers_calls;return registered_buffer_error; }
inline int io_uring_unregister_files(io_uring*) { ++unregister_files_calls;return 0; }
inline int io_uring_unregister_buffers(io_uring*) { return 0; }
inline io_uring_probe* io_uring_get_probe_ring(io_uring*) { static io_uring_probe p;return &p; }
inline int io_uring_opcode_supported(io_uring_probe*,int) { return opcode_supported; }
inline void io_uring_free_probe(io_uring_probe*) {}
"""

_FAKE_CASES = r"""
#include "moe/expert_stream/host/uring_reader.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
using namespace sglang::expert_stream;
void env(const char* key,const char* value) { std::string k="SGLANG_EXPERT_STREAM_URING_";setenv((k+key).c_str(),value,1); }
template<class F> void rejected(F f) { bool bad=false;try {f();}catch(const std::exception&){bad=true;}assert(bad); }
int main() {
 char arena[8192]{}; std::vector<iovec> buffers{{arena,sizeof(arena)}}; std::vector<int> files{99,13};
 {
 UringReader r;assert(r.init(8));r.configure_resources(files,buffers,false);
 assert(last_ring->flags==0 && register_files_calls==0 && register_buffers_calls==0);
 assert(r.prep_read(99,arena,16,0,1));assert(r.submit(1)==1);std::vector<ReadCompletion> out;
 assert(r.reap(out)==1 && out[0].data==1 && out[0].res==16);r.close();r.close();
 }
 env("FIXED_FILES","1");env("READ_MODE","fixed");
 {
 UringReader r;assert(r.init(8));r.configure_resources(files,buffers,true);
 rejected([&]{r.prep_read(98,arena,16,0,1);});assert(last_ring->sq.empty());
 rejected([&]{r.prep_read(99,arena+8190,16,0,1);});assert(last_ring->sq.empty());
 const iovec v[]{{arena,16},{arena+32,16}};
 rejected([&]{r.prep_readv(99,v,2,0,1);});assert(last_ring->sq.empty());
 assert(r.prep_read(13,arena,16,0,2));assert(last_ring->sq.front().fd==1);
 assert(last_ring->sq.front().flags & IOSQE_FIXED_FILE);assert(last_ring->sq.front().buf_index==0);
 const int old_setups=setups, old_registration=register_files_calls;
 r.drain(1);assert(setups==old_setups+1 && register_files_calls==old_registration+1);
 assert(r.prep_read(13,arena,16,0,3));r.submit(0);r.close();assert(!r.ready());
 }
#ifndef FAKE_OLD_HEADERS
 env("READ_MODE","readv_fixed");
 {
 UringReader r;assert(r.init(8));r.configure_resources(files,buffers,true);
 assert(r.prep_read(99,arena,17,0,10));assert(r.prep_read(13,arena+64,29,0,11));
 const iovec v[]{{arena+128,13},{arena+256,19}};assert(r.prep_readv(99,v,2,0,12));
 r.submit(0);std::vector<ReadCompletion> out;assert(r.reap(out)==3);
 assert(out[0].res==17 && out[1].res==29 && out[2].res==32);
 }
 {
 UringReader r;assert(r.init(8));r.configure_resources(files,{{arena,64},{arena+64,64}},true);
 const iovec v[]{{arena,16},{arena+64,16}};
 rejected([&]{r.prep_readv(99,v,2,0,5);});assert(last_ring->sq.empty());
 }
 opcode_supported=false;{ UringReader r;rejected([&]{r.init(8);});assert(!r.ready()); }opcode_supported=true;
#else
 env("READ_MODE","readv_fixed");{ UringReader r;rejected([&]{r.init(8);}); }
#endif
 env("READ_MODE","normal");env("MODE","sqpoll");env("WAIT_MODE","spin");
 {
 UringReader r;assert(r.init(8));r.configure_resources(files,buffers,true);
 for(unsigned i=0;i<3;++i)assert(r.prep_read(99,arena+i*16,16,0,20+i));
 consume_limit=1;submit_error_once=-EINTR;const int old_setups=setups;
 r.drain(3);assert(last_ring->sq.empty() && last_ring->cq.empty() && setups==old_setups);
 assert(r.prep_read(99,arena,16,0,25));const int old_waits=wait_calls;
 assert(r.submit(1)>=0 && wait_calls==old_waits);std::vector<ReadCompletion> out;r.reap(out);
 }
 env("MODE","iopoll");
 { UringReader r;assert(r.init(8));rejected([&]{r.configure_resources(files,buffers,false);}); }
 env("MODE","default");
 registered_file_error=-EMFILE;{UringReader r;assert(r.init(8));rejected([&]{r.configure_resources(files,buffers,true);});}
 registered_file_error=0;env("READ_MODE","fixed");registered_buffer_error=-ENOMEM;
 {int old=unregister_files_calls;{UringReader r;assert(r.init(8));rejected([&]{r.configure_resources(files,buffers,true);});}assert(unregister_files_calls==old+1);}
 registered_buffer_error=0;env("MODE","sqpoll");setup_error=-EINVAL;
 {UringReader r;rejected([&]{r.init(8);});}
 std::cout<<"PASS defaults, registrations, range rejection, reset, SQPOLL partial drain, spin, unsupported capability\n";
}
"""


@pytest.mark.parametrize("old_headers", [False, True])
def test_uring_reader_driver_contract(tmp_path, old_headers):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    (tmp_path / "liburing.h").write_text(_FAKE_LIBURING)
    source = tmp_path / "reader.cpp"
    source.write_text(_FAKE_CASES)
    binary = tmp_path / "reader"
    args = [
        compiler,
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-I",
        str(tmp_path),
        "-I",
        str(INCLUDE),
    ]
    if old_headers:
        args.append("-DFAKE_OLD_HEADERS")
    subprocess.run(args + [str(source), "-o", str(binary)], check=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(PREFIX)}
    completed = subprocess.run(
        [binary], env=env, text=True, capture_output=True, timeout=10
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASS defaults" in completed.stdout
