// CPU-only: which calls on a DEFER_TASKRUN ring make a deferred completion visible? No drive, no GPU. TOPOLOGY.md Appendix E.
//   gcc -O2 -pthread defer_probe.c -o defer_probe -luring ; taskset -c 2 ./defer_probe
#define _GNU_SOURCE
#include <liburing.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void* w(void*a){usleep(50000);char c='x';(void)!write(*(int*)a,&c,1);return 0;}
static void run(const char*name,unsigned flags,int step){
  struct io_uring r;struct io_uring_params p;memset(&p,0,sizeof p);p.flags=flags;
  if(io_uring_queue_init_params(8,&r,&p)<0){printf("%s init failed\n",name);return;}
  int pf[2];(void)!pipe(pf);char b[1];
  struct io_uring_sqe*s=io_uring_get_sqe(&r);io_uring_prep_read(s,pf[0],b,1,0);io_uring_submit(&r);
  pthread_t t;pthread_create(&t,0,w,&pf[1]);usleep(120000); // write landed at 50 ms; owner has not entered since
  unsigned before=io_uring_cq_ready(&r);
  const char*what="";int rc=0;
  switch(step){
    case 0: what="cq_ready only (no syscall)";break;
    case 1: what="io_uring_submit() with an empty SQ";rc=io_uring_submit(&r);break;
    case 2: {what="io_uring_submit() with a NOP prepared";struct io_uring_sqe*n=io_uring_get_sqe(&r);io_uring_prep_nop(n);rc=io_uring_submit(&r);break;}
    case 3: what="io_uring_get_events()";rc=io_uring_get_events(&r);break;
    case 4: what="io_uring_submit_and_wait(0)";rc=io_uring_submit_and_wait(&r,0);break;
    case 5: {struct io_uring_cqe*c;what="io_uring_peek_cqe()";rc=io_uring_peek_cqe(&r,&c);break;}
    case 6: {what="io_uring_submit_and_wait(1)";rc=io_uring_submit_and_wait(&r,1);break;}
  }
  unsigned after=io_uring_cq_ready(&r);
  printf("%-32s %-40s rc=%d cq_ready before=%u after=%u\n",name,what,rc,before,after);
  pthread_join(t,0);close(pf[0]);close(pf[1]);io_uring_queue_exit(&r);
}
int main(void){
  for(int st=0;st<=6;st++){
    run("flags=0",0,st);
    run("SI|DTR",IORING_SETUP_SINGLE_ISSUER|IORING_SETUP_DEFER_TASKRUN,st);
    run("SI|DTR|TASKRUN_FLAG",IORING_SETUP_SINGLE_ISSUER|IORING_SETUP_DEFER_TASKRUN|IORING_SETUP_TASKRUN_FLAG,st);
  }
  return 0;}
