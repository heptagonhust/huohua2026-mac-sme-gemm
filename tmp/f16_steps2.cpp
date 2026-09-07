#include <cstdio>
#include <cstring>
#include <cstdint>
#include <signal.h>
#include <setjmp.h>

static sigjmp_buf jb;
static void on_sig(int){ siglongjmp(jb, 1); }
static __fp16 A[32], B[32];
static float row[16][16];

// full: smstart+zero+ld1h+fmopa+readback via mov za0h.s (like working FP32 kernel)
static void full_mov(){
  asm volatile(
    "smstart\n\tzero {za}\n\tptrue p1.s\n\tptrue p2.h\n\t"
    "ld1h {z0.h}, p2/z, [%1]\n\tld1h {z1.h}, p2/z, [%2]\n\t"
    "fmopa za0.s, p1/m, p1/m, z0.h, z1.h\n\t"
    "mov w12, #0\n\tmov x16, %0\n\t"
    "1:\n\tmov z2.s, p1/m, za0h.s[w12, 0]\n\t"
    "st1w {z2.s}, p1, [x16]\n\tadd x16, x16, #64\n\tadd w12, w12, #1\n\t"
    "cmp w12, #16\n\tblt 1b\n\tsmstop\n\t"
    :: "r"(row), "r"(A), "r"(B) : "p1","p2","z0","z1","z2","w12","x16","za","memory");
}

int main(){
  for(int i=0;i<32;i++){ A[i]=(__fp16)(i+1); B[i]=(__fp16)(i+101); }
  struct sigaction sa; memset(&sa,0,sizeof sa); sa.sa_handler=on_sig; sigemptyset(&sa.sa_mask); sigaction(SIGILL,&sa,0);
  if(sigsetjmp(jb,1)==0){ full_mov(); printf("full_mov OK\n"); for(int r=0;r<3;r++){for(int c=0;c<16;c++)printf("%6.0f",row[r][c]);printf("\n");} }
  else printf("full_mov SIGILL\n");
  return 0;
}
