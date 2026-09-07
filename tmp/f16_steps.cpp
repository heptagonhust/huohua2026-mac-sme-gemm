#include <cstdio>
#include <cstring>
#include <cstdint>
#include <signal.h>
#include <setjmp.h>

static sigjmp_buf jb;
static void on_sig(int){ siglongjmp(jb, 1); }

static __fp16 A[32], B[32];

// step1: smstart + zero za only
static void step1(){ asm volatile("smstart\n\tzero {za}\n\tsmstop\n\t" :: : "za","memory"); }
// step2: + ptrue/ld1h (no za access beyond zero)
static void step2(){ asm volatile("smstart\n\tzero {za}\n\tptrue p2.h\n\tld1h {z0.h}, p2/z, [%0]\n\tsmstop\n\t" :: "r"(A) : "p2","z0","za","memory"); }
// step3: + fmopa .h
static void step3(){ asm volatile("smstart\n\tzero {za}\n\tptrue p1.s\n\tptrue p2.h\n\tld1h {z0.h}, p2/z, [%0]\n\tld1h {z1.h}, p2/z, [%1]\n\tfmopa za0.s, p1/m, p1/m, z0.h, z1.h\n\tsmstop\n\t" :: "r"(A), "r"(B) : "p1","p2","z0","z1","za","memory"); }
// step4: smstart ZA variant via .inst 0xd503477f then fmopa
static void step4(){ asm volatile(".inst 0xd503477f\n\tzero {za}\n\tptrue p1.s\n\tptrue p2.h\n\tld1h {z0.h}, p2/z, [%0]\n\tld1h {z1.h}, p2/z, [%1]\n\tfmopa za0.s, p1/m, p1/m, z0.h, z1.h\n\t.inst 0xd503467f\n\t" :: "r"(A), "r"(B) : "p1","p2","z0","z1","za","memory"); }

int main(){
  for(int i=0;i<32;i++){ A[i]=(__fp16)(i+1); B[i]=(__fp16)(i+101); }
  struct sigaction sa; memset(&sa,0,sizeof sa); sa.sa_handler=on_sig; sigemptyset(&sa.sa_mask); sigaction(SIGILL,&sa,0);
  struct { const char* n; void(*f)(); } t[] = { {"step1 smstart+zero",step1}, {"step2 +ld1h",step2}, {"step3 +fmopa.h",step3}, {"step4 smstart_za+fmopa",step4} };
  for(auto& e:t){
    if(sigsetjmp(jb,1)==0){ e.f(); printf("%-24s OK\n", e.n); }
    else printf("%-24s SIGILL\n", e.n);
  }
  return 0;
}
