#include <cstdio>
int main() {
    unsigned long long svl = 0;
    __asm__ volatile("rdsvl %0, #1" : "=r"(svl));   // streaming VL in bytes
    std::printf("streaming VL = %llu bytes = %llu bits -> %llu fp32 lanes\n",
                svl, svl * 8, svl / 4);
    return 0;
}
