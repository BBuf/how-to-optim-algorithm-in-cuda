#include "hopper-gemm-ws/hopper_gemm_kernel_launch.h"

int main(int argc, char**argv) {

  int m = 8192;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 8192;
  if (argc >= 3)
    sscanf(argv[2], "%d", &n);

  int k = 8192;
  if (argc >= 4)
    sscanf(argv[3], "%d", &k);

  int iterations = 10;

  print("M N K = [%d %d %d].\n", m, n, k);
  
  gemm_tn_launch(m, n, k, iterations);

  return 0;
}
