#include <iostream>
#include "src/math.h"
#include <cuda_runtime.h>
using namespace std;

int main() {
    bool flag = InitCUDA();
    cout << flag << endl;
    int ans = zxy_cal_squares();
    cout << ans << endl;
    return 0;
}