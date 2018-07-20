#include <iostream>
#include "src/math.h"
using namespace std;

int main() {
    bool flag = InitCUDA();
//    cout << flag << endl;
//    int ans = zxy_cal_squares();
//    cout << ans << endl;
    zxy_matrix_mul();
    return 0;
}