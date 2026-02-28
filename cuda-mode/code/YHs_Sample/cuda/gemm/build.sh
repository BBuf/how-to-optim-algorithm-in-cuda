nvcc ${1} -std=c++11 -arch sm_${2} --ptxas-options=-v -lcublas

