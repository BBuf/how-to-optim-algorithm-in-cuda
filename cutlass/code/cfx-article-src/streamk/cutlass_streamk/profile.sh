M=1024
N_INCREMENT=128
MAX_N=8096
K_LIST="2048 4096 6144 8192"
SPLITS_LIST="1 2 3 4"

echo "M,N,K,RasterOption,Swizzle,Decomposition,Splits,ms,GFLOP/s,num_worktiles"
for K in $K_LIST; do
  N=$N_INCREMENT
  while [ "$N" -le  "$MAX_N" ] ; do
    ./benchmark --m=$M --n=$N --k=$K --decomposition=dataparallel --raster=H --csv
    for SPLITS in $SPLITS_LIST; do
      ./benchmark --m=$M --n=$N --k=$K --decomposition=splitk --splits=$SPLITS --raster=H --csv
    done
    ./benchmark --m=$M --n=$N --k=$K --decomposition=streamk --raster=H --csv
    ./benchmark --m=$M --n=$N --k=$K --decomposition=heuristic --raster=H --csv
    N=$(( N + N_INCREMENT ))
  done
done
