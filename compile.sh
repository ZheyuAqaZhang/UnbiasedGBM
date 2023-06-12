cache_dir=$HOME/cache
if [ ! -d $cache_dir ]; then
    mkdir $cache_dir
fi

g++ UnbiasedGBM.cpp -o $HOME/cache/ugb -fopenmp -std=c++11 -O2 -w