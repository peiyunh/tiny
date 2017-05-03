%  FILE:   cnn_widerface.m
%
%    This script compiles compute_dense_overlap.cc 
%

mex compute_dense_overlap.cc CXXFLAGS='$CXXFLAGS -fopenmp' ...
    LDFLAGS='$LDFLAGS -fopenmp' CXXOPTIMFLAGS='-O3 -DNDEBUG' 