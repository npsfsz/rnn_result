#!/bin/bash 

time ./rnn_parallel.sh 1 64 > 1.result
time ./rnn_parallel.sh 2 64 > 2.result
time ./rnn_parallel.sh 4 64 > 4.result
time ./rnn_parallel.sh 8 64 > 8.result
time ./rnn_parallel.sh 16 64  > 16.result
