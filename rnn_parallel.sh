#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "Usage: ./rnn_parallel.sh <num_processes> <num_files>"
    exit
fi

rm -f temp_signals/*
rm -f output_signal

num_processes=$1
num_files=$2

chunk_size=$(( $num_files/$num_processes ))

remainder=$(( $num_files%$num_processes ))
#echo $remainder

start=1
core=0
for i in `seq 1 $num_processes`;do
    script="taskset -c "$core" python rnn.py "$start" "$chunk_size

    if [ "$i" -eq "$num_processes" ]; then
	if [ "$remainder" -eq 0 ]; then
	    m=1 #just a placeholder
	else
	    (( chunk_size=chunk_size+remainder ))
	    script="taskset -c "$core" python rnn.py "$start" "$chunk_size
	fi
    fi
    echo $script
    $script &
    (( start=start+chunk_size ))
    (( core=core+1 ))
    if [ "$core" -eq 32 ];then
	$core=0
    fi
done 

wait

#run the reduction
rm -f output_signal

if [ $1 -eq 1 ]
  then
    cp temp_signals/* output_signal
    exit
fi
python rnn_reduce.py
