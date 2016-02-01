#!/bin/bash

RUNS=5
SIZE="80 84 88 92 96 100 104 108 112 116 120 124 128 132 136 140 144 148 152 156 160 164 168 172 176 180 184 188 192 196 200 204 208 212 216 220 224 228 232 236 240 244 248 252 256 260 264 268 272 276"
FILE="o3.txt"

# All tests
for t in `seq 1 5`; do

	echo "Running Test $t ... "
	
	for i in $SIZE; do

		echo "    $t $run $i ..."

		for run in `seq 1 $RUNS`; do

			echo "$t $run $i ..."
			./bicgstab_cl --gpu --stats -t $t -n $i >>$FILE

		done
	
	done

done

