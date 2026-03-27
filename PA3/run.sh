#!/bin/bash

prog=$1
sizes=(1024 2048 4096 8192 16384 32768)

outfile="${prog}.txt"
> "$outfile"

for N in "${sizes[@]}"
do
    echo "N=$N" >> "$outfile"

    # timeout 600s /usr/bin/time -f "time: real=%E user=%U sys=%S" "./$prog" "$N" \
    #     1>/dev/null 2>>"$outfile"
    timeout 600s "./$prog" "$N" 1>>"$outfile"

    status=$?

    if [ $status -eq 0 ]; then
        echo "status: completed" >> "$outfile"
    elif [ $status -eq 124 ]; then
        echo "status: timeout" >> "$outfile"
    else
        echo "status: error ($status)" >> "$outfile"
    fi

    echo "-----------------------------" >> "$outfile"
done