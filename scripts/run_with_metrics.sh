#!/bin/bash
# Wrapper script to run ray tracer and capture peak usage metrics
# Usage: ./run_with_metrics.sh <binary> <args...>

if [ $# -lt 1 ]; then
    echo "Usage: $0 <binary> <args...>"
    exit 1
fi

BINARY=$1
shift

NUM_CPUS=$(nproc)

echo "Running: $BINARY $@"

echo "Logical CPUs: ${NUM_CPUS}"

# For CPU binaries (serial, openmp)
if [[ "$BINARY" == *"serial"* ]] || [[ "$BINARY" == *"openmp"* ]]; then
    TIME_OUTPUT=$(/usr/bin/time -v $BINARY "$@" 2>&1)
    echo "$TIME_OUTPUT" | grep -v "Command being timed"
    PEAK_MEMORY=$(echo "$TIME_OUTPUT" | grep "Maximum resident set size" | awk '{print $6 / 1024}')  # KB to MB
    CPU_PERCENT=$(echo "$TIME_OUTPUT" | grep "Percent of CPU this job got" | awk '{print $7}' | tr -d '%')
    AVG_CPU_PERCENT=$(awk -v p="$CPU_PERCENT" -v n="$NUM_CPUS" 'BEGIN { printf "%.2f", p / n }')
    echo "Peak memory usage: ${PEAK_MEMORY} MB"
    echo "CPU utilization (aggregate): ${CPU_PERCENT}%"
    echo "CPU utilization (avg per core): ${AVG_CPU_PERCENT}%"
fi

# For CUDA binary
if [[ "$BINARY" == *"cuda"* ]]; then
    TIME_OUTPUT=$(/usr/bin/time -v $BINARY "$@" 2>&1)
    echo "$TIME_OUTPUT" | grep -v "Command being timed"
    PEAK_MEMORY=$(echo "$TIME_OUTPUT" | grep "Maximum resident set size" | awk '{print $6 / 1024}')  # KB to MB
    CPU_PERCENT=$(echo "$TIME_OUTPUT" | grep "Percent of CPU this job got" | awk '{print $7}' | tr -d '%')
    AVG_CPU_PERCENT=$(awk -v p="$CPU_PERCENT" -v n="$NUM_CPUS" 'BEGIN { printf "%.2f", p / n }')
    echo "Peak memory usage: ${PEAK_MEMORY} MB"
    echo "CPU utilization (aggregate): ${CPU_PERCENT}%"
    echo "CPU utilization (avg per core): ${AVG_CPU_PERCENT}%"
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
    echo "GPU memory used: ${GPU_MEMORY} MB"
    echo "GPU utilization: ${GPU_UTIL}%"
    echo "Power draw: ${POWER} W"
fi