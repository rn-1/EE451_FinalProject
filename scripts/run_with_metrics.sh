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
    # Start GPU monitoring in background
    GPU_LOG=$(mktemp)
    nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw \
               --format=csv,noheader,nounits --loop-ms=100 > "$GPU_LOG" &
    SMI_PID=$!
    
    # Run the CUDA program
    TIME_OUTPUT=$(/usr/bin/time -v $BINARY "$@" 2>&1)
    
    # Stop GPU monitoring
    kill $SMI_PID 2>/dev/null
    wait $SMI_PID 2>/dev/null
    
    echo "$TIME_OUTPUT" | grep -v "Command being timed"
    PEAK_MEMORY=$(echo "$TIME_OUTPUT" | grep "Maximum resident set size" | awk '{print $6 / 1024}')  # KB to MB
    CPU_PERCENT=$(echo "$TIME_OUTPUT" | grep "Percent of CPU this job got" | awk '{print $7}' | tr -d '%')
    AVG_CPU_PERCENT=$(awk -v p="$CPU_PERCENT" -v n="$NUM_CPUS" 'BEGIN { printf "%.2f", p / n }')
    
    # Calculate peak GPU metrics from samples
    GPU_UTIL=$(cut -d',' -f1 "$GPU_LOG" | sort -n | tail -1)
    GPU_MEMORY=$(cut -d',' -f2 "$GPU_LOG" | sort -n | tail -1)
    POWER=$(cut -d',' -f3 "$GPU_LOG" | sort -n | tail -1)
    
    # Cleanup
    rm -f "$GPU_LOG"
    
    echo "Peak memory usage: ${PEAK_MEMORY} MB"
    echo "CPU utilization (aggregate): ${CPU_PERCENT}%"
    echo "CPU utilization (avg per core): ${AVG_CPU_PERCENT}%"
    echo "Peak GPU memory used: ${GPU_MEMORY} MB"
    echo "Peak GPU utilization: ${GPU_UTIL}%"
    echo "Peak power draw: ${POWER} W"
fi