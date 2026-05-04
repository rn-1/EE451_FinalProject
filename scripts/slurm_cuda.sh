#!/bin/bash
#SBATCH --job-name=rt_cuda
#SBATCH --account=ee451_grp          # UPDATE: your project allocation
#SBATCH --partition=gpu              # GPU partition — check CARC docs for correct name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm_%j_cuda.out
#SBATCH --error=results/slurm_%j_cuda.err

module purge
module load gcc/13.3.0
module load cuda/12.6.3

cd "$SLURM_SUBMIT_DIR"
mkdir -p output results

BINARY=./bin/cuda_rt
RESULTS=results/timings.csv

if [ ! -f "$RESULTS" ]; then
    echo "impl,scene,width,height,spp,depth,kernel_ms,total_ms,ray_count,rays_per_sec,gpu_memory_peak_mb,gpu_memory_avg_mb,gpu_util_peak_pct,gpu_util_avg_pct,power_peak_w,power_avg_w,gpu_name" > "$RESULTS"
fi

DEPTH=50
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
echo "GPU: $GPU_NAME"

for SCENE in random cornell; do
    for DIMS in "400 225" "800 450" "1920 1080"; do
        W=$(echo $DIMS | cut -d' ' -f1)
        H=$(echo $DIMS | cut -d' ' -f2)
        for SPP in 10 50 100 500; do
            OUTFILE=output/cuda_${SCENE}_${W}x${H}_spp${SPP}.ppm
            echo "Running: cuda scene=$SCENE res=${W}x${H} spp=$SPP ..."
            
            # Start GPU monitoring in background
            GPU_LOG=$(mktemp)
            nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw \
                       --format=csv,noheader,nounits --loop-ms=100 > "$GPU_LOG" &
            SMI_PID=$!
            
            # Run the CUDA program
            TIMING=$($BINARY --scene $SCENE --width $W --spp $SPP \
                             --depth $DEPTH --output $OUTFILE --timing-only)
            
            # Stop GPU monitoring
            kill $SMI_PID 2>/dev/null
            wait $SMI_PID 2>/dev/null
            
            # Parse timing output
            KERNEL_MS=$(echo "$TIMING" | cut -f1)
            TOTAL_MS=$(echo "$TIMING"  | cut -f2)
            RAY_COUNT=$(echo "$TIMING" | cut -f3)
            RAYS_PER_SEC=$(echo "$TIMING" | cut -f4)
            
            # Calculate peak GPU metrics from samples
            GPU_UTIL_PEAK=$(cut -d',' -f1 "$GPU_LOG" | sort -n | tail -1)
            GPU_MEMORY_PEAK=$(cut -d',' -f2 "$GPU_LOG" | sort -n | tail -1)
            POWER_PEAK=$(cut -d',' -f3 "$GPU_LOG" | sort -n | tail -1)
            
            # Calculate average GPU metrics
            GPU_UTIL_AVG=$(cut -d',' -f1 "$GPU_LOG" | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "0"}')
            GPU_MEMORY_AVG=$(cut -d',' -f2 "$GPU_LOG" | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "0"}')
            POWER_AVG=$(cut -d',' -f3 "$GPU_LOG" | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "0"}')
            
            # Cleanup
            rm -f "$GPU_LOG"
            echo "cuda,$SCENE,$W,$H,$SPP,$DEPTH,$KERNEL_MS,$TOTAL_MS,$RAY_COUNT,$RAYS_PER_SEC,$GPU_MEMORY_PEAK,$GPU_MEMORY_AVG,$GPU_UTIL_PEAK,$GPU_UTIL_AVG,$POWER_PEAK,$POWER_AVG,$GPU_NAME" >> "$RESULTS"
            echo "  -> kernel=${KERNEL_MS} ms  total=${TOTAL_MS} ms  ${RAY_COUNT} rays ($(echo "$RAYS_PER_SEC / 1e9" | bc -l) Grays/s), GPU: ${GPU_MEMORY_PEAK} MB peak mem (${GPU_MEMORY_AVG} MB avg), ${GPU_UTIL_PEAK}% peak util (${GPU_UTIL_AVG}% avg), ${POWER_PEAK} W peak (${POWER_AVG} W avg)"
        done
    done
done

echo "CUDA benchmark complete."
