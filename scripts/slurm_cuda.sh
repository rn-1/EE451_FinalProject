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
    echo "impl,scene,width,height,spp,depth,kernel_ms,total_ms,ray_count,rays_per_sec,gpu_memory_mb,gpu_util_pct,power_w,gpu_name" > "$RESULTS"
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
            # cuda_rt --timing-only prints: kernel_ms<TAB>total_ms<TAB>ray_count<TAB>rays_per_sec
            TIMING=$($BINARY --scene $SCENE --width $W --spp $SPP \
                             --depth $DEPTH --output $OUTFILE --timing-only)
            KERNEL_MS=$(echo "$TIMING" | cut -f1)
            TOTAL_MS=$(echo "$TIMING"  | cut -f2)
            RAY_COUNT=$(echo "$TIMING" | cut -f3)
            RAYS_PER_SEC=$(echo "$TIMING" | cut -f4)
            # Query GPU metrics after run
            GPU_MEMORY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
            GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
            POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
            echo "cuda,$SCENE,$W,$H,$SPP,$DEPTH,$KERNEL_MS,$TOTAL_MS,$RAY_COUNT,$RAYS_PER_SEC,$GPU_MEMORY,$GPU_UTIL,$POWER,$GPU_NAME" >> "$RESULTS"
            echo "  -> kernel=${KERNEL_MS} ms  total=${TOTAL_MS} ms  ${RAY_COUNT} rays ($(echo "$RAYS_PER_SEC / 1e9" | bc -l) Grays/s), GPU: ${GPU_MEMORY} MB mem, ${GPU_UTIL}% util, ${POWER} W"
        done
    done
done

echo "CUDA benchmark complete."
