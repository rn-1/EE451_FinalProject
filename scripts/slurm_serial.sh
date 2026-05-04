#!/bin/bash
#SBATCH --job-name=rt_serial
#SBATCH --account=ee451_grp          # UPDATE: your project allocation
#SBATCH --partition=main             # CPU-only partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00              # Serial at 1920x1080 spp=500 can take hours
#SBATCH --output=results/slurm_%j_serial.out
#SBATCH --error=results/slurm_%j_serial.err

module purge
module load gcc/13.3.0

cd "$SLURM_SUBMIT_DIR"
mkdir -p output results

BINARY=./bin/serial_rt
RESULTS=results/timings.csv

# Write CSV header if file is new
if [ ! -f "$RESULTS" ]; then
    echo "impl,scene,width,height,spp,depth,render_ms,ray_count,rays_per_sec,peak_memory_mb" > "$RESULTS"
fi

DEPTH=50

for SCENE in random cornell; do
    for DIMS in "400 225" "800 450" "1920 1080"; do
        W=$(echo $DIMS | cut -d' ' -f1)
        H=$(echo $DIMS | cut -d' ' -f2)
        for SPP in 10 50 100 500; do
            OUTFILE=output/serial_${SCENE}_${W}x${H}_spp${SPP}.ppm
            echo "Running: serial scene=$SCENE res=${W}x${H} spp=$SPP ..."
            TIME_OUTPUT=$(/usr/bin/time -v $BINARY --scene $SCENE --width $W --spp $SPP \
                             --depth $DEPTH --output $OUTFILE --timing-only 2>&1)
            TIMING=$(echo "$TIME_OUTPUT" | grep -E "^[0-9]+\.[0-9]+\t[0-9]+\t[0-9]+\.[0-9]+$" | head -1)
            PEAK_MEMORY=$(echo "$TIME_OUTPUT" | grep "Maximum resident set size" | awk '{print $6 / 1024}')  # Convert KB to MB
            RENDER_MS=$(echo "$TIMING" | cut -f1)
            RAY_COUNT=$(echo "$TIMING" | cut -f2)
            RAYS_PER_SEC=$(echo "$TIMING" | cut -f3)
            echo "serial,$SCENE,$W,$H,$SPP,$DEPTH,$RENDER_MS,$RAY_COUNT,$RAYS_PER_SEC,$PEAK_MEMORY" >> "$RESULTS"
            echo "  -> ${RENDER_MS} ms, ${RAY_COUNT} rays ($(echo "$RAYS_PER_SEC / 1e9" | bc -l) Grays/s), ${PEAK_MEMORY} MB peak memory"
        done
    done
done

echo "Serial benchmark complete. Results in $RESULTS"
