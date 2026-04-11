# EE451 — CUDA-Accelerated Ray Tracing Engine

Team: Julia Wang, Sahil Pandit, Akul Jindal, Abhishek Kakolla, Ryan Nene

A ray tracing engine implemented in three progressively parallelized versions:
- **Serial** — single-threaded C++
- **OpenMP** — multi-threaded CPU
- **CUDA** — GPU-accelerated

Renders to PPM image files. Benchmark scenes: RTIOW random sphere scene and Cornell Box.

---

## Directory Structure

```
EE451_PROJ/
├── include/               # Shared headers (all three implementations)
│   ├── vec3.h             # 3D vector math
│   ├── ray.h              # Ray class
│   ├── interval.h         # [tmin, tmax] interval
│   ├── utils.h            # PPM writer, timer, CLI arg parser
│   ├── hittable.h         # Abstract Hittable base + hit_record
│   ├── hittable_list.h    # Scene container (CPU)
│   ├── sphere.h           # Sphere primitive
│   ├── quad.h             # Quad/rectangle primitive (Cornell Box walls)
│   ├── material.h         # Lambertian, Metal, Dielectric, DiffuseLight
│   └── camera.h           # Camera params, ray generation, render loop
├── scenes/
│   ├── random_spheres.h   # ~484-sphere RTIOW final scene
│   └── cornell_box.h      # Classic Cornell Box (quads + emissive light)
├── src/
│   ├── serial/main.cpp    # Serial implementation entry point
│   ├── openmp/main.cpp    # OpenMP implementation entry point
│   └── cuda/
│       ├── main.cpp       # CUDA host entry point
│       ├── render.cuh     # cuda_render() declaration
│       ├── render.cu      # CUDA kernels (init_curand, render_kernel)
│       └── device_scene.cuh/.cu  # Device-side scene builder (device new)
├── scripts/
│   ├── slurm_serial.sh    # SLURM job: serial benchmark sweep
│   ├── slurm_openmp.sh    # SLURM job: OpenMP thread-count sweep
│   └── slurm_cuda.sh      # SLURM job: CUDA benchmark sweep
├── output/                # Rendered PPM images (auto-created)
├── results/               # timings.csv and SLURM logs (auto-created)
└── Makefile
```

---

## Building

### On CARC (recommended)

Load the required modules first:

```bash
module load gcc/13.3.0
module load cuda/12.6.3   # only needed for CUDA build
```

Then build whichever target(s) you need:

```bash
make serial    # builds bin/serial_rt
make openmp    # builds bin/openmp_rt
make cuda      # builds bin/cuda_rt  (requires nvcc)
make all       # builds all three
```

Binaries are placed in `bin/`.

> **Note:** `make cuda` must be run on a node with `nvcc` available. On CARC, submit it as a job or use an interactive GPU session rather than the login node.

---

## Running

All three binaries share the same command-line interface:

```
./bin/<binary> --scene <scene> --width <W> --spp <N>
               [--depth <D>] --output <path.ppm> [--timing-only]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--scene` | `random` (RTIOW spheres) or `cornell` (Cornell Box) | `random` |
| `--width` | Image width in pixels (height is derived from aspect ratio) | `400` |
| `--spp` | Samples per pixel (more = less noise, slower) | `100` |
| `--depth` | Maximum ray bounce depth | `50` |
| `--output` | Output PPM file path | `output/<impl>_out.ppm` |
| `--timing-only` | Print render time to stdout only, skip file write | off |
| `--device` | GPU device index *(CUDA only)* | `0` |

### Serial

```bash
./bin/serial_rt --scene random --width 800 --spp 100 --output output/serial.ppm
```

### OpenMP

Control thread count with `OMP_NUM_THREADS`:

```bash
OMP_NUM_THREADS=16 ./bin/openmp_rt --scene random --width 800 --spp 100 \
                                    --output output/openmp.ppm
```

### CUDA

```bash
./bin/cuda_rt --scene random --width 1920 --spp 500 --output output/cuda.ppm
```

Prints both kernel-only time (GPU compute via `cudaEvent`) and total time (including transfers) to stderr.

### Quick smoke tests

```bash
make test-serial    # 200px, 10 spp, random scene
make test-openmp    # same, 4 threads
make test-cuda      # same, GPU
make test-all       # all three
```

---

## Converting PPM to PNG

ImageMagick is available on CARC:

```bash
convert output/cuda.ppm output/cuda.png
```

---

## Benchmarking on CARC

The SLURM scripts sweep all combinations of scene, resolution, and SPP automatically. Results are appended to `results/timings.csv`.

**Before submitting**, update the `--account` line in each script to your allocation:

```bash
# In scripts/slurm_*.sh, change:
#SBATCH --account=ee451_grp   →   #SBATCH --account=<your_account>
```

Check your account name with:
```bash
myaccount
```

### Submit jobs

```bash
sbatch scripts/slurm_serial.sh    # ~8h walltime for full sweep
sbatch scripts/slurm_openmp.sh    # ~4h walltime
sbatch scripts/slurm_cuda.sh      # ~2h walltime
```

Check job status:
```bash
squeue -u $USER
```

### Benchmark sweep

Each script iterates over:
- **Scenes:** `random`, `cornell`
- **Resolutions:** 400×225, 800×450, 1920×1080
- **SPP:** 10, 50, 100, 500
- **Threads (OpenMP only):** 1, 2, 4, 8, 16, 32

Output is appended to `results/timings.csv` with columns:
- Serial/OpenMP: `impl, scene, width, height, spp, depth, render_ms`
- CUDA: `impl, scene, width, height, spp, depth, kernel_ms, total_ms, gpu_name`

`kernel_ms` is measured with `cudaEvent` (GPU compute only). `total_ms` includes cuRAND initialization and host↔device memory transfers.

---

## Implementation Notes

### Parallelization strategy

- **Serial:** Standard recursive `ray_color()` over all pixels.
- **OpenMP:** `#pragma omp parallel for schedule(dynamic, 4)` over scanline rows. Each thread has its own `std::mt19937` seeded by thread ID to avoid data races on the RNG.
- **CUDA:** One thread per pixel (16×16 thread blocks). `ray_color()` is implemented as an iterative bounce loop (no recursion) to avoid device call-stack growth. Each thread has its own `curandState`, initialized in a separate kernel before rendering.

### CUDA virtual dispatch

GPU geometry objects (`DeviceSphere`, `DeviceQuad`) use C++ virtual functions. This requires them to be allocated on the device using device-side `new` inside a single-thread `<<<1,1>>>` setup kernel, so their vtables reside in device memory. The Makefile uses `-rdc=true` (relocatable device code) which is required for this to link correctly.

Materials on the GPU use a flat `MaterialData` struct array + a `switch` statement instead of virtual dispatch, which is more register-friendly in the hot render path.
