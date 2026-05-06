"""Generate report plots from the static benchmark CSV.

Hardware: Intel i7-12700H + NVIDIA RTX 3060.
"""
import csv
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "/Users/abhishekkakolla/Downloads/Benchmarks - Static.csv"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(OUT_DIR, exist_ok=True)


def load_rows():
    rows = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "impl": r["Implementation"].strip(),
                "scene": r["Scene"].strip(),
                "width": int(r["Image Width"]),
                "spp": int(r["Samples Per Pixel"]),
                "time_ms": float(r["Execution Time (ms)"]),
                "rays": int(r["Traced Rays"]),
                "throughput": float(r["Throughput (GRays/s)"]),
            })
    return rows


def get(rows, impl=None, scene=None):
    out = rows
    if impl is not None:
        out = [r for r in out if r["impl"] == impl]
    if scene is not None:
        out = [r for r in out if r["scene"] == scene]
    return sorted(out, key=lambda r: r["spp"])


def omp_threads(label):
    m = re.search(r"n=(\d+)", label)
    return int(m.group(1)) if m else None


IMPL_COLORS = {
    "Serial": "#444444",
    "OpenMP (n=2)": "#fdae61",
    "OpenMP (n=4)": "#f46d43",
    "OpenMP (n=8)": "#d73027",
    "OpenMP (n=16)": "#a50026",
    "CUDA": "#1a9850",
}


def plot_exec_time_complex(rows):
    impls = ["Serial", "OpenMP (n=2)", "OpenMP (n=4)", "OpenMP (n=8)",
             "OpenMP (n=16)", "CUDA"]
    plt.figure(figsize=(8, 5.5))
    for impl in impls:
        data = get(rows, impl=impl, scene="Complex")
        if not data:
            continue
        xs = [d["spp"] for d in data]
        ys = [d["time_ms"] / 1000.0 for d in data]
        plt.plot(xs, ys, "o-", label=impl, color=IMPL_COLORS[impl], lw=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Samples Per Pixel")
    plt.ylabel("Execution Time (s, log scale)")
    plt.title("Execution Time vs SPP — Complex Scene (1280px)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "exec_time_complex.png"), dpi=150)
    plt.close()


def plot_throughput_complex(rows):
    impls = ["Serial", "OpenMP (n=2)", "OpenMP (n=4)", "OpenMP (n=8)",
             "OpenMP (n=16)", "CUDA"]
    plt.figure(figsize=(8, 5.5))
    for impl in impls:
        data = get(rows, impl=impl, scene="Complex")
        if not data:
            continue
        xs = [d["spp"] for d in data]
        ys = [d["throughput"] * 1000.0 for d in data]  # GRays/s -> MRays/s
        plt.plot(xs, ys, "o-", label=impl, color=IMPL_COLORS[impl], lw=2)
    plt.xscale("log")
    plt.xlabel("Samples Per Pixel")
    plt.ylabel("Throughput (MRays/s)")
    plt.title("Throughput vs SPP — Complex Scene (1280px)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "throughput_complex.png"), dpi=150)
    plt.close()


def plot_speedup_complex(rows):
    impls = ["OpenMP (n=2)", "OpenMP (n=4)", "OpenMP (n=8)",
             "OpenMP (n=16)", "CUDA"]
    serial = {d["spp"]: d["time_ms"] for d in get(rows, impl="Serial", scene="Complex")}
    plt.figure(figsize=(8, 5.5))
    for impl in impls:
        data = get(rows, impl=impl, scene="Complex")
        xs, ys = [], []
        for d in data:
            if d["spp"] in serial:
                xs.append(d["spp"])
                ys.append(serial[d["spp"]] / d["time_ms"])
        plt.plot(xs, ys, "o-", label=impl, color=IMPL_COLORS[impl], lw=2)
    plt.xscale("log")
    plt.xlabel("Samples Per Pixel")
    plt.ylabel("Speedup over Serial")
    plt.title("Speedup vs Serial — Complex Scene (1280px)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "speedup_complex.png"), dpi=150)
    plt.close()


def plot_omp_scaling(rows):
    """OpenMP speedup and efficiency vs thread count, averaged across SPP."""
    serial = {d["spp"]: d["time_ms"] for d in get(rows, impl="Serial", scene="Complex")}
    threads = [1, 2, 4, 8, 16]
    avg_speedup = []
    for t in threads:
        if t == 1:
            avg_speedup.append(1.0)
            continue
        data = get(rows, impl=f"OpenMP (n={t})", scene="Complex")
        sp = [serial[d["spp"]] / d["time_ms"] for d in data if d["spp"] in serial]
        avg_speedup.append(np.mean(sp))

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.plot(threads, avg_speedup, "o-", color="#d73027", lw=2, label="Measured Speedup")
    ax1.plot(threads, threads, "--", color="gray", lw=1.5, label="Ideal (linear)")
    ax1.set_xlabel("OpenMP Threads")
    ax1.set_ylabel("Speedup over Serial")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)
    ax1.set_xticks(threads)
    ax1.set_xticklabels(threads)
    ax1.grid(True, which="both", ls=":", alpha=0.5)

    ax2 = ax1.twinx()
    eff = [s / t * 100 for s, t in zip(avg_speedup, threads)]
    ax2.plot(threads, eff, "s--", color="#1a9850", lw=1.8, label="Efficiency (%)")
    ax2.set_ylabel("Parallel Efficiency (%)")
    ax2.set_ylim(0, 110)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    plt.title("OpenMP Strong Scaling — Complex Scene (avg across SPP)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "openmp_scaling.png"), dpi=150)
    plt.close()


def plot_serial_scenes(rows):
    """Serial throughput by scene."""
    scenes = ["Simple", "Medium", "Complex", "Cornell"]
    plt.figure(figsize=(8, 5.5))
    for scene in scenes:
        data = get(rows, impl="Serial", scene=scene)
        xs = [d["spp"] for d in data]
        ys = [d["throughput"] * 1000.0 for d in data]
        plt.plot(xs, ys, "o-", label=scene, lw=2)
    plt.xscale("log")
    plt.xlabel("Samples Per Pixel")
    plt.ylabel("Throughput (MRays/s)")
    plt.title("Serial Throughput by Scene Complexity")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "serial_scenes.png"), dpi=150)
    plt.close()


def plot_summary_bar(rows):
    """Bar chart: average throughput per implementation on Complex."""
    impls = ["Serial", "OpenMP (n=2)", "OpenMP (n=4)", "OpenMP (n=8)",
             "OpenMP (n=16)", "CUDA"]
    means = []
    for impl in impls:
        data = get(rows, impl=impl, scene="Complex")
        means.append(np.mean([d["throughput"] * 1000.0 for d in data]))
    colors = [IMPL_COLORS[i] for i in impls]
    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(impls, means, color=colors, edgecolor="black")
    for b, m in zip(bars, means):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                 f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Avg Throughput (MRays/s)")
    plt.title("Mean Throughput by Implementation — Complex Scene")
    plt.xticks(rotation=20)
    plt.yscale("log")
    plt.grid(True, axis="y", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "throughput_summary.png"), dpi=150)
    plt.close()


def main():
    rows = load_rows()
    plot_exec_time_complex(rows)
    plot_throughput_complex(rows)
    plot_speedup_complex(rows)
    plot_omp_scaling(rows)
    plot_serial_scenes(rows)
    plot_summary_bar(rows)
    print(f"Plots written to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
