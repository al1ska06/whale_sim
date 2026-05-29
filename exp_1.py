import subprocess
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── CONFIG ────────────────────────────────────────────────────────────────────

SLIM_BINARY = "slim"
SLIM_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whale_sim.slim")

# How many independent replicates to run per bottleneck size
N_REPLICATES = 15

# Bottleneck sizes to sweep, meaning that the population is reduced to this size for 100 generations before recovery.
BOTTLENECK_SIZES = [10, 50, 100, 250, 500, 1000, 2000]
# ─────────────────────────────────────────────────────────────────────────────

# calling whale_sim.slim, the output of which is the pi value (or, biologically,the mean pairwise diversity)
def run_single_simulation(bottleneck_size: int, replicate: int = 0) -> float | None:
    """
    Run one SLiM replicate and return the parsed π value, or None on failure.
    SLiM is seeded differently each call via the OS random seed.
    """
    cmd = [
        SLIM_BINARY,
        "-d", f"BSIZE={bottleneck_size}",
        SLIM_SCRIPT,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"SLiM binary not found at '{SLIM_BINARY}'.\n"
            "Edit SLIM_BINARY at the top of this script to point at your slim executable.\n"
            "Run `which slim` or `find ~ -name slim -type f` to locate it."
        )
    except subprocess.TimeoutExpired:
        print(f"  [WARN] Replicate {replicate} timed out for BSIZE={bottleneck_size}")
        return None

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        print(f"  [WARN] SLiM exited with code {result.returncode} for BSIZE={bottleneck_size}")
        if stderr:
            print(f"  SLiM stderr: {stderr[:300]}")
        return None

    # Parse the clearly-marked output line:  RESULT_PI: 0.000123
    match = re.search(r"RESULT_PI:\s*([\d.eE+\-]+)", stdout)
    if match:
        return float(match.group(1))

    print(f"  [WARN] Could not parse RESULT_PI from stdout for BSIZE={bottleneck_size}")
    print(f"  Raw stdout: {stdout[:300]}")
    return None


def run_parameter_sweep(
    bottleneck_sizes: list[int],
    n_replicates: int,
) -> dict[int, list[float]]:
    """
    Run the full sweep. Returns {bsize: [pi_rep1, pi_rep2, ...]}.
    """
    results: dict[int, list[float]] = {}

    for bsize in bottleneck_sizes:
        print(f"\nBottleneck size N={bsize}")
        pi_values = []

        for rep in range(n_replicates):
            pi = run_single_simulation(bsize, replicate=rep)
            if pi is not None:
                pi_values.append(pi)
                print(f"  rep {rep+1}/{n_replicates}: π = {pi:.6f}")
            else:
                print(f"  rep {rep+1}/{n_replicates}: failed / extinct")

        results[bsize] = pi_values

    return results


def plot_results(results: dict[int, list[float]]) -> None:
    """
    Produce two panels:
      1. Mean π vs bottleneck size  (with ± 1 SD error bars)
      2. Relative diversity loss compared to largest bottleneck size
    """
    sizes  = [s for s in sorted(results.keys()) if results[s]]
    means  = [np.mean(results[s]) for s in sizes]
    stds   = [np.std(results[s])  for s in sizes]

    # Normalise against the largest (least-bottlenecked) population
    baseline = means[-1] if means[-1] > 0 else 1.0
    relative = [m / baseline for m in means]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Simulated genomic footprint of a population bottleneck\n"
        "(cf. Wayne et al. 2023, Nat Commun — fin whale whaling collapse)",
        fontsize=12, y=1.02,
    )

    # ── Panel 1: absolute π ──────────────────────────────────────────────────
    ax1.errorbar(sizes, means, yerr=stds, fmt="o-", color="#1b6ca8",
                 capsize=4, linewidth=2, markersize=7, label="Mean π ± SD")
    ax1.set_xscale("log")
    ax1.set_xlabel("Bottleneck population size  (log scale)", fontsize=11)
    ax1.set_ylabel("Nucleotide diversity  π", fontsize=11)
    ax1.set_title("Absolute nucleotide diversity", fontsize=11)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.set_xticks(sizes)
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ── Panel 2: relative diversity ──────────────────────────────────────────
    ax2.plot(sizes, [r * 100 for r in relative], "s-", color="#c0392b",
             linewidth=2, markersize=7, label="% of pre-bottleneck diversity")
    ax2.axhline(100, linestyle="--", color="gray", linewidth=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("Bottleneck population size  (log scale)", fontsize=11)
    ax2.set_ylabel("Diversity retained  (%)", fontsize=11)
    ax2.set_title("Relative diversity loss", fontsize=11)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.set_xticks(sizes)
    ax2.tick_params(axis="x", rotation=45)
    ax2.set_ylim(0, 115)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bottleneck_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.show()


def print_summary(results: dict[int, list[float]]) -> None:
    print("\n" + "=" * 55)
    print(f"{'BSIZE':>8}  {'n_reps':>6}  {'mean π':>12}  {'SD':>10}")
    print("-" * 55)
    for bsize in sorted(results.keys()):
        vals = results[bsize]
        if vals:
            print(f"{bsize:>8}  {len(vals):>6}  {np.mean(vals):>12.6f}  {np.std(vals):>10.6f}")
        else:
            print(f"{bsize:>8}  {0:>6}  {'EXTINCT/FAIL':>12}")
    print("=" * 55)


if __name__ == "__main__":
    print(f"SLiM binary : {SLIM_BINARY}")
    print(f"SLiM script : {SLIM_SCRIPT}")
    print(f"Replicates  : {N_REPLICATES}")
    print(f"Sweep       : {BOTTLENECK_SIZES}\n")

    results = run_parameter_sweep(BOTTLENECK_SIZES, N_REPLICATES)
    print_summary(results)
    plot_results(results)