import subprocess
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── CONFIG ────────────────────────────────────────────────────────────────────

SLIM_BINARY = "slim"
SLIM_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "whale_sim_p3.slim")

# Bottleneck size fixed at the paper's empirical ENP value (Wayne et al. 2023)
BSIZE = 305

# How many independent replicates per duration value
N_REPLICATES = 15

# Recovery threshold (90 % of pre-bottleneck π, per our stated fallback)
RECOVERY_THRESHOLD = 0.90

# Bottleneck durations to sweep (generations).
# Paper: 2 gens and 20 gens.  We add every integer in between and extend
# to 30 to capture any unrecoverable transition beyond the paper's window.
BGENS_RANGE = list(range(2, 92, 10))  # [2, 4, 6, ..., 50] — every 2 gens
# ─────────────────────────────────────────────────────────────────────────────


def run_single_simulation(bgens: int, replicate: int = 0) -> dict | None:

    cmd = [
        SLIM_BINARY,
        "-d", f"BSIZE={BSIZE}",
        "-d", f"BGENS={bgens}",
        SLIM_SCRIPT,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=240,        # P3 runs up to gen ~1050; generous timeout
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"SLiM binary not found at '{SLIM_BINARY}'.\n"
            "Edit SLIM_BINARY at the top of this script.\n"
            "Run `which slim` to locate it."
        )
    except subprocess.TimeoutExpired:
        print(f"  [WARN] Replicate {replicate} timed out for BGENS={bgens}")
        return None

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        print(f"  [WARN] SLiM exited {result.returncode} for BGENS={bgens}")
        if stderr:
            print(f"  stderr: {stderr[:300]}")
        return None

    pre_match  = re.search(r"PRE_PI:\s*([\d.eE+\-]+)",  stdout)
    post_match = re.search(r"POST_PI:\s*([\d.eE+\-]+)", stdout)

    if pre_match and post_match:
        return {
            "pi_pre":  float(pre_match.group(1)),
            "pi_post": float(post_match.group(1)),
        }

    print(f"  [WARN] Could not parse PRE_PI / POST_PI for BGENS={bgens}")
    print(f"  Raw stdout: {stdout[:300]}")
    return None


def run_sweep(bgens_range: list[int],
              n_replicates: int) -> dict[int, list[dict]]:
    """
    Sweep across bottleneck durations.
    Returns {bgens: [{'pi_pre': ..., 'pi_post': ...}, ...]}
    """
    results: dict[int, list[dict]] = {}

    for bgens in bgens_range:
        print(f"\nBottleneck duration = {bgens} generation(s)  (BSIZE = {BSIZE})")
        reps = []
        for rep in range(n_replicates):
            out = run_single_simulation(bgens, replicate=rep)
            if out is not None:
                ratio = (out["pi_post"] / out["pi_pre"]
                         if out["pi_pre"] > 0 else 0.0)
                print(f"  rep {rep+1}/{n_replicates}: "
                      f"π_pre={out['pi_pre']:.6f}  "
                      f"π_post={out['pi_post']:.6f}  "
                      f"ratio={ratio:.3f}")
                reps.append(out)
            else:
                print(f"  rep {rep+1}/{n_replicates}: failed / extinct")
        results[bgens] = reps

    return results


def summarise(results: dict[int, list[dict]]) -> dict:
    """
    Compute per-duration summary statistics:
      mean_ratio, std_ratio, mean_pi_post, std_pi_post, recoverable, n_valid
    """
    summary = {}
    for bgens, reps in results.items():
        if not reps:
            summary[bgens] = dict(mean_ratio=np.nan, std_ratio=np.nan,
                                  mean_pi_post=np.nan, std_pi_post=np.nan,
                                  recoverable=False, n_valid=0)
            continue
        ratios    = [(r["pi_post"] / r["pi_pre"] if r["pi_pre"] > 0 else 0.0)
                     for r in reps]
        pi_posts  = [r["pi_post"] for r in reps]
        mean_r    = float(np.mean(ratios))
        summary[bgens] = dict(
            mean_ratio=mean_r,
            std_ratio=float(np.std(ratios)),
            mean_pi_post=float(np.mean(pi_posts)),
            std_pi_post=float(np.std(pi_posts)),
            recoverable=(mean_r >= RECOVERY_THRESHOLD),
            n_valid=len(reps),
        )
    return summary


def print_summary_table(summary: dict) -> None:
    print("\n" + "=" * 68)
    print(f"{'BGENS':>6}  {'n_reps':>6}  "
          f"{'mean π_post/π_pre':>18}  {'SD':>8}  {'recoverable':>12}")
    print("-" * 68)
    for bgens in sorted(summary):
        d = summary[bgens]
        rec_str = "YES" if d["recoverable"] else "NO"
        # Flag the paper's two reported timepoints
        marker = " ◀ paper" if bgens in (2, 20) else ""
        print(f"{bgens:>6}  {d['n_valid']:>6}  "
              f"{d['mean_ratio']:>18.4f}  {d['std_ratio']:>8.4f}  "
              f"{rec_str:>12}{marker}")
    print("=" * 68)


def plot_results(summary: dict) -> None:
    """
    Three-panel figure:
      Panel 1 — mean π_post vs bottleneck duration  (absolute diversity)
      Panel 2 — mean π_post / π_pre vs duration  (relative; threshold line)
      Panel 3 — recoverability (binary) vs duration
    """
    bgens_vals  = sorted(summary)
    means_ratio = [summary[b]["mean_ratio"]    for b in bgens_vals]
    stds_ratio  = [summary[b]["std_ratio"]     for b in bgens_vals]
    means_post  = [summary[b]["mean_pi_post"]  for b in bgens_vals]
    stds_post   = [summary[b]["std_pi_post"]   for b in bgens_vals]
    recoverable = [summary[b]["recoverable"]   for b in bgens_vals]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "P3 — Effect of bottleneck duration on post-recovery genetic diversity\n"
        f"(BSIZE = {BSIZE}, neutral mutations only; cf. Wayne et al. 2023 Fig. 5A)",
        fontsize=12, y=1.02,
    )

    # ── Panel 1: absolute π_post ─────────────────────────────────────────────
    ax1 = axes[0]
    ax1.errorbar(bgens_vals, means_post, yerr=stds_post,
                 fmt="o-", color="#1b6ca8",
                 capsize=4, linewidth=2, markersize=6,
                 label="Mean π_post ± SD")
    # Mark the paper's two timepoints
    for ref_gen in (2, 20):
        if ref_gen in summary and not np.isnan(summary[ref_gen]["mean_pi_post"]):
            ax1.axvline(ref_gen, linestyle=":", color="#7f8c8d",
                        linewidth=1.2, alpha=0.8)
            ax1.annotate(f"Paper: {ref_gen} gen",
                         xy=(ref_gen, summary[ref_gen]["mean_pi_post"]),
                         xytext=(ref_gen + 0.5,
                                 summary[ref_gen]["mean_pi_post"] * 1.03),
                         fontsize=8, color="#7f8c8d")
    ax1.set_xlabel("Bottleneck duration (generations)", fontsize=11)
    ax1.set_ylabel("Nucleotide diversity  π_post", fontsize=11)
    ax1.set_title("Absolute diversity after recovery", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ── Panel 2: relative diversity ratio ────────────────────────────────────
    ax2 = axes[1]
    ax2.errorbar(bgens_vals, means_ratio, yerr=stds_ratio,
                 fmt="s-", color="#c0392b",
                 capsize=4, linewidth=2, markersize=6,
                 label="Mean π_post / π_pre ± SD")
    ax2.axhline(RECOVERY_THRESHOLD, linestyle="--", color="#e67e22",
                linewidth=1.5,
                label=f"Recovery threshold ({int(RECOVERY_THRESHOLD*100)} %)")
    for ref_gen in (2, 20):
        if ref_gen in summary:
            ax2.axvline(ref_gen, linestyle=":", color="#7f8c8d",
                        linewidth=1.2, alpha=0.8)
    ax2.set_xlabel("Bottleneck duration (generations)", fontsize=11)
    ax2.set_ylabel("π_post / π_pre", fontsize=11)
    ax2.set_title("Relative diversity recovery", fontsize=11)
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Annotate threshold crossing (first generation where recoverable = False)
    for i, (b, rec) in enumerate(zip(bgens_vals, recoverable)):
        if i > 0 and not rec and recoverable[i - 1]:
            ax2.axvline(b, linestyle="--", color="#8e44ad",
                        linewidth=1.3, alpha=0.8,
                        label=f"Threshold crossing (~{b} gens)")
            ax1.axvline(b, linestyle="--", color="#8e44ad",
                        linewidth=1.3, alpha=0.8)
            ax2.legend(fontsize=9)
            break

    # ── Panel 3: binary recoverability ───────────────────────────────────────
    ax3 = axes[2]
    colors = ["#27ae60" if r else "#c0392b" for r in recoverable]
    ax3.bar(bgens_vals, [int(r) for r in recoverable],
            color=colors, edgecolor="white", linewidth=0.5)
    ax3.axhline(0.5, linestyle=":", color="gray", linewidth=1)
    ax3.set_xlabel("Bottleneck duration (generations)", fontsize=11)
    ax3.set_ylabel(f"Recoverable  (π_post/π_pre ≥ {RECOVERY_THRESHOLD})",
                   fontsize=11)
    ax3.set_title("Recoverability by duration", fontsize=11)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["No", "Yes"])
    ax3.set_ylim(-0.2, 1.3)
    ax3.set_xticks(bgens_vals)
    ax3.tick_params(axis="x", rotation=45)
    # Reference lines at paper's timepoints
    for ref_gen in (2, 20):
        ax3.axvline(ref_gen, linestyle=":", color="#7f8c8d",
                    linewidth=1.2, alpha=0.8)
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#27ae60", label="Recoverable"),
        Patch(facecolor="#c0392b", label="Not recoverable"),
    ]
    ax3.legend(handles=legend_elements, fontsize=9)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "bottleneck_p3_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    print(f"SLiM binary      : {SLIM_BINARY}")
    print(f"SLiM script      : {SLIM_SCRIPT}")
    print(f"Replicates       : {N_REPLICATES}")
    print(f"BSIZE (fixed)    : {BSIZE}")
    print(f"Duration sweep   : {BGENS_RANGE[0]}–{BGENS_RANGE[-1]} generations")
    print(f"Recovery threshold: π_post/π_pre >= {RECOVERY_THRESHOLD}\n")

    raw_results = run_sweep(BGENS_RANGE, N_REPLICATES)
    summary     = summarise(raw_results)
    print_summary_table(summary)
    plot_results(summary)
