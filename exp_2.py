import subprocess
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── CONFIG ────────────────────────────────────────────────────────────────────

SLIM_BINARY = "slim"
SLIM_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "whale_sim_p2.slim")

# Bottleneck size fixed at the paper's empirical ENP estimate
BSIZE = 350

# How many independent replicates per selection coefficient
# (stochasticity is high near the recoverability boundary)
N_REPLICATES = 20

# Recovery threshold: π_post / π_pre must reach this to count as "recoverable"
RECOVERY_THRESHOLD = 0.90

# Selection coefficients to sweep (magnitudes; SLiM script negates them).
# Range: weakly deleterious (s = -1e-5) → strongly deleterious (s = -0.5).
# Log-spaced so we get good resolution across orders of magnitude.
SEL_COEFFS = [
    1e-5,   # nearly neutral
    5e-5,
    1e-4,
    5e-4,
    1e-3,   # weakly deleterious (Wayne et al. threshold: s > -0.001)
    5e-3,
    1e-2,   # moderately deleterious boundary (Wayne et al.: s = -0.01)
    5e-2,
    1e-1,   # strongly deleterious
    2e-1,
    5e-1,   # severely deleterious
]

# ─────────────────────────────────────────────────────────────────────────────


def run_single_simulation(sel_s: float, replicate: int = 0) -> dict | None:
    """
    Run one SLiM replicate for a given |s| and return a dict with
    {'pi_pre': float, 'pi_post': float}, or None on failure.

    SEL_S is passed as a *positive* magnitude; the SLiM script negates it.
    """
    cmd = [
        SLIM_BINARY,
        "-d", f"BSIZE={BSIZE}",
        "-d", f"SEL_S={sel_s}",
        SLIM_SCRIPT,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"SLiM binary not found at '{SLIM_BINARY}'.\n"
            "Edit SLIM_BINARY at the top of this script.\n"
            "Run `which slim` to locate it."
        )
    except subprocess.TimeoutExpired:
        print(f"  [WARN] Replicate {replicate} timed out for |s|={sel_s:.2e}")
        return None

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        print(f"  [WARN] SLiM exited {result.returncode} for |s|={sel_s:.2e}")
        if stderr:
            print(f"  stderr: {stderr[:300]}")
        return None

    # Parse both output lines
    pre_match  = re.search(r"PRE_PI:\s*([\d.eE+\-]+)",  stdout)
    post_match = re.search(r"POST_PI:\s*([\d.eE+\-]+)", stdout)

    if pre_match and post_match:
        pi_pre  = float(pre_match.group(1))
        pi_post = float(post_match.group(1))
        return {"pi_pre": pi_pre, "pi_post": pi_post}

    print(f"  [WARN] Could not parse PRE_PI / POST_PI for |s|={sel_s:.2e}")
    print(f"  Raw stdout: {stdout[:300]}")
    return None


def run_sweep(sel_coeffs: list[float],
              n_replicates: int) -> dict[float, list[dict]]:
    NEUTRAL_PI_PRE = 0.000185  # From neutral control runs; used to sanity-check that the script is working

    results: dict[float, list[dict]] = {}

    for s_mag in sel_coeffs:
        print(f"\n|s| = {s_mag:.2e}  (BSIZE = {BSIZE})")
        reps = []
        for rep in range(n_replicates):
            out = run_single_simulation(s_mag, replicate=rep)
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
        results[s_mag] = reps

    return results


def summarise(results: dict[float, list[dict]]) -> dict:
    """
    From raw replicate dicts, compute per-s summary statistics:
      mean_ratio, std_ratio, recoverable (bool), n_valid
    """
    summary = {}
    for s_mag, reps in results.items():
        if not reps:
            summary[s_mag] = dict(mean_ratio=np.nan, std_ratio=np.nan,
                                  recoverable=False, n_valid=0)
            continue
        ratios = [
            (r["pi_post"] / r["pi_pre"] if r["pi_pre"] > 0 else 0.0)
            for r in reps
        ]
        mean_r = float(np.mean(ratios))
        std_r  = float(np.std(ratios))
        summary[s_mag] = dict(
            mean_ratio=mean_r,
            std_ratio=std_r,
            recoverable=(mean_r >= RECOVERY_THRESHOLD),
            n_valid=len(reps),
        )
    return summary


def print_summary_table(summary: dict) -> None:
    print("\n" + "=" * 65)
    print(f"{'|s|':>10}  {'n_reps':>6}  "
          f"{'mean π_post/π_pre':>18}  {'SD':>8}  {'recoverable':>12}")
    print("-" * 65)
    for s_mag in sorted(summary):
        d = summary[s_mag]
        rec_str = "YES" if d["recoverable"] else "NO"
        print(f"{s_mag:>10.2e}  {d['n_valid']:>6}  "
              f"{d['mean_ratio']:>18.4f}  {d['std_ratio']:>8.4f}  "
              f"{rec_str:>12}")
    print("=" * 65)


def plot_results(summary: dict) -> None:
    """
    Two-panel figure:
      Panel 1 — mean π_post/π_pre vs |s|  (error bars = ± 1 SD)
      Panel 2 — recoverability (binary pass/fail) vs |s|
    """
    s_vals    = sorted(summary)
    means     = [summary[s]["mean_ratio"]  for s in s_vals]
    stds      = [summary[s]["std_ratio"]   for s in s_vals]
    recoverable = [summary[s]["recoverable"] for s in s_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "P2 — Effect of deleterious mutation strength on post-bottleneck recovery\n"
        f"(BSIZE = {BSIZE}, based on Wayne et al. 2023 Nat Commun)",
        fontsize=12, y=1.02,
    )

    # ── Panel 1: π ratio ─────────────────────────────────────────────────────
    ax1.errorbar(s_vals, means, yerr=stds,
                 fmt="o-", color="#1b6ca8",
                 capsize=4, linewidth=2, markersize=7,
                 label="Mean π_post / π_pre ± SD")
    ax1.axhline(RECOVERY_THRESHOLD, linestyle="--", color="#e67e22",
                linewidth=1.5,
                label=f"Recovery threshold ({int(RECOVERY_THRESHOLD*100)} %)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Selection coefficient magnitude  |s|  (log scale)",
                   fontsize=11)
    ax1.set_ylabel("π_post / π_pre", fontsize=11)
    ax1.set_title("Relative diversity recovery", fontsize=11)
    ax1.set_ylim(0, 1.15)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ── Panel 2: binary recoverability ───────────────────────────────────────
    colors = ["#27ae60" if r else "#c0392b" for r in recoverable]
    ax2.scatter(s_vals, [int(r) for r in recoverable],
                c=colors, s=120, zorder=3,
                label="Recoverable (green) / Not recoverable (red)")
    ax2.axhline(0.5, linestyle=":", color="gray", linewidth=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("Selection coefficient magnitude  |s|  (log scale)",
                   fontsize=11)
    ax2.set_ylabel(f"Recoverable  (π_post/π_pre ≥ {RECOVERY_THRESHOLD})",
                   fontsize=11)
    ax2.set_title("Recoverability by mutation strength", fontsize=11)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["No", "Yes"])
    ax2.set_ylim(-0.3, 1.3)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Annotate the approximate threshold crossing
    for i, (s, rec) in enumerate(zip(s_vals, recoverable)):
        if i > 0 and recoverable[i - 1] != rec:
            ax2.axvline(s, linestyle="--", color="#8e44ad",
                        linewidth=1.2, alpha=0.7, label="Threshold crossing")
            ax1.axvline(s, linestyle="--", color="#8e44ad",
                        linewidth=1.2, alpha=0.7)
            break

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "bottleneck_p2_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    print(f"SLiM binary   : {SLIM_BINARY}")
    print(f"SLiM script   : {SLIM_SCRIPT}")
    print(f"Replicates    : {N_REPLICATES}")
    print(f"BSIZE (fixed) : {BSIZE}")
    print(f"Sweep |s|     : {SEL_COEFFS}")
    print(f"Recovery threshold: π_post/π_pre >= {RECOVERY_THRESHOLD}\n")

    raw_results = run_sweep(SEL_COEFFS, N_REPLICATES)
    summary     = summarise(raw_results)
    print_summary_table(summary)
    plot_results(summary)
