"""Discrete vs continuous barrier monitoring for a down-and-out call.

Prices the contract with 4 to 252 monitoring dates, with and without the
Broadie-Glasserman-Kou continuity correction, and compares each price with
the continuously monitored closed form. Prints a Markdown table.

Run with: uv run python scripts/barrier_continuity_correction.py
"""

from mc_pricer import RNG, bs_down_and_out_call, mc_price_barrier, simulate_gbm_paths

S0, K, H, T, r, sigma = 100.0, 100.0, 90.0, 1.0, 0.05, 0.2
N_PATHS = 400_000


def main():
    closed = bs_down_and_out_call(S0, K, H, T, r, sigma)
    print(f"Down-and-out call: S0={S0:g}, K={K:g}, H={H:g}, r={r:g}, sigma={sigma:g}, T={T:g}")
    print(f"Continuous closed form: {closed:.4f}   ({N_PATHS:,} paths per row)\n")
    print("| dates | discrete MC | error | BGK-corrected MC | error | stderr |")
    print("|---:|---:|---:|---:|---:|---:|")
    contract = {"barrier_type": "down_out", "option_type": "call"}
    for steps in (4, 12, 52, 252):
        paths = simulate_gbm_paths(
            S0, T, r, sigma, steps=steps, n_paths=N_PATHS, rng=RNG(seed=steps)
        )
        plain = mc_price_barrier(paths, K, H, T, r, **contract)
        bgk = mc_price_barrier(
            paths, K, H, T, r, **contract, sigma=sigma, continuity_correction=True
        )
        print(
            f"| {steps} | {plain['price']:.4f} | {plain['price'] - closed:+.4f} "
            f"| {bgk['price']:.4f} | {bgk['price'] - closed:+.4f} | {plain['stderr']:.4f} |"
        )


if __name__ == "__main__":
    main()
