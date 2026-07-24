# PLAYBOOK — the best-evidence path to profitability

This is not a guru strategy. Every number below was **measured by this project on real
data** — the backtests, sweeps, walk-forwards and live paper sessions in this repo.
Profitability is not a signal you find; it is three things multiplied together:

> **positive expectancy × survival × time**

Miss any factor and the product is zero. Here is what our own data says about each.

---

## 1. What YOUR data actually found (the evidence table)

| Finding | Measured result | What it means |
|---|---|---|
| **Vol-targeting beats buy & hold on Sharpe** | Sensex 0.75 vs 0.64, S&P 0.51 vs 0.40, **~half the drawdown** (forecast.py, full-cycle) | The single most robust edge we ever measured. Size by volatility, not by feeling. |
| **Selectivity is the real lever** | Crypto conviction gate 40→70: expectancy **+0.52% → +2.63%**/trade, PF **1.19 → 1.88**, trades 5027→340 (signal_perf) | Fewer, better trades. The gate IS the strategy. |
| **Consensus cuts drawdown ~in half** | BTC committee: DD **23% vs 51%** B&H, Sharpe 0.94 vs 0.84 | The 45-agent gate doesn't beat the market; it survives it better. |
| **Walk-forward evaporation** | BTC 1d consensus: in-sample **+0.64%**/trade → out-of-sample **+0.07%**, PF 1.03 (built 2026-07) | ~89% of any backtest "edge" is curve-fit. Trust only the OOS residue. |
| **Frequency kills** | Weekly Sharpe 1.05 (best); hourly **−81%** (fees); 15m MTF even with alignment **−97%** | Trade slower. Every halving of frequency roughly doubles the fee-survival odds. |
| **Shorts are regime-dependent** | Bear basket: long/short +20.7% vs long-only −7.9%. Bull: BTC long-only +119% vs L/S +1.6% | Short only in measured downtrends (regime.py), never by default. |
| **Leverage arithmetic** | 1-hr live session @20x: fees = 4% of margin per round-trip; @125x liquidation sits **inside** every stop | Leverage scales variance and fees, never expectancy. ≤5x tactical, ≤1.5x core. |
| **Calibration is real** | Forecast intervals: 89–90% coverage at 90% target, PIT ≈ 0.50, on ~98yr of S&P | We can't predict direction (~52–55%), but we CAN price the size of tomorrow's move. |
| **Options premium mispricing is detectable** | Bank Nifty: implied 12.97 vs forecast 18.49 → premium CHEAP | A relative-value read; unvalidated P&L (no historical chains) — smallest sleeve. |

**Measured ruin (never again):** BB middle-cross (−87% to −100%), unfiltered intraday,
india_daily ensemble (−6.6%), copying Telegram "80% accurate" channels (our audit of the
same rule class: 48–55%), forcing trades in VOLATILE CHOP.

---

## 2. The strategy — three sleeves and a desk

### Sleeve A — CORE (≈70% of capital): vol-targeted index
The only thing that beat buy & hold on Sharpe in our tests over full cycles.

- Hold ONE broad exposure you believe in for years (Nifty / S&P / BTC — pick per horizon).
- Weekly, size it: `weight = min(1.5, 12% / forecast_vol_annualised)` (forecast.py).
- Vol spikes → the position shrinks *before* the crash deepens; calm → full size.
- Expected: market-like return, **roughly half the drawdown**, Sharpe +0.10–0.15 vs B&H.
- This sleeve is boring. Boring compounds.

### Sleeve B — TACTICAL (≈20%): consensus swing trades, daily bars only
The walk-forward survivor, traded exactly as validated:

- Universe: BTC, ETH + top-liquidity names only. Daily bars. Nothing faster.
- Entry: consensus crossing ≥35/45 **AND** regime agreement (regime.py — with-trend only,
  stand aside in VOLATILE CHOP). Alts additionally need conviction ≥70 (the 2.63%/PF 1.88 gate).
- Exit: 2×ATR target / 1×ATR stop, next-bar-open entries, max-hold timeout — the exact
  signal_backtest recipe, nothing improvised.
- Risk: 1% of equity per trade (1R), max 3 concurrent, shorts only in TRENDING DOWN.
- **Kill rule:** if the journal's rolling 20-trade expectancy goes negative, the sleeve
  stops until the next quarterly re-validation. The sleeve must re-earn its capital.

### Sleeve C — OPTIONS RELATIVE VALUE (≤10%): the lens sleeve
Only when the options lens shows forecast vol vs implied divergence ≥ 3–4 vol points:

- Forecast ≫ implied → premium is cheap → **buy** defined-risk premium (long straddle/option).
- Forecast ≪ implied → premium rich → stay out (selling premium = unlimited risk; not before
  20+ journaled paper trades prove the read).
- This is the *least validated* edge (no historical chains) — hence the smallest sleeve and
  paper-first discipline.

### The desk (risk_pro.py) — non-negotiable, always on
- 1R = 1% of equity. The stop sets the size, never the reverse.
- Portfolio heat ≤ 5%. Daily stop at −3R. Three straight losses → half size.
- Every trade passes the pre-trade checklist (TAKE / HALF / SKIP) and enters the journal.
- Quarterly: re-run walk-forward + auto_learn. **Parameters that fail the OOS gate are
  dropped, even if they "feel" like they work.** The system must keep the right to say no.

---

## 3. What profitable actually looks like

With these sleeves holding their measured numbers out-of-sample:

- **Realistic target: ~12–20%/year at roughly half the market's drawdown.** The out-years
  compound because the down-years are shallow — that is where the wealth comes from.
- NOT 80% win rates (measured reality: 48–55%). NOT doubling monthly (that's the leverage
  lottery — we measured its fee/liquidation arithmetic live). NOT a promise — a hypothesis
  with the strongest evidence this repo has produced, to be validated forward in the journal.

The uncomfortable truth that took this project months of honest measurement:
**the signals were never the edge. Surviving long enough for a modest edge to compound —
that is the entire game.** Everything above is engineered for survival first.
