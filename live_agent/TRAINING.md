# How to Train This Agent to Its Best

A practical, honest playbook for making the system as good as it can be.

> **Read this first.** "Training" here means making the agent *adapt and stay
> validated* — not conjuring guaranteed profit. No training loop can do that;
> markets are adversarial and non-stationary. What you *can* achieve is a
> disciplined, risk-controlled system whose edge (mostly lower drawdown and
> selectivity) is **measured, not assumed**. The goal is to find the most
> defensible configuration and to *know its real numbers* — including when the
> honest answer is "no edge here, don't trade it."

---

## The core training loop

```
   paper-trade / backtest  →  MEASURE (honest metrics)  →  tune ONE knob
        ↑                                                        │
        └──────────  re-validate OUT-OF-SAMPLE  ←────────────────┘
```

The only rule that matters: **every change must prove itself on data it did not
see.** If a tweak only looks good in-sample, throw it away. The tools below all
have an out-of-sample (OOS) discipline built in.

---

## Step 1 — Establish the honest baseline

Before tuning anything, know where you stand.

```bash
source ../venv/bin/activate

# Real directional hit-rate + expectancy (the truth vs "80% channels")
python signal_perf.py --market crypto_daily --years 3 --horizon 5
python signal_perf.py --market us_tech_daily --years 3
python signal_perf.py --market india_daily   --years 3

# Strategy vs buy-and-hold, per market
python backtest.py --all --years 3
```

Write down, per market: **hit rate, expectancy, profit factor, Sharpe, max
drawdown.** These are your scorecard. A change is only an improvement if it
beats *these* out-of-sample.

**What "good" looks like (be realistic):**
- Hit rate **50–56%** (NOT 80% — anyone claiming that is lying).
- **Expectancy > 0** and **profit factor > 1** — these decide profit, not hit rate.
- Sharpe **> 1** is good; the system's real strength is usually **lower drawdown** than buy-and-hold.

---

## Step 2 — Let it learn (the automated trainers)

Two trainers adapt the model with an OOS gate that *refuses to overfit*:

```bash
# Re-fit the ensemble signal weights, walk-forward, adopt only if it generalises
python auto_learn.py

# Search the committee consensus threshold, OOS-validated, adopt only past the gate
python train_strategy.py --market crypto_daily --years 3
```

Run these **weekly**. Sometimes they will **refuse to update** ("not adopted")
— that is correct behaviour, not a failure. A trainer that always "finds a
winner" is fooling you.

---

## Step 3 — Tune the knobs (one at a time)

All live in `config.yaml`. Change **one**, re-run Step 1, keep it only if the
OOS scorecard improves.

| Knob | Where | Effect | Honest note |
|---|---|---|---|
| `strategy.weights` | config | momentum vs mean-rev vs trend mix | let `auto_learn.py` set these |
| `committee.threshold` | config | how many of 45 agents must agree | higher = fewer, higher-quality trades |
| `risk.entry_threshold` | config | min signal strength to act | raises selectivity |
| Conviction gate | Hit-rate tab / signals | min conviction to fire | **the biggest lever** — see Step 4 |
| `risk.max_position_weight`, `max_gross_exposure` | config | sizing / leverage | keep conservative |
| `risk.volatility_penalty_weight` | config | risk aversion in reward | higher = calmer equity |
| `universe` / market presets | config | what you trade | more liquid = better fills |

---

## Step 4 — The single most effective lever: **selectivity**

The honest way to raise your odds is *trade less, but only the best setups.*
Open the **📊 Hit rate** tab (or `signal_perf.py`) and look at the conviction
sweep. You will see the same pattern everywhere:

```
 conviction ≥ 40  → many signals, ~49% hit, low expectancy
 conviction ≥ 70  → few signals,  ~55% hit, much higher expectancy & profit factor
```

**Pick the conviction level with the best expectancy / profit factor you can
still get enough trades at.** That single choice usually does more than any
fancy feature. It works identically for **long and short**.

---

## Step 5 — Forward-validate on paper (the part that actually matters)

Backtests flatter themselves. The real test is **time**.

1. Set up the scheduled jobs so the system runs itself:
   ```bash
   cp com.rohitraj.*.plist ~/Library/LaunchAgents/
   launchctl load ~/Library/LaunchAgents/com.rohitraj.scan.plist
   launchctl load ~/Library/LaunchAgents/com.rohitraj.signals.plist
   # (plus autolearn weekly, alerts, daily trade)
   ```
2. Run **autonomous paper sessions** (🟢 Live trading) regularly.
3. **Watch for weeks.** Track the live equity curves and the Hit-rate tab over
   time. Judge on **risk-adjusted** results, not a single good day.

Only if the *forward* paper record holds up for weeks/months should you even
think about real money — and start tiny.

---

## The discipline (what separates training from self-deception)

- **Out-of-sample or it didn't happen.** In-sample results are worthless.
- **Don't chase the in-sample winner.** The trainer shows this trap: the setting
  that looked best in-sample usually generalises *worse*.
- **Model costs honestly.** Slippage + fees kill intraday edges (the hourly
  backtest lost ~80% to costs). Keep `transaction_cost` realistic or higher.
- **Beware overfitting.** 45 agents × many knobs × many markets = easy to find a
  fluke. The OOS gates exist to stop you.
- **Expectancy > hit rate.** A 48% hit rate with big wins beats 70% with big losses.
- **Leverage is a magnifier of ruin.** The 125× sim shows why — use it to learn,
  not to trade.
- **No external signal channel is the missing piece.** "80% correct" is marketing;
  your edge, if any, is selectivity + risk control + patience.

---

## A simple weekly routine

1. `python universe.py` — refresh symbols (or let the nightly job do it).
2. `python auto_learn.py` and `python train_strategy.py --market <m>` — adapt + OOS-gate.
3. `python signal_perf.py --market <m>` — re-check the honest hit-rate.
4. `python backtest.py --all` — strategy vs buy-and-hold per market.
5. Glance at the live paper equity curves. **Change at most one knob.**
6. Write down the scorecard. Compare to last week. Keep only what improved OOS.

Do this for a couple of months before drawing any conclusion. That patience —
not another feature — is what "training it to the best" actually means.
