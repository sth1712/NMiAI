# Astar Island — Complete Analysis Data (R2-R8)

## Round Summary

| Round | Exp | Surv | sf/se | Ruin | Score | Rank | Model |
|-------|------|------|-------|------|-------|------|-------|
| R2 | 0.207 | 0.410 | 0.438 | 0.034 | 49.3 | #77/153 | v2 |
| R3 | 0.003 | 0.018 | 0.435 | 0.004 | 71.2 | #10/100 | v4 |
| R4 | 0.097 | 0.234 | 0.516 | 0.021 | 81.1 | #25/86 | v4 |
| R5 | 0.136 | 0.328 | 0.476 | 0.027 | 68.9 | #65/144 | v5 (no cross-seed) |
| R6 | 0.265 | 0.413 | 0.439 | 0.042 | 68.8 | #56/186 | v6 (clamped) |
| R7 | 0.161 | 0.422 | 0.443 | 0.034 | 63.0 | #59/199 | v6.2 (surv wrong) |
| R8 | 0.026 | 0.069 | 0.497 | 0.011 | 81.0 | #61/214 | v7 (two-param) |

## R8 Detailed Analysis (v7, score=81.0)

### Rates
exp=0.0259 f2s=0.0262 p2s=0.0256 ss=0.0686 se=0.6145 sf=0.3054 sr=0.0106

### Detected vs Actual
- Detected: exp=0.031, surv=0.063
- Actual: exp=0.026, surv=0.069
- Expected surv from exp: 0.115 (would have been WRONG — actual is 0.069)
- Two-param detection gave surv=0.063, much closer to actual 0.069

### Bias (GT - Pred, positive = underpredicted)
- Forest: KL=0.066 [0.025, -0.001, -0.007, -0.005, -0.005, -0.008]
  → Small overshoot on empty, small undershoot on forest. Very good.
- Settlement: KL=0.093 [0.094, 0.004, -0.008, -0.001, -0.080, -0.008]
  → Underpredicted empty by 9.4%, overpredicted forest by 8.0% for settlements
- Plains: KL=0.058 [0.007, -0.001, -0.007, -0.005, 0.013, -0.008]
  → Almost perfect

### R8 Settlement Issue
settlements: se=0.615 (GT), but our model: sf/se invariant gives forest_share
We predicted too much forest (8% over) and too little empty (9.4% under) for settlements.
sf/se ratio for R8 = 0.497 (higher than avg 0.45). Our invariant assumed 0.45.

### Backtesting Ceilings (with PERFECT profiles)
- R2: 84.3, R3: 93.5, R4: 89.9
- R5: 79.1, R6: 79.1, R7: 63.4
- R8: not computed yet

## Cross-Round Invariants

1. forest_to_sett ≈ plains_to_sett (ratio 1.00-1.20)
2. sf/se ratio: 0.435, 0.435, 0.516, 0.476, 0.439, 0.443, 0.497 → avg 0.463, range 0.435-0.516
3. Mountain prob = 0 for non-mountain cells (always)
4. Expansion varies: 0.003 - 0.265
5. Survival varies: 0.018 - 0.422
6. surv = 1.588*exp + 0.074 (R²=0.84) — correlated but NOT identical

## R8 Forest Distance Profile
d=1: [0.099, 0.038, 0.000, 0.006, 0.857, 0]
d=2: [0.089, 0.035, 0.001, 0.006, 0.870, 0]
d=3: [0.077, 0.029, 0.001, 0.005, 0.888, 0]
d=4: [0.068, 0.022, 0.001, 0.004, 0.905, 0]
d=5: [0.053, 0.015, 0.001, 0.002, 0.928, 0]
d=6: [0.018, 0.009, 0.001, 0.002, 0.971, 0]
d=7: [0.005, 0.003, 0.000, 0.001, 0.991, 0]

## What Went RIGHT in R8 (score 81.0)
1. Two-parameter detection: surv=0.063 vs expected 0.115 — caught the harsh winter
2. Low expansion → most cells static → high ceiling
3. Forest bias very small (KL=0.066)
4. Plains almost perfect (KL=0.058)

## What's Still Wrong
1. Settlement forest/empty split: predicted sf/se=0.45 invariant, actual was 0.497
   → 8% forest overprediction, 9.4% empty underprediction
2. Mountain floor still wastes ~0.8% per cell
3. Port/ruin small-sample noise
