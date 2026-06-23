# Synthetic vs real mFDA report audit (D1)

## Summary

- **Synthetic (train+val pooled) mean category coverage:** 1.0
- **Real test mean category coverage:** 1.0
- **Coverage delta (synthetic − real):** 0.0
- **All-7-slots rate — synthetic:** 1.0 | **real:** 1.0
- **Target duplicate rate — synthetic:** 0.952 | **real:** 0.5
- **Breathing presence delta (synthetic − real):** 0.0


## TRAIN (n=99893, is_real=False)

- Mean category coverage: **1.0**
- All-7-slots rate: **1.0**
- Target length (chars): p50 **494**, mean **501.5**
- Duplicate target rate: **0.9493**

| Category | presence rate |
|----------|---------------|
| Breathing | 1.0 |
| Lips | 1.0 |
| Palate | 1.0 |
| Larynx | 1.0 |
| Monotonicity | 1.0 |
| Tongue | 1.0 |
| Intelligibility | 1.0 |

## VAL (n=10000, is_real=False)

- Mean category coverage: **1.0**
- All-7-slots rate: **1.0**
- Target length (chars): p50 **494**, mean **501.9**
- Duplicate target rate: **0.8447**

| Category | presence rate |
|----------|---------------|
| Breathing | 1.0 |
| Lips | 1.0 |
| Palate | 1.0 |
| Larynx | 1.0 |
| Monotonicity | 1.0 |
| Tongue | 1.0 |
| Intelligibility | 1.0 |

## TEST (n=96, is_real=True)

- Mean category coverage: **1.0**
- All-7-slots rate: **1.0**
- Target length (chars): p50 **438**, mean **449.1**
- Duplicate target rate: **0.5**

| Category | presence rate |
|----------|---------------|
| Breathing | 1.0 |
| Lips | 1.0 |
| Palate | 1.0 |
| Larynx | 1.0 |
| Monotonicity | 1.0 |
| Tongue | 1.0 |
| Intelligibility | 1.0 |

### By group (test)
- **PD** (n=48): coverage **1.0**, len p50 **478**
- **HC** (n=48): coverage **1.0**, len p50 **404**

## Interpretation hints

- If **synthetic coverage ≈ 1.0** but **model decode coverage ≈ 0.14**, the gap is likely generation/training dynamics, not missing structure in training targets.
- If **synthetic coverage is well below real test**, re-prepare or fix ETL/report templates before B23.
- High **duplicate_rate** on synthetic train/val suggests memorizable repeated phrases.
