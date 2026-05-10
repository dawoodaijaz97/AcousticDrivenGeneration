# ETL report

This report summarizes the outputs created by the ETL pipeline in `etl/`.

## Dataset sizes and splits

| train_size | split | rows | is_real | group_counts |
| --- | --- | --- | --- | --- |
| 1k | train | 1000 | `{"False": 1000}` | `{"None": 1000}` |
| 1k | val | 10000 | `{"False": 10000}` | `{"None": 10000}` |
| 1k | test | 96 | `{"True": 96}` | `{"HC": 48, "PD": 48}` |
| 10k | train | 10000 | `{"False": 10000}` | `{"None": 10000}` |
| 10k | val | 9998 | `{"False": 9998}` | `{"None": 9998}` |
| 10k | test | 96 | `{"True": 96}` | `{"HC": 48, "PD": 48}` |
| 100k | train | 99893 | `{"False": 99893}` | `{"None": 99893}` |
| 100k | val | 9974 | `{"False": 9974}` | `{"None": 9974}` |
| 100k | test | 96 | `{"True": 96}` | `{"HC": 48, "PD": 48}` |

## Leakage / overlap handling

| train_size | val_dropped_due_to_train_overlap | test_dropped_due_to_train_or_val_overlap |
| --- | --- | --- |
| 1k | 0 | 0 |
| 10k | 2 | 0 |
| 100k | 26 | 0 |

## Text length stats (chars)

| train_size | split | input_min | input_p50 | input_max | target_min | target_p50 | target_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1k | train | 100 | 106 | 106 | 384 | 491 | 646 |
| 1k | val | 98 | 106 | 106 | 384 | 494 | 646 |
| 1k | test | 100 | 106 | 106 | 396 | 438 | 570 |
| 10k | train | 98 | 106 | 106 | 384 | 494 | 646 |
| 10k | val | 98 | 106 | 106 | 384 | 494 | 646 |
| 10k | test | 100 | 106 | 106 | 396 | 438 | 570 |
| 100k | train | 96 | 106 | 106 | 375 | 494 | 659 |
| 100k | val | 98 | 106 | 106 | 384 | 494 | 646 |
| 100k | test | 100 | 106 | 106 | 396 | 438 | 570 |

## Figures (per split)

For each `train_size`, the following figures are generated under `processed/<train_size>/dashboard_report/`.

### 1k

- Features: `train_features.png`, `val_features.png`, `test_features.png`
- Lengths: `train_lengths.png`, `val_lengths.png`, `test_lengths.png`

![1k train features](../../processed/1k/dashboard_report/train_features.png)
![1k train lengths](../../processed/1k/dashboard_report/train_lengths.png)

![1k val features](../../processed/1k/dashboard_report/val_features.png)
![1k val lengths](../../processed/1k/dashboard_report/val_lengths.png)

![1k test features](../../processed/1k/dashboard_report/test_features.png)
![1k test lengths](../../processed/1k/dashboard_report/test_lengths.png)

### 10k

- Features: `train_features.png`, `val_features.png`, `test_features.png`
- Lengths: `train_lengths.png`, `val_lengths.png`, `test_lengths.png`

![10k train features](../../processed/10k/dashboard_report/train_features.png)
![10k train lengths](../../processed/10k/dashboard_report/train_lengths.png)

![10k val features](../../processed/10k/dashboard_report/val_features.png)
![10k val lengths](../../processed/10k/dashboard_report/val_lengths.png)

![10k test features](../../processed/10k/dashboard_report/test_features.png)
![10k test lengths](../../processed/10k/dashboard_report/test_lengths.png)

### 100k

- Features: `train_features.png`, `val_features.png`, `test_features.png`
- Lengths: `train_lengths.png`, `val_lengths.png`, `test_lengths.png`

![100k train features](../../processed/100k/dashboard_report/train_features.png)
![100k train lengths](../../processed/100k/dashboard_report/train_lengths.png)

![100k val features](../../processed/100k/dashboard_report/val_features.png)
![100k val lengths](../../processed/100k/dashboard_report/val_lengths.png)

![100k test features](../../processed/100k/dashboard_report/test_features.png)
![100k test lengths](../../processed/100k/dashboard_report/test_lengths.png)

