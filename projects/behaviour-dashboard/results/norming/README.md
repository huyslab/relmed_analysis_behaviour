# Behaviour Analysis Dashboard

Generated on: 2025-10-24 21:39:34

This dashboard contains all the generated figures from the behaviour analysis.

## 1. Reversal Learning Accuracy Curve

![Reversal Learning Accuracy Curve](reversal_accuracy_curve.svg)


---

**Summary**: Generated 1 figures from the behaviour analysis pipeline.

**Figure files**: All figures are saved as SVG files in the `results/` directory.


### Data Quality Overview

<details><summary>Click to expand</summary>

```text
┌──────────────────────────┬───────────┬───────────────────────┬───────────────────┬──────────────────────────────┬─────────────────────────────┬──────────────────┬───────────────────┬─────────────────────────┬───────────────┬─────────────────────┬──────────────────────┬────────────────┬───────────────────────────┬─────────────┬───────────┬───────────────────┬────────────────────────┐
│             PROLIFIC_PID │   session │ prop_missing_reversal │ prop_missing_pilt │ prop_missing_control_presses │ prop_missing_control_choice │ prop_missing_all │ reversal_accuracy │ reversal_critical_value │ pilt_accuracy │ pilt_critical_value │ n_pilt_quiz_attempts │ max_press_rate │ completion_time_screening │ rt_reversal │   rt_pilt │ focus_loss_events │ fullscreen_exit_events │
│                   String │    String │              Float64? │          Float64? │                     Float64? │                    Float64? │         Float64? │          Float64? │                Float64? │      Float64? │             Missing │               Int64? │       Float64? │                   String? │     String? │   String? │            Int64? │                 Int64? │
├──────────────────────────┼───────────┼───────────────────────┼───────────────────┼──────────────────────────────┼─────────────────────────────┼──────────────────┼───────────────────┼─────────────────────────┼───────────────┼─────────────────────┼──────────────────────┼────────────────┼───────────────────────────┼─────────────┼───────────┼───────────────────┼────────────────────────┤
│ 63d169a794b8af9770a39e7e │ screening │                   0.0 │               0.0 │                    0.0416667 │                        0.25 │         0.021978 │              0.66 │                    0.64 │          0.83 │             missing │                    1 │        11.1429 │                     20:17 │   151 (233) │ 459 (127) │                 0 │                      0 │
│ 6775f7bf80cba9df1032a472 │ screening │                   0.0 │               0.0 │                        0.125 │                         0.0 │        0.0211268 │              0.72 │                    0.62 │          0.42 │             missing │                    2 │        6.85714 │                     16:51 │   221 (259) │ 427 (301) │                 3 │                      0 │
│ 67657a1d50f6c9eb353769fa │ screening │                   0.0 │               0.0 │                    0.0416667 │                         0.0 │        0.0111111 │              0.74 │                    0.62 │           1.0 │             missing │                    1 │        7.57143 │                     12:33 │   186 (158) │ 662 (153) │                 0 │                      0 │
│ 6767202a0793489cda4738d2 │ screening │                   0.0 │               0.0 │                          0.0 │                        0.25 │       0.00892857 │              0.72 │                    0.64 │           0.6 │             missing │                    2 │        5.85714 │                     20:27 │   331 (271) │ 479 (149) │                 1 │                      0 │
└──────────────────────────┴───────────┴───────────────────────┴───────────────────┴──────────────────────────────┴─────────────────────────────┴──────────────────┴───────────────────┴─────────────────────────┴───────────────┴─────────────────────┴──────────────────────┴────────────────┴───────────────────────────┴─────────────┴───────────┴───────────────────┴────────────────────────┘
```

</details>
