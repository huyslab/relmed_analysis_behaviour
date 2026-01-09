Refactor helpers to reduce duplication in PILT_recovery.R

Files:
- models_registry.R: R-only registry of models (Stan paths, parameter names, defaults, prior selector)
- data_prep.R: `build_data_list()` and `inject_choices()` helpers
- recovery_pipeline.R: `simulate_prior()`, `fit_model()`, `run_recovery()` orchestrators
- plotting.R: adapters to parameterize existing `caterpillar_recovery()` and `hyperprarameter_recovery()`

Quick usage inside a notebook or R script:

```r
# Load dependencies
library(cmdstanr)
library(data.table)
library(ggplot2)
library(cowplot)
source("projects/PILT_modeling/R/models_registry.R")
source("projects/PILT_modeling/R/data_prep.R")
source("projects/PILT_modeling/R/recovery_pipeline.R")
source("projects/PILT_modeling/R/plotting.R")
options(mc.cores = parallel::detectCores())

# Read task sequence and run recovery for selected models
task_sequence <- read.csv("tmp/PILT_sequence.csv")
res <- run_recovery(
  model_ids = c("running_average", "running_average_rs", "q_learning_rs"),
  task_sequence = task_sequence,
  N_participants = 50,
  per_model_opts = list(
    running_average = list(seed = 1234),
    running_average_rs = list(seed = 1234, threads_per_chain = 4),
    q_learning_rs = list(seed = 1234, threads_per_chain = 4, prior_iter_sampling = 1000)
  )
)

# Access plots, fits, and prior draws
res$running_average$participant_plot
res$running_average$hyper_plot
res$q_learning_rs$participant_plot
res$q_learning_rs$hyper_plot
```

To add a new model:
1) Add a `model_registry[["new_model_id"]]` entry in models_registry.R with `fit_stan`, `predict_stan`, `participant_regex`, `hyperparams`, `sampling_defaults`, and a `prior_selector(draws)` function.
2) Call `run_recovery(model_ids = c("new_model_id"), ...)` without changing pipeline code.

Note: If a model requires a different synthetic cohort size (e.g., Q-learning using 300 participants), call `run_recovery()` separately with the desired `N_participants` value for that model set.
