# Load dependencies and plotting theme; recovery helpers for diagnostics
library(cmdstanr)
library(data.table)
library(ggplot2)
library(cowplot)
theme_set(theme_minimal() + theme_cowplot(8))
source("projects/PILT_modeling/R/models_registry.R")
source("projects/PILT_modeling/R/data_prep.R")
source("projects/PILT_modeling/R/recovery_pipeline.R")
source("projects/PILT_modeling/R/plotting.R")
source("projects/PILT_modeling/R/models_registry.R")
source("projects/PILT_modeling/R/data_prep.R")
source("projects/PILT_modeling/R/recovery_pipeline.R")
source("projects/PILT_modeling/R/plotting.R")
options(mc.cores = parallel::detectCores())

# Read task sequence and run recovery for selected models
task_sequence <- read.csv("tmp/PILT_sequence.csv")
res <- run_recovery(
  model_ids = c("running_average_rs", "running_average_pr_rs", "q_learning_rs"),
  task_sequence = task_sequence,
  N_participants = 100,
  per_model_opts = list(
    running_average = list(seed = 1234),
    running_average_rs = list(seed = 1234, threads_per_chain = 4),
    q_learning_rs = list(seed = 1234, threads_per_chain = 4, prior_iter_sampling = 1000)
  )
)

