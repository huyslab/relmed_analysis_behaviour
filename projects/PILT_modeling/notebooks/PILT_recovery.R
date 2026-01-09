# Load dependencies and plotting theme; recovery helpers for diagnostics
library(cmdstanr)
library(data.table)
library(ggplot2)
library(cowplot)
theme_set(theme_minimal() + theme_cowplot(8))
source("projects/PILT_modeling/utils/recovery.R")

# Use all available CPU cores for CmdStan sampling
options(mc.cores = parallel::detectCores())

# Running average model recovery ----
# Path to hierarchical running-average Stan model
hierarchical_running_average_file <- file.path(
  "projects",
  "PILT_modeling",
  "models",
  "pilt_hierarchical_running_average_blockloop.stan"
)

# Compile model once; output stored under tmp/
hierarchical_running_average <- cmdstan_model(hierarchical_running_average_file,
  dir = "tmp")

# Read a fixed task sequence used for prior-predictive simulation
task_sequence <- read.csv("tmp/PILT_sequence.csv")
# Number of synthetic participants to simulate for recovery
N_participants <- 50
# Precompute trial/block indices and outcomes for Stan
prepared_sequences <- prepare_task_sequences(task_sequence, N_participants)

# Stan data for prior draws (prior_only = 1 disables likelihood on choices)
data_list <- list(
  N_trials = length(prepared_sequences$trial),
  N_actions = 2,
  N_blocks = length(prepared_sequences$block_starts),
  N_participants = N_participants,
  block_starts = prepared_sequences$block_starts,
  block_ends = prepared_sequences$block_ends,
  trial = prepared_sequences$trial,
  choice = rep(1L, length(prepared_sequences$trial)),
  outcomes = prepared_sequences$outcomes,
  participant_per_block = prepared_sequences$participant_per_block,
  initial_value = 0.0,
  prior_only = 1L
)

# Draw from the prior to obtain parameter samples
prior <- hierarchical_running_average$sample(
  data = data_list,
  iter_warmup = 500,
  iter_sampling = 100,
  chains = 1,
  seed = 1,
  refresh = 1000
)

# Select one prior draw close to target hyperparameters (simple distance heuristic)
prior_draws <- prior$draws()
prior_draws <- prior_draws[which.min(abs(prior_draws[, , "logrho"] - 1.5) + abs(prior_draws[, , "tau"] - 0.6)),,]

# Model used to generate prior-predictive choices given parameters
hierarchical_running_average_predict_file <- file.path(
  "projects",
  "PILT_modeling",
  "models",
  "pilt_hierarchical_running_average_predict.stan"
)

# Compile the predictive model
hierarchical_running_average_predict <- cmdstan_model(hierarchical_running_average_predict_file,
  dir = "tmp")

# Simulate choices from prior parameters via generated quantities
prior_predictive <- hierarchical_running_average_predict$generate_quantities(
  prior_draws,
  data = data_list,
  seed = 123
)

# Flatten simulated choices to integer vector for Stan data
prior_predictive <- as.vector(prior_predictive$draws())

# Switch to likelihood mode and inject simulated choices
prior_predictive_list <- copy(data_list)
prior_predictive_list$choice <- prior_predictive
prior_predictive_list$prior_only <- 0L

# Fit the hierarchical model to its own prior-predictive data (recovery)
fit_recovery <- hierarchical_running_average$sample(
  data = prior_predictive_list,
  iter_warmup = 1000,
  iter_sampling = 1000,
  chains = 4,
  seed = 1234
)

# Recover subject-level parameters (regex "r\\[" selects per-participant r)
(running_average_caterpillar <- caterpillar_recovery(fit_recovery$draws(), "r\\[" , prior_draws))

# Recover hyperparameters logrho, tau
(running_average_hyper <- hyperprarameter_recovery(
  fit_recovery$draws(),
  c("logrho", "tau"),
  prior_draws
))

# Threaded RS parameterization for speed; enable within-chain parallelism
hierarchical_running_average_rs <- cmdstan_model(
  "projects/PILT_modeling/models/pilt_hierarchical_running_average_rs.stan",
  cpp_options = list(stan_threads = TRUE),
  dir = "tmp"
)

# Fit RS variant with threads per chain
fit_recovery_rs <- hierarchical_running_average_rs$sample(
  data = prior_predictive_list,
  iter_warmup = 1000,
  iter_sampling = 1000,
  chains = 4,
  seed = 1234,
  threads_per_chain = 4
)

# Subject-level RS recovery (rho_s)
(running_average_rs_caterpillar <- caterpillar_recovery(fit_recovery_rs$draws(), "rhos\\[" , prior_draws))

# Hyperparameter recovery for RS variant (logrho, tau)
(running_average_rs_hyper <- hyperprarameter_recovery(
  fit_recovery_rs$draws(),
  c("logrho", "tau"),
  prior_draws
))

# Hierarchical Q-learning RS model path and compilation
hierarchical_Q_learning_file <- file.path(
  "projects",
  "PILT_modeling",
  "models",
  "pilt_hierarchical_Q_learning_rs.stan"
)

# Compile Q-learning model with threading
hierarchical_Q_learning <- cmdstan_model(hierarchical_Q_learning_file,
  cpp_options = list(stan_threads = TRUE),
  dir = "tmp")

# Use a larger synthetic cohort to stabilize hyperparameter estimation
N_participants <- 300
prepared_sequences <- prepare_task_sequences(task_sequence, N_participants)

# Stan data for Q-learning prior draws (prior_only = 1)
data_list <- list(
  N_trials = length(prepared_sequences$trial),
  N_actions = 2,
  N_blocks = length(prepared_sequences$block_starts),
  N_participants = N_participants,
  block_starts = prepared_sequences$block_starts,
  block_ends = prepared_sequences$block_ends,
  trial = prepared_sequences$trial,
  choice = rep(1L, length(prepared_sequences$trial)),
  outcomes = prepared_sequences$outcomes,
  participant_per_block = prepared_sequences$participant_per_block,
  initial_value = 0.0,
  prior_only = 1L
)

# Draw from prior for Q-learning RS
prior <- hierarchical_Q_learning$sample(
  data = data_list,
  iter_warmup = 500,
  iter_sampling = 1000,
  chains = 1,
  seed = 1,
  threads_per_chain = 1
)

# Pick a prior draw near target hyperparameters (weighted L1 distances)
prior_draws <- prior$draws()
prior_draws <- prior_draws[
  which.min(abs(prior_draws[, , "logrho_mu"] - 2.) / 2. + 
  abs(prior_draws[, , "logrho_tau"] - 0.3) / 0.3 +
  abs(prior_draws[, , "logitalpha_mu"] + 0.6) / 0.6 +
  abs(prior_draws[, , "logitalpha_tau"] - 0.1) / 0.1)
  ,,]

# Inspect distribution of subject-level alphas and rhos for selected draw
dists <- as.data.table(prior_draws)[grepl("alphas\\[", variable) | grepl("rhos\\[", variable), ]
dists[, parameter := fifelse(grepl("alphas\\[", variable), "alpha", "rho")]

# Quick diagnostic histogram by parameter type
ggplot(dists, aes(x = value)) + geom_histogram() + facet_wrap(~parameter, scales = "free")

# Predictive model used to generate choices from Q-learning parameters
hierarchical_Q_learning_predict_file <- file.path(
  "projects",
  "PILT_modeling",
  "models",
  "pilt_hierarchical_Q_learning_predict.stan"
)

# Compile predictive model
hierarchical_Q_learning_predict <- cmdstan_model(hierarchical_Q_learning_predict_file,
  dir = "tmp")

# Simulate prior-predictive choices for Q-learning
prior_predictive <- hierarchical_Q_learning_predict$generate_quantities(
  prior_draws,
  data = data_list,
  seed = 123
)

# Flatten choices vector and prepare Stan data for recovery fit
prior_predictive <- as.vector(prior_predictive$draws())
prior_predictive_list <- copy(data_list)
prior_predictive_list$choice <- prior_predictive
prior_predictive_list$prior_only <- 0L

# Fit Q-learning to prior-predictive data with threading
fit_recovery <- hierarchical_Q_learning$sample(
  data = prior_predictive_list,
  iter_warmup = 1000,
  iter_sampling = 1000,
  chains = 4,
  seed = 1234,
  threads_per_chain = 4
)

# Recover subject-level learning rates (regex "a\\[" selects alpha entries)
(running_average_rs_caterpillar <- caterpillar_recovery(fit_recovery$draws(), "a\\[" , prior_draws))

# Recover Q-learning hyperparameters: logrho_* and logitalpha_*
(running_average_rs_hyper <- hyperprarameter_recovery(
  fit_recovery$draws(),
  c("logrho_mu", "logrho_tau", "logitalpha_mu", "logitalpha_tau"),
  prior_draws
))
