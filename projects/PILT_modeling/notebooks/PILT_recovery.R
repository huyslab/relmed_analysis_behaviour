library(cmdstanr)
library(data.table)
library(ggplot2)
library(cowplot)
theme_set(theme_minimal() + theme_cowplot(8))
source("projects/PILT_modeling/utils/recovery.R")

options(mc.cores = parallel::detectCores())

# Running average model recovery ----
# Path to Stan model
hierarchical_running_average_file <- file.path(
  "projects",
  "PILT_modeling",
  "models",
  "pilt_hierarchical_running_average_blockloop.stan"
)

# Compile model
hierarchical_running_average <- cmdstan_model(hierarchical_running_average_file,
  dir = "tmp")

# Create Stan data of task sequence for prior sampling
task_sequence <- read.csv("tmp/PILT_sequence.csv")
N_participants <- 50
prepared_sequences <- prepare_task_sequences(task_sequence, N_participants)


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

# Draw from the prior
prior <- hierarchical_running_average$sample(
  data = data_list,
  iter_warmup = 500,
  iter_sampling = 100,
  chains = 1,
  seed = 1,
  refresh = 1000
)

# Choose reasonable logrho for face data
prior_draws <- prior$draws()
prior_draws <- prior_draws[which.min(abs(prior_draws[, , "logrho"] - 1.5) + abs(prior_draws[, , "tau"] - 0.6)),,]

# Path to Stan model
hierarchical_running_average_predict_file <- file.path(
  "projects",
  "PILT_modeling",
  "models",
  "pilt_hierarchical_running_average_predict.stan"
)

# Compile model
hierarchical_running_average_predict <- cmdstan_model(hierarchical_running_average_predict_file,
  dir = "tmp")

# Draw data from the prior
prior_predictive <- hierarchical_running_average_predict$generate_quantities(
  prior_draws,
  data = data_list,
  seed = 123
)

prior_predictive <- as.vector(prior_predictive$draws())

# Prepare data list for fitting to prior predictive data
prior_predictive_list <- copy(data_list)
prior_predictive_list$choice <- prior_predictive
prior_predictive_list$prior_only <- 0L

# Fit model to prior predictive data
fit_recovery <- hierarchical_running_average$sample(
  data = prior_predictive_list,
  iter_warmup = 1000,
  iter_sampling = 1000,
  chains = 4,
  seed = 1234
)

(running_average_caterpillar <- caterpillar_recovery(fit_recovery$draws(), "r\\[" , prior_draws))

(running_average_hyper <- hyperprarameter_recovery(
  fit_recovery$draws(),
  c("logrho", "tau"),
  prior_draws
))

hierarchical_running_average_rs <- cmdstan_model(
  "projects/PILT_modeling/models/pilt_hierarchical_running_average_rs.stan",
  cpp_options = list(stan_threads = TRUE),
  dir = "tmp"
)

fit_recovery_rs <- hierarchical_running_average_rs$sample(
  data = prior_predictive_list,
  iter_warmup = 1000,
  iter_sampling = 1000,
  chains = 4,
  seed = 1234,
  threads_per_chain = 4
)

(running_average_rs_caterpillar <- caterpillar_recovery(fit_recovery_rs$draws(), "rhos\\[" , prior_draws))

(running_average_rs_hyper <- hyperprarameter_recovery(
  fit_recovery_rs$draws(),
  c("logrho", "tau"),
  prior_draws
))

# Path to Stan model
hierarchical_Q_learning_file <- file.path(
  "projects",
  "PILT_modeling",
  "models",
  "pilt_hierarchical_Q_learning_rs.stan"
)

# Compile model
hierarchical_Q_learning <- cmdstan_model(hierarchical_Q_learning_file,
  cpp_options = list(stan_threads = TRUE),
  dir = "tmp")

# Draw from the prior
N_participants <- 300
prepared_sequences <- prepare_task_sequences(task_sequence, N_participants)


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

prior <- hierarchical_Q_learning$sample(
  data = data_list,
  iter_warmup = 500,
  iter_sampling = 100,
  chains = 1,
  seed = 1,
  refresh = 1000,
  threads_per_chain = 1
)

# Choose reasonable parameters for face data
prior_draws <- prior$draws()
prior_draws <- prior_draws[
  which.min(abs(prior_draws[, , "logrho_mu"] - 1.5) / 1.5 + 
  abs(prior_draws[, , "logrho_tau"] - 0.6) / 0.6 +
  abs(prior_draws[, , "logitalpha_mu"] + 0.3) / 0.3 +
  abs(prior_draws[, , "logitalpha_tau"] - 0.1) / 0.1)
  ,,]

# Path to Stan model
hierarchical_Q_learning_predict_file <- file.path(
  "projects",
  "PILT_modeling",
  "models",
  "pilt_hierarchical_Q_learning_predict.stan"
)

# Compile model
hierarchical_Q_learning_predict <- cmdstan_model(hierarchical_Q_learning_predict_file,
  dir = "tmp")

# Draw data from the prior
prior_predictive <- hierarchical_Q_learning_predict$generate_quantities(
  prior_draws,
  data = data_list,
  seed = 123
)

prior_predictive <- as.vector(prior_predictive$draws())

# Prepare data list for fitting to prior predictive data
prior_predictive_list <- copy(data_list)
prior_predictive_list$choice <- prior_predictive
prior_predictive_list$prior_only <- 0L

fit_recovery <- hierarchical_Q_learning$sample(
  data = prior_predictive_list,
  iter_warmup = 1000,
  iter_sampling = 1000,
  chains = 4,
  seed = 1234,
  threads_per_chain = 4
)

(running_average_rs_caterpillar <- caterpillar_recovery(fit_recovery$draws(), "alphas\\[" , prior_draws))

(running_average_rs_hyper <- hyperprarameter_recovery(
  fit_recovery$draws(),
  c("logrho_mu", "logrho_tau", "logitalpha_mu", "logitalpha_tau"),
  prior_draws
))
