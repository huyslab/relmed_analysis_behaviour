# R-only model registry for PILT hierarchical models
# Each model entry defines Stan file paths, parameter conventions, and default sampling options.

model_registry <- local({
  list(
    running_average = list(
      fit_stan = "projects/PILT_modeling/models/pilt_hierarchical_running_average_blockloop.stan",
      predict_stan = "projects/PILT_modeling/models/pilt_hierarchical_running_average_predict.stan",
      participant_regex = "r\\[",
      hyperparams = c("logrho", "tau"),
      sampling_defaults = list(
        iter_warmup = 1000,
        iter_sampling = 1000,
        chains = 4,
        threads_per_chain = NULL,
        seed = 1234,
        prior_iter_warmup = 500,
        prior_iter_sampling = 100,
        prior_chains = 1,
        prior_seed = 1,
        prior_predictive_seed = 123
      ),
      prior_selector = function(draws) {
        # Choose draw near target logrho ~ 1.5 and tau ~ 0.6
        idx <- which.min(abs(draws[, , "logrho"] - 1.5) + abs(draws[, , "tau"] - 0.6))
        draws[idx, , ]
      }
    ),
    running_average_rs = list(
      fit_stan = "projects/PILT_modeling/models/pilt_hierarchical_running_average_rs.stan",
      predict_stan = "projects/PILT_modeling/models/pilt_hierarchical_running_average_predict.stan",
      participant_regex = "rhos\\[",
      hyperparams = c("logrho", "tau"),
      sampling_defaults = list(
        iter_warmup = 1000,
        iter_sampling = 1000,
        chains = 4,
        threads_per_chain = 4,
        seed = 1234,
        prior_iter_warmup = 500,
        prior_iter_sampling = 100,
        prior_chains = 1,
        prior_seed = 1,
        prior_predictive_seed = 123
      ),
      prior_selector = function(draws) {
        idx <- which.min(abs(draws[, , "logrho"] - 1.5) + abs(draws[, , "tau"] - 0.6))
        draws[idx, ,]
      }
    ),
    running_average_pr_rs = list(
      fit_stan = "projects/PILT_modeling/models/pilt_hierarchical_running_average_pr_rs.stan",
      predict_stan = "projects/PILT_modeling/models/pilt_hierarchical_running_average_pr_predict.stan",
      participant_regex = c("rhos_p\\[", "rhos_r\\["),
      hyperparams = c("logrho_int", "logrho_int_tau", "logrho_beta", "logrho_beta_tau"),
      sampling_defaults = list(
        iter_warmup = 1000,
        iter_sampling = 1000,
        chains = 4,
        threads_per_chain = 4,
        seed = 1234,
        prior_iter_warmup = 500,
        prior_iter_sampling = 1000,
        prior_chains = 1,
        prior_seed = 1,
        prior_predictive_seed = 123
      ),
      prior_selector = function(draws) {
        idx <- which.min(abs(draws[, , "logrho_int"] - 1.5) / 1.5 + 
            abs(draws[, , "logrho_int_tau"] - 0.6) / 0.6 +
            abs(draws[, , "logrho_beta"] - 0.5) / 0.5 +
            abs(draws[, , "logrho_beta_tau"] - 0.2) / 0.2)
        draws[idx, ,]
      }
    ),
    q_learning_rs = list(
      fit_stan = "projects/PILT_modeling/models/pilt_hierarchical_Q_learning_rs.stan",
      predict_stan = "projects/PILT_modeling/models/pilt_hierarchical_Q_learning_predict.stan",
      participant_regex = "a\\[",
      hyperparams = c("logrho_mu", "logrho_tau", "logitalpha_mu", "logitalpha_tau"),
      sampling_defaults = list(
        iter_warmup = 1000,
        iter_sampling = 1000,
        chains = 4,
        threads_per_chain = 4,
        seed = 1234,
        prior_iter_warmup = 500,
        prior_iter_sampling = 1000,
        prior_chains = 1,
        prior_seed = 1,
        prior_predictive_seed = 123
      ),
      prior_selector = function(draws) {
        idx <- which.min(
          abs(draws[, , "logrho_mu"] - 2.) / 2. +
            abs(draws[, , "logrho_tau"] - 0.3) / 0.3 +
            abs(draws[, , "logitalpha_mu"] + 0.6) / 0.6 +
            abs(draws[, , "logitalpha_tau"] - 0.1) / 0.1
        )
        draws[idx, ,]
      }
    )
  )
})

get_model <- function(model_id) {
  if (!model_id %in% names(model_registry)) stop(sprintf("Unknown model_id: %s", model_id))
  model_registry[[model_id]]
}
