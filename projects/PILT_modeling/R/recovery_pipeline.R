# Recovery pipeline: prior simulation and fitting wrappers (R-only)
# Depends on: cmdstanr, projects/PILT_modeling/utils/recovery.R, and R/models_registry.R

library(cmdstanr)

# Merge named lists with defaults (x overrides y)
merge_opts <- function(defaults, override = NULL) {
  if (is.null(override)) return(defaults)
  defaults[names(override)] <- override
  defaults
}

compile_model <- function(stan_path, threads_per_chain = NULL) {
  cpp_opts <- if (is.null(threads_per_chain)) NULL else list(stan_threads = TRUE)
  cmdstan_model(stan_path, cpp_options = cpp_opts, dir = "tmp")
}

simulate_prior <- function(model_id,
                           prepared_sequences,
                           N_participants,
                           registry = model_registry,
                           prior_opts = NULL) {
  mdl <- registry[[model_id]]
  stopifnot(!is.null(mdl))
  sdef <- mdl$sampling_defaults
  sopt <- merge_opts(sdef, prior_opts)

  # Build Stan data in prior-only mode
  data_list <- build_seq_data_list(prepared_sequences, N_participants, prior_only = TRUE)

  # Compile hierarchical fit model for prior sampling
  fit_model <- compile_model(mdl$fit_stan, threads_per_chain = sopt$threads_per_chain)

  # Draw from prior
  prior <- fit_model$sample(
    data = data_list,
    iter_warmup = sopt$prior_iter_warmup,
    iter_sampling = sopt$prior_iter_sampling,
    chains = sopt$prior_chains,
    seed = sopt$prior_seed,
    threads_per_chain = sopt$threads_per_chain
  )
  all_draws <- prior$draws()
  chosen_draw <- mdl$prior_selector(all_draws)

  stopifnot('Expecting exactly one draw from prior_selector' = nrow(chosen_draw) == 1)

  # Compile predictive model and generate choices
  predict_model <- compile_model(mdl$predict_stan)
  prior_pred <- predict_model$generate_quantities(
    chosen_draw,
    data = data_list,
    seed = sopt$prior_predictive_seed
  )
  choices <- as.vector(prior_pred$draws())
  prior_predictive_list <- inject_choices(data_list, choices)

  list(
    prior = prior,
    prior_draws = chosen_draw,
    data_list = data_list,
    prior_predictive_list = prior_predictive_list
  )
}

fit_model <- function(model_id,
                      prior_predictive_list,
                      fit_opts = NULL,
                      registry = model_registry) {
  mdl <- registry[[model_id]]
  stopifnot(!is.null(mdl))
  sdef <- mdl$sampling_defaults
  sopt <- merge_opts(sdef, fit_opts)

  fit_model <- compile_model(mdl$fit_stan, threads_per_chain = sopt$threads_per_chain)
  fit <- fit_model$sample(
    data = prior_predictive_list,
    iter_warmup = sopt$iter_warmup,
    iter_sampling = sopt$iter_sampling,
    chains = sopt$chains,
    seed = sopt$seed,
    threads_per_chain = sopt$threads_per_chain
  )
  fit
}

run_recovery <- function(model_ids,
                         task_sequence,
                         N_participants,
                         per_model_opts = list(),
                         registry = model_registry) {
  # Prepare sequences via existing helper
  prepared_sequences <- prepare_task_sequences(task_sequence, N_participants)

  results <- list()
  for (mid in model_ids) {
    prior_res <- simulate_prior(mid, prepared_sequences, N_participants, registry, prior_opts = per_model_opts[[mid]])
    fit <- fit_model(mid, prior_res$prior_predictive_list, fit_opts = per_model_opts[[mid]], registry)

    participant_regex <- registry[[mid]]$participant_regex
    hyperparams <- registry[[mid]]$hyperparams

    if (is.character(participant_regex) && length(participant_regex) > 1) {
      participant_plot <- lapply(participant_regex, function(pr) {
        caterpillar_recovery(fit$draws(), pr, prior_res$prior_draws)
      })
    } else {
      participant_plot <- caterpillar_recovery(fit$draws(), participant_regex, prior_res$prior_draws)
    }
    hyper_plot <- hyperprarameter_recovery(fit$draws(), hyperparams, prior_res$prior_draws)

    results[[mid]] <- list(
      prior = prior_res$prior,
      prior_draws = prior_res$prior_draws,
      fit = fit,
      participant_plot = participant_plot,
      hyper_plot = hyper_plot
    )
  }
  results
}
