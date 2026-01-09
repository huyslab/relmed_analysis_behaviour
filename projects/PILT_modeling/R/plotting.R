# Plotting adapters to parameterize evaluation by registry entries
# Wrap existing helpers from projects/PILT_modeling/utils/recovery.R

plot_participant_recovery <- function(fit, prior_draws, participant_regex) {
  caterpillar_recovery(fit$draws(), participant_regex, prior_draws)
}

plot_hyper_recovery <- function(fit, prior_draws, hyperparams) {
  hyperprarameter_recovery(fit$draws(), hyperparams, prior_draws)
}
