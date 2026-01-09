# Plotting helpers and adapters for recovery diagnostics
# Moved from projects/PILT_modeling/utils/recovery.R

caterpillar_recovery <- function(
  draws,
  param,
  true_values
) {

  # Select rows
  draws <- as.data.table(draws)[grepl(param, variable),]
  true_values <- as.data.table(true_values)[grepl(param, variable),]

  # Summarize draws
  draws_sum <- draws[, .(
    llb = quantile(value, 0.025),
    lb = quantile(value, 0.25),
    ub = quantile(value, 0.75),
    uub = quantile(value, 0.975),
    median = median(value)
  ), by = .(variable)]

  # Merge summaries with true values
  recovery_df <- merge(draws_sum, true_values, by = "variable")
  
  # Sort by true value
  recovery_df <- recovery_df[order(median)]

  # Plot caterpillar plot
  p <- ggplot(recovery_df, aes(y = seq_len(nrow(recovery_df)), x = median)) +
    geom_point() +
    geom_linerange(aes(xmin = lb, xmax = ub), color = "blue", linewidth = 1) +
    geom_linerange(aes(xmin = llb, xmax = uub), color = "lightblue", linewidth = 0.5) +
    geom_point(aes(x = value), color = "red", shape = 4, size = 3) +
    labs(x = "Parameter Value") +
    scale_y_continuous(name = NULL, breaks = NULL) +
    theme(
      axis.title.y = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.line.y = element_blank(),
    )

  return(p)
}

hyperprarameter_recovery <- function(
  draws,
  params,
  true_values
) {
  draws <- as.data.table(draws)[variable %in% params,]
  true_values <- as.data.table(true_values)[variable %in% params,]

  draws_sum <- draws[, .(
    llb = quantile(value, 0.025),
    lb = quantile(value, 0.25),
    ub = quantile(value, 0.75),
    uub = quantile(value, 0.975)
  ), by = .(variable)]

  p <- ggplot(draws, aes(x = value)) + geom_density() + facet_wrap(~variable, scales = "free", ncol = 1) +
    geom_vline(data = true_values, aes(xintercept = value), color = "red", linetype = "dashed") +
    geom_segment(data = draws_sum, aes(x = llb, xend = uub, y = 0, yend = 0), color = "lightblue", linewidth = 2) +
    geom_segment(data = draws_sum, aes(x = lb, xend = ub, y = 0, yend = 0), color = "blue", linewidth = 4) +
    theme(
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.line.y = element_blank(),
      )

    return(p)
}

plot_participant_recovery <- function(fit, prior_draws, participant_regex) {
  caterpillar_recovery(fit$draws(), participant_regex, prior_draws)
}

plot_hyper_recovery <- function(fit, prior_draws, hyperparams) {
  hyperprarameter_recovery(fit$draws(), hyperparams, prior_draws)
}
