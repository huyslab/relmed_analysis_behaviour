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
    geom_segment(data = draws_sum, aes(x = llb, xend = uub, y = 0, yend = 0), color = "lightblue", size = 2) +
    geom_segment(data = draws_sum, aes(x = lb, xend = ub, y = 0, yend = 0), color = "blue", size = 4) +
    theme(
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.line.y = element_blank(),
      )

    return(p)
}

prepare_task_sequences <- function(task_sequence, N_participants) {
    # Cross-join participants with the task sequence
    participants_df <- data.frame(participant = seq_len(N_participants))
    task_sequences <- merge(participants_df, task_sequence, all = TRUE)

    # Sorty by participant, block, trial
    task_sequences <- as.data.table(task_sequences)[order(participant, block, trial)]

    # Identify block starts and ends
    block_starts <- task_sequences[, .I[1], by = .(participant, block)]$V1
    block_ends <- task_sequences[, .I[.N], by = .(participant, block)]$V1

    # Map participant per block
    participant_per_block <- task_sequences[, .(participant = unique(participant)), by = .(participant, block)]$participant

    return(list(
      trial = task_sequences$trial,
      outcomes = cbind(task_sequences$feedback_left, task_sequences$feedback_right),
      block_starts = block_starts,
      block_ends = block_ends,
      participant_per_block = participant_per_block
    ))
}