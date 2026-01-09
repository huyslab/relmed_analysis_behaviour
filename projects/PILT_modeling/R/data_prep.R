# Data preparation helpers for PILT models
# Relies on prepare_task_sequences() already provided in projects/PILT_modeling/utils/recovery.R

build_data_list <- function(prepared_sequences, N_participants, prior_only = TRUE) {
  list(
    N_trials = length(prepared_sequences$trial),
    N_actions = 2L,
    N_blocks = length(prepared_sequences$block_starts),
    N_participants = as.integer(N_participants),
    block_starts = prepared_sequences$block_starts,
    block_ends = prepared_sequences$block_ends,
    trial = prepared_sequences$trial,
    choice = rep(1L, length(prepared_sequences$trial)),
    outcomes = prepared_sequences$outcomes,
    participant_per_block = prepared_sequences$participant_per_block,
    initial_value = 0.0,
    prior_only = if (prior_only) 1L else 0L
  )
}

inject_choices <- function(data_list, choices) {
  out <- data.table::copy(data_list)
  out$choice <- as.integer(choices)
  out$prior_only <- 0L
  out
}

# Task sequence preparation moved from projects/PILT_modeling/utils/recovery.R
prepare_task_sequences <- function(task_sequence, N_participants) {
    # Cross-join participants with the task sequence
    participants_df <- data.frame(participant = seq_len(N_participants))
    task_sequences <- merge(participants_df, task_sequence, all = TRUE)

    # Sort by participant, block, trial
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
