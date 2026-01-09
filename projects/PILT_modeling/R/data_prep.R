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
