# Read a JS file containing a JSON string assignment and return as data.table
# Example expected format:
#   const PILT_json = '...JSON...';
# Supports single ('), double ("), or backtick (`) quoted strings.
read_task_sequence_js_to_dt <- function(path) {
  # Dependencies
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite package is required")
  if (!requireNamespace("data.table", quietly = TRUE)) stop("data.table package is required")

  # Read entire file as a single string
  lines <- base::readLines(path, warn = FALSE, encoding = "UTF-8")
  content <- paste(lines, collapse = "\n")

  # Find first '=' then extract the quoted JSON string that follows
  eq_pos <- base::regexpr("=", content)
  if (eq_pos[1] == -1) stop("No assignment '=' found in file: ", path)
  after <- base::substring(content, eq_pos[1] + 1)
  after <- base::sub("^\\s+", "", after)

  # Determine quote type
  quote <- base::substr(after, 1, 1)
  if (!quote %in% c("'", '"', "`")) stop("Expected a quoted JSON string after '=' in: ", path)

  # Extract quoted segment accounting for simple escapes
  i <- 2L
  esc <- FALSE
  n <- base::nchar(after, type = "bytes")
  while (i <= n) {
    ch <- base::substr(after, i, i)
    if (!esc && ch == quote) break
    if (!esc && ch == "\\") esc <- TRUE else esc <- FALSE
    i <- i + 1L
  }
  if (i > n) stop("Closing quote not found in: ", path)
  json_str <- base::substr(after, 2L, i - 1L)

  # Parse JSON
  parsed <- jsonlite::fromJSON(json_str)

  # If nested list of data.frames, rbind into one data.table
  if (is.list(parsed) && !is.data.frame(parsed)) {
    # Attempt to coerce each element to data.frame and row-bind
    dfs <- lapply(parsed, function(x) {
      if (is.data.frame(x)) return(x)
      # If x is itself a list of rows, convert
      tryCatch(as.data.frame(x), error = function(e) {
        # Fallback to fromJSON again for nested structures
        jsonlite::fromJSON(jsonlite::toJSON(x))
      })
    })
    dt <- data.table::rbindlist(dfs, use.names = TRUE, fill = TRUE)
  } else {
    # Already a data.frame
    dt <- data.table::as.data.table(parsed)
  }

  dt
}

# Parse all .js files in a folder and rbind into one data.table.
# Adds/overwrites a `session` column derived from the filename:
# the substring after the first '_' and before the '.js'. Also adds `source_file`.
read_task_sequence_folder_to_dt <- function(folder, recursive = TRUE) {
  if (!requireNamespace("data.table", quietly = TRUE)) stop("data.table package is required")

  files <- base::list.files(folder, pattern = "\\.js$", full.names = TRUE, recursive = recursive)
  if (length(files) == 0L) stop("No .js files found in folder: ", folder)

  dts <- lapply(files, function(f) {
    dt <- read_task_sequence_js_to_dt(f)
    base <- base::basename(f)
    dt$source_file <- base
    dt
  })

  data.table::rbindlist(dts, use.names = TRUE, fill = TRUE)
}

# Data preparation helpers for PILT models
# Relies on prepare_task_sequences() already provided in projects/PILT_modeling/utils/recovery.R

build_seq_data_list <- function(prepared_sequences, N_participants, prior_only = TRUE) {
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
prepare_task_sequences <- function(task_sequence, N_participants, return_data_list = TRUE) {
    # Cross-join participants with the task sequence
    participants_df <- data.frame(participant = seq_len(N_participants))
    task_sequences <- merge(participants_df, task_sequence, all = TRUE)

    if (!return_data_list) {
      return(task_sequences)
    }

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

