## convert csv to list for cmdstan

data <- read.csv("sim_data_all.csv")
ids <- unique(data$PID)

par_df <- tibble::tibble(
  PID = data |> dplyr::distinct(PID) |> dplyr::pull(PID),
  alpha = data |> dplyr::distinct(PID, α) |> dplyr::pull(α),
  C = data |> dplyr::distinct(PID, C) |> dplyr::pull(C),
  w = data |> dplyr::distinct(PID, w) |> dplyr::pull(w),
  rho = data |> dplyr::distinct(PID, ρ) |> dplyr::pull(ρ)
)

rec_par_df <- par_df |>
  dplyr::mutate(across(c(alpha, C, w, rho), ~ 0))

prep_data <- function(data, id) {
  set_size <- block <- PID <- NA
  df <- data |> dplyr::filter(PID == id)
  outcome_mat <- cbind(df$feedback_optimal, df$feedback_suboptimal)
  unique_out <- unique(c(df$feedback_optimal, df$feedback_suboptimal))
  ls <- list(
    N = nrow(df),
    B = max(df$block),
    S = max(df$set_size),
    nT = max(df$trial),
    K = length(unique_out),
    choice = df$choice,
    block = df$block,
    pair = df$pair,
    set_size = df |>
      dplyr::distinct(set_size, block) |>
      dplyr::select(set_size) |>
      dplyr::pull(),
    valence = df$valence,
    outcomes = outcome_mat,
    unq_outc = unique_out
  )
  return(ls)
}

sgd_mod <- cmdstanr::cmdstan_model("working_memory/rlwm_sgd.stan")
l <- prep_data(data, ids[1])

m <- sgd_mod$sample(
  data = l, chains = 4, iter_warmup = 500, iter_sampling = 1000,
  parallel_chains = 4, refresh = 0
)

m <- sgd_mod$variational(data = l)
m$draws(variables = c("alpha", "C", "w", "rho")) |> colMeans()

for (id in ids) {
  ls <- prep_data(data, id)
  mod_fit <- sgd_mod$variational(data = ls)
  s <- mod_fit$draws(variables = c("alpha", "C", "w", "rho")) |> colMeans()
  rec_par_df[rec_par_df$PID == id, c("alpha", "C", "w", "rho")] <- as.list(s)
}

par_df <- par_df |>
  tidyr::pivot_longer(-PID, names_to = "parameter", values_to = "vb_mean_true")
rec_par_df <- rec_par_df |>
  tidyr::pivot_longer(-PID, names_to = "parameter", values_to = "vb_mean_rec")

pars <- dplyr::left_join(par_df, rec_par_df, by = c("PID", "parameter"))

library(ggplot2)
pars |>
  dplyr::filter(parameter == "w") |>
  ggplot(aes(x = vb_mean_true, y = vb_mean_rec, colour = parameter)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  theme_minimal()