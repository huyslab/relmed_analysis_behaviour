data {
  int<lower=1> N_trials;
  int<lower=1> N_actions;
  int<lower=0> N_blocks;
  int<lower=1> N_participants;

  array[N_blocks] int<lower=1, upper=N_trials> block_starts;
  array[N_blocks] int<lower=1, upper=N_trials> block_ends;
  array[N_trials] int<lower=1> trial;
  array[N_trials] int<lower=1, upper=N_actions> choice;
  matrix[N_trials, N_actions] outcomes;
  array[N_blocks] int<lower=1, upper=N_participants> participant_per_block;
  
  int<lower=0, upper=1> prior_only;
}

transformed data {
  array[N_blocks] int block_ids;
  for (b in 1:N_blocks) block_ids[b] = b;
}

parameters {
  real logrho_int;
  real<lower=0> logrho_int_tau;
  real logrho_beta;
  real<lower=0> logrho_beta_tau;
  vector[N_participants] r_ints;
  vector[N_participants] r_betas;
}

transformed parameters {
  // Compute punishment and reward rho for each participant, scaling by 0.5 so that beta is difference between reward and punishment
  vector[N_participants] rhos_p  = exp(logrho_int + logrho_int_tau * r_ints - 0.5 * (logrho_beta + logrho_beta_tau * r_betas));
  vector[N_participants] rhos_r  = exp(logrho_int + logrho_int_tau * r_ints + 0.5 * (logrho_beta + logrho_beta_tau * r_betas));
}
