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
  
  real initial_value;
  int<lower=0, upper=1> prior_only;
}

transformed data {
  array[N_blocks] int block_ids;
  for (b in 1:N_blocks) block_ids[b] = b;
}

parameters {
  real logrho;
  real<lower=0> tau;
  vector[N_participants] r;
}

transformed parameters {
  vector[N_participants] rhos  = exp(logrho + tau * r);
}
