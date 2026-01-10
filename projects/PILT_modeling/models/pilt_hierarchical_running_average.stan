// Shared utility functions
#include "pilt_utils.stan"

#include "pilt_hierarchical_running_average_data_params.stan"

model {
  // Priors
  logrho ~ normal(0, 1.5);
  tau ~ normal(0, 0.5);
  r ~ normal(0, 1);

  // Likelihood via block-wise accumulation
  if (prior_only == 0) {
    vector[N_actions] Q0 = rep_vector(initial_value * rhos[1], N_actions);
    for (bi in 1:N_blocks) {
      target += running_average_block_ll(
        block_starts[bi],
        block_ends[bi],
        N_actions,
        choice,
        trial,
        outcomes,
        Q0,
        rhos[participant_per_block[bi]]
      );
    }
  }
}
