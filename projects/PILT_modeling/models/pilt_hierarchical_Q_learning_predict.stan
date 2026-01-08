// Shared utility functions
#include "pilt_utils.stan"

#include "pilt_hierarchical_Q_learning_data_params.stan"

generated quantities {
  array[N_trials] int<lower=1, upper=N_actions> sim_choice;

  {
    vector[N_actions] Q0 = rep_vector(initial_value * rhos[1], N_actions);
    vector[N_actions] q;
    for (bi in 1:N_blocks) {
      int start_idx = block_starts[bi];
      int end_idx = block_ends[bi];
      q = Q0;
      real rho = rhos[participant_per_block[bi]];
      real alpha = alphas[participant_per_block[bi]];

      for (i in start_idx:end_idx) {
        sim_choice[i] = categorical_logit_rng(q);
        int c = sim_choice[i];
        real o = outcomes[i, c];
        q[c] += compute_update(alpha, o, rho, q[c]);
      }
    }
  }
}
