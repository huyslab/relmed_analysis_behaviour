// Shared utility functions
#include "pilt_utils.stan"

#include "pilt_hierarchical_running_average_pr_data_params.stan"

model {
    // Priors
    logrho_int ~ normal(0, 1.5);
    logrho_int_tau ~ normal(0, 0.5);
    logrho_beta ~ normal(0, 1);
    logrho_beta_tau ~ normal(0, 0.5);

    r_ints ~ normal(0, 1);
    r_betas ~ normal(0, 1);

    // Likelihood via block-wise accumulation
    if (prior_only == 0) {
        vector[N_actions] Q0 = rep_vector(0, N_actions);
        int grainsize = 1; // tune if desired
        target += reduce_sum(running_average_pr_partial_sum,
            block_ids, grainsize,
            N_actions,
            choice,
            trial,
            outcomes,
            rhos_p,
            rhos_r,
            block_starts,
            block_ends,
            participant_per_block,
            Q0);
    }
}
