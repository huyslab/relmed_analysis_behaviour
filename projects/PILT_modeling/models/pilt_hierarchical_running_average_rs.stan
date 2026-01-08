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
        int grainsize = 1; // tune if desired
        target += reduce_sum(running_average_partial_sum,
            block_ids, grainsize,
            N_actions,
            choice,
            trial,
            outcomes,
            rhos,
            block_starts,
            block_ends,
            participant_per_block,
            Q0);
    }
}
