functions {
  real compute_update(real alpha, real outcome, real rho, real Q_val) {
    return alpha * (outcome * rho - Q_val);
  }

  real running_average_block_ll(
      int start_idx,
      int end_idx,
      int K,
      array[] int choice,
      array[] int trial,
      matrix outcomes,
      vector Q_init,
      real rho
  ) {
    vector[K] q = Q_init;
    real ll = 0;
    for (i in start_idx:end_idx) {
      ll += categorical_logit_lpmf(choice[i] | q);
      int a;
      real alpha;
      real o;
      a = choice[i];
      alpha = 1.0 / trial[i];
      o = outcomes[i, a];
      q[a] += compute_update(alpha, o, rho, q[a]);
    }
    return ll;
  }

  real blocks_partial_sum(array[] int block_ids_slice,
                          int start, int end,
                          int N_actions,
                          array[] int choice,
                          array[] int trial,
                          matrix outcomes,
                          vector rhos,
                          array[] int block_starts,
                          array[] int block_ends,
                          array[] int participant_per_block,
                          vector Q0) {
    real lp = 0;
    for (n in block_ids_slice) {
      lp += running_average_block_ll(
        block_starts[n],
        block_ends[n],
        N_actions,
        choice,
        trial,
        outcomes,
        Q0,
        rhos[participant_per_block[n]]
      );
    }
    return lp;
  }
}