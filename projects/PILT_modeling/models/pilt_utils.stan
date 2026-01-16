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

  real running_average_partial_sum(array[] int block_ids_slice,
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

  real pearcehall_block_ll(
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
    vector[K] pe = rep_vector(1.0, K);
    real ll = 0;
    int c;
    real alpha;
    real o;

    for (i in start_idx:end_idx) {
      ll += categorical_logit_lpmf(choice[i] | q);
      c = choice[i];
      alpha = sqrt(pe[c] * pe[c] + 1e-8); // Smooth abs to avoid discontinuities
      o = outcomes[i, c];
      pe[c] = o - q[c];
      q[c] += compute_update(alpha, o, rho, q[c]);
    }
    return ll;
  }

  real pearcehall_partial_sum(array[] int block_ids_slice,
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
      lp += pearcehall_block_ll(
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

  real running_average_pr_block_ll(
      int start_idx,
      int end_idx,
      int K,
      array[] int choice,
      array[] int trial,
      matrix outcomes,
      vector Q_init,
      real rho_p,
      real rho_r
  ) {
    vector[K] q = Q_init;
    real ll = 0;
    for (i in start_idx:end_idx) {
      ll += categorical_logit_lpmf(choice[i] | q);
      int a;
      real alpha;
      real o;
      real rho;
      a = choice[i];
      alpha = 1.0 / trial[i];
      o = outcomes[i, a];
      rho = o < 0 ? rho_p : rho_r;
      q[a] += compute_update(alpha, o, rho, q[a]);
    }
    return ll;
  }

  real running_average_pr_partial_sum(array[] int block_ids_slice,
        int start, int end,
        int N_actions,
        array[] int choice,
        array[] int trial,
        matrix outcomes,
        vector rhos_p,
        vector rhos_r,
        array[] int block_starts,
        array[] int block_ends,
        array[] int participant_per_block,
        vector Q0) {
    real lp = 0;
    for (n in block_ids_slice) {
      lp += running_average_pr_block_ll(
        block_starts[n],
        block_ends[n],
        N_actions,
        choice,
        trial,
        outcomes,
        Q0,
        rhos_p[participant_per_block[n]],
        rhos_r[participant_per_block[n]]
      );
    }
    return lp;
  }

  real Q_learning_block_ll(
      int start_idx,
      int end_idx,
      int K,
      array[] int choice,
      matrix outcomes,
      vector Q_init,
      real rho,
      real alpha
  ) {
    vector[K] q = Q_init;
    real ll = 0;
    for (i in start_idx:end_idx) {
      ll += categorical_logit_lpmf(choice[i] | q);
      int a;
      real o;
      a = choice[i];
      o = outcomes[i, a];
      q[a] += compute_update(alpha, o, rho, q[a]);
    }
    return ll;
  }

  real Q_learning_partial_sum(array[] int block_ids_slice,
                        int start, int end,
                        int N_actions,
                        array[] int choice,
                        matrix outcomes,
                        vector rhos,
                        vector alphas,
                        array[] int block_starts,
                        array[] int block_ends,
                        array[] int participant_per_block,
                        vector Q0) {
    real lp = 0;
    for (n in block_ids_slice) {
      lp += Q_learning_block_ll(
        block_starts[n],
        block_ends[n],
        N_actions,
        choice,
        outcomes,
        Q0,
        rhos[participant_per_block[n]],
        alphas[participant_per_block[n]]
      );
    }
    return lp;
  }


}