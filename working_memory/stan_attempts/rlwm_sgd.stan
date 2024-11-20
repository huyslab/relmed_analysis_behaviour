functions {
  int get_index(vector arr, real val) {
    for (i in 1:size(arr)) {
      if (arr[i] == val) {
        return i;
      }
    }
    return -1;
  }
}

data {
  int<lower=1> N; // number of trials
  int<lower=1> B; // number of blocks
  int<lower=1> S; // maximum set size
  int<lower=1> nT; // maximum number of trials per block
  int<lower=1> K; // number of unique outcomes
  array[N] int<lower=0, upper=1> choice; // choices
  array[N] int<lower=1> block; // block indices
  array[N] int<lower=1> pair;// pair indices
  array[B] int<lower=1> set_size; // set sizes
  vector<lower=-1, upper=1>[N] valence; // valence values
  matrix[N, 2] outcomes; // outcomes
  vector[K] unq_outc; // unique outcomes
}

parameters {
  real<lower=0> rho;
  real A;
  real W;
  real<lower=1> C;
}

transformed parameters {
  vector[nT] wt;
  // matrix[K, nT] outc_wts;
  
  // Initialize outcome weights
  real k = 3; // sharpness of the sigmoid
  for (t in 1:nT) {
    wt[(nT+1-t)] = 1 / (1 + exp((t - C) * k)); // reverse filled
  }

  // outc_wts = rho * unq_outc * wt;

  // initialize transformed parameters
  real alpha = Phi_approx(A);
  real w = Phi_approx(W);
}

model {
  // Priors
  rho ~ exponential(0.5);
  A ~ normal(0, 0.5);
  W ~ normal(0, 0.5);
  C ~ normal(4, 2);

  matrix[N, S] Qs;
  matrix[N, S] Ws;
  vector[N] initial;
  initial = rep_vector(0.50375, N);
  real loglike = 0;
  
  // Initialize Q and W values
  for (s in 1:S) {
    Qs[:, s] = rho * initial .* valence;
    Ws[:, s] = rho * initial .* valence;
  }

  matrix[nT, S] outc_mat;
  // array[S, nT] int outc_lag;
  array[S] int outc_num;
  outc_mat = rep_matrix(0, nT, S);
  outc_num = rep_array(0, S);
  // outc_lag = rep_array(0, S, nT);
  
  // Loop over trials
  for (i in 1:N) {
    int pri = 2 * pair[i];
    real pi_rl = 1 / (1 + exp(-(Qs[i, pri] - Qs[i, pri - 1])));
    real pi_wm = 1 / (1 + exp(-(Ws[i, pri] - Ws[i, pri - 1])));
    real pi = w * pi_wm + (1 - w) * pi_rl;
    
    // Choice
    choice[i] ~ bernoulli(pi);
    int choice_idx = choice[i] + pri - 1;
    
    // Log likelihood
    loglike += bernoulli_lpmf(choice[i] | pi);
    
    // Prediction error
    real delta = (outcomes[i, choice[i] + 1] * rho) - Qs[i, choice[i] + pri - 1];
    
    // Update Qs and Ws and decay Ws
    if (i < N && block[i] == block[i + 1]) {
      Qs[i + 1] = Qs[i];
      Qs[i + 1, choice_idx] += alpha * delta;
      Ws[i + 1] = Ws[i];
      
      // // Update outcome buffers
      // for (t in 1:nT) {
      //   if (outc_lag[choice_idx, t] != 0) {
      //     outc_lag[choice_idx, t] += 1;
      //   }
      // }
      outc_num[choice_idx] += 1;
      int outc_no = outc_num[choice_idx];
      //outc_lag[choice_idx, outc_no] = 1;
      outc_mat[outc_no, choice_idx] = outcomes[i, choice[i] + 1] * rho;
      
      // Calculate weighted average of recent outcomes
      Ws[i + 1, choice_idx] = sum(outc_mat[1:outc_no, choice_idx] .* wt[(nT-outc_no+1):nT]) / outc_no;
    } else if (i < N) {
      // Reset buffers at the start of a new block
      outc_mat = rep_matrix(0, nT, S);
      //outc_lag = rep_array(0, S, nT);
      outc_num = rep_array(0, S);
    }
  }
}
