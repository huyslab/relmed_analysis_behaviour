##------------------------------------------------------------------------------
# RL models --------------------------------------------------------------------
## -----------------------------------------------------------------------------
# Transform unconstrainted a to learning rate α
a2α(a) = logistic(π/sqrt(3) * a)
α2a(α) = logit(α) / (π/sqrt(3))

@assert α2a(a2α(0.5)) ≈ 0.5

# Get data into correct format for the model -----------------------------------
function unpack_data(data::DataFrame)
    data_tuple = (
        block = data.block, # length = number of trials
        valence = unique(data[!, [:block, :valence]]).valence, # length = number of blocks
        pair = data.pair, # length = number of trials
        outcomes = hcat(data.feedback_suboptimal, data.feedback_optimal), # length = number of trials
        set_size = filter(x -> x.trial == 1 && x.pair == 1, data).set_size, # length = number of blocks
    )
    return data_tuple
end

# a version of the Collins and Frank model
@model function RLWM_ss(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 1.), lower = 0.),
        :a => Normal(0., 0.5),
        :F_wm => Normal(0., 0.5),
        :W => Normal(0., 0.5),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)
    # initial values
    aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
    initial = isnothing(initial) ? aao : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    parameters = collect(keys(priors))

    ## Priors on RL parameters ------------------------------------------------
    # reward sensitivity or inverse temp?
    ρ ~ priors[:ρ]
    
    # learning rate
    if :a in parameters
        a ~ priors[:a]
        α = a2α(a)
    elseif :α in parameters
        α ~ priors[:α]
    end

    ## Priors on WM parameters ------------------------------------------------
    # forgetting rate
    if :F_wm in parameters
        F_wm ~ priors[:F_wm]
        φ_wm = a2α(F_wm)
    elseif :φ_wm in parameters
        φ_wm ~ priors[:φ_wm]
    else
        φ_wm = 0
    end

    # initial weight of WM vs RL
    if :W in parameters
        W ~ priors[:W]
        w0 = a2α(W)
    elseif :w0 in parameters
        w0 ~ priors[:w0]
    end

    C ~ priors[:C] # capacity

    # Weight of WM vs RL
    w = Dict(s => w0 * min(1, C / s) for s in unique(data.set_size))

    # Initialization -----------------------------------------------------------
    # Q and W values
    Qs = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Q0, Q = copy(Qs), copy(Qs[:, 1:2])
    W0, Wv = copy(Ws), copy(Ws[:, 1:2])

    # Initial set-size
    ssz = data.set_size[data.block[1]]
    N = length(data.block)

    # Initialize log-likelihood
    loglike = 0.

    # Model --------------------------------------------------------------------

    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        pri = 2 * data.pair[i] # for indexing
        # RL and WM policies (softmax with directed and undirected noise)
        π_rl = 1 / (1 + exp(-(Qs[i, pri] - Qs[i, pri - 1])))
        π_wm = 1 / (1 + exp(-(Ws[i, pri] - Ws[i, pri - 1])))

        # Weighted policy
        π = w[ssz] * π_wm + (1 - w[ssz]) * π_rl
        logit_π = log(π / (1 - π))

		# Choice
        # n.b. can't do Bernoulli(π) because it's not numerically stable
		choice[i] ~ BernoulliLogit(logit_π)
		
        # Useful indices
        choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i] + 1

        # Log likelihood
        loglike += loglikelihood(BernoulliLogit(π), choice[i])

		# Prediction error
		δ = data.outcomes[i, choice_1id] * ρ - Qs[i, choice_idx]

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Qs[i + 1, :] = Qs[i, :] # store previous Q for all options
			Qs[i + 1, choice_idx] += α * δ # update chosen option
            Ws[i + 1, :] = Ws[i, :] + φ_wm * (W0[i, :] .- Ws[i, :]) # decay all W-values
            Ws[i + 1, choice_idx] += data.outcomes[i, choice_1id] * ρ # update chosen option
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
        end
        # store Q- and W- values for output
        Q[i, :] = Qs[i, (pri-1):pri]
        Wv[i, :] = Ws[i, (pri-1):pri]
    end

    return (choice = choice, Qs = Q, Ws = Wv, loglike = loglike)

end

@model function RLWM_pmst(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 1.), lower = 0.),
        :a => Normal(0., 0.5),
        :F_wm => Normal(0., 0.5),
        :W => Normal(0., 0.5),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)
    # initial values
    aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
    initial = isnothing(initial) ? aao : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    parameters = collect(keys(priors))

    ## Priors on RL parameters -------------------------------------------------
	# reward sensitivity or inverse temp?
    ρ ~ priors[:ρ]
    
    # learning rate
    if :a in parameters
        a ~ priors[:a]
        α = a2α(a)
    elseif :α in parameters
        α ~ priors[:α]
    end

    ## Priors on WM parameters ------------------------------------------------
    if :W in parameters
        W ~ priors[:W]
        w0 = a2α(W)
    elseif :w0 in parameters
        w0 ~ priors[:w0]
    end
    C ~ priors[:C] # capacity

    # Weight of WM vs RL
    w = Dict(s => w0 * min(1, C / s) for s in unique(data.set_size))

    # Initialisation -----------------------------------------------------------
    # Initialize Q and W values
    Qs = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Q0, Q = copy(Qs), copy(Qs[:, 1:2])
    Wv = copy(Ws[:, 1:2])

    # Initial set-size
    N = length(data.block)
    ssz = data.set_size[data.block[1]]

    # Initialize circular buffers and running sums for outcomes
    buffer_size = floor(Int, convert(Float64, C))
    buffers = [zeros(Float64, ssz) for _ in 1:buffer_size]
    buffer_sums = zeros(Float64, ssz)
    buffer_counts = zeros(Int, ssz)
    buffer_indicies = ones(Int, ssz)
    outc_no = ones(Int, ssz)

    # Initialize log-likelihood
    loglike = 0.

    # Model --------------------------------------------------------------------
    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        pri = 2 * data.pair[i]
        # RL and WM policies (softmax with directed and undirected noise)
        π_rl = 1 / (1 + exp(-(Qs[i, pri] - Qs[i, pri - 1])))
        π_wm = 1 / (1 + exp(-(Ws[i, pri] - Ws[i, pri - 1])))

        # Weighted policy
        π = w[ssz] * π_wm + (1 - w[ssz]) * π_rl
        logit_π = log(π / (1 - π))

		# Choice
		choice[i] ~ BernoulliLogit(logit_π) # again, because Bernoulli(π) is not numerically stable
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i] + 1

        # Log likelihood
        loglike += loglikelihood(BernoulliLogit(π), choice[i])

		# Prediction error
		δ = data.outcomes[i, choice_1id] * ρ - Qs[i, choice_idx]

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Qs[i + 1, :] = Qs[i, :] # store previous Q for all options
			Qs[i + 1, choice_idx] += α * δ # update chosen option
            Ws[i + 1, :] = Ws[i, :] # store previous W for all options
            
            ## Update circular buffer and running sum for the chosen option
            # 1. remove outcome C outcomes (for that choice) back from buffer
            buffer_upd_idx = buffer_indicies[choice_idx]
            buffer_sums[choice_idx] -= buffers[buffer_upd_idx][choice_idx]
            # 2. store the new outcome, and update the counts and sum of the chosen option
            buffers[buffer_upd_idx][choice_idx] = data.outcomes[i, choice_1id] * ρ # store the new outcome
            buffer_sums[choice_idx] += data.outcomes[i, choice_1id] * ρ # update the running sum of the chosen option
            buffer_counts[choice_idx] = min(buffer_counts[choice_idx] + 1, buffer_size)
            # 3. update the buffer index - mod1 gets remainder after division
            #    so e.g., on outcome 6 for that stimulus with C=4, mod1(7, 4) = 3, so we will be overwriting the 3rd element of the buffer
            buffer_indicies[choice_idx] = mod1(outc_no[choice_idx] + 1, buffer_size)
            outc_no[choice_idx] += 1
            # 4. compute the running average for each option using the running sum and buffer count
            Ws[i + 1, choice_idx] = buffer_sums[choice_idx] / buffer_counts[choice_idx]
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            # Reset buffers at the start of a new block
            buffers = [zeros(Float64, ssz) for _ in 1:buffer_size]
            buffer_sums = zeros(Float64, ssz)
            buffer_counts = zeros(Int, ssz)
            buffer_indicies = ones(Int, ssz)
            buffer_upd_idx = ones(Int, ssz)
            outc_no = ones(Int, ssz)
        end
        # store Q- and W- values for output
        Q[i, :] = Qs[i, (pri-1):pri]
        Wv[i, :] = Ws[i, (pri-1):pri]
    end

    return (choice = choice, Qs = Q, Ws = Wv, loglike = loglike)

end

@model function RLWM_pmst_sgd(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 1.), lower = 0.),
        :a => Normal(0., 0.5),
        :W => Normal(0., 0.5),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)

    # initial values
    aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
    initial = isnothing(initial) ? aao : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    parameters = collect(keys(priors))

    ## Priors on RL parameters ------------------------------------------------
	# reward sensitivity
    ρ ~ priors[:ρ]
    
    # learning rate
    if :a in parameters
        a ~ priors[:a]
        α = a2α(a)
    elseif :α in parameters
        α ~ priors[:α]
    end

    ## Priors on WM parameters ------------------------------------------------
    if :W in parameters
        W ~ priors[:W]
        w0 = a2α(W)
    elseif :w0 in parameters
        w0 ~ priors[:w0]
    end
    C ~ priors[:C] # capacity

    # Weight of WM vs RL
    w = Dict(s => w0 * min(1, C / s) for s in unique(data.set_size))

    ## Initialisation ----------------------------------------------------------
    # sigmoid transformation using C
    k = 5 # sharpness of the sigmoid
    nT = sum(data.block .== 1)
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k)) |> reverse

    # Initialize Q and W values
    Qs = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Q0, Q = copy(Qs), copy(Qs[:, 1:2])
    Wv = copy(Ws[:, 1:2])

    # Initial set-size
    N = length(data.block)
    ssz = data.set_size[data.block[1]]

    # Initialize outcome buffers
    outc_mat = zeros(Float64, nT, ssz)
    outc_nos = zeros(Int, ssz)
    outc_num = 0

    # Initialize log-loglikelihood
    loglike = 0.

    ## Model -------------------------------------------------------------------
    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        pri = 2 * data.pair[i]
        # RL and WM policies (softmax with directed and undirected noise)
        π_rl = 1 / (1 + exp(-β * (Qs[i, pri] - Qs[i, pri - 1])))
        π_wm = 1 / (1 + exp(-β * (Ws[i, pri] - Ws[i, pri - 1])))

        # Weighted policy
        π = w[ssz] * π_wm + (1 - w[ssz]) * π_rl

        # done it this way because Bernoulli() is not numerically stable
        # but the weights are defined by weighting the policy probabilities
        logit_π = log(π / (1 - π))

		# Choice
		choice[i] ~ BernoulliLogit(logit_π)
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i] + 1

        # log likelihood
        loglike += loglikelihood(BernoulliLogit(π), choice[i])

		# Prediction error
		δ = data.outcomes[i, choice_1id] * ρ - Qs[i, choice_idx]

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Qs[i + 1, :] = Qs[i, :] # store previous Q for all options
			Qs[i + 1, choice_idx] += α * δ # update chosen option
            Ws[i + 1, :] = Ws[i, :] # store previous W

            # we're going to do this deeply inefficiently...
            # 1. add outcome to buffer
            outc_nos[choice_idx] += 1
            outc_num = outc_nos[choice_idx]
            outc_mat[outc_num, choice_idx] = data.outcomes[i, choice_1id] * ρ
            # 2. multiply the buffer by the last outc_num weights in the weight vector and get the average
            Ws[i + 1, choice_idx] = sum(outc_mat[1:outc_num, choice_idx] .* wt[(end-outc_num+1):end]) / outc_num
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            # Reset buffers at the start of a new block
            outc_mat = zeros(Float64, nT, ssz)
            outc_nos = zeros(Int, ssz)
            outc_num = 0
        end
        # store Q- and W- values for output
        Q[i, :] = Qs[i, (pri-1):pri]
        Wv[i, :] = Ws[i, (pri-1):pri]
    end

    return (choice = choice, Qs = Q, Ws = Wv, loglike = loglike)

end