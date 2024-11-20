##------------------------------------------------------------------------------
# RL models --------------------------------------------------------------------
## -----------------------------------------------------------------------------

@model function RL_ss(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 1.), lower = 0.),
        :a => Normal(0., 0.5)
    ),
    initial::Union{Nothing, Float64} = nothing
)
    # initial values
    aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
    initial = isnothing(initial) ? aao : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    parameters = collect(keys(priors))
    @submodel a, E, F, β, ρ, α, ε, φ = rl_pars(priors, parameters)

    # Initialize Q values
	Qs = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]
    Q0, Q = copy(Qs), copy(Qs[:, 1:2])

    N = length(data.block)
    loglike = 0.

	# Loop over trials, updating Q values and incrementing log-density
	for i in eachindex(data.block)

        # Policy (softmax) - β=1 if we're using reward sensitivity and ε=0 if we're not using lapse rate
        # in Collins et al. terminology, these are directed and undirected noise
        pri = 2 * data.pair[i]
        π = (1 - ε) * (β * (Qs[i, pri] - Qs[i, pri - 1])) + ε * 0.5

		# Choice
		choice[i] ~ BernoulliLogit(π)
        
        # useful indexes
        choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i] + 1
        
        # log likelihood
        loglike += loglikelihood(BernoulliLogit(π), choice[i])

		# Prediction error
		δ = data.outcomes[i, choice_1id] * ρ - Qs[i, choice_idx]

		# Update Q value
		if (i != N) && (data.block[i] == data.block[i+1])
            Qs[i + 1, :] = Qs[i, :] + φ * (Q0[i, :] .- Qs[i, :]) # decay or just store previous Q
			Qs[i + 1, choice_idx] += α * δ
		end
        # store Q values for output (n.b. these are the values for pair[i] *before* the update)
        Q[i, :] = Qs[i, (pri-1):pri]
	end

	return (choice = choice, Qs = Q, loglike = loglike)

end

@model function RL_recip_ss(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 1.), lower = 0.),
        :a => Normal(0., 0.5)
    ),
    initial::Union{Nothing, Float64} = nothing
)
    # initial values
    aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
    initial = isnothing(initial) ? aao : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    parameters = collect(keys(priors))
    @submodel a, E, F, β, ρ, α, ε, φ = rl_pars(priors, parameters)

	# Initialize Q values
	Qs = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]
    Q0, Q = copy(Qs), copy(Qs[:, 1:2])

    N = length(data.block)
    loglike = 0.

	# Loop over trials, updating Q values and incrementing log-density
	for i in eachindex(data.block)

        # Policy (softmax) - β=1 if we're using reward sensitivity and ε=0 if we're not using lapse rate
        # in Collins et al. terminology, these are directed and undirected noise
        pri = 2 * data.pair[i]
        π = (1 - ε) * (β * (Qs[i, pri] - Qs[i, pri - 1])) + ε * 0.5

		# Choice
		choice[i] ~ BernoulliLogit(π)
		
        # useful indexes
        choice_idx::Int64 = choice[i] + pri - 1
        alt_idx::Int64 = pri - choice[i]
        choice_1id::Int64 = choice[i] + 1
        
        # log likelihood
        loglike += loglikelihood(BernoulliLogit(π), choice[i])

		# Prediction error
		δ = data.outcomes[i, choice_1id] * ρ - Qs[i, choice_idx]

		# Update Q value
		if (i != N) && (data.block[i] == data.block[i+1])
            Qs[i + 1, :] = Qs[i, :] + φ * (Q0[i, :] .- Qs[i, :]) # decay or just store previous Q
			Qs[i + 1, choice_idx] += α * δ
            Qs[i + 1, alt_idx] -= α * δ
		end
        # store Q values for output (n.b. these are the values for pair[i] *before* the update) 
        Q[i, :] = Qs[i, (pri-1):pri]
	end

	return (choice = choice, Qs = Q, loglike = loglike)

end

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
    set_sizes = unique(data.set_size)
    @submodel a, E, F_rl, β, ρ, α, ε, φ_rl = rl_pars(priors, parameters)
    @submodel F_wm, φ_wm, W = wm_pars(priors, parameters, set_sizes)

    C ~ priors[:C] # capacity
    w = Dict(s => a2α(W) * min(1, C / s) for s in set_sizes)

    # Initialize Q and W values
    Qs = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Q0, Q = copy(Qs), copy(Qs[:, 1:2])
    W0, Wv = copy(Ws), copy(Ws[:, 1:2])

    # Initial set-size
    ssz = data.set_size[data.block[1]]
    N = length(data.block)
    loglike = 0.

    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        pri = 2 * data.pair[i]
        # RL and WM policies (softmax with directed and undirected noise)
        π_rl = (1 - ε) * (1 / (1 + exp(-β * (Qs[i, pri] - Qs[i, pri - 1])))) + ε * 0.5
        π_wm = (1 - ε) * (1 / (1 + exp(-β * (Ws[i, pri] - Ws[i, pri - 1])))) + ε * 0.5

        # Weighted policy
        π = w[ssz] * π_wm + (1 - w[ssz]) * π_rl
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
            Qs[i + 1, :] = Qs[i, :] + φ_rl * (Q0[i, :] .- Qs[i, :]) # decay or just store previous Q
			Qs[i + 1, choice_idx] += α * δ
            Ws[i + 1, :] = Ws[i, :] + φ_wm * (W0[i, :] .- Ws[i, :]) # decay or just store previous W
            Ws[i + 1, choice_idx] += data.outcomes[i, choice_1id] * ρ
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
        end
        # store Q- and W-values for output (n.b. these are the values for pair[i] *before* the update)
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
    set_sizes = unique(data.set_size)
    @submodel a, E, F_rl, β, ρ, α, ε, φ_rl = rl_pars(priors, parameters)
    @submodel F_wm, φ_wm, W = wm_pars(priors, parameters, set_sizes)

    C ~ priors[:C] # capacity
    w = isa(W, Dict) ? Dict(s => a2α(W[s]) for s in set_sizes) : Dict(s => a2α(W) for s in set_sizes)

    # Initialize Q and W values
    Qs = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Q0, Q = copy(Qs), copy(Qs[:, 1:2])
    Wv = copy(Ws[:, 1:2])

    # Initial set-size
    N = length(data.block)
    ssz = data.set_size[data.block[1]]
    loglike = 0.
    gd = groupby(DataFrame("block" => data.block, "pair" => data.pair), :block)
    nT = maximum(combine(gd, :pair => (x -> maximum(values(countmap(x)))) => :max_count).max_count)

    # Initialize circular buffers and running sums for outcomes
    bs = clamp(C, 1., nT)
    buffer_size = floor(Int, bs)
    buffers = [Vector{Any}(nothing, ssz) for _ in 1:buffer_size]
    buffer_sums = Vector{Any}(nothing, ssz)
    buffer_counts = zeros(Int, ssz)
    buffer_indicies = ones(Int, ssz)
    outc_no = ones(Int, ssz)

    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        pri = 2 * data.pair[i]
        # RL and WM policies (softmax with directed and undirected noise)
        π_rl = (1 - ε) * (1 / (1 + exp(-β * (Qs[i, pri] - Qs[i, pri - 1])))) + ε * 0.5
        π_wm = (1 - ε) * (1 / (1 + exp(-β * (Ws[i, pri] - Ws[i, pri - 1])))) + ε * 0.5

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
            Qs[i + 1, :] = Qs[i, :] + φ_rl * (Q0[i, :] .- Qs[i, :]) # decay or just store previous Q
			Qs[i + 1, choice_idx] += α * δ
            Ws[i + 1, :] = Ws[i, :] # store previous W
            
            ## Update circular buffer and running sum for the chosen option
            # 1. remove outcome C outcomes (for that choice) back from buffer
            buffer_upd_idx = buffer_indicies[choice_idx]
            buffer_sums[choice_idx] = (
                (buffer_sums[choice_idx] === nothing ? 0 : buffer_sums[choice_idx]) - 
                (buffers[buffer_upd_idx][choice_idx] === nothing ? 0 : buffers[buffer_upd_idx][choice_idx])
            )
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
            buffers = [Vector{Any}(nothing, ssz) for _ in 1:buffer_size]
            buffer_sums = Vector{Any}(nothing, ssz)
            buffer_counts = zeros(Int, ssz)
            buffer_indicies = ones(Int, ssz)
            buffer_upd_idx = ones(Int, ssz)
            outc_no = ones(Int, ssz)
        end
        # store Q- and W-values for output (n.b. these are the values for pair[i] *before* the update)
        Q[i, :] = Qs[i, (pri-1):pri]
        Wv[i, :] = Ws[i, (pri-1):pri]
    end

    return (choice = choice, Qs = Q, Ws = Wv, loglike = loglike)

end

@model function WM_pmst_sgd(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 2.), lower = 0.),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)

    # initial values
    aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
    initial = isnothing(initial) ? aao : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    C ~ priors[:C] # capacity

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    gd = groupby(DataFrame("block" => data.block, "pair" => data.pair), :block)
    nT = maximum(combine(gd, :pair => (x -> maximum(values(countmap(x)))) => :max_count).max_count)
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in data.outcomes, premultiply by ρ and wt
    unq_outc = unique(data.outcomes)
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Wv = copy(Ws[:, 1:2])

    # Initial set-size
    N = length(data.block)
    ssz = data.set_size[data.block[1]]

    # Initialize outcome buffers
    outc_no = 0
    outc_mat = Matrix{Any}(nothing, ssz, nT) # matrix of recent outcomes for each stimulus
    outc_lag = zeros(Int, ssz, nT) # how many outcomes back was this outcome?
    outc_num = zeros(Int, ssz) # how many outcomes have we seen for this stimulus?

    # store log-loglikelihood
    loglike = 0.

    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        pri = 2 * data.pair[i]

		# Choice
		choice[i] ~ BernoulliLogit(Ws[i, pri] - Ws[i, pri - 1])
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i] + 1

        # log likelihood
        loglike += loglikelihood(BernoulliLogit(π), choice[i])

		# Prediction error
        chce = data.outcomes[i, choice_1id]

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Ws[i + 1, :] = Ws[i, :] # store previous W
            # 1. iterate lags for prior outcomes and update outcome number for this stimulus
            outc_lag[choice_idx] += outc_lag[choice_idx] .!== 0
            outc_num[choice_idx] += 1
            outc_no = outc_num[choice_idx]
            # 2. initialise the lag for this outcome number and store the relevant outcome
            outc_lag[choice_idx, outc_no], outc_mat[choice_idx, outc_no] = 1, chce
            # 3. use the pre-computed weights to calculate the sigmoid weighted average of recent outcomes
            Ws[i + 1, choice_idx] = sum([outc_wts[outc_key[outc_mat[choice_idx, j]], outc_lag[choice_idx, j]] for j in 1:outc_no]) / min(outc_no, C)
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            # reset buffers at the start of a new block
            outc_no = 0
            outc_mat = Matrix{Any}(nothing, ssz, nT)
            outc_lag = zeros(Int, ssz, nT)
            outc_num = zeros(Int, ssz)
        end
        # store Q- and W-values for output (n.b. these are the values for pair[i] *before* the update)
        Wv[i, :] = Ws[i, (pri-1):pri]
    end

    return (choice = choice, Qs = nothing, Ws = Wv, loglike = loglike)

end

@model function WM_pmst_sgd_sum(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 2.), lower = 0.),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)

    # initial values
    aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
    initial = isnothing(initial) ? aao : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    C ~ priors[:C] # capacity

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    gd = groupby(DataFrame("block" => data.block, "pair" => data.pair), :block)
    nT = maximum(combine(gd, :pair => (x -> maximum(values(countmap(x)))) => :max_count).max_count)
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in data.outcomes, premultiply by ρ and wt
    unq_outc = unique(data.outcomes)
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Wv = copy(Ws[:, 1:2])

    # Initial set-size
    N = length(data.block)
    ssz = data.set_size[data.block[1]]

    # Initialize outcome buffers
    outc_no = 0
    outc_mat = Matrix{Any}(nothing, ssz, nT) # matrix of recent outcomes for each stimulus
    outc_lag = zeros(Int, ssz, nT) # how many outcomes back was this outcome?
    outc_num = zeros(Int, ssz) # how many outcomes have we seen for this stimulus?

    # store log-loglikelihood
    loglike = 0.

    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        pri = 2 * data.pair[i]

        # Choice
        choice[i] ~ BernoulliLogit(Ws[i, pri] - Ws[i, pri - 1])
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i] + 1

        # log likelihood
        loglike += loglikelihood(BernoulliLogit(π), choice[i])

		# Prediction error
        chce = data.outcomes[i, choice_1id]

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Ws[i + 1, :] = Ws[i, :] # store previous W
            # 1. iterate lags for prior outcomes and update outcome number for this stimulus
            outc_lag[choice_idx] += outc_lag[choice_idx] .!== 0
            outc_num[choice_idx] += 1
            outc_no = outc_num[choice_idx]
            # 2. initialise the lag for this outcome number and store the relevant outcome
            outc_lag[choice_idx, outc_no], outc_mat[choice_idx, outc_no] = 1, chce
            # 3. use the pre-computed weights to calculate the sigmoid weighted average of recent outcomes
            Ws[i + 1, choice_idx] = sum([outc_wts[outc_key[outc_mat[choice_idx, j]], outc_lag[choice_idx, j]] for j in 1:outc_no])
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            # reset buffers at the start of a new block
            outc_no = 0
            outc_mat = Matrix{Any}(nothing, ssz, nT)
            outc_lag = zeros(Int, ssz, nT)
            outc_num = zeros(Int, ssz)
        end
        # store Q- and W-values for output (n.b. these are the values for pair[i] *before* the update)
        Wv[i, :] = Ws[i, (pri-1):pri]
    end

    return (choice = choice, Qs = nothing, Ws = Wv, loglike = loglike)

end

@model function WM_pmst_sgd_recip(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 2.), lower = 0.),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)

    # initial values
    aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
    initial = isnothing(initial) ? aao : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    C ~ priors[:C] # capacity

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    gd = groupby(DataFrame("block" => data.block, "pair" => data.pair), :block)
    nT = maximum(combine(gd, :pair => (x -> maximum(values(countmap(x)))) => :max_count).max_count)
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in data.outcomes, premultiply by ρ and wt
    unq_outc = unique(data.outcomes)
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Wv = copy(Ws[:, 1:2])

    # Initial set-size
    N = length(data.block)
    ssz = data.set_size[data.block[1]]
    val = data.valence[data.block[1]]

    # Initialize outcome buffers
    outc_no = 0
    outc_mat = Matrix{Any}(nothing, ssz, nT) # matrix of recent outcomes for each stimulus
    outc_lag = zeros(Int, ssz, nT) # how many outcomes back was this outcome?
    outc_num = zeros(Int, ssz) # how many outcomes have we seen for this stimulus?

    # store log-loglikelihood
    loglike = 0.

    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        pri = 2 * data.pair[i]
        
        # Choice
        choice[i] ~ BernoulliLogit(Ws[i, pri] - Ws[i, pri - 1])
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i] + 1
        alt_idx::Int64 = pri - choice[i]

        # log likelihood
        loglike += loglikelihood(BernoulliLogit(π), choice[i])

		# Prediction error
        chce = data.outcomes[i, choice_1id]
        # use pseudorandom number generator because optimizer doesn't like rand() (computes is hash(i) even or odd + 1)
        alt = abs(chce) >= 0.5 ? val * 0.01 : val * 1.0 # [0.5, 1.0][mod(hash(i), 2) + 1]

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Ws[i + 1, :] = Ws[i, :] # store previous W
            # 1. iterate lags for prior outcomes and update outcome number for this stimulus
            outc_lag[pri-1:pri] += outc_lag[pri-1:pri] .!== 0
            outc_num[pri-1:pri] += [1, 1]
            outc_no = outc_num[choice_idx]
            # 2. initialise the lag for this outcome number and store the relevant outcome
            outc_lag[choice_idx, outc_no], outc_mat[choice_idx, outc_no] = 1, chce
            outc_lag[alt_idx, outc_no], outc_mat[alt_idx, outc_no] = 1, alt
            # 3. use the pre-computed weights to calculate the sigmoid weighted average of recent outcomes
            Ws[i + 1, choice_idx] = sum([outc_wts[outc_key[outc_mat[choice_idx, j]], outc_lag[choice_idx, j]] for j in 1:outc_no]) / min(outc_no, C)
            Ws[i + 1, alt_idx] = sum([outc_wts[outc_key[outc_mat[alt_idx, j]], outc_lag[alt_idx, j]] for j in 1:outc_no]) / min(outc_no, C)
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            val = data.valence[data.block[i+1]]
            # reset buffers at the start of a new block
            outc_no = 0
            outc_mat = Matrix{Any}(nothing, ssz, nT)
            outc_lag = zeros(Int, ssz, nT)
            outc_num = zeros(Int, ssz)
        end
        # store Q- and W-values for output (n.b. these are the values for pair[i] *before* the update)
        Wv[i, :] = Ws[i, (pri-1):pri]
    end

    return (choice = choice, Qs = nothing, Ws = Wv, loglike = loglike)

end

@model function WM_pmst_sgd_sum_recip(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 2.), lower = 0.),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)

    # initial values
    aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
    initial = isnothing(initial) ? aao : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    C ~ priors[:C] # capacity

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    gd = groupby(DataFrame("block" => data.block, "pair" => data.pair), :block)
    nT = maximum(combine(gd, :pair => (x -> maximum(values(countmap(x)))) => :max_count).max_count)
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in data.outcomes, premultiply by ρ and wt
    unq_outc = unique(data.outcomes)
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Wv = copy(Ws[:, 1:2])

    # Initial set-size
    N = length(data.block)
    ssz = data.set_size[data.block[1]]
    val = data.valence[data.block[1]]

    # Initialize outcome buffers
    outc_no = 0
    outc_mat = Matrix{Any}(nothing, ssz, nT) # matrix of recent outcomes for each stimulus
    outc_lag = zeros(Int, ssz, nT) # how many outcomes back was this outcome?
    outc_num = zeros(Int, ssz) # how many outcomes have we seen for this stimulus?

    # store log-loglikelihood
    loglike = 0.

    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        pri = 2 * data.pair[i]
        # WM policy (softmax with directed and undirected noise)
        π = 1 / (1 + exp(-(Ws[i, pri] - Ws[i, pri - 1])))

        # done it this way because Bernoulli() is not numerically stable
        # but the weights are defined by weighting the policy probabilities
        logit_π = log(π / (1 - π))

		# Choice
		choice[i] ~ BernoulliLogit(logit_π)
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i] + 1
        alt_idx::Int64 = pri - choice[i]

        # log likelihood
        loglike += loglikelihood(BernoulliLogit(π), choice[i])

		# Prediction error
        chce = data.outcomes[i, choice_1id]
        # use pseudorandom number generator because optimizer doesn't like rand() (computes is hash(i) even or odd + 1)
        alt = abs(chce) >= 0.5 ? val * 0.01 : val * 1.0 # [0.5, 1.0][mod(hash(i), 2) + 1]

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Ws[i + 1, :] = Ws[i, :] # store previous W
            # 1. iterate lags for prior outcomes and update outcome number for this stimulus
            outc_lag[pri-1:pri] += outc_lag[pri-1:pri] .!== 0
            outc_num[pri-1:pri] += [1, 1]
            outc_no = outc_num[choice_idx]
            # 2. initialise the lag for this outcome number and store the relevant outcome
            outc_lag[choice_idx, outc_no], outc_mat[choice_idx, outc_no] = 1, chce
            outc_lag[alt_idx, outc_no], outc_mat[alt_idx, outc_no] = 1, alt
            # 3. use the pre-computed weights to calculate the sigmoid weighted average of recent outcomes
            Ws[i + 1, choice_idx] = sum([outc_wts[outc_key[outc_mat[choice_idx, j]], outc_lag[choice_idx, j]] for j in 1:outc_no])
            Ws[i + 1, alt_idx] = sum([outc_wts[outc_key[outc_mat[alt_idx, j]], outc_lag[alt_idx, j]] for j in 1:outc_no])
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            val = data.valence[data.block[i+1]]
            # reset buffers at the start of a new block
            outc_no = 0
            outc_mat = Matrix{Any}(nothing, ssz, nT)
            outc_lag = zeros(Int, ssz, nT)
            outc_num = zeros(Int, ssz)
        end
        # store Q- and W-values for output (n.b. these are the values for pair[i] *before* the update)
        Wv[i, :] = Ws[i, (pri-1):pri]
    end

    return (choice = choice, Qs = nothing, Ws = Wv, loglike = loglike)

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
    set_sizes = unique(data.set_size)
    @submodel a, E, F_rl, β, ρ, α, ε, φ_rl = rl_pars(priors, parameters)
    @submodel F_wm, φ_wm, W = wm_pars(priors, parameters, set_sizes)

    C ~ priors[:C] # capacity
    w = isa(W, Dict) ? Dict(s => a2α(W[s]) for s in set_sizes) : Dict(s => a2α(W) for s in set_sizes)

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    gd = groupby(DataFrame("block" => data.block, "pair" => data.pair), :block)
    nT = maximum(combine(gd, :pair => (x -> maximum(values(countmap(x)))) => :max_count).max_count)
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in data.outcomes, premultiply by ρ and wt
    unq_outc = unique(data.outcomes)
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

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
    outc_no = 0
    outc_mat = Matrix{Any}(nothing, ssz, nT) # matrix of recent outcomes for each stimulus
    outc_lag = zeros(Int, ssz, nT) # how many outcomes back was this outcome?
    outc_num = zeros(Int, ssz) # how many outcomes have we seen for this stimulus?

    # store log-loglikelihood
    loglike = 0.

    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        pri = 2 * data.pair[i]
        # RL and WM policies (softmax with directed and undirected noise)
        π_rl = (1 - ε) * (1 / (1 + exp(-β * (Qs[i, pri] - Qs[i, pri - 1])))) + ε * 0.5
        π_wm = (1 - ε) * (1 / (1 + exp(-β * (Ws[i, pri] - Ws[i, pri - 1])))) + ε * 0.5

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
        chce = data.outcomes[i, choice_1id]
		δ = chce * ρ - Qs[i, choice_idx]

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Qs[i + 1, :] = Qs[i, :] + φ_rl * (Q0[i, :] .- Qs[i, :]) # decay or just store previous Q
			Qs[i + 1, choice_idx] += α * δ
            Ws[i + 1, :] = Ws[i, :] # store previous W
            # 1. iterate lags for prior outcomes and update outcome number for this stimulus
            outc_lag[choice_idx] += outc_lag[choice_idx] .!== 0
            outc_num[choice_idx] += 1
            outc_no = outc_num[choice_idx]
            # 2. initialise the lag for this outcome number and store the relevant outcome
            outc_lag[choice_idx, outc_no], outc_mat[choice_idx, outc_no] = 1, chce
            # 3. use the pre-computed weights to calculate the sigmoid weighted average of recent outcomes
            Ws[i + 1, choice_idx] = sum([outc_wts[outc_key[outc_mat[choice_idx, j]], outc_lag[choice_idx, j]] for j in 1:outc_no]) / min(outc_no, C)
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            # reset buffers at the start of a new block
            outc_no = 0
            outc_mat = Matrix{Any}(nothing, ssz, nT)
            outc_lag = zeros(Int, ssz, nT)
            outc_num = zeros(Int, ssz)
        end
        # store Q- and W-values for output (n.b. these are the values for pair[i] *before* the update)
        Q[i, :] = Qs[i, (pri-1):pri]
        Wv[i, :] = Ws[i, (pri-1):pri]
    end

    return (choice = choice, Qs = Q, Ws = Wv, loglike = loglike)

end

@model function RLWM_pmst_sgd_recip(
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
    set_sizes = unique(data.set_size)
    @submodel a, E, F_rl, β, ρ, α, ε, φ_rl = rl_pars(priors, parameters)
    @submodel F_wm, φ_wm, W = wm_pars(priors, parameters, set_sizes)

    C ~ priors[:C] # capacity
    w = isa(W, Dict) ? Dict(s => a2α(W[s]) for s in set_sizes) : Dict(s => a2α(W) for s in set_sizes)

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    gd = groupby(DataFrame("block" => data.block, "pair" => data.pair), :block)
    nT = maximum(combine(gd, :pair => (x -> maximum(values(countmap(x)))) => :max_count).max_count)
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in data.outcomes, premultiply by ρ and wt
    unq_outc = unique(data.outcomes)
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Qs = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Q0, Q = copy(Qs), copy(Qs[:, 1:2])
    Wv = copy(Ws[:, 1:2])

    # Initial set-size
    N = length(data.block)
    ssz = data.set_size[data.block[1]]
    val = data.valence[data.block[1]]

    # Initialize outcome buffers
    outc_no = 0
    outc_mat = Matrix{Any}(nothing, ssz, nT) # matrix of recent outcomes for each stimulus
    outc_lag = zeros(Int, ssz, nT) # how many outcomes back was this outcome?
    outc_num = zeros(Int, ssz) # how many outcomes have we seen for this stimulus?

    # store log-loglikelihood
    loglike = 0.

    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        pri = 2 * data.pair[i]
        # RL and WM policies (softmax with directed and undirected noise)
        π_rl = (1 - ε) * (1 / (1 + exp(-β * (Qs[i, pri] - Qs[i, pri - 1])))) + ε * 0.5
        π_wm = (1 - ε) * (1 / (1 + exp(-β * (Ws[i, pri] - Ws[i, pri - 1])))) + ε * 0.5

        # Weighted policy
        π = w[ssz] * π_wm + (1 - w[ssz]) * π_rl

        # done it this way because Bernoulli() is not numerically stable
        # but the weights are defined by weighting the policy probabilities
        logit_π = log(π / (1 - π))

		# Choice
		choice[i] ~ BernoulliLogit(logit_π)
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i] + 1
        alt_idx::Int64 = pri - choice[i]

        # log likelihood
        loglike += loglikelihood(BernoulliLogit(π), choice[i])

		# Prediction error
        chce = data.outcomes[i, choice_1id]
        # use pseudorandom number generator because optimizer doesn't like rand() (computes is hash(i) even or odd + 1)
        alt = abs(chce) >= 0.5 ? val * 0.01 : val * 1.0 # [0.5, 1.0][mod(hash(i), 2) + 1]
		δ = chce * ρ - Qs[i, choice_idx]

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Qs[i + 1, :] = Qs[i, :] + φ_rl * (Q0[i, :] .- Qs[i, :]) # decay or just store previous Q
			Qs[i + 1, choice_idx] += α * δ
            Qs[i + 1, alt_idx] -= α * δ
            Ws[i + 1, :] = Ws[i, :] # store previous W
            # 1. iterate lags for prior outcomes and update outcome number for this stimulus
            outc_lag[pri-1:pri] += outc_lag[pri-1:pri] .!== 0
            outc_num[pri-1:pri] += [1, 1]
            outc_no = outc_num[choice_idx]
            # 2. initialise the lag for this outcome number and store the relevant outcome
            outc_lag[choice_idx, outc_no], outc_mat[choice_idx, outc_no] = 1, chce
            outc_lag[alt_idx, outc_no], outc_mat[alt_idx, outc_no] = 1, alt
            # 3. use the pre-computed weights to calculate the sigmoid weighted average of recent outcomes
            Ws[i + 1, choice_idx] = sum([outc_wts[outc_key[outc_mat[choice_idx, j]], outc_lag[choice_idx, j]] for j in 1:outc_no]) / min(outc_no, C)
            Ws[i + 1, alt_idx] = sum([outc_wts[outc_key[outc_mat[alt_idx, j]], outc_lag[alt_idx, j]] for j in 1:outc_no]) / min(outc_no, C)
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            val = data.valence[data.block[i+1]]
            # reset buffers at the start of a new block
            outc_no = 0
            outc_mat = Matrix{Any}(nothing, ssz, nT)
            outc_lag = zeros(Int, ssz, nT)
            outc_num = zeros(Int, ssz)
        end
        # store Q- and W-values for output (n.b. these are the values for pair[i] *before* the update)
        Q[i, :] = Qs[i, (pri-1):pri]
        Wv[i, :] = Ws[i, (pri-1):pri]
    end

    return (choice = choice, Qs = Q, Ws = Wv, loglike = loglike)

end

@model function RLWM_pmst_sgd_sum_recip(
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
    set_sizes = unique(data.set_size)
    @submodel a, E, F_rl, β, ρ, α, ε, φ_rl = rl_pars(priors, parameters)
    @submodel F_wm, φ_wm, W = wm_pars(priors, parameters, set_sizes)

    C ~ priors[:C] # capacity
    w = isa(W, Dict) ? Dict(s => a2α(W[s]) for s in set_sizes) : Dict(s => a2α(W) for s in set_sizes)

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    gd = groupby(DataFrame("block" => data.block, "pair" => data.pair), :block)
    nT = maximum(combine(gd, :pair => (x -> maximum(values(countmap(x)))) => :max_count).max_count)
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in data.outcomes, premultiply by ρ and wt
    unq_outc = unique(data.outcomes)
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Qs = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Q0, Q = copy(Qs), copy(Qs[:, 1:2])
    Wv = copy(Ws[:, 1:2])

    # Initial set-size
    N = length(data.block)
    ssz = data.set_size[data.block[1]]
    val = data.valence[data.block[1]]

    # Initialize outcome buffers
    outc_no = 0
    outc_mat = Matrix{Any}(nothing, ssz, nT) # matrix of recent outcomes for each stimulus
    outc_lag = zeros(Int, ssz, nT) # how many outcomes back was this outcome?
    outc_num = zeros(Int, ssz) # how many outcomes have we seen for this stimulus?

    # store log-loglikelihood
    loglike = 0.

    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        pri = 2 * data.pair[i]
        # RL and WM policies (softmax with directed and undirected noise)
        π_rl = (1 - ε) * (1 / (1 + exp(-β * (Qs[i, pri] - Qs[i, pri - 1])))) + ε * 0.5
        π_wm = (1 - ε) * (1 / (1 + exp(-β * (Ws[i, pri] - Ws[i, pri - 1])))) + ε * 0.5

        # Weighted policy
        π = w[ssz] * π_wm + (1 - w[ssz]) * π_rl

        # done it this way because Bernoulli() is not numerically stable
        # but the weights are defined by weighting the policy probabilities
        logit_π = log(π / (1 - π))

		# Choice
		choice[i] ~ BernoulliLogit(logit_π)
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i] + 1
        alt_idx::Int64 = pri - choice[i]

        # log likelihood
        loglike += loglikelihood(BernoulliLogit(π), choice[i])

		# Prediction error
        chce = data.outcomes[i, choice_1id]
        # use pseudorandom number generator because optimizer doesn't like rand() (computes is hash(i) even or odd + 1)
        alt = abs(chce) >= 0.5 ? val * 0.01 : val * 1.0 # [0.5, 1.0][mod(hash(i), 2) + 1]
		δ = chce * ρ - Qs[i, choice_idx]

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Qs[i + 1, :] = Qs[i, :] + φ_rl * (Q0[i, :] .- Qs[i, :]) # decay or just store previous Q
			Qs[i + 1, choice_idx] += α * δ
            Qs[i + 1, alt_idx] -= α * δ
            Ws[i + 1, :] = Ws[i, :] # store previous W
            # 1. iterate lags for prior outcomes and update outcome number for this stimulus
            outc_lag[pri-1:pri] += outc_lag[pri-1:pri] .!== 0
            outc_num[pri-1:pri] += [1, 1]
            outc_no = outc_num[choice_idx]
            # 2. initialise the lag for this outcome number and store the relevant outcome
            outc_lag[choice_idx, outc_no], outc_mat[choice_idx, outc_no] = 1, chce
            outc_lag[alt_idx, outc_no], outc_mat[alt_idx, outc_no] = 1, alt
            # 3. use the pre-computed weights to calculate the sigmoid weighted average of recent outcomes
            Ws[i + 1, choice_idx] = sum([outc_wts[outc_key[outc_mat[choice_idx, j]], outc_lag[choice_idx, j]] for j in 1:outc_no])
            Ws[i + 1, alt_idx] = sum([outc_wts[outc_key[outc_mat[alt_idx, j]], outc_lag[alt_idx, j]] for j in 1:outc_no])
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            val = data.valence[data.block[i+1]]
            # reset buffers at the start of a new block
            outc_no = 0
            outc_mat = Matrix{Any}(nothing, ssz, nT)
            outc_lag = zeros(Int, ssz, nT)
            outc_num = zeros(Int, ssz)
        end
        # store Q- and W-values for output (n.b. these are the values for pair[i] *before* the update)
        Q[i, :] = Qs[i, (pri-1):pri]
        Wv[i, :] = Ws[i, (pri-1):pri]
    end

    return (choice = choice, Qs = Q, Ws = Wv, loglike = loglike)

end

@model function WM_all_outc_pmst_sgd(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 2.), lower = 0.),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)

    # initial values
    aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
    initial = isnothing(initial) ? aao : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    C ~ priors[:C] # capacity

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    nT = sum(data.block .== mode(data.block))
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in data.outcomes, premultiply by ρ and wt
    unq_outc = unique(data.outcomes)
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Wv = copy(Ws[:, 1:2])

    # Initial set-size
    N = length(data.block)
    ssz = data.set_size[data.block[1]]

    # Initialize outcome buffers
    outc_no = 0
    outc_mat = Matrix{Any}(nothing, ssz, nT) # matrix of recent outcomes for each stimulus
    outc_lag = zeros(Int, ssz, nT) # how many outcomes back was this outcome?
    outc_num = zeros(Int, ssz) # how many outcomes have we seen for this stimulus?

    # store log-loglikelihood
    loglike = 0.

    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        pri = 2 * data.pair[i]

        # Choice
        choice[i] ~ BernoulliLogit(Ws[i, pri] - Ws[i, pri - 1])
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i] + 1

        # log likelihood
        loglike += loglikelihood(BernoulliLogit(π), choice[i])

		# Prediction error
        chce = data.outcomes[i, choice_1id]

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Ws[i + 1, :] = Ws[i, :] # store previous W
            # 1. iterate lags for prior outcomes and update outcome number for this stimulus
            outc_lag += outc_lag .!== 0
            outc_num[choice_idx] += 1
            outc_no = outc_num[choice_idx]
            # 2. initialise the lag for this outcome number and store the relevant outcome
            outc_lag[choice_idx, outc_no], outc_mat[choice_idx, outc_no] = 1, chce
            # 3. use the pre-computed weights to calculate the sigmoid weighted average of recent outcomes
            Ws[i + 1, choice_idx] = sum([outc_wts[outc_key[outc_mat[choice_idx, j]], outc_lag[choice_idx, j]] for j in 1:outc_no]) / min(outc_no, C)
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            # reset buffers at the start of a new block
            outc_no = 0
            outc_mat = Matrix{Any}(nothing, ssz, nT)
            outc_lag = zeros(Int, ssz, nT)
            outc_num = zeros(Int, ssz)
        end
        # store W-values for output (n.b. these are the values for pair[i] *before* the update)
        Wv[i, :] = Ws[i, (pri-1):pri]
    end

    return (choice = choice, Qs = nothing, Ws = Wv, loglike = loglike)

end

@model function WM_all_outc_pmst_sgd_sum(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 2.), lower = 0.),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)

    # initial values
    aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
    initial = isnothing(initial) ? aao : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    C ~ priors[:C] # capacity

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    nT = sum(data.block .== mode(data.block))
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in data.outcomes, premultiply by ρ and wt
    unq_outc = unique(data.outcomes)
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Wv = copy(Ws[:, 1:2])

    # Initial set-size
    N = length(data.block)
    ssz = data.set_size[data.block[1]]

    # Initialize outcome buffers
    outc_no = 0
    outc_mat = Matrix{Any}(nothing, ssz, nT) # matrix of recent outcomes for each stimulus
    outc_lag = zeros(Int, ssz, nT) # how many outcomes back was this outcome?
    outc_num = zeros(Int, ssz) # how many outcomes have we seen for this stimulus?

    # store log-loglikelihood
    loglike = 0.

    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        pri = 2 * data.pair[i]
        
        # Choice
        choice[i] ~ BernoulliLogit(Ws[i, pri] - Ws[i, pri - 1])
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i] + 1

        # log likelihood
        loglike += loglikelihood(BernoulliLogit(π), choice[i])

		# Prediction error
        chce = data.outcomes[i, choice_1id]

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Ws[i + 1, :] = Ws[i, :] # store previous W
            # 1. iterate lags for prior outcomes and update outcome number for this stimulus
            outc_lag += outc_lag .!== 0
            outc_num[choice_idx] += 1
            outc_no = outc_num[choice_idx]
            # 2. initialise the lag for this outcome number and store the relevant outcome
            outc_lag[choice_idx, outc_no], outc_mat[choice_idx, outc_no] = 1, chce
            # 3. use the pre-computed weights to calculate the sigmoid weighted average of recent outcomes
            Ws[i + 1, choice_idx] = sum([outc_wts[outc_key[outc_mat[choice_idx, j]], outc_lag[choice_idx, j]] for j in 1:outc_no])
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            # reset buffers at the start of a new block
            outc_no = 0
            outc_mat = Matrix{Any}(nothing, ssz, nT)
            outc_lag = zeros(Int, ssz, nT)
            outc_num = zeros(Int, ssz)
        end
        # store W-values for output (n.b. these are the values for pair[i] *before* the update)
        Wv[i, :] = Ws[i, (pri-1):pri]
    end

    return (choice = choice, Qs = nothing, Ws = Wv, loglike = loglike)

end

@model function RLWM_all_outc_pmst_sgd(
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
    set_sizes = unique(data.set_size)
    @submodel a, E, F_rl, β, ρ, α, ε, φ_rl = rl_pars(priors, parameters)
    @submodel F_wm, φ_wm, W = wm_pars(priors, parameters, set_sizes)

    C ~ priors[:C] # capacity
    w = isa(W, Dict) ? Dict(s => a2α(W[s]) for s in set_sizes) : Dict(s => a2α(W) for s in set_sizes)

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    nT = sum(data.block .== mode(data.block))
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in data.outcomes, premultiply by ρ and wt
    unq_outc = unique(data.outcomes)
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

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
    outc_no = 0
    outc_mat = Matrix{Any}(nothing, ssz, nT) # matrix of recent outcomes for each stimulus
    outc_lag = zeros(Int, ssz, nT) # how many outcomes back was this outcome?
    outc_num = zeros(Int, ssz) # how many outcomes have we seen for this stimulus?

    # store log-loglikelihood
    loglike = 0.

    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        pri = 2 * data.pair[i]
        # RL and WM policies (softmax with directed and undirected noise)
        π_rl = (1 - ε) * (1 / (1 + exp(-β * (Qs[i, pri] - Qs[i, pri - 1])))) + ε * 0.5
        π_wm = (1 - ε) * (1 / (1 + exp(-β * (Ws[i, pri] - Ws[i, pri - 1])))) + ε * 0.5

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
        chce = data.outcomes[i, choice_1id]
		δ = chce * ρ - Qs[i, choice_idx]

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Qs[i + 1, :] = Qs[i, :] + φ_rl * (Q0[i, :] .- Qs[i, :]) # decay or just store previous Q
			Qs[i + 1, choice_idx] += α * δ
            Ws[i + 1, :] = Ws[i, :] # store previous W
            # 1. iterate lags for all prior outcomes and update outcome number for this stimulus
            outc_lag += outc_lag .!== 0
            outc_num[choice_idx] += 1
            outc_no = outc_num[choice_idx]
            # 2. initialise the lag for this outcome number and store the relevant outcome
            outc_lag[choice_idx, outc_no], outc_mat[choice_idx, outc_no] = 1, chce
            # 3. use the pre-computed weights to calculate the sigmoid weighted average of recent outcomes
            Ws[i + 1, choice_idx] = sum([outc_wts[outc_key[outc_mat[choice_idx, j]], outc_lag[choice_idx, j]] for j in 1:outc_no]) / min(outc_no, C)
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            # reset buffers at the start of a new block
            outc_no = 0
            outc_mat = Matrix{Any}(nothing, ssz, nT)
            outc_lag = zeros(Int, ssz, nT)
            outc_num = zeros(Int, ssz)
        end
        # store Q- and W-values for output (n.b. these are the values for pair[i] *before* the update)
        Q[i, :] = Qs[i, (pri-1):pri]
        Wv[i, :] = Ws[i, (pri-1):pri]
    end

    return (choice = choice, Qs = Q, Ws = Wv, loglike = loglike)

end

@model function WM_all_outc_pmst_sgd_recip(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 2.), lower = 0.),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)

    # initial values
    aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
    initial = isnothing(initial) ? aao : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    C ~ priors[:C] # capacity

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    nT = sum(data.block .== mode(data.block))
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in data.outcomes, premultiply by ρ and wt
    unq_outc = unique(data.outcomes)
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Wv = copy(Ws[:, 1:2])

    # Initial set-size
    N = length(data.block)
    ssz = data.set_size[data.block[1]]
    val = data.valence[data.block[1]]

    # Initialize outcome buffers
    outc_no = 0
    outc_mat = Matrix{Any}(nothing, ssz, nT) # matrix of recent outcomes for each stimulus
    outc_lag = zeros(Int, ssz, nT) # how many outcomes back was this outcome?
    outc_num = zeros(Int, ssz) # how many outcomes have we seen for this stimulus?

    # store log-loglikelihood
    loglike = 0.

    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        pri = 2 * data.pair[i]
        
        # Choice
        choice[i] ~ BernoulliLogit(Ws[i, pri] - Ws[i, pri - 1])
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i] + 1
        alt_idx::Int64 = pri - choice[i]

        # log likelihood
        loglike += loglikelihood(BernoulliLogit(π), choice[i])

		# Prediction error
        chce = data.outcomes[i, choice_1id]
        # use pseudorandom number generator because optimizer doesn't like rand() (computes is hash(i) even or odd + 1)
        alt = abs(chce) >= 0.5 ? val * 0.01 : val * 1.0 # [0.5, 1.0][mod(hash(i), 2) + 1]

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Ws[i + 1, :] = Ws[i, :] # store previous W
            # 1. iterate lags for prior outcomes and update outcome number for this stimulus
            outc_lag += outc_lag .!== 0
            outc_num[pri-1:pri] += [1, 1]
            outc_no = outc_num[choice_idx]
            # 2. initialise the lag for this outcome number and store the relevant outcome
            outc_lag[choice_idx, outc_no], outc_mat[choice_idx, outc_no] = 1, chce
            outc_lag[alt_idx, outc_no], outc_mat[alt_idx, outc_no] = 1, alt
            # 3. use the pre-computed weights to calculate the sigmoid weighted average of recent outcomes
            Ws[i + 1, choice_idx] = sum([outc_wts[outc_key[outc_mat[choice_idx, j]], outc_lag[choice_idx, j]] for j in 1:outc_no]) / min(outc_no, C)
            Ws[i + 1, alt_idx] = sum([outc_wts[outc_key[outc_mat[alt_idx, j]], outc_lag[alt_idx, j]] for j in 1:outc_no]) / min(outc_no, C)
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            val = data.valence[data.block[i+1]]
            # reset buffers at the start of a new block
            outc_no = 0
            outc_mat = Matrix{Any}(nothing, ssz, nT)
            outc_lag = zeros(Int, ssz, nT)
            outc_num = zeros(Int, ssz)
        end
        # store W-values for output (n.b. these are the values for pair[i] *before* the update)
        Wv[i, :] = Ws[i, (pri-1):pri]
    end

    return (choice = choice, Qs = nothing, Ws = Wv, loglike = loglike)

end

@model function WM_all_outc_pmst_sgd_sum_recip(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 2.), lower = 0.),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)

    # initial values
    aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
    initial = isnothing(initial) ? aao : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    C ~ priors[:C] # capacity

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    nT = sum(data.block .== mode(data.block))
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in data.outcomes, premultiply by ρ and wt
    unq_outc = unique(data.outcomes)
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Wv = copy(Ws[:, 1:2])

    # Initial set-size
    N = length(data.block)
    ssz = data.set_size[data.block[1]]
    val = data.valence[data.block[1]]

    # Initialize outcome buffers
    outc_no = 0
    outc_mat = Matrix{Any}(nothing, ssz, nT) # matrix of recent outcomes for each stimulus
    outc_lag = zeros(Int, ssz, nT) # how many outcomes back was this outcome?
    outc_num = zeros(Int, ssz) # how many outcomes have we seen for this stimulus?

    # store log-loglikelihood
    loglike = 0.

    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        pri = 2 * data.pair[i]
        
        # Choice
        choice[i] ~ BernoulliLogit(Ws[i, pri] - Ws[i, pri - 1])
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i] + 1
        alt_idx::Int64 = pri - choice[i]

        # log likelihood
        loglike += loglikelihood(BernoulliLogit(π), choice[i])

		# Prediction error
        chce = data.outcomes[i, choice_1id]
        # use pseudorandom number generator because optimizer doesn't like rand() (computes is hash(i) even or odd + 1)
        alt = abs(chce) >= 0.5 ? val * 0.01 : val * 1.0 # [0.5, 1.0][mod(hash(i), 2) + 1]

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Ws[i + 1, :] = Ws[i, :] # store previous W
            # 1. iterate lags for prior outcomes and update outcome number for this stimulus
            outc_lag += outc_lag .!== 0
            outc_num[pri-1:pri] += [1, 1]
            outc_no = outc_num[choice_idx]
            # 2. initialise the lag for this outcome number and store the relevant outcome
            outc_lag[choice_idx, outc_no], outc_mat[choice_idx, outc_no] = 1, chce
            outc_lag[alt_idx, outc_no], outc_mat[alt_idx, outc_no] = 1, alt
            # 3. use the pre-computed weights to calculate the sigmoid weighted average of recent outcomes
            Ws[i + 1, choice_idx] = sum([outc_wts[outc_key[outc_mat[choice_idx, j]], outc_lag[choice_idx, j]] for j in 1:outc_no])
            Ws[i + 1, alt_idx] = sum([outc_wts[outc_key[outc_mat[alt_idx, j]], outc_lag[alt_idx, j]] for j in 1:outc_no])
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            val = data.valence[data.block[i+1]]
            # reset buffers at the start of a new block
            outc_no = 0
            outc_mat = Matrix{Any}(nothing, ssz, nT)
            outc_lag = zeros(Int, ssz, nT)
            outc_num = zeros(Int, ssz)
        end
        # store W-values for output (n.b. these are the values for pair[i] *before* the update)
        Wv[i, :] = Ws[i, (pri-1):pri]
    end

    return (choice = choice, Qs = nothing, Ws = Wv, loglike = loglike)

end

# ##------------------------------------------------------------------------------
# # Hierarchical RL models -------------------------------------------------------
# ##------------------------------------------------------------------------------
@model function RL(
	data::DataFrame;
    hyperpriors::Dict = Dict(
        :ρ_μ => Normal(0., 2.),
        :ρ_σ => Exponential(1.),
        :a_μ => Normal(0., 2.),
        :a_σ => Exponential(1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)
    a_μ ~ hyperpriors[:a_μ]
    a_σ ~ hyperpriors[:a_σ]
    ρ_μ ~ hyperpriors[:ρ_μ]
    ρ_σ ~ hyperpriors[:ρ_σ]

    ppts = unique(data.PID)
    choices = Array{Vector{Any}}(undef, length(ppts))
    Qs = Vector{Matrix{Any}}(undef, length(ppts))
    loglike = Vector{Any}(undef, length(ppts))

    d = [unpack_data(filter(x -> x.PID == ppts[p], data)) for p in eachindex(ppts)]
    c = [filter(x -> x.PID == ppts[p], data).choice for p in eachindex(ppts)]

    lk = ReentrantLock()

    Threads.@threads for p in eachindex(ppts)
        lock(lk) do
            @submodel prefix="pt$p" choices[p], Qs[p], loglike[p] = RL_ss(
                d[p],
                c[p];
                priors = Dict(
                    :ρ => truncated(Normal(ρ_μ, ρ_σ), lower = 0.),
                    :a => Normal(a_μ, a_σ)
                ),
                initial = initial
            )
        end
    end
    return (choice = choices, Qs = Qs, loglike = loglike)
end

# @model function RLWM(;
#     block::Vector{Vector{Int64}}, # Block number per trial
# 	valence::Vector{Vector{Float64}}, # Valence of each block. Vector of lenth maximum(block) per participant
# 	choice, # Binary choice, coded true for stimulus A. Not typed so that it can be simulated
# 	outcomes::Vector{Matrix{Float64}}, # Outcomes for options, second column optimal
# 	initV::Matrix{Float64}, # Initial Q values
#     set_size::Vector{Vector{Int64}}, # Set size for each block
#     parameters::Vector{Symbol} = [:ρ, :a, :F_wm, :W, :C], # Group-level parameters to estimate
#     sigmas::Dict{Symbol, Float64} = Dict(:ρ => 1., :a => 0.5, :F_wm => 0.5, :W => 0.5, :C => 1.)
# )

#     p = length(block) # number of participants

#     ## Set priors and transform bounded parameters
#     # reward sensitivity or inverse temp?
#     if :ρ in parameters
#         ρ ~ filldist(truncated(Normal(0., sigmas[:ρ]), lower = 0.), p)
#         β = ones(p)
#     elseif :β in parameters
#         β ~ filldist(truncated(Normal(0., sigmas[:β]), lower = 0.), p)
#         ρ = ones(p)
#     end
#     # learning rate
#     a ~ MvNormal(fill(0., p), I(p) * sigmas[:a])
#     α = a2α.(a)
#     # undirected noise or lapse rate
#     if :E in parameters
#         E ~ MvNormal(fill(0., p), I(p) * sigmas[:E])
#         ε = a2α.(E)
#     else
#         ε = zeros(p)
#     end
#     # forgetting rate
#     if :F in parameters
#         F ~ MvNormal(fill(0., p), I(p) * sigmas[:F])
#         φ_rl = a2α.(F)
#     else
#         φ_rl = zeros(p)
#     end

#     # Priors on WM parameters
#     W ~ MvNormal(fill(0., p), I(p) * sigmas[:W])
#     F_wm ~ MvNormal(fill(0., p), I(p) * sigmas[:F_wm])
#     C ~ filldist(truncated(Normal(0., sigmas[:C]), lower = 0.), p)

#     # Transform bounded parameters
#     w0, φ_wm = a2α.(W), a2α.(F_wm) # initial weight of WM vs RL, forgetting rate for WM

#     # Weight of WM vs RL
#     w = [Dict(sz => w0[sz] * min(1, C[sz] / sz) for sz in unique(set_size[p])) for p in 1:p]

#     # Initialize Q and W values
#     Qs = [initV .* (ρ[s] .* valence[s][block[s]]) for s in eachindex(valence)]
#     Ws = [initV .* (ρ[s] .* valence[s][block[s]]) for s in eachindex(valence)]

#     # Initial values
#     if @isdefined(φ_rl)
#         Q0 = copy(Qs)
#     end
#     W0 = 0.5 # initial weight of WM vs RL (1 / n_actions)

#     for s in eachindex(block)
#         ssz = set_size[s][block[s][1]]
#         for i in eachindex(block[s])

#             # RL and WM policies (softmax with directed and undirected noise)
#             π_rl = (1 - ε[s]) * (1 / (1 + exp(-β[s] * (Qs[s][i, 2] - Qs[s][i, 1])))) + ε[s] * 0.5
#             π_wm = (1 - ε[s]) * (1 / (1 + exp(-β[s] * (Ws[s][i, 2] - Ws[s][i, 1])))) + ε[s] * 0.5

#             # Weighted policy
#             π = w[s][ssz] * π_wm + (1 - w[s][ssz]) * π_rl

#             # Choice
#             choice[s][i] ~ Bernoulli(π)
#             choice_idx::Int64 = choice[s][i] + 1

#             # Prediction error
#             δ = outcomes[s][i, choice_idx] * ρ[s] - Qs[s][i, choice_idx]

#             dcy_rl = φ_rl[s] * (Q0[s][i, :] .- Qs[s][i, :])
#             dcy_wm = φ_wm[s] * (W0 .- Ws[s][i, :])

#             # Update Qs and Ws and decay Ws
#             if (i != length(block[s])) && (block[s][i] == block[s][i+1])
#                 Qs[s][i + 1, choice_idx] = Qs[s][i, choice_idx] + α[s] * δ + dcy_rl[choice_idx]
#                 Qs[s][i + 1, 3 - choice_idx] = Qs[s][i, 3 - choice_idx] + dcy_rl[3 - choice_idx]
#                 Ws[s][i + 1, choice_idx] = outcomes[s][i, choice_idx] * ρ[s] + dcy_wm[choice_idx]
#                 Ws[s][i + 1, 3 - choice_idx] = dcy_wm[3 - choice_idx]
#             elseif (i != length(block[s]))
#                 ssz = set_size[s][block[s][i+1]]
#             end
#         end
#     end

#     return (choice = choice, Qs = Qs, Ws = Ws)

# end


### HELPER FUNCTIONS -----------------------------------------------------------

# Transform unconstrainted a to learning rate α
a2α(a) = logistic(π/sqrt(3) * a)
α2a(α) = logit(α) / (π/sqrt(3))

@assert α2a(a2α(0.5)) ≈ 0.5

# Get data into correct format for the model -----------------------------------
function unpack_data(data::DataFrame)
    # sort by block and trial
    DataFrames.sort!(data, [:block, :trial])
    data_tuple = (
        block = data.block, # length = number of trials
        valence = unique(data[!, [:block, :valence]]).valence, # length = number of blocks
        pair = data.pair, # length = number of trials
        outcomes = hcat(data.feedback_suboptimal, data.feedback_optimal), # length = number of trials
        set_size = unique(data[!, [:block, :set_size]]).set_size, # length = number of blocks
    )
    return data_tuple
end

@model function rl_pars(
    priors::Dict,
    parameters::Vector{Symbol}
)
    # Set priors and transform bounded parameters
    # reward sensitivity or inverse temp?
    if :ρ in parameters
        ρ ~ priors[:ρ]
        β = 1.
    elseif :β in parameters
        β ~ priors[:β]
        ρ = 1.
    end
    
    # learning rate
    if :a in parameters
        a ~ priors[:a]
        α = a2α(a)
    elseif :α in parameters
        a = nothing
        α ~ priors[:α]
    else
        a, α = nothing, 0
    end

    # undirected noise or lapse rate
    if :E in parameters
        E ~ priors[:E]
        ε = a2α(E)
    elseif :ε in parameters
        E = 0
        ε ~ priors[:ε]
    else
        E, ε = nothing, 0
    end

    # forgetting rate
    if :F in parameters
        F ~ priors[:F]
        φ = a2α(F)
    elseif :φ in parameters
        φ ~ priors[:φ]
    else
        F, φ = nothing, 0
    end

    return (a, E, F, β, ρ, α, ε, φ)
end

@model function wm_pars(
    priors::Dict,
    parameters::Vector{Symbol},
    set_sizes::Vector{Int}
)
    # forgetting rate
    if :F_wm in parameters
        F_wm ~ priors[:F_wm]
        φ_wm = a2α(F_wm)
    elseif :φ_wm in parameters
        F_wm = nothing
        φ_wm ~ priors[:φ_wm]
    else
        F_wm, φ_wm = nothing, 0
    end

    # initial weight of WM vs RL
    w_pr = :W in parameters ? priors[:W] : Dict{Int, Any}()
    if isa(w_pr, Dict)
        W = Dict{Int, Any}()
        for s in set_sizes
            W[s] ~ priors[Symbol("W[$s]")]
        end
    else
        W ~ w_pr
    end

    return (F_wm, φ_wm, W)
end