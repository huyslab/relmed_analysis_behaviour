##------------------------------------------------------------------------------
# RL models --------------------------------------------------------------------
## -----------------------------------------------------------------------------

@model function RL_multi_2set(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 1.), lower = 0.),
        :a1 => Normal(0., 0.5),
        :a2 => Normal(0., 0.5)
    ),
    initial::Union{Nothing, Float64} = nothing
)
    # initial values
    n_options = size(data.outcomes, 2)
    aao = sum([
        mean([0.01, mean([0.5, 1.])]) * n_options-1,
        mean([1., mean([0.5, 0.01])])
    ])
    initial = isnothing(initial) ? aao / n_options : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    a1 ~ priors[:a1] # learning rate
    a2 ~ priors[:a2] # learning rate with higher set-size

    # Transformed parameters
    α1, α2 = a2α(a1), a2α(a2) 

    # Initialize Q values
	Qs = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]
    Q = copy(Qs[:, 1:n_options])

    N = length(data.block)
    bss, ssz = data.set_size[data.block[1]], data.set_size[data.block[1]]
    loglike = 0.

	# Loop over trials, updating Q values and incrementing log-density
	for i in eachindex(data.block)
        pri = 1 + (n_options * (data.stimset[i]-1)) # index of first outcome for this stimulus group

        # more than 2 options so need to use proper softmax, but the last option is always the optimal one
        π = Turing.softmax(Qs[i, pri:(pri+n_options-1)]) # softmax of W-values
        
        # Choice
        choice[i] ~ Turing.Categorical(π) # 1, 2, 3...
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i]
        
        # log likelihood
        loglike += loglikelihood(Turing.Categorical(π), choice[i])

		# Prediction error
		δ = data.outcomes[i, choice_1id] * ρ - Qs[i, choice_idx]

		# Update Q value
		if (i != N) && (data.block[i] == data.block[i+1])
            Qs[i + 1, :] = Qs[i, :] # store previous Q
			Qs[i + 1, choice_idx] += (data.set_size[data.block[i]] == bss ? α1 : α2) * δ
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
		end
        # store Q values for output (n.b. these are the values for stimset[i] *before* the update)
        Q[i, :] = Qs[i, pri:(pri+n_options-1)]
	end

	return (choice = choice, Qs = Q, loglike = loglike)

end

@model function RL_multi_2set_recip(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 1.), lower = 0.),
        :a1 => Normal(0., 0.5),
        :a2 => Normal(0., 0.5)
    ),
    initial::Union{Nothing, Float64} = nothing
)
    # initial values
    n_options = size(data.outcomes, 2)
    aao = sum([
        mean([0.01, mean([0.5, 1.])]) * n_options-1,
        mean([1., mean([0.5, 0.01])])
    ])
    initial = isnothing(initial) ? aao / n_options : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    a1 ~ priors[:a1] # learning rate
    a2 ~ priors[:a2] # learning rate with higher set-size

    # Transformed parameters
    α1, α2 = a2α(a1), a2α(a2) 

    # Initialize Q values
	Qs = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]
    Q = copy(Qs[:, 1:n_options])

    N = length(data.block)
    bss, ssz = data.set_size[data.block[1]], data.set_size[data.block[1]]
    loglike = 0.

	# Loop over trials, updating Q values and incrementing log-density
	for i in eachindex(data.block)
        pri = 1 + (n_options * (data.stimset[i]-1)) # index of first outcome for this stimulus group

        # more than 2 options so need to use proper softmax, but the last option is always the optimal one
        π = Turing.softmax(Qs[i, pri:(pri+n_options-1)]) # softmax of W-values
        
        # Choice
        choice[i] ~ Turing.Categorical(π) # 1, 2, 3...
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i]
        alt_idx::Vector{Int64} = setdiff(collect(pri:(pri+n_options-1)), [choice_idx])
        
        # log likelihood
        loglike += loglikelihood(Turing.Categorical(π), choice[i])

		# Prediction error
		δ = data.outcomes[i, choice_1id] * ρ - Qs[i, choice_idx]

		# Update Q value
		if (i != N) && (data.block[i] == data.block[i+1])
            Qs[i + 1, :] = Qs[i, :] # store previous Q
			Qs[i + 1, choice_idx] += (data.set_size[data.block[i]] == bss ? α1 : α2) * δ
            Qs[i + 1, alt_idx] .-= (data.set_size[data.block[i]] == bss ? α1 : α2) * δ / length(alt_idx)
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
		end
        # store Q values for output (n.b. these are the values for stimset[i] *before* the update)
        Q[i, :] = Qs[i, pri:(pri+n_options-1)]
	end

	return (choice = choice, Qs = Q, loglike = loglike)

end

@model function RL_multi_2set_diff(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 1.), lower = 0.),
        :a => Normal(0., 0.5),
        :Δa => Normal(0, 0.5) # difference in learning rate with higher set-size
    ),
    initial::Union{Nothing, Float64} = nothing
)
    # initial values
    n_options = size(data.outcomes, 2)
    aao = sum([
        mean([0.01, mean([0.5, 1.])]) * n_options-1,
        mean([1., mean([0.5, 0.01])])
    ])
    initial = isnothing(initial) ? aao / n_options : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    a ~ priors[:a] # learning rate
    Δa ~ priors[:Δa] # difference in learning rate with higher set-size

    # Transformed parameters
    α1 = a2α(a)
    α2 = a2α(a + Δa)

    # Initialize Q values
	Qs = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]
    Q = copy(Qs[:, 1:n_options])

    N = length(data.block)
    bss, ssz = data.set_size[data.block[1]], data.set_size[data.block[1]]
    loglike = 0.

	# Loop over trials, updating Q values and incrementing log-density
	for i in eachindex(data.block)
        pri = 1 + (n_options * (data.stimset[i]-1)) # index of first outcome for this stimulus group

        # more than 2 options so need to use proper softmax, but the last option is always the optimal one
        π = Turing.softmax(Qs[i, pri:(pri+n_options-1)]) # softmax of W-values
        
        # Choice
        choice[i] ~ Turing.Categorical(π) # 1, 2, 3...
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i]
        
        # log likelihood
        loglike += loglikelihood(Turing.Categorical(π), choice[i])

		# Prediction error
		δ = data.outcomes[i, choice_1id] * ρ - Qs[i, choice_idx]

		# Update Q value
		if (i != N) && (data.block[i] == data.block[i+1])
            Qs[i + 1, :] = Qs[i, :] # store previous Q
			Qs[i + 1, choice_idx] += (data.set_size[data.block[i]] == bss ? α1 : α2) * δ
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
		end
        # store Q values for output (n.b. these are the values for stimset[i] *before* the update)
        Q[i, :] = Qs[i, pri:(pri+n_options-1)]
	end

	return (choice = choice, Qs = Q, loglike = loglike)

end

@model function RL_multi_2set_diff_recip(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 1.), lower = 0.),
        :a => Normal(0., 0.5),
        :Δa => Normal(0., 0.5) # difference in learning rate with higher set-size
    ),
    initial::Union{Nothing, Float64} = nothing
)
    # initial values
    n_options = size(data.outcomes, 2)
    aao = sum([
        mean([0.01, mean([0.5, 1.])]) * n_options-1,
        mean([1., mean([0.5, 0.01])])
    ])
    initial = isnothing(initial) ? aao / n_options : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    a ~ priors[:a] # learning rate
    Δa ~ priors[:Δa] # difference in learning rate with higher set-size

    # Transformed parameters
    α1 = a2α(a)
    α2 = a2α(a + Δa)

    # Initialize Q values
	Qs = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]
    Q = copy(Qs[:, 1:n_options])

    N = length(data.block)
    bss, ssz = data.set_size[data.block[1]], data.set_size[data.block[1]]
    loglike = 0.

	# Loop over trials, updating Q values and incrementing log-density
	for i in eachindex(data.block)
        pri = 1 + (n_options * (data.stimset[i]-1)) # index of first outcome for this stimulus group

        # more than 2 options so need to use proper softmax, but the last option is always the optimal one
        π = Turing.softmax(Qs[i, pri:(pri+n_options-1)]) # softmax of W-values
        
        # Choice
        choice[i] ~ Turing.Categorical(π) # 1, 2, 3...
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i]
        alt_idx::Vector{Int64} = setdiff(collect(pri:(pri+n_options-1)), [choice_idx])
        
        # log likelihood
        loglike += loglikelihood(Turing.Categorical(π), choice[i])

		# Prediction error
		δ = data.outcomes[i, choice_1id] * ρ - Qs[i, choice_idx]

		# Update Q value
		if (i != N) && (data.block[i] == data.block[i+1])
            Qs[i + 1, :] = Qs[i, :] # store previous Q
			Qs[i + 1, choice_idx] += (data.set_size[data.block[i]] == bss ? α1 : α2) * δ
            Qs[i + 1, alt_idx] .-= (data.set_size[data.block[i]] == bss ? α1 : α2) * δ / length(alt_idx)
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
		end
        # store Q values for output (n.b. these are the values for stimset[i] *before* the update) 
        Q[i, :] = Qs[i, pri:(pri+n_options-1)]
	end

	return (choice = choice, Qs = Q, loglike = loglike)

end

## ##------------------------------------------------------------------------------
# WM models --------------------------------------------------------------------
## -----------------------------------------------------------------------------

@model function WM_multi_pmst_sgd(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 2.), lower = 0.),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)

    # initial values
    n_options = size(data.outcomes, 2)
    aao = sum([
        mean([0.01, mean([0.5, 1.])]) * n_options-1,
        mean([1., mean([0.5, 0.01])])
    ])
    initial = isnothing(initial) ? aao / n_options : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    C ~ priors[:C] # capacity

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    nT = sum(data.block .== mode(data.block))
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in outcomes, premultiply by ρ and wt
    unq_outc = unique(data.outcomes)
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Wv = copy(Ws[:, 1:n_options])

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
        pri = 1 + (n_options * (data.stimset[i]-1)) # index of first outcome for this stimulus group

        # more than 2 options so need to use proper softmax, but the last option is always the optimal one
        π = Turing.softmax(Ws[i, pri:(pri+n_options-1)]) # softmax of W-values

        # Choice
        choice[i] ~ Turing.Categorical(π) # 1, 2, 3...
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i]

        # log likelihood
        loglike += loglikelihood(Turing.Categorical(π), choice[i])

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
        # store Q- and W-values for output (n.b. these are the values for stimset[i] *before* the update)
        Wv[i, :] = Ws[i, pri:(pri+n_options-1)]
    end

    return (choice = choice, Qs = nothing, Ws = Wv, loglike = loglike)

end

@model function WM_multi_pmst_sgd_sum(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 2.), lower = 0.),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)

    # initial values
    n_options = size(data.outcomes, 2)
    aao = sum([
        mean([0.01, mean([0.5, 1.])]) * n_options-1,
        mean([1., mean([0.5, 0.01])])
    ])
    initial = isnothing(initial) ? aao / n_options : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    C ~ priors[:C] # capacity

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    nT = sum(data.block .== mode(data.block))
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in outcomes, premultiply by ρ and wt
    unq_outc = unique(data.outcomes)
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Wv = copy(Ws[:, 1:n_options])

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
        pri = 1 + (n_options * (data.stimset[i]-1)) # index of first outcome for this stimulus group

        # more than 2 options so need to use proper softmax, but the last option is always the optimal one
        π = Turing.softmax(Ws[i, pri:(pri+n_options-1)]) # softmax of W-values

        # Choice
        choice[i] ~ Turing.Categorical(π) # 1, 2, 3...
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i]

        # log likelihood
        loglike += loglikelihood(Turing.Categorical(π), choice[i])

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
        # store Q- and W-values for output (n.b. these are the values for stimset[i] *before* the update)
        Wv[i, :] = Ws[i, pri:(pri+n_options-1)]
    end

    return (choice = choice, Qs = nothing, Ws = Wv, loglike = loglike)

end

@model function WM_multi_pmst_sgd_recip(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 2.), lower = 0.),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)

    # initial values
    n_options = size(data.outcomes, 2)
    aao = sum([
        mean([0.01, mean([0.5, 1.])]) * n_options-1,
        mean([1., mean([0.5, 0.01])])
    ])
    initial = isnothing(initial) ? aao / n_options : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    C ~ priors[:C] # capacity

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    nT = sum(data.block .== mode(data.block))
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in outcomes, premultiply by ρ and wt
    unq_outc = [unique(data.outcomes)..., 0.505, -0.505]
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Wv = copy(Ws[:, 1:n_options])

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
        pri = 1 + (n_options * (data.stimset[i]-1)) # index of first outcome for this stimulus group

        # more than 2 options so need to use proper softmax, but the last option is always the optimal one
        π = Turing.softmax(Ws[i, pri:(pri+n_options-1)]) # softmax of W-values

        # Choice
        choice[i] ~ Turing.Categorical(π) # 1, 2, 3...
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i]
        alt_idx::Vector{Int64} = setdiff(collect(pri:(pri+n_options-1)), [choice_idx])

        # log likelihood
        loglike += loglikelihood(Turing.Categorical(π), choice[i])

		# Prediction error
        chce = data.outcomes[i, choice_1id]
        # use pseudorandom number generator because optimizer doesn't like rand() (computes is hash(i) even or odd + 1)
        alt = abs(chce) >= 0.5 ? val * 0.01 : val * 0.505 # average of 1.0 and 0.01

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Ws[i + 1, :] = Ws[i, :] # store previous W
            # 1. iterate lags for prior outcomes and update outcome number for this stimulus
            outc_lag[pri:(pri+n_options-1)] += outc_lag[pri:(pri+n_options-1)] .!== 0
            outc_num[pri:(pri+n_options-1)] += ones(n_options)
            outc_no = outc_num[choice_idx]
            # 2. initialise the lag for this outcome number and store the relevant outcome
            outc_lag[choice_idx, outc_no], outc_mat[choice_idx, outc_no] = 1, chce
            # 3. use the pre-computed weights to calculate the sigmoid weighted average of recent outcomes
            Ws[i + 1, choice_idx] = sum([outc_wts[outc_key[outc_mat[choice_idx, j]], outc_lag[choice_idx, j]] for j in 1:outc_no]) / min(outc_no, C)
            # repeat for both alternative options
            for aidx in alt_idx
                outc_lag[aidx, outc_no], outc_mat[aidx, outc_no] = 1, alt
                Ws[i + 1, aidx] = sum([outc_wts[outc_key[outc_mat[aidx, j]], outc_lag[aidx, j]] for j in 1:outc_no]) / min(outc_no, C)
            end
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            val = data.valence[data.block[i+1]]
            # reset buffers at the start of a new block
            outc_no = 0
            outc_mat = Matrix{Any}(nothing, ssz, nT)
            outc_lag = zeros(Int, ssz, nT)
            outc_num = zeros(Int, ssz)
        end
        # store Q- and W-values for output (n.b. these are the values for stimset[i] *before* the update)
        Wv[i, :] = Ws[i, pri:(pri+n_options-1)]
    end

    return (choice = choice, Qs = nothing, Ws = Wv, loglike = loglike)

end

@model function WM_multi_pmst_sgd_sum_recip(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 2.), lower = 0.),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)

    # initial values
    n_options = size(data.outcomes, 2)
    aao = sum([
        mean([0.01, mean([0.5, 1.])]) * n_options-1,
        mean([1., mean([0.5, 0.01])])
    ])
    initial = isnothing(initial) ? aao / n_options : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))

    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    C ~ priors[:C] # capacity

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    gd = groupby(DataFrame("block" => data.block, "stimset" => data.stimset), :block)
    nT = maximum(combine(gd, :stimset => (x -> maximum(values(countmap(x)))) => :max_count).max_count)
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in data.outcomes, premultiply by ρ and wt
    unq_outc = [unique(data.outcomes)..., 0.505, -0.505]
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Wv = copy(Ws[:, 1:n_options])

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
        pri = 1 + (n_options * (data.stimset[i]-1)) # index of first outcome for this stimulus group

        # more than 2 options so need to use proper softmax, but the last option is always the optimal one
        π = Turing.softmax(Ws[i, pri:(pri+n_options-1)]) # softmax of W-values

        # Choice
        choice[i] ~ Turing.Categorical(π) # 1, 2, 3...
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i]
        alt_idx::Vector{Int64} = setdiff(collect(pri:(pri+n_options-1)), [choice_idx])

        # log likelihood
        loglike += loglikelihood(Turing.Categorical(π), choice[i])

		# Prediction error
        chce = data.outcomes[i, choice_1id]
        alt = abs(chce) >= 0.5 ? val * 0.01 : val * 0.505 # average of 1.0 and 0.01

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Ws[i + 1, :] = Ws[i, :] # store previous W
            # 1. iterate lags for prior outcomes and update outcome number for this stimulus
            outc_lag[pri:(pri+n_options-1)] += outc_lag[pri:(pri+n_options-1)] .!== 0
            outc_num[pri:(pri+n_options-1)] += ones(n_options)
            outc_no = outc_num[choice_idx]
            # 2. initialise the lag for this outcome number and store the relevant outcome
            outc_lag[choice_idx, outc_no], outc_mat[choice_idx, outc_no] = 1, chce
            # 3. use the pre-computed weights to calculate the sigmoid weighted average of recent outcomes
            Ws[i + 1, choice_idx] = sum([outc_wts[outc_key[outc_mat[choice_idx, j]], outc_lag[choice_idx, j]] for j in 1:outc_no])
            # repeat for both alternative options
            for aidx in alt_idx
                outc_lag[aidx, outc_no], outc_mat[aidx, outc_no] = 1, alt
                Ws[i + 1, aidx] = sum([outc_wts[outc_key[outc_mat[aidx, j]], outc_lag[aidx, j]] for j in 1:outc_no])
            end
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            val = data.valence[data.block[i+1]]
            # reset buffers at the start of a new block
            outc_no = 0
            outc_mat = Matrix{Any}(nothing, ssz, nT)
            outc_lag = zeros(Int, ssz, nT)
            outc_num = zeros(Int, ssz)
        end
        # store Q- and W-values for output (n.b. these are the values for stimset[i] *before* the update)
        Wv[i, :] = Ws[i, pri:(pri+n_options-1)]
    end

    return (choice = choice, Qs = nothing, Ws = Wv, loglike = loglike)

end

@model function WM_multi_all_outc_pmst_sgd(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 2.), lower = 0.),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)

    # initial values
    n_options = size(data.outcomes, 2)
    aao = sum([
        mean([0.01, mean([0.5, 1.])]) * n_options-1,
        mean([1., mean([0.5, 0.01])])
    ])
    initial = isnothing(initial) ? aao / n_options : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    C ~ priors[:C] # capacity

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    nT = sum(data.block .== mode(data.block))
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in outcomes, premultiply by ρ and wt
    unq_outc = unique(data.outcomes)
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Wv = copy(Ws[:, 1:n_options])

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
        pri = 1 + (n_options * (data.stimset[i]-1)) # index of first outcome for this stimulus group

        # more than 2 options so need to use proper softmax, but the last option is always the optimal one
        π = Turing.softmax(Ws[i, pri:(pri+n_options-1)]) # softmax of W-values

        # Choice
        choice[i] ~ Turing.Categorical(π) # 1, 2, 3...
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i]

        # log likelihood
        loglike += loglikelihood(Turing.Categorical(π), choice[i])

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
        # store W-values for output (n.b. these are the values for stimset[i] *before* the update)
        Wv[i, :] = Ws[i, pri:(pri+n_options-1)]
    end

    return (choice = choice, Qs = nothing, Ws = Wv, loglike = loglike)

end

@model function WM_multi_all_outc_pmst_sgd_sum(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 2.), lower = 0.),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)

    # initial values
    n_options = size(data.outcomes, 2)
    aao = sum([
        mean([0.01, mean([0.5, 1.])]) * n_options-1,
        mean([1., mean([0.5, 0.01])])
    ])
    initial = isnothing(initial) ? aao / n_options : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    C ~ priors[:C] # capacity

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    nT = sum(data.block .== mode(data.block))
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in outcomes, premultiply by ρ and wt
    unq_outc = unique(data.outcomes)
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Wv = copy(Ws[:, 1:n_options])

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
        pri = 1 + (n_options * (data.stimset[i]-1)) # index of first outcome for this stimulus group

        # more than 2 options so need to use proper softmax, but the last option is always the optimal one
        π = Turing.softmax(Ws[i, pri:(pri+n_options-1)]) # softmax of W-values

        # Choice
        choice[i] ~ Turing.Categorical(π) # 1, 2, 3...
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i]

        # log likelihood
        loglike += loglikelihood(Turing.Categorical(π), choice[i])

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
        # store W-values for output (n.b. these are the values for stimset[i] *before* the update)
        Wv[i, :] = Ws[i, pri:(pri+n_options-1)]
    end

    return (choice = choice, Qs = nothing, Ws = Wv, loglike = loglike)

end

@model function WM_multi_all_outc_pmst_sgd_recip(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 2.), lower = 0.),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)
    # initial values
    n_options = size(data.outcomes, 2)
    aao = sum([
        mean([0.01, mean([0.5, 1.])]) * n_options-1,
        mean([1., mean([0.5, 0.01])])
    ])
    initial = isnothing(initial) ? aao / n_options : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))

    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    C ~ priors[:C] # capacity

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    nT = sum(data.block .== mode(data.block))
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in data.outcomes, premultiply by ρ and wt
    unq_outc = [unique(data.outcomes)..., 0.505, -0.505]
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Wv = copy(Ws[:, 1:n_options])

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
        pri = 1 + (n_options * (data.stimset[i]-1)) # index of first outcome for this stimulus group

        # more than 2 options so need to use proper softmax, but the last option is always the optimal one
        π = Turing.softmax(Ws[i, pri:(pri+n_options-1)]) # softmax of W-values

        # Choice
        choice[i] ~ Turing.Categorical(π) # 1, 2, 3...
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i]
        alt_idx::Vector{Int64} = setdiff(collect(pri:(pri+n_options-1)), [choice_idx])

        # log likelihood
        loglike += loglikelihood(Turing.Categorical(π), choice[i])

		# Prediction error
        chce = data.outcomes[i, choice_1id]
        # use pseudorandom number generator because optimizer doesn't like rand() (computes is hash(i) even or odd + 1)
        alt = abs(chce) >= 0.5 ? val * 0.01 : val * 0.505 # average of 1.0 and 0.01

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Ws[i + 1, :] = Ws[i, :] # store previous W
            # 1. iterate lags for prior outcomes and update outcome number for this stimulus
            outc_lag += outc_lag .!== 0
            outc_num[pri:(pri+n_options-1)] += ones(n_options)
            outc_no = outc_num[choice_idx]
            # 2. initialise the lag for this outcome number and store the relevant outcome
            outc_lag[choice_idx, outc_no], outc_mat[choice_idx, outc_no] = 1, chce
            # 3. use the pre-computed weights to calculate the sigmoid weighted average of recent outcomes
            Ws[i + 1, choice_idx] = sum([outc_wts[outc_key[outc_mat[choice_idx, j]], outc_lag[choice_idx, j]] for j in 1:outc_no]) / min(outc_no, C)
            # repeat for both alternative options
            for aidx in alt_idx
                outc_lag[aidx, outc_no], outc_mat[aidx, outc_no] = 1, alt
                Ws[i + 1, aidx] = sum([outc_wts[outc_key[outc_mat[aidx, j]], outc_lag[aidx, j]] for j in 1:outc_no]) / min(outc_no, C)
            end
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            val = data.valence[data.block[i+1]]
            # reset buffers at the start of a new block
            outc_no = 0
            outc_mat = Matrix{Any}(nothing, ssz, nT)
            outc_lag = zeros(Int, ssz, nT)
            outc_num = zeros(Int, ssz)
        end
        # store W-values for output (n.b. these are the values for stimset[i] *before* the update)
        Wv[i, :] = Ws[i, pri:(pri+n_options-1)]
    end

    return (choice = choice, Qs = nothing, Ws = Wv, loglike = loglike)

end

@model function WM_multi_all_outc_pmst_sgd_sum_recip(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 2.), lower = 0.),
        :C => truncated(Normal(3., 2.), lower = 1.)
    ),
    initial::Union{Nothing, Float64} = nothing
)

    # initial values
    n_options = size(data.outcomes, 2)
    aao = sum([
        mean([0.01, mean([0.5, 1.])]) * n_options-1,
        mean([1., mean([0.5, 0.01])])
    ])
    initial = isnothing(initial) ? aao / n_options : initial
    initV::AbstractArray{Float64} = fill(initial, 1, maximum(data.set_size))
    
    # Parameters to estimate
    ρ ~ priors[:ρ] # sensitivity
    C ~ priors[:C] # capacity

    # sigmoid transformation using C
    k = 3 # sharpness of the sigmoid
    nT = sum(data.block .== mode(data.block))
    wt = 1 ./ (1 .+ exp.((collect(1:nT) .- C) * k))

    # for each unique outcome in data.outcomes, premultiply by ρ and wt
    unq_outc = [unique(data.outcomes)..., 0.505, -0.505]
    outc_wts = unq_outc * ρ .* wt' # matrix of outcome weights
    outc_key = Dict(o => i for (i, o) in enumerate(unq_outc)) # map outcomes to indices

    # Initialize Q and W values
    Ws = repeat(initV .* ρ, length(data.block)) .* data.valence[data.block]

    # Initial values
    Wv = copy(Ws[:, 1:n_options])

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
        pri = 1 + (n_options * (data.stimset[i]-1)) # index of first outcome for this stimulus group

        # more than 2 options so need to use proper softmax, but the last option is always the optimal one
        π = Turing.softmax(Ws[i, pri:(pri+n_options-1)]) # softmax of W-values

        # Choice
        choice[i] ~ Turing.Categorical(π) # 1, 2, 3...
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i]
        alt_idx::Vector{Int64} = setdiff(collect(pri:(pri+n_options-1)), [choice_idx])

        # log likelihood
        loglike += loglikelihood(Turing.Categorical(π), choice[i])

		# Prediction error
        chce = data.outcomes[i, choice_1id]
        # use pseudorandom number generator because optimizer doesn't like rand() (computes is hash(i) even or odd + 1)
        alt = abs(chce) >= 0.5 ? val * 0.01 : val * 0.505 # average of 1.0 and 0.01

        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Ws[i + 1, :] = Ws[i, :] # store previous W
            # 1. iterate lags for prior outcomes and update outcome number for this stimulus
            outc_lag += outc_lag .!== 0
            outc_num[pri:(pri+n_options-1)] += ones(n_options)
            outc_no = outc_num[choice_idx]
            # 2. initialise the lag for this outcome number and store the relevant outcome
            outc_lag[choice_idx, outc_no], outc_mat[choice_idx, outc_no] = 1, chce
            # 3. use the pre-computed weights to calculate the sigmoid weighted average of recent outcomes
            Ws[i + 1, choice_idx] = sum([outc_wts[outc_key[outc_mat[choice_idx, j]], outc_lag[choice_idx, j]] for j in 1:outc_no])
            # repeat for both alternative options
            for aidx in alt_idx
                outc_lag[aidx, outc_no], outc_mat[aidx, outc_no] = 1, alt
                Ws[i + 1, aidx] = sum([outc_wts[outc_key[outc_mat[aidx, j]], outc_lag[aidx, j]] for j in 1:outc_no])
            end
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
            val = data.valence[data.block[i+1]]
            # reset buffers at the start of a new block
            outc_no = 0
            outc_mat = Matrix{Any}(nothing, ssz, nT)
            outc_lag = zeros(Int, ssz, nT)
            outc_num = zeros(Int, ssz)
        end
        # store W-values for output (n.b. these are the values for stimset[i] *before* the update)
        Wv[i, :] = Ws[i, pri:(pri+n_options-1)]
    end

    return (choice = choice, Qs = nothing, Ws = Wv, loglike = loglike)

end

### HELPER FUNCTIONS -----------------------------------------------------------

# Transform unconstrained a to learning rate α
a2α(a) = logistic(π/sqrt(3) * a)
α2a(α) = logit(α) / (π/sqrt(3))

@assert α2a(a2α(0.5)) ≈ 0.5

# Get data into correct format for the model -----------------------------------
function unpack_data(data::DataFrame)
    # sort by block and trial
    DataFrames.sort!(data, [:block, :trial])
    
    # deal with multiple suboptimal feedback
    n_subopt = count(i -> occursin("feedback_subopt", i), names(data))
    feedback_suboptimal = n_subopt == 1 ? 
        data.feedback_suboptimal :
        hcat([data[!, "feedback_suboptimal$i"] for i in 1:n_subopt]...)

    # renumber blocks if necessary to start at 1
    if minimum(data.block) != 1
        data.block .-= minimum(data.block) - 1
    end

    data_tuple = (
        block = data.block, # length = number of trials
        valence = unique(data[!, [:block, :valence]]).valence, # length = number of blocks
        stimset = data.stimset, # length = number of trials
        outcomes = hcat(feedback_suboptimal, data.feedback_optimal), # length = number of trials
        set_size = unique(data[!, [:block, :set_size]]).set_size .* (1 + n_subopt), # length = number of blocks
    )
    return data_tuple
end