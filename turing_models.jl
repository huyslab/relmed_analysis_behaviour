##------------------------------------------------------------------------------
# RL models --------------------------------------------------------------------
## -----------------------------------------------------------------------------
@model function RL_ss(;
	block::Vector{Int64}, # Block number
	valence::AbstractVector, # Valence of each block
	choice, # Binary choice, coded true for stimulus A. Not typed so that it can be simulated
	outcomes::Matrix{Float64}, # Outcomes for options, second column optimal
	initV::Matrix{Float64}, # Initial Q values,
    set_size::Nothing = nothing, # here for compatibility with RLWM
    parameters::Vector{Symbol} = [:ρ, :a], # Group-level parameters to estimate
    sigmas::Dict{Symbol, Float64} = Dict(:ρ => 1., :a => 0.5),
    fixed_params::Dict = Dict(:ρ => nothing, :α => nothing)
)
    ## Set priors and transform bounded parameters
    # reward sensitivity or inverse temp?
    if :ρ in keys(fixed_params)
        ρ = fixed_params[:ρ]
        β = 1
    elseif :ρ in parameters
        ρ ~ truncated(Normal(0., sigmas[:ρ]), lower = 0.)
        β = 1
    elseif :β in parameters
        β ~ truncated(Normal(0., sigmas[:β]), lower = 0.)
        ρ = 1
    end
    # learning rate
    if :α in keys(fixed_params)
        a ~ Normal(α2a(fixed_params[:α]), 0.)
        α = fixed_params[:α]
    else
        a ~ Normal(0., sigmas[:a])
        α = a2α(a)
    end 
    # undirected noise or lapse rate
    if :E in parameters
        E ~ Normal(0., sigmas[:E])
        ε = a2α(E)
    else
        ε = 0
    end
    # forgetting rate
    if :F in parameters
        F ~ Normal(0., sigmas[:F])
        φ = a2α(F)
    else
        φ = 0
    end
    # # perseveration
    # if :P in parameters
    #     P ~ Normal(0., sigmas[:P])
    #     pers = a2α(P)
    # end

	# Initialize Q values
	Qs = repeat(initV .* ρ, length(block)) .* valence[block]
    if @isdefined(φ)
        Q0 = copy(Qs)
    end

    # fix parameters if provided
    if !isnothing(fixed_params)
        for (k, v) in fixed_params
            if k == :ρ
                ρ = v
            elseif k == :a
                α = a2α(v)
            elseif k == :E
                ε = a2α(v)
            elseif k == :F
                φ = a2α(v)
            end
        end
    end

    N = length(block)

	# Loop over trials, updating Q values and incrementing log-density
	for i in eachindex(block)

        # Policy (softmax) - β=1 if we're using reward sensitivity and ε=0 if we're not using lapse rate
        # in Collins et al. terminology, these are directed and undirected noise
        π = (1 - ε) * (1 / (1 + exp(-β * (Qs[i, 2] - Qs[i, 1])))) + ε * 0.5

		# Choice
		choice[i] ~ Bernoulli(π)
		choice_idx::Int64 = choice[i] + 1

		# Prediction error
		δ = outcomes[i, choice_idx] * ρ - Qs[i, choice_idx]
        # if @isdefined(pers) && δ < 0
        #     α *= 1 - pers
        # end

        dcy = φ * (Q0[i, :] .- Qs[i, :])

		# Update Q value
		if (i != N) && (block[i] == block[i+1])
			Qs[i + 1, choice_idx] = Qs[i, choice_idx] + α * δ + dcy[choice_idx]
			Qs[i + 1, 3 - choice_idx] = Qs[i, 3 - choice_idx] + dcy[3 - choice_idx]
		end
	end

	return (choice = choice, Qs = Qs)

end

@model function RL(;
	block::Vector{Vector{Int64}}, # Block number per trial
	valence::Vector{Vector{Float64}}, # Valence of each block. Vector of lenth maximum(block) per participant
	choice, # Binary choice, coded true for stimulus A. Not typed so that it can be simulated
	outcomes::Vector{Matrix{Float64}}, # Outcomes for options, second column optimal
	initV::Matrix{Float64}, # Initial Q values
    set_size::Nothing = nothing, # here for compatibility with RLWM
    parameters::Vector{Symbol} = [:ρ, :a], # Group-level parameters to estimate
    sigmas::Dict{Symbol, Float64} = Dict(:ρ => 1., :a => 0.5)
)
    p = length(block) # number of participants

    ## Set priors and transform bounded parameters
    # reward sensitivity or inverse temp?
    if :ρ in parameters
        ρ ~ filldist(truncated(Normal(0., sigmas[:ρ]), lower = 0.), p)
        β = ones(p)
    elseif :β in parameters
        β ~ filldist(truncated(Normal(0., sigmas[:β]), lower = 0.), p)
        ρ = ones(p)
    end
    # learning rate
	a ~ MvNormal(fill(0., p), I(p) * sigmas[:a])
    α = a2α.(a)
    # undirected noise or lapse rate
    if :E in parameters
        E ~ MvNormal(fill(0., p), I(p) * sigmas[:E])
        ε = a2α.(E)
    else
        ε = zeros(p)
    end
    # forgetting rate
    if :F in parameters
        F ~ MvNormal(fill(0., p), I(p) * sigmas[:F])
        φ = a2α.(F)
    else
        φ = zeros(p)
    end

	# Initialize Q values
	Qs = [initV .* (ρ[s] .* valence[s][block[s]]) for s in eachindex(valence)]
    if @isdefined(φ)
        Q0 = copy(Qs)
    end

	# Loop over trials, updating Q values and incrementing log-density
    for s in eachindex(block)
        for i in eachindex(block[s])

            # Policy (softmax) - β=1 if we're using reward sensitivity and ε=0 if we're not using lapse rate
            # in Collins et al. terminology, these are directed and undirected noise
            π = (1 - ε[s]) * (1 / (1 + exp(-β[s] * (Qs[s][i, 2] - Qs[s][i, 1])))) + ε[s] * 0.5

            # Choice
            choice[s][i] ~ Bernoulli(π)
            choice_idx::Int64 = choice[s][i] + 1

            # Prediction error
            δ = outcomes[s][i, choice_idx] * ρ[s] - Qs[s][i, choice_idx]

            dcy = φ[s] * (Q0[s][i, :] .- Qs[s][i, :])

            # Update Q value
            if (i != length(block[s])) && (block[s][i] == block[s][i+1])
                Qs[s][i + 1, choice_idx] = Qs[s][i, choice_idx] + α[s] * δ + dcy[choice_idx]
                Qs[s][i + 1, 3 - choice_idx] = Qs[s][i, 3 - choice_idx] + dcy[3 - choice_idx]
            end
        end
    end

	return (choice = choice, Qs = Qs)

end

@model function RLWM_ss(;
    block::Vector{Int64}, # Block number
    valence::AbstractVector, # Valence of each block
    choice, # Binary choice, coded true for stimulus A. Not typed so that it can be simulated
    outcomes::Matrix{Float64}, # Outcomes for options, second column optimal
    initV::Matrix{Float64}, # Initial Q values,
    set_size::Vector{Int64}, # Set size for each block
    parameters::Vector{Symbol} = [:ρ, :a, :F_wm, :W, :C], # Group-level parameters to estimate
    sigmas::Dict{Symbol, Float64} = Dict(:ρ => 1., :a => 0.5, :F_wm => 0.5, :W => 0.5, :C => 1.),
    fixed_params::Dict = Dict(:ρ => nothing, :α => nothing)
)
    ## Priors on RL and WM parameters
	# reward sensitivity or inverse temp?
    if :ρ in keys(fixed_params)
        ρ = fixed_params[:ρ]
        β = 1
    elseif :ρ in parameters
        ρ ~ truncated(Normal(0., sigmas[:ρ]), lower = 0.)
        β = 1
    elseif :β in parameters
        β ~ truncated(Normal(0., sigmas[:β]), lower = 0.)
        ρ = 1
    end
    # learning rate
    if :α in keys(fixed_params)
        a ~ Normal(α2a(fixed_params[:α]), 0.)
        α = fixed_params[:α]
    else
        a ~ Normal(0., sigmas[:a])
        α = a2α(a)
    end
    # undirected noise or lapse rate
    if :E in parameters
        E ~ Normal(0., sigmas[:E])
        ε = a2α(E)
    else
        ε = 0
    end
    # forgetting rate for Q-values
    if :F_rl in parameters
        F_rl ~ Normal(0., sigmas[:F_rl])
        φ_rl = a2α(F_rl)
    else
        φ_rl = 0
    end
    # # perseveration
    # if :P in parameters
    #     P ~ Normal(0., sigmas[:P])
    #     pers = a2α(P)
    # end

    W ~ Normal(0., sigmas[:W]) # initial weight of WM vs RL
    F_wm ~ Normal(0., sigmas[:F_wm]) # forgetting rate
    C ~ truncated(Normal(0., sigmas[:C]), lower = 0.) # capacity

    # Transform bounded parameters
    w0 = a2α(W) # initial weight of WM vs RL
    φ_wm = a2α(F_wm) # forgetting rate for WM (required as we assumed perfect retention)

    # Weight of WM vs RL
    w = Dict(s => w0 * min(1, C / s) for s in unique(set_size))

    # Initialize Q and W values
    Qs = repeat(initV .* ρ, length(block)) .* valence[block]
    Ws = repeat(initV .* ρ, length(block)) .* valence[block]

    # Initial values
    if @isdefined(φ_rl)
        Q0 = copy(Qs)
    end
    W0 = 0.5 # initial weight of WM vs RL (1 / n_actions)


    # Initial set-size
    ssz = set_size[block[1]]
    N = length(block)

    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N

        # RL and WM policies (softmax with directed and undirected noise)
        π_rl = (1 - ε) * (1 / (1 + exp(-β * (Qs[i, 2] - Qs[i, 1])))) + ε * 0.5
        π_wm = (1 - ε) * (1 / (1 + exp(-β * (Ws[i, 2] - Ws[i, 1])))) + ε * 0.5

        # Weighted policy
        π = w[ssz] * π_wm + (1 - w[ssz]) * π_rl

		# Choice
		choice[i] ~ Bernoulli(π)
		choice_idx::Int64 = choice[i] + 1

		# Prediction error
		δ = outcomes[i, choice_idx] * ρ - Qs[i, choice_idx]
        # if @isdefined(pers) && δ < 0
        #     α *= 1 - pers
        # end

        dcy_rl = φ_rl * (Q0[i, :] .- Qs[i, :])
        dcy_wm = φ_wm * (W0 .- Ws[i, :])

        # Update Qs and Ws and decay Ws
        if (i != N) && (block[i] == block[i+1])
            Qs[i + 1, choice_idx] = Qs[i, choice_idx] + α * δ + dcy_rl[choice_idx]
			Qs[i + 1, 3 - choice_idx] = Qs[i, 3 - choice_idx] + dcy_rl[3 - choice_idx]
            Ws[i + 1, choice_idx] = outcomes[i, choice_idx] * ρ + dcy_wm[choice_idx]
            Ws[i + 1, 3 - choice_idx] = dcy_wm[3 - choice_idx]
        elseif (i != N)
            ssz = set_size[block[i+1]]
        end
    end

    return (choice = choice, Qs = Qs, Ws = Ws)

end

@model function RLWM(;
    block::Vector{Vector{Int64}}, # Block number per trial
	valence::Vector{Vector{Float64}}, # Valence of each block. Vector of lenth maximum(block) per participant
	choice, # Binary choice, coded true for stimulus A. Not typed so that it can be simulated
	outcomes::Vector{Matrix{Float64}}, # Outcomes for options, second column optimal
	initV::Matrix{Float64}, # Initial Q values
    set_size::Vector{Vector{Int64}}, # Set size for each block
    parameters::Vector{Symbol} = [:ρ, :a, :F_wm, :W, :C], # Group-level parameters to estimate
    sigmas::Dict{Symbol, Float64} = Dict(:ρ => 1., :a => 0.5, :F_wm => 0.5, :W => 0.5, :C => 1.)
)

    p = length(block) # number of participants

    ## Set priors and transform bounded parameters
    # reward sensitivity or inverse temp?
    if :ρ in parameters
        ρ ~ filldist(truncated(Normal(0., sigmas[:ρ]), lower = 0.), p)
        β = ones(p)
    elseif :β in parameters
        β ~ filldist(truncated(Normal(0., sigmas[:β]), lower = 0.), p)
        ρ = ones(p)
    end
    # learning rate
    a ~ MvNormal(fill(0., p), I(p) * sigmas[:a])
    α = a2α.(a)
    # undirected noise or lapse rate
    if :E in parameters
        E ~ MvNormal(fill(0., p), I(p) * sigmas[:E])
        ε = a2α.(E)
    else
        ε = zeros(p)
    end
    # forgetting rate
    if :F in parameters
        F ~ MvNormal(fill(0., p), I(p) * sigmas[:F])
        φ_rl = a2α.(F)
    else
        φ_rl = zeros(p)
    end

    # Priors on WM parameters
    W ~ MvNormal(fill(0., p), I(p) * sigmas[:W])
    F_wm ~ MvNormal(fill(0., p), I(p) * sigmas[:F_wm])
    C ~ filldist(truncated(Normal(0., sigmas[:C]), lower = 0.), p)

    # Transform bounded parameters
    w0, φ_wm = a2α.(W), a2α.(F_wm) # initial weight of WM vs RL, forgetting rate for WM

    # Weight of WM vs RL
    w = [Dict(sz => w0[sz] * min(1, C[sz] / sz) for sz in unique(set_size[p])) for p in 1:p]

    # Initialize Q and W values
    Qs = [initV .* (ρ[s] .* valence[s][block[s]]) for s in eachindex(valence)]
    Ws = [initV .* (ρ[s] .* valence[s][block[s]]) for s in eachindex(valence)]

    # Initial values
    if @isdefined(φ_rl)
        Q0 = copy(Qs)
    end
    W0 = 0.5 # initial weight of WM vs RL (1 / n_actions)

    for s in eachindex(block)
        ssz = set_size[s][block[s][1]]
        for i in eachindex(block[s])

            # RL and WM policies (softmax with directed and undirected noise)
            π_rl = (1 - ε[s]) * (1 / (1 + exp(-β[s] * (Qs[s][i, 2] - Qs[s][i, 1])))) + ε[s] * 0.5
            π_wm = (1 - ε[s]) * (1 / (1 + exp(-β[s] * (Ws[s][i, 2] - Ws[s][i, 1])))) + ε[s] * 0.5

            # Weighted policy
            π = w[s][ssz] * π_wm + (1 - w[s][ssz]) * π_rl

            # Choice
            choice[s][i] ~ Bernoulli(π)
            choice_idx::Int64 = choice[s][i] + 1

            # Prediction error
            δ = outcomes[s][i, choice_idx] * ρ[s] - Qs[s][i, choice_idx]

            dcy_rl = φ_rl[s] * (Q0[s][i, :] .- Qs[s][i, :])
            dcy_wm = φ_wm[s] * (W0 .- Ws[s][i, :])

            # Update Qs and Ws and decay Ws
            if (i != length(block[s])) && (block[s][i] == block[s][i+1])
                Qs[s][i + 1, choice_idx] = Qs[s][i, choice_idx] + α[s] * δ + dcy_rl[choice_idx]
                Qs[s][i + 1, 3 - choice_idx] = Qs[s][i, 3 - choice_idx] + dcy_rl[3 - choice_idx]
                Ws[s][i + 1, choice_idx] = outcomes[s][i, choice_idx] * ρ[s] + dcy_wm[choice_idx]
                Ws[s][i + 1, 3 - choice_idx] = dcy_wm[3 - choice_idx]
            elseif (i != length(block[s]))
                ssz = set_size[s][block[s][i+1]]
            end
        end
    end

    return (choice = choice, Qs = Qs, Ws = Ws)

end