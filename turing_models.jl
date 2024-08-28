##------------------------------------------------------------------------------
# RL models --------------------------------------------------------------------
## -----------------------------------------------------------------------------
@model function RL(;
	N::Int64, # Total number of trials
	block::Vector{Int64}, # Block number
	valence::AbstractVector, # Valence of each block
	choice, # Binary choice, coded true for stimulus A. Not typed so that it can be simulated
	outcomes::Matrix{Float64}, # Outcomes for options, second column optimal
	initV::Matrix{Float64}, # Initial Q values,
    parameters::Vector{Symbol} = [:ρ, :a], # Group-level parameters to estimate
    sigmas::Dict{Symbol, Float64} = Dict(:ρ => 1., :a => 0.5)
)
    ## Set priors and transform bounded parameters
    # reward sensitivity or inverse temp?
    if :ρ in parameters
        ρ ~ truncated(Normal(0., sigmas[:ρ]), lower = 0.)
        β = 1
    elseif :β in parameters
        β ~ truncated(Normal(0., sigmas[:β]), lower = 0.)
        ρ = 1
    end
    # learning rate
	a ~ Normal(0., sigmas[:a])
    α = a2α(a)
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

	# Loop over trials, updating Q values and incrementing log-density
	for i in 1:N

        # Policy (softmax) - β is 1 if we're using reward sensitivity and ε=0 if we're not using lapse rate
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

@model function RLWM(;
    N::Int64, # Total number of trials
    block::Vector{Int64}, # Block number
    valence::AbstractVector, # Valence of each block
    choice, # Binary choice, coded true for stimulus A. Not typed so that it can be simulated
    outcomes::Matrix{Float64}, # Outcomes for options, second column optimal
    initV::Matrix{Float64}, # Initial Q values,
    set_size::Vector{Int64}, # Set size for each block
    parameters::Vector{Symbol} = [:ρ, :a, :F_wm, :W, :C], # Group-level parameters to estimate
    sigmas::Dict{Symbol, Float64} = Dict(:ρ => 1., :a => 0.5, :F_wm => 0.5, :w => 0.5, :c => 1.)
)
    ## Priors on RL and WM parameters
	a ~ Normal(0., sigmas[:a])
    α = a2α(a)
    # reward sensitivity or inverse temp?
    if :ρ in parameters
        ρ ~ truncated(Normal(0., sigmas[:ρ]), lower = 0.)
        β = 1
    elseif :β in parameters
        β ~ truncated(Normal(0., sigmas[:β]), lower = 0.)
        ρ = 1
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