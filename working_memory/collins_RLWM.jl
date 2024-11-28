# Attempt to (almost) exactly replicate the RLWM/HLWM models as described in
# Collins et al. (2024): https://osf.io/preprints/psyarxiv/he3pm

@model function RLWM_collins24(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :β => Dirac(25.), # fixed inverse temperature
        :a => Normal(0., 2.),
        :bRL => Normal(0., 1.),
        :bWM => Normal(0., 1.),
        :E => Normal(0., 0.5),
        :F_wm => Normal(1., 1.),
        :w0 => Normal(0., 2.),
        :rlw => Normal(0., 2.), # wm/rl interaction
        :C => Distributions.Categorical([fill(0.25, 4)]) # 1, 2, 3, or 4
    ),
    HLWM::Bool = false,
    initial::Union{Nothing, Float64} = nothing
)
    # initial values
    n_options = size(data.outcomes, 2)
    init = isnothing(initial) ? 0. : initial
    initV::AbstractArray{Float64} = fill(init, 1, maximum(data.set_size))

    # optimal outcomes in reward/punishment blocks
    unq_outc = unique(data.outcomes)
    opt_outc = vcat(
        unq_outc[unq_outc .>= 0.5], # reward blocks
        unq_outc[(unq_outc .> -0.5) .& (unq_outc .< 0)] # punishment blocks
    )
    
    # Parameters to estimate
    parameters = collect(keys(priors))
    set_sizes = unique(data.set_size)
    @submodel a_pos, a_neg, bRL, bWM, E, F_rl, F_wm, w0, rlw, β, α_pos, α_neg, α_neg_wm, ε, φ_rl, φ_wm, rlw_int = rlwm_pars(priors, parameters)

    # Fix working memory weights for different set sizes
    C ~ priors[:C] + 1 # 2, 3, 4, or 5
    w = Dict(s => a2α(w0) * min(1, C / s) for s in set_sizes)

    # Initialize Q and W values
    Qs::Matrix{Any} = repeat(initV, length(data.block)) .* data.valence[data.block]
    Ws::Matrix{Any} = repeat(initV, length(data.block)) .* data.valence[data.block]

    # Initial values
    Q0, Q = copy(Qs), copy(Qs[:, 1:n_options])
    W0, Wv = copy(Ws), copy(Ws[:, 1:n_options])

    # Initial set-size
    ssz = data.set_size[data.block[1]]
    N = length(data.block)
    loglike = 0.

    # Loop over trials, updating Q values and incrementing log-density
    for i in 1:N
        # index of first outcome for this stimulus group, because we have a flat Q-table
        pri = 1 + (n_options * (data.pair[i]-1))

        # setup RL and WM policies (softmax with directed and undirected noise)
        # Collins (2024) assumes directed noise β is fixed at 25.
        π_rl = (1 - ε) * Turing.softmax(β .* Qs[i, pri:(pri+n_options-1)]) .+ ε / n_options
        π_wm = (1 - ε) * Turing.softmax(β .* Ws[i, pri:(pri+n_options-1)]) .+ ε / n_options

        # Weighted policy
        π = w[ssz] * π_wm + (1 - w[ssz]) * π_rl

		# Choice
        choice[i] ~ Turing.Categorical(π) # 1, 2, 3... n_options
		choice_idx::Int64 = choice[i] + pri - 1
        choice_1id::Int64 = choice[i]

        # log likelihood
        loglike += loglikelihood(Turing.Categorical(π), choice[i])

		# prediction error
        outc_wm = data.outcomes[i, choice_1id]
        outc_rl = HLWM ? 1. : outc_wm
		δ = outc_rl - (rlw_int * Ws[i, choice_idx] + (1 - rlw_int) * Qs[i, choice_idx])
        α_rl, α_wm = if any(x -> x == outc_rl, opt_outc)
            α_pos, 1.
        else
            α_neg, α_neg_wm
        end
        
        # Update Qs and Ws and decay Ws
        if (i != N) && (data.block[i] == data.block[i+1])
            Qs[i + 1, :] = Qs[i, :] + φ_rl * (Q0[i, :] .- Qs[i, :]) # decay or just store previous Q
			Qs[i + 1, choice_idx] += α_rl * δ
            Ws[i + 1, :] = Ws[i, :] + φ_wm * (W0[i, :] .- Ws[i, :]) # decay or just store previous W
            Ws[i + 1, choice_idx] += α_wm * (outc_wm - Ws[i, choice_idx])
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
        end

        # store Q- and W-values for output (n.b. these are the values for pair[i] *before* the update)
        Q[i, :] = Qs[i, pri:(pri+n_options-1)]
        Wv[i, :] = Ws[i, pri:(pri+n_options-1)]
    end

    return (choice = choice, Qs = Q, Ws = Wv, loglike = loglike)

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
        pair = data.pair, # length = number of trials
        outcomes = hcat(feedback_suboptimal, data.feedback_optimal), # length = number of trials
        set_size = unique(data[!, [:block, :set_size]]).set_size # length = number of blocks
        # action = data.action # length = number of trials
    )
    return data_tuple
end

@model function rlwm_pars(
    priors::Dict,
    parameters::Vector{Symbol}
)
    # reward sensitivity or inverse temp?
    if :β in parameters
        β ~ priors[:β]
    else
        β = 25. # this is what AGC fixes it to
    end
    
    # RL learning rates
    if :a in parameters
        a_pos ~ priors[:a]
        if :bRL in parameters
            bRL ~ priors[:bRL]
            a_neg = a_pos * bRL
        else
            bRL, a_neg = nothing, a_pos
        end
        α_pos, α_neg = a2α(a_pos), a2α(a_neg)
    elseif :α_pos in parameters
        α_pos ~ priors[:α_pos]
        a_pos = α2a(α_pos)
        if :bRL in parameters
            bRL ~ priors[:bRL]
            a_neg = a_pos * bRL
        else
            bRL, a_neg = nothing, a_pos
        end
        α_neg = a2α(a_neg)
    elseif :α in parameters
        a_pos, a_neg = nothing, nothing
        α_pos ~ priors[:α]
        α_neg = α_pos
    end

    # WM learning rate bias
    if :bWM in parameters
        bWM ~ priors[:bWM]
    else
        bWM = nothing
    end
    α_neg_wm = isnothing(bWM) ? 1 : a2α(bWM)

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

    # forgetting rate for RL
    if :F_rl in parameters
        F_rl ~ priors[:F_rl]
        φ_rl = a2α(F_rl)
    elseif :φ in parameters
        φ_rl ~ priors[:φ_rl]
    else
        F_rl, φ_rl = nothing, 0
    end

    # forgetting rate for WM
    if :F_wm in parameters
        F_wm ~ priors[:F_wm]
        φ_wm = a2α(F_wm)
    elseif :φ_wm in parameters
        F_wm = nothing
        φ_wm ~ priors[:φ_wm]
    else
        F_wm, φ_wm = nothing, 0
    end

    # going to ignore the modified choice kernels for now
    # there is also policy compression implemented, but again complex
    # # stickiness
    # if :K in parameters
    #     K ~ priors[:K]
    #     κ = sign(K) # +1 suggests stay; -1 suggests switch
    # else
    #     K, κ = nothing, 1
    # end

    # initial weight of WM vs RL
    if :w0 in parameters
        w0 ~ priors[:w0]
    else
        w0 = 0.
    end

    if :rlw in parameters
        rlw ~ priors[:rlw]
        rlw_int = a2α(rlw)
    else
        rlw, rlw_int = nothing, 0
    end

    return (
        a_pos, a_neg, bRL, bWM, E, F_rl, F_wm, w0, rlw, # unconstrained parameters
        β, α_pos, α_neg, α_neg_wm, ε, φ_rl, φ_wm, rlw_int # constrained parameters
    )
end