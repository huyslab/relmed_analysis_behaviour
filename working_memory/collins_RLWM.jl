# Attempt to (almost) exactly replicate the RLWM/HLWM models as described in
# Collins et al. (2024): https://osf.io/preprints/psyarxiv/he3pm

@model function RLWM_collins24(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :β => 25., # fixed inverse temperature
        :a => Normal(0., 2.),
        :bRL => Normal(0., 1.),
        :bWM => Normal(0., 1.),
        :E => Normal(0., 0.5),
        :F_wm => Normal(1., 1.),
        :w0 => Normal(0., 2.),
        :C => Uniform(2., 5.)
    ),
    HLWM::Bool = false,
    initial::Union{Nothing, Float64} = nothing
)
    # initial values
    n_actions = size(data.outcomes, 2)
    init = isnothing(initial) ? 1. / n_actions : initial
    mss::Int64 = maximum(data.set_size) * n_actions
    initV::Matrix{Float64} = fill(init, 1, mss)

    # optimal outcomes in reward/punishment blocks
    unq_outc = unique(data.outcomes)
    opt_outc = vcat(
        unq_outc[unq_outc .>= 0.5], # reward blocks
        unq_outc[(unq_outc .> -0.5) .& (unq_outc .< 0)] # punishment blocks
    )
    
    # Parameters to estimate
    parameters = collect(keys(priors))
    set_sizes = unique(data.set_size)
    @submodel a_pos, a_neg, bRL, bWM, E, F_rl, F_wm, w0, β, α_pos, α_neg, α_neg_wm, ε, φ_rl, φ_wm = rlwm_pars(priors, parameters)

    # Fix working memory weights for different set sizes
    C ~ priors[:C]
    w = Dict(s => a2α(w0) * min(1., C / s) for s in set_sizes)

    # Get type parameter from priors to match ForwardDiff
    T = typeof(α_pos)

    # Initialize Q and W values
    Qs = repeat(initV, length(data.block)) .* T.(data.valence[data.block])
    Ws = repeat(initV, length(data.block)) .* T.(data.valence[data.block])
    
    # Initial values
    Q = copy(Qs[:, 1:n_actions])
    W0, Wv = copy(Ws), copy(Ws[:, 1:n_actions])
    
    # Initial set-size
    ssz = data.set_size[data.block[1]]
    N = length(data.block)
    loglike = zero(T)

    # Main loop
    for i in 1:N
        # index of first outcome for this stimulus group, because we have a flat Q-table
        pri::Int = 1 + (n_actions * (data.stimset[i]-1))

        # setup RL and WM policies (softmax with directed and undirected noise)
        # Collins (2024) assumes directed noise β is fixed at 25.
        π_rl = (1 - ε) * Turing.softmax(β .* Qs[i, pri:(pri+n_actions-1)]) .+ ε / n_actions
        π_wm = (1 - ε) * Turing.softmax(β .* Ws[i, pri:(pri+n_actions-1)]) .+ ε / n_actions

        # Weighted policy
        π = @. w[ssz] * π_wm + (1 - w[ssz]) * π_rl
        
        # Choice and likelihood
        choice[i] ~ Turing.Categorical(π)
        loglike += loglikelihood(Turing.Categorical(π), choice[i])
        
        # Updates
        choice_idx::Int = choice[i] + pri - 1
        choice_1id::Int = choice[i]
        outc_wm = data.outcomes[i, choice_1id]
        outc_rl = HLWM ? 1.0 : outc_wm
        α_rl, α_wm = in(outc_rl, opt_outc) ? (α_pos, 1.0) : (α_neg, α_neg_wm)
        
        if (i != N) && (data.block[i] == data.block[i+1])
            # In-place updates
            @views begin
                copyto!(Qs[i + 1, :], Qs[i, :])
                Qs[i + 1, choice_idx] += α_rl * (outc_rl - Qs[i, choice_idx])
                
                copyto!(Ws[i + 1, :], Ws[i, :])
                un_idx = setdiff(1:mss, choice_idx)
                Ws[i + 1, un_idx] .+= φ_wm .* (W0[i, un_idx] .- Ws[i, un_idx])
                Ws[i + 1, choice_idx] += α_wm * (outc_wm - Ws[i, choice_idx])
            end
        elseif (i != N)
            ssz = data.set_size[data.block[i+1]]
        end
        
        # Store values
        @views begin
            Q[i, :] .= Qs[i, pri:(pri+n_actions-1)]
            Wv[i, :] .= Ws[i, pri:(pri+n_actions-1)]
        end
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
        stimset = data.stimset, # length = number of trials
        outcomes = hcat(feedback_suboptimal, data.feedback_optimal), # length = number of trials
        set_size = unique(data[!, [:block, :set_size]]).set_size # length = number of blocks
    )
    return data_tuple
end

@model function rlwm_pars(
    priors::Dict,
    parameters::Vector{Symbol}
)
    # reward sensitivity or inverse temp?
    if isa(priors[:β], Number)
        β = priors[:β]
    elseif :β in parameters && isa(priors[:β], Distributions.Distribution)
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
        α_neg_wm = a2α(bWM)
    else
        bWM, α_neg_wm = nothing, 1.
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

    return (
        a_pos, a_neg, bRL, bWM, E, F_rl, F_wm, w0, # unconstrained parameters
        β, α_pos, α_neg, α_neg_wm, ε, φ_rl, φ_wm # constrained parameters
    )
end