@model function HLWM_collins_continuous(
    data::NamedTuple,
    choice;
    priors::Dict = Dict(
        :β => 25., # fixed inverse temperature
        :a => Normal(0., 2.),
        :bWM => Normal(0., 1.),
        :E => Normal(0., 0.5),
        :F_wm => Normal(-1., 1.), # working memory forgetting rate
        :w0 => Beta(1., 2.) # prop. of WM to RL weight (i.e., 0.5 ===)
    ),
    initial::Union{Nothing, Float64} = nothing
)
    # initial values
    n_actions = size(data.outcomes, 2)
    init = isnothing(initial) ? 1. / n_actions : initial
    mss::Int64 = maximum(data.stimset) * n_actions
    initV::Matrix{Float64} = fill(init, 1, mss)

    # optimal outcomes in reward/punishment blocks
    unq_outc = unique(data.outcomes)
    opt_outc = vcat(
        unq_outc[unq_outc .>= 0.5], # reward blocks
        unq_outc[(unq_outc .> -0.5) .& (unq_outc .< 0)] # punishment blocks
    )
    
    # Parameters to estimate
    parameters = collect(keys(priors))
    @submodel a_pos, a_neg, bRL, bWM, E, F_hl, F_wm, w0, β, α_pos, α_neg, α_neg_wm, ε, φ_hl, φ_wm = rlwm_pars(priors, parameters)

    # Get type parameter from priors to match ForwardDiff
    T = typeof(α_pos)

    # Initialize Q and W values
    Qs = T.(repeat(initV .* repeat(data.valence, inner = n_actions)', length(data.block)))
    Ws = T.(repeat(initV .* repeat(data.valence, inner = n_actions)', length(data.block)))

    # Initial values
    Q = copy(Qs[:, 1:n_actions])
    W0, Wv = copy(Ws), copy(Ws[:, 1:n_actions])
    N = length(data.stimset)
    loglike = zero(T)

    # Main loop
    for i in 1:N
        # index of first outcome for this stimulus group, because we have a flat Q-table
        pri::Int = 1 + (n_actions * (data.stimset[i]-1))
        opts::UnitRange{Int64} = pri:(pri+n_actions-1)

        # setup RL and WM policies (softmax with directed and undirected noise)
        π_hl = Turing.softmax(β .* Qs[i, opts])
        π_wm = Turing.softmax(β .* Ws[i, opts])
        π = @. (1 - ε) * (w0 * π_wm + (1 - w0) * π_hl) + ε / n_actions

        # weightings inside the softmax?
        # π = (1 - ε) * Turing.softmax(β * ((1 - w0) * Qs[i, opts] .+ w0 * Ws[i, opts])) .+ ε / n_actions
        
        # Choice and likelihood
        choice[i] ~ Turing.Categorical(π)
        loglike += loglikelihood(Turing.Categorical(π), choice[i])
        
        # Updates
        choice_idx::Int = choice[i] + pri - 1
        choice_1id::Int = choice[i]
        outc_hl, outc_wm = 1.0, data.outcomes[i, choice_1id] # HLWM means 1.0 for chosen option, regardless of valence
        α_hl, α_wm = in(outc_wm, opt_outc) ? (α_pos, 1.0) : (α_neg, α_neg_wm)
        
        if (i != N)
            # In-place updates
            @views begin
                copyto!(Qs[i + 1, :], Qs[i, :])
                Qs[i + 1, choice_idx] += α_hl * (outc_hl - Qs[i, choice_idx])
                
                copyto!(Ws[i + 1, :], Ws[i, :])
                un_idx = setdiff(1:mss, choice_idx)
                Ws[i + 1, un_idx] .+= φ_wm .* (W0[i, un_idx] .- Ws[i, un_idx])
                Ws[i + 1, choice_idx] += α_wm * (outc_wm - Ws[i, choice_idx])
            end
        end
        
        # Store values
        @views begin
            Q[i, :] .= Qs[i, opts]
            Wv[i, :] .= Ws[i, opts]
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
    data.val_grp = data.block
    if minimum(data.block) != 1
        data.block .-= minimum(data.block) - 1
    elseif length(unique(data.block)) == 1
        data.val_grp .= data.stimset
    end

    data_tuple = (
        block = data.block, # length = number of trials
        valence = unique(data[!, [:val_grp, :valence]]).valence, # length = number of blocks
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
    if :β in parameters && isa(priors[:β], Float64)
        β = priors[:β]
    elseif :β in parameters && isa(priors[:β], Distributions.Distribution)
        β ~ priors[:β]
    else
        β = 25. # this is what AGC fixes it to
    end
    
    # RL learning rates
    if :a in parameters || :a_pos in parameters
        if :a in parameters
            a_pos ~ priors[:a]
        else
            a_pos ~ priors[:a_pos]
        end
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