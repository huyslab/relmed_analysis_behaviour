function task_vars_for_condition(condition::String)
    # Load sequence from file
    task = DataFrame(CSV.File("data/PLT_task_structure_" * condition * ".csv"))

    # Renumber block
    task.block = task.block .+ (task.session .- 1) * maximum(task.block)

    # Arrange feedback by optimal / suboptimal
    task.feedback_optimal = 
        ifelse.(task.optimal_A .== 1, task.feedback_A, task.feedback_B)

    task.feedback_suboptimal = 
        ifelse.(task.optimal_A .== 0, task.feedback_A, task.feedback_B)


    # Arrange outcomes such as second column is optimal
    outcomes = hcat(
        task.feedback_suboptimal,
        task.feedback_optimal,
    )

    return (
        task = task,
        block = task.block,
        valence = unique(task[!, [:block, :valence]]).valence,
        outcomes = outcomes
    )

end

# Simulate data from model prior
function simulate_single_p(
    n::Int64; # How many datasets to simulate
    wm::Bool = false, # WM model
    block::Vector{Int64}, # Block number
    valence::AbstractVector, # Valence of each block
    outcomes::Matrix{Float64}, # Outcomes for options, first column optimal
    initV::Matrix{Float64}, # Initial Q (and W) values
    set_size::Union{Vector{Int64}, Nothing} = nothing, # Set size for each block, required for WM models
    parameters::Vector{Symbol} = [:ρ, :a], # Group-level parameters to estimate
    transformed::Dict{Symbol, Symbol} = Dict(:a => :α), # Transformed parameters
    sigmas::Dict{Symbol, Float64} = Dict(:ρ => 2., :a => 0.5),
    random_seed::Union{Int64, Nothing} = nothing
)

    # Total trial number
    N = length(block)

    # Trials per block
    n_trials = div(length(block), maximum(block))
    if length(block) % maximum(block) != 0
        n_trials += 1
    end

    # Prepare model for simulation
    if !wm
        prior_model = RL(
            N = N,
            block = block,
            valence = valence,
            choice = fill(missing, length(block)),
            outcomes = outcomes,
            initV = initV,
            parameters = parameters,
            sigmas = sigmas 
        )
    else
        @assert set_size !== nothing
        prior_model = RLWM(
            N = N,
            block = block,
            valence = valence,
            choice = fill(missing, length(block)),
            outcomes = outcomes,
            initV = initV,
            set_size = set_size,
            parameters = parameters,
            sigmas = sigmas 
        )
    end

    # Draw parameters and simulate choice
    prior_sample = sample(
        isnothing(random_seed) ? Random.default_rng() : Xoshiro(random_seed),
        prior_model,
        Prior(),
        n
    )

    # Arrange choice for return
    sim_data = DataFrame(
        PID = repeat(1:n, inner = N),
        block = repeat(block, n),
        valence = repeat(valence, inner = n_trials, outer = n),
        trial = repeat(1:n_trials, n * maximum(block)),
        choice = prior_sample[:, [Symbol("choice[$i]") for i in 1:N], 1] |>
            Array |> transpose |> vec
    )

    for p in parameters
        v = repeat(prior_sample[:, p, 1], inner = N)
        if haskey(transformed, p)
            v = v .|> a2α # assume all transformations are to [0, 1]
            sim_data[!, transformed[p]] = v
        else
            sim_data[!, p] = v
        end
    end

    # Compute Q values
    gq = generated_quantities(prior_model, prior_sample)
    Qs = [pt.Qs for pt in gq] |> vec

    sim_data.Q_optimal = vcat([qs[:, 2] for qs in Qs]...)
    sim_data.Q_suboptimal = vcat([qs[:, 1] for qs in Qs]...)

    if wm
        Ws = [pt.Ws for pt in gq] |> vec
        sim_data.W_optimal = vcat([ws[:, 2] for ws in Ws]...)
        sim_data.W_suboptimal = vcat([ws[:, 1] for ws in Ws]...)
    end

    return sim_data
end

# # Sample from posterior conditioned on DataFrame with data for single participant
function posterior_sample_single_p(
	data::AbstractDataFrame;
    wm::Bool = false,
	initV::Float64,
    set_size::Union{Vector{Int64}, Nothing} = nothing,
    parameters::Vector{Symbol} = [:ρ, :a],
    sigmas::Dict{Symbol, Float64} = Dict(:ρ => 2., :a => 0.5),
	random_seed::Union{Int64, Nothing} = nothing,
	iter_sampling = 1000
)
    if wm
        model = RLWM(;
            N = nrow(data),
            block = data.block,
            valence = unique(data[!, [:block, :valence]]).valence,
            choice = data.choice,
            outcomes = hcat(
                data.feedback_suboptimal,
                data.feedback_optimal,
            ),
            initV = fill(initV, 1, 2),
            set_size = set_size,
            parameters = parameters,
            sigmas = sigmas
        )
    else
        model = RL(;
            N = nrow(data),
            block = data.block,
            valence = unique(data[!, [:block, :valence]]).valence,
            choice = data.choice,
            outcomes = hcat(
                data.feedback_suboptimal,
                data.feedback_optimal,
            ),
            initV = fill(initV, 1, 2),
            parameters = parameters,
            sigmas = sigmas
        )
    end

	fit = sample(
		isnothing(random_seed) ? Random.default_rng() : Xoshiro(random_seed),
		model, 
		NUTS(), 
		MCMCThreads(), 
		iter_sampling, 
		4)

	return fit
end