begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, Distributions, StatsBase,
		CSV, Turing
	using LogExpFunctions: logistic, logit, softmax
  using Tidier, JLD2

	# include("$(pwd())/controltask_models.jl")
	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/stats_utils.jl")
	include("$(pwd())/model_utils.jl")

  include(joinpath(pwd(), "fetch_preprocess_data.jl"))
	# include(joinpath(pwd(), "plotting_utils.jl"))
  include(joinpath(pwd(), "data_analysis", "control_exploration_fn.jl"))
	Turing.setprogress!(false)
end

begin
  # Load and clean data
  @load "data/explore_choice_df.jld2" explore_choice_df
end

begin
  # Model function
  @model function control_model_beta(;
    choice,
    controlled::AbstractVector{<:Integer},
    options::AbstractMatrix{<:Integer},
    n_options::Union{Nothing, Int} = nothing, # Number of options in the task
    alpha::Union{Nothing, AbstractVector{<:Real}} = nothing, # initial α per option
    beta::Real = 1.0, # fixed β per option
    priors::Dict = Dict(
      :γ => Beta(1, 1)
    )
  )
    N = length(choice) # Number of trials
    K_total = something(n_options, maximum(options))  # total unique options (e.g., 4),
    α0 = alpha === nothing ? ones(K_total) : collect(float.(alpha))

    # Priors
    γ ~ priors[:γ]

    Tγ = typeof(γ)
    α = Vector{Tγ}(undef, K_total)
    β = convert(Tγ, beta)
    for k in 1:K_total
      α[k] = convert(Tγ, α0[k])
    end

    for n in 1:N
      u = entropy.([Beta(α[k], β) for k in options[n,:]])
      p = softmax(u)
      choice[n] ~ Categorical(p)

      # Update beliefs
      if controlled[n] == 1
        chosen_a = options[n, choice[n]]
        α[chosen_a] += 1.0
        for a in 1:K_total
          a == chosen_a && continue
          α[a] += (1.0 - α[a]) * γ
        end
      else
        for a in 1:K_total
          α[a] += (1.0 - α[a]) * γ
        end
      end
      
    end

  end

  # Unpack function
  function unpack_control_model_beta(
    df::AbstractDataFrame;
    columns::Dict{String, Any} = Dict(
        "choice" => :response,
        "control" => :control_rule_used,
        "options" => [:left, :right]
      )
    )
    data = dropmissing(df, columns["choice"])
    choice = ifelse.(data[:, columns["choice"]] .== "left", 1, 2)
    controlled = ifelse.(data[:, columns["control"]] .== "control", 1, 0)
    options = Matrix{Int64}(
        select(data, columns["options"] .=> 
        ByRow(function case_match(x)
            x == "blue" && return 1
            x == "green" && return 2
            x == "yellow" && return 3
            x == "red" && return 4
          end)
      ))
    return (;choice, controlled, options)
  end
end

begin
  # Fit model to a single participant for testing
  fit = optimize(
    unpack_control_model_beta(
      explore_choice_df[explore_choice_df.prolific_pid .== "5d0827ec51969e001989475c" .&& explore_choice_df.session .== "1" , :]);
      model = control_model_beta,
      priors = Dict(:γ => Beta(1, 1))
      )

  γ_est = fit.values[:γ]
end

begin
  # Fit model to multiple participants, session 1 only
  fit_mult = optimize_multiple(
    filter(x -> x.session .== "1", explore_choice_df);
    model = control_model_beta,
    unpack_function = unpack_control_model_beta,
    priors = Dict(:γ => Beta(1, 1)),
    grouping_col = :prolific_pid
  )
end

begin
  # Fit model to multiple participants by sessions
  fit_mult = optimize_multiple_by_factor(
    explore_choice_df;
    model = control_model_beta,
    unpack_function = unpack_control_model_beta,
    priors = Dict(:γ => Beta(1, 1)),
    factor = :session,
    remap_columns = Dict(
        "choice" => :response,
        "control" => :control_rule_used,
        "options" => [:left, :right]
      )
  )
end