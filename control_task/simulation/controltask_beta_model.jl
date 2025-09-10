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
    n_options = n_options === nothing ? maximum(options) : n_options # Number of options in the task
    alpha = alpha === nothing ? ones(n_options) : alpha

    beliefs = Vector{Beta}(undef, n_options)
    for i in 1:n_options
      beliefs[i] = Beta(alpha[i], beta)
    end

    # Priors
    γ ~ priors[:γ]

    for n in 1:N
      choice[n] ~ Categorical(softmax(entropy.(beliefs[options[n,:]])))

      # Update beliefs
      if controlled[n] == 1
        chosen_a = options[n, choice[n]]
        unchosen_a = setdiff(1:n_options, chosen_a)
        beliefs[chosen_a] = Beta(beliefs[chosen_a].α + 1, beliefs[chosen_a].β)
        for a in unchosen_a
          beliefs[a] = Beta(beliefs[a].α + (1 - beliefs[a].α) * γ, beliefs[a].β)
        end
      else
        for a in 1:n_options
          beliefs[a] = Beta(beliefs[a].α + (1 - beliefs[a].α) * γ, beliefs[a].β)
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
      explore_choice_df[explore_choice_df.prolific_pid .== "5d0827ec51969e001989475c", :]);
      model = control_model_beta,
      priors = Dict(:γ => Beta(1, 1))
      )

  γ_est = fit.values[:γ]
end

begin
  fit_mult = optimize_multiple(
    filter(x -> x.session .== "1", explore_choice_df);
    model = control_model_beta,
    unpack_function = unpack_control_model_beta,
    priors = Dict(:γ => Beta(1, 1)),
    grouping_col = :prolific_pid
  )
end