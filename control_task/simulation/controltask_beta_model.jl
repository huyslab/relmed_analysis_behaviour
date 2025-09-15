begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
  Pkg.activate("$(pwd())/relmed_environment")
  # instantiate, i.e. make sure that all packages are downloaded
  Pkg.instantiate()
  using Random, DataFrames, Distributions, StatsBase,
  CSV, Turing
  using Tidier, JLD2, AlgebraOfGraphics, CairoMakie, Printf, Statistics
	using LogExpFunctions: logistic, logit, softmax

	# include("$(pwd())/controltask_models.jl")
	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/stats_utils.jl")
	include("$(pwd())/model_utils.jl")

  include(joinpath(pwd(), "fetch_preprocess_data.jl"))
	include(joinpath(pwd(), "plotting_utils.jl"))
  include(joinpath(pwd(), "data_analysis", "control_exploration_fn.jl"))
	Turing.setprogress!(false)
end

begin
  # Set theme
	inter_bold = assetpath(pwd() * "/fonts/Inter/Inter-Bold.ttf")
	
	th = Theme(
		font = "Helvetica",
		fontsize = 16,
		Axis = (
			xgridvisible = false,
			ygridvisible = false,
			rightspinevisible = false,
			topspinevisible = false,
			xticklabelsize = 14,
			yticklabelsize = 14,
			spinewidth = 1.5,
			xtickwidth = 1.5,
			ytickwidth = 1.5
		)
	)
	set_theme!(th)
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
      :γ_pr => Normal(0., 1.),
      :β_0 => Normal(0., 1.),
      :β_H => Normal(0., 1.)
    )
  )
    N = length(choice) # Number of trials
    K_total = something(n_options, maximum(options))  # total unique options (e.g., 4),
    α0 = alpha === nothing ? ones(K_total) : collect(float.(alpha))

    # Priors
    γ_pr ~ priors[:γ_pr]
    γ := a2α(γ_pr)  # decay parameter between 0 and 1
    β_0 ~ priors[:β_0]
    β_H ~ priors[:β_H]

    Tγ = typeof(γ)
    α = Vector{Tγ}(undef, K_total)
    β = convert(Tγ, beta)
    for k in 1:K_total
      α[k] = convert(Tγ, α0[k])
    end

    for n in 1:N
      u = entropy.([Beta(α[k], β) for k in options[n,:]])
      choice[n] ~ BernoulliLogit(β_0 + β_H * (u[2] - u[1]))

      # Update beliefs
      if controlled[n] == 1
        chosen_a = options[n, Int(choice[n] + 1)]  # choice is 0/1, index is 1/2 => [:left, :right]
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

  @model function control_model_interval(;
    choice,
    controlled::AbstractVector{<:Integer},
    options::AbstractMatrix{<:Integer},
    n_options::Union{Nothing, Int} = nothing, # Number of options in the task
    alpha::Union{Nothing, AbstractVector{<:Real}} = nothing, # initial α per option
    priors::Dict = Dict(
      :β_0 => Normal(0., 1.),
      :β_H => Normal(0., 1.)
    )
  )
    N = length(choice) # Number of trials
    K_total = something(n_options, maximum(options))  # total unique options (e.g., 4),
    α = alpha === nothing ? fill(72, K_total) : collect(float.(alpha))

    # Priors
    β_0 ~ priors[:β_0]
    β_H ~ priors[:β_H]

    for n in 1:N
      u = [α[k] for k in options[n,:]]
      choice[n] ~ BernoulliLogit(β_0 + β_H * (u[2] - u[1]))

      # Update interval
      α[options[n, :]] .= 1.
      α[setdiff(1:K_total, options[n, :])] .= α[setdiff(1:K_total, options[n, :])] .+ 1.
      
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
    choice = ifelse.(data[:, columns["choice"]] .== "right", 1, 0)
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
  control_beta_prior = Dict(
      :γ_pr => Normal(0., 2.),
      :β_0 => Normal(0., 2.),
      :β_H => Normal(0., 2.)
    )
  # Fit model to a single participant for testing
  fit = optimize(
    unpack_control_model_beta(
      explore_choice_df[explore_choice_df.prolific_pid .== "681106ac93d01f1615c6f003" .&& explore_choice_df.session .== "1" , :]);
      model = control_model_beta,
      priors = control_beta_prior
      )

  fit.values
end

begin
  # Fit model to multiple participants, session 1 only
  fit_mult = optimize_multiple(
    filter(x -> x.session .== "1", explore_choice_df);
    model = control_model_beta,
    unpack_function = unpack_control_model_beta,
    priors = control_beta_prior,
    grouping_col = :prolific_pid
  )
  DataFrames.transform!(fit_mult, :γ_pr => ByRow(a2α) => :γ)
end

begin
  # Fit model to multiple participants by sessions
  fit_mult_by_f = optimize_multiple_by_factor(
    explore_choice_df;
    model = control_model_beta,
    unpack_function = unpack_control_model_beta,
    priors = control_beta_prior,
    factor = :session,
    remap_columns = Dict(
        "choice" => :response,
        "control" => :control_rule_used,
        "options" => [:left, :right]
      )
  )
  DataFrames.transform!(fit_mult_by_f, :γ_pr => ByRow(a2α) => :γ)
end


begin
  vars = [:γ, :β_0, :β_H]
  panels = DataFrame[]
  for y in vars, x in vars
    if x == y
      df = select(fit_mult_by_f, x, :session)
      rename!(df, x => :x)
      df.y = df.x
      df.xvar .= string(x); df.yvar .= string(y); df.kind .= "diag"
      panels = [panels; df]
    else
      df = select(fit_mult_by_f, x, y, :session)
      rename!(df, x => :x, y => :y)
      df.xvar .= string(x); df.yvar .= string(y); df.kind .= "off"
      panels = [panels; df]
    end
  end
  pg = vcat(panels...)
  off_df = pg[pg.kind .== "off", :]
  diag_df = pg[pg.kind .== "diag", :]

  layer_off = AlgebraOfGraphics.data(off_df) *
  mapping(:x, :y, col = :xvar, row = :yvar, color = :session) *
  visual(Scatter)

  layer_diag = AlgebraOfGraphics.data(diag_df) *
  mapping(:x, col = :xvar, row = :yvar, color = :session) *
  visual(Hist)

  (layer_off + layer_diag) |> draw(facet = (; linkxaxes = :none, linkyaxes = :none))
end

begin
  # Test-retest reliability
  retest_pars = @chain fit_mult_by_f begin
    stack([:γ, :β_0, :β_H], variable_name = :parameter, value_name = :value)
    unstack([:prolific_pid, :parameter], :session, :value, renamecols=x->Symbol(:sess_, x))
  end

  f = Figure(size = (19.5 * 3, 19.5) .* 72 ./ 2.54 ./ 2)
	
  for (i, v) in enumerate([:γ, :β_0, :β_H])
    workshop_reliability_scatter!(
      f[1, i];
      df = dropmissing!(retest_pars) |> filter(x -> x.parameter .== string(v)),
      xcol = :sess_1,
      ycol = :sess_2,
      xlabel = "Session 1",
      ylabel = "Session 2",
      subtitle = string(v),
      correct_r = false,
      markersize = 6
    )
  end
  f
end

begin
  # Prior predictive checking
  # - Simulate choices from the prior for a single participant/session
  # - Compare simulated left-choice rate distribution to observed

  # Select one participant and session (same PID used above for consistency)
  ppc_df = explore_choice_df[
    explore_choice_df.prolific_pid .== "681106ac93d01f1615c6f003" .&&
    explore_choice_df.session .== "1",
    :
  ]

  # Prepare model data and replace outcome with missings for prior simulation
  data_nt = unpack_control_model_beta(ppc_df)
  task_nt = (; data_nt..., choice = fill(missing, length(data_nt.choice)))

  # Draw prior predictive simulations (replicates)
  n_sims = 400
  preds = prior_sample(
    task_nt;
    model = control_model_beta,
    n = n_sims,
    priors = control_beta_prior,
    outcome_name = :choice,
  )

  # Summaries: proportion of left choices in each simulated replicate
  p_right_sim = vec(mean(preds .== 1; dims = 1))
  p_right_obs = mean(data_nt.choice .== 1)

  # Plot histogram of simulated left-choice rate with observed marked
  f = Figure(size = (16, 9) .* 72 ./ 2.54)
  gp = f[1, 1]
  mp = AlgebraOfGraphics.data(DataFrame(p_left = p_right_sim)) *
       mapping(:p_left) *
       visual(Hist)
  draw!(gp, mp; axis = (; xlabel = "P(choose right)", ylabel = "Count", subtitle = "Prior predictive"))
  ax = extract_axis(gp)
  vlines!(ax, [p_right_obs]; color = :red, linewidth = 3)
  vlines!(ax, [mean(p_right_sim)]; color = :gray80, linewidth = 3)
  display(f)
end

begin
  # Aggregate prior predictive across participants, by session
  sessions = unique(explore_choice_df.session) |> collect
  n_sims_agg = 400
  f2 = Figure(size = (24 * length(sessions), 16) .* 72 ./ 2.54 ./ 2)
  for (i, s) in enumerate(sessions)
    sess_df = explore_choice_df[explore_choice_df.session .== s, :]
    pids = unique(sess_df.prolific_pid)

    sim_list = Vector{Vector{Float64}}()
    obs_list = Float64[]

    for pid in pids
      pid_df = sess_df[sess_df.prolific_pid .== pid, :]
      data_pid = unpack_control_model_beta(pid_df)
      if length(data_pid.choice) == 0
        continue
      end
      task_pid = (; data_pid..., choice = fill(missing, length(data_pid.choice)))
      preds_pid = prior_sample(
        task_pid;
        model = control_model_beta,
        n = n_sims_agg,
        priors = control_beta_prior,
        outcome_name = :choice,
      )
      push!(sim_list, vec(mean(preds_pid .== 1; dims = 1)))
      push!(obs_list, mean(data_pid.choice .== 1))
    end

    if isempty(sim_list)
      continue
    end

    P = reduce(hcat, sim_list)  # (n_sims_agg, n_pids) or Vector if single pid
    if ndims(P) == 1
      P = reshape(P, :, 1)
    end
    sess_sim_mean = vec(mean(P; dims = 2))
    obs_mean = mean(obs_list)

    gp2 = f2[1, i]
    mp2 = AlgebraOfGraphics.data(DataFrame(p_right_mean = sess_sim_mean)) *
          mapping(:p_right_mean) *
          visual(Hist)
    draw!(gp2, mp2; axis = (; xlabel = "Mean P(choose right)", ylabel = "Count", subtitle = "Session $(s)"))
    ax2 = extract_axis(gp2)
    vlines!(ax2, [obs_mean]; color = :red, linewidth = 3)
    vlines!(ax2, [mean(sess_sim_mean)]; color = :grey80, linewidth = 3)
  end

  f2

end

begin
  # Parameter recovery for control_model_beta
  # - Simulate datasets with known parameters using real task structure
  # - Fit the model to each simulated dataset
  # - Compare recovered parameters to ground truth

  # Use a single participant's task structure for speed/consistency
  pr_pid = "681106ac93d01f1615c6f003"
  pr_sess = "1"
  pr_df = explore_choice_df[
    explore_choice_df.prolific_pid .== pr_pid .&&
    explore_choice_df.session .== pr_sess,
    :
  ]

  data_pr = unpack_control_model_beta(pr_df)
  Ntr = length(data_pr.choice)
  if Ntr == 0
    @warn "No trials found for selected PID/session; skipping parameter recovery."
  else
    # Simulation settings (keep modest for speed; increase if needed)
    n_reps = 1000
    rng = Random.default_rng()

    # Containers
    recs = Vector{NamedTuple}(undef, n_reps)

    # Pre-allocate outcome placeholder
    task_missing = (; data_pr..., choice = fill(missing, Ntr))

    for i in 1:n_reps
      # Sample ground-truth parameters from prior
      par_recover_prior = Dict(
        :γ_pr => Normal(0., 2.),
        :β_0 => Normal(0., 2.),
        :β_H => Normal(0., 2.)
      )
      true_γ_pr = rand(rng, par_recover_prior[:γ_pr])
      true_β_0  = rand(rng, par_recover_prior[:β_0])
      true_β_H  = rand(rng, par_recover_prior[:β_H])

      # Dirac priors at true values for simulation
      priors_true = Dict(
        :γ_pr => Distributions.Dirac(true_γ_pr),
        :β_0  => Distributions.Dirac(true_β_0),
        :β_H  => Distributions.Dirac(true_β_H),
      )

      # Simulate choices
      sim_choice = prior_sample(
        task_missing;
        model = control_model_beta,
        n = 1,
        priors = priors_true,
        outcome_name = :choice,
        rng = rng,
      )

      # Fit model to simulated data (MAP, few starts for speed)
      fit_i = optimize(
        (; data_pr..., choice = sim_choice);
        model = control_model_beta,
        priors = par_recover_prior,
        estimate = "MAP",
        n_starts = 3,
      )

      # Store true and recovered (also transformed γ)
      recs[i] = (
        true_γ_pr = true_γ_pr,
        true_γ    = a2α(true_γ_pr),
        true_β_0  = true_β_0,
        true_β_H  = true_β_H,
        est_γ_pr  = fit_i.values[:γ_pr],
        est_γ     = a2α(fit_i.values[:γ_pr]),
        est_β_0   = fit_i.values[:β_0],
        est_β_H   = fit_i.values[:β_H],
        lp        = fit_i.lp,
      )
    end

    recovery_df = DataFrame(recs)

    # Quick numeric summaries (Pearson r)
    corr_γ_pr = cor(recovery_df.true_γ_pr, recovery_df.est_γ_pr)
    corr_γ    = cor(recovery_df.true_γ,   recovery_df.est_γ)
    corr_β0   = cor(recovery_df.true_β_0, recovery_df.est_β_0)
    corr_βH   = cor(recovery_df.true_β_H, recovery_df.est_β_H)
    @info "Recovery correlations" γ_pr = corr_γ_pr γ=corr_γ β0=corr_β0 βH=corr_βH

    # Scatter plots: estimated vs true
    f = Figure(size = (21, 7) .* 72 ./ 2.54)

    # γ (transformed)
    ax1 = Axis(f[1, 1], xlabel = "True γ", ylabel = "Estimated γ", subtitle = @sprintf("γ (r = %.2f)", corr_γ))
    scatter!(ax1, recovery_df.true_γ, recovery_df.est_γ; markersize = 6, color = :dodgerblue, alpha = 0.2)
    par_lims = extrema(vcat(recovery_df.true_γ, recovery_df.est_γ))
    lines!(ax1, [par_lims...], [par_lims...]; color = :gray50, linestyle = :dash)

    # β_0
    ax2 = Axis(f[1, 2], xlabel = "True β_0", ylabel = "Estimated β_0", subtitle = @sprintf("β_0 (r = %.2f)", corr_β0))
    scatter!(ax2, recovery_df.true_β_0, recovery_df.est_β_0; markersize = 6, color = :seagreen, alpha = 0.2)
    par_lims2 = extrema(vcat(recovery_df.true_β_0, recovery_df.est_β_0))
    lines!(ax2, [par_lims2...], [par_lims2...]; color = :gray50, linestyle = :dash)

    # β_H
    ax3 = Axis(f[1, 3], xlabel = "True β_H", ylabel = "Estimated β_H", subtitle = @sprintf("β_H (r = %.2f)", corr_βH))
    scatter!(ax3, recovery_df.true_β_H, recovery_df.est_β_H; markersize = 6, color = :tomato, alpha = 0.2)
    par_lims3 = extrema(vcat(recovery_df.true_β_H, recovery_df.est_β_H))
    lines!(ax3, [par_lims3...], [par_lims3...]; color = :gray50, linestyle = :dash)

    display(f)

    recovery_df
  end
end
end