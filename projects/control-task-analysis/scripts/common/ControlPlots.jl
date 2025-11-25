module ControlPlots

using DataFrames, CairoMakie, AlgebraOfGraphics, Statistics, GLM, TidierData, Effects, MixedModels
using LogExpFunctions: logistic, logit

export plot_explore_trend, plot_explore_trend_reliability, 
       plot_bias_and_sensitivity_reliability, plot_glmm_effects, plot_choice_pred_curves, plot_effort_interaction

function plot_explore_trend(
  df::DataFrame;
  xlabel::String="Trial",
  ylabel::String="P(Explorative)",
  xcol::Symbol=:trial_number,
  ycol::Symbol=:explorative_cat,
  title::String="Exploration Trend by ...",
  caption::String=""
)
  f = Figure(size=(600, 600))

  # Group average plot
  df_grp = dropmissing(df, [xcol, ycol])
  df_grp = combine(groupby(df_grp, [:session, xcol]),
    :explorative_cat => mean => :explorative)
  sort!(df_grp, [:session, xcol])
  p_grp = data(df_grp) *
          mapping(xcol, :explorative, color=:session => :Session) * (visual(ScatterLines) + linear())
  fig = draw!(f[1, 1], p_grp;
    axis=(; xlabel=xlabel, ylabel=ylabel))
  legend!(f[1, 2], fig)

  # Individual participant plot
  df_ind = dropmissing(df, [xcol, ycol])
  df_ind = combine(groupby(df_ind, [:prolific_pid, :session, xcol]),
    :explorative_cat => mean => :explorative)
  sort!(df_ind, [:prolific_pid, :session, xcol])
  p_ind = data(df_ind) *
          mapping(xcol, :explorative, group=:prolific_pid, color=:session, col=:session) * linear(interval=nothing)
  draw!(f[2, :], p_ind, scales(Col=(; categories=["1" => "Session 1", "2" => "Session 2"]));
    axis=(; xlabel=xlabel, ylabel=ylabel))

  Label(f[0, :], title; tellheight=true, tellwidth=false)
  Label(f[end+1, :], caption; tellheight=true, tellwidth=false)
  f
end

function plot_explore_trend_reliability(
  df::DataFrame;
  title::String="Test-retest: ...",
  ycol::Symbol=:explorative_cat,
  xcol::Symbol=:trial_number,
  caption::String=""
)
  glm_coef(data) = coef(lm(Term(ycol) ~ Term(xcol), data))

  retest_df = @chain df begin
    transform(xcol => (x -> x .- mean(x)), renamecols=false)
    @drop_missing(explorative_cat)
    groupby([:prolific_pid, :session])
    combine(AsTable([ycol, xcol]) => (x -> [glm_coef(x)]) => [:β0, :β_trial])
  end

  fig = Figure(; size=(12, 6) .* 144 ./ 2.54)
  workshop_reliability_scatter!(
    fig[1, 1];
    df=dropmissing(unstack(retest_df, [:prolific_pid], :session, :β0)),
    xlabel="Session 1",
    ylabel="Session 2",
    xcol=Symbol(string(1)),
    ycol=Symbol(string(2)),
    subtitle="Avg. exploration",
    correct_r=false
  )

  workshop_reliability_scatter!(
    fig[1, 2];
    df=dropmissing(unstack(retest_df, [:prolific_pid], :session, :β_trial)),
    xlabel="Session 1",
    ylabel="Session 2",
    xcol=Symbol(string(1)),
    ycol=Symbol(string(2)),
    subtitle="Exploration trend",
    correct_r=false
  )
  Label(fig[0, :], title)
  Label(fig[end+1, :], caption, halign=:right)
  fig
end

function plot_bias_and_sensitivity_reliability(
  df::DataFrame;
  xcol::Symbol=:explorative_val,
  ycol::Symbol=:response_left,
  title="Test-retest: ...",
  caption=""
)
  # Predict probabilities for a range of x

  glm_model(df) = glm(Term(ycol) ~ Term(xcol), df, Bernoulli(), LogitLink())
  glm_coef(df) = coef(glm_model(df))

  df = dropmissing(df, [ycol, xcol])
  subset!(df, xcol => ByRow(x -> !isinf(x) && !isnan(x)))

  coef_df = groupby(df, [:prolific_pid, :session]) |>
            x -> combine(x, AsTable([ycol, xcol]) => (x -> [glm_coef(x)]) => [:β0, :β_explorative])

  # Coefficient distribution plot
  coef_long = stack(coef_df, [:β0, :β_explorative], variable_name=:parameter, value_name=:value)
  coef_long = transform(coef_long, :parameter => ByRow(x -> x == "β0" ? "Bias" : "Sensitivity") => :parameter_label)

  coef_plot = data(coef_long) * 
    mapping(:session => nonnumeric, :value, group=:prolific_pid, col=:parameter_label) * 
    (visual(Scatter, markersize=8, alpha=0.5) + visual(Lines, linewidth=1.5, alpha=0.3)) +
    mapping(0) * visual(HLines, color=:gray, linestyle=:dash, linewidth=1)

  fig = Figure(; size=(12, 10) .* 144 ./ 2.54)
  draw!(fig[1, 1:2], coef_plot;
  facet=(; linkyaxes=:none),
  axis=(; xlabel="Session", ylabel="Parameter Value"))

  # println(filter(x -> (x.session .== "2" .&& x.β0 .< -15), coef_df))

  workshop_reliability_scatter!(
    fig[2, 1];
    df=dropmissing(unstack(coef_df, [:prolific_pid], :session, :β0)),
    xlabel="Session 1",
    ylabel="Session 2",
    xcol=Symbol(string(1)),
    ycol=Symbol(string(2)),
    subtitle="Bias",
    correct_r=false
  )

  workshop_reliability_scatter!(
    fig[2, 2];
    df=dropmissing(unstack(coef_df, [:prolific_pid], :session, :β_explorative)),
    xlabel="Session 1",
    ylabel="Session 2",
    xcol=Symbol(string(1)),
    ycol=Symbol(string(2)),
    subtitle="Sensitivity",
    correct_r=false
  )
  Label(fig[0, :], title)
  Label(fig[end+1, :], caption, halign=:right)
  fig
end

function plot_glmm_effects(model, predictor_vars::Vector{String};
  predictor_labels = Dict(), predictor_colors = Dict())
  
    ## Visualize fixed-effect estimates and marginal predictions using Effects.jl
  coef_tbl = DataFrame(coeftable(model))
  rename!(coef_tbl, Symbol("Coef.") => :coef, Symbol("Std. Error") => :se)
  coef_tbl.term = string.(coefnames(model))
  coef_tbl.lower = coef_tbl.coef .- 1.96 .* coef_tbl.se
  coef_tbl.upper = coef_tbl.coef .+ 1.96 .* coef_tbl.se

  term_order = reverse(coef_tbl.term)
  order_idx = reverse(1:nrow(coef_tbl))

  coef_ax_ticks = (1:length(term_order), [predictor_labels[i] for i in term_order])

  # Generate effects for all predictors using Effects.jl
  x_range = range(-2.5, 2.5; length=200)

  # Compute effects for each predictor
  predictor_effects = Dict(
    var => begin
      effects(Dict(Symbol(var) => x_range), model; invlink=identity, level=0.95) |>
      x -> transform(x, [:response_left, :err, :lower, :upper] .=> ByRow(logistic), renamecols=false)
    end
    for var in predictor_vars
  )

  # Extract coefficients for labels
  coef_names = string.(coefnames(model))
  fixefs = collect(fixef(model))
  predictor_coefs = Dict(
    var => fixefs[findfirst(==(var), coef_names)]
    for var in predictor_vars
  )

  f = Figure(size=(1200, 300))

  # Fixed-effects coefficient plot
  ax_coef = Axis(f[1, 1];
    title="Fixed-effects (log-odds)",
    xlabel="Estimate",
    yticks=coef_ax_ticks,
    yreversed=true)

  vlines!(ax_coef, [0]; color=:black, linestyle=:dash)
  xerr_lower = coef_tbl.coef[order_idx] .- coef_tbl.lower[order_idx]
  xerr_upper = coef_tbl.upper[order_idx] .- coef_tbl.coef[order_idx]
  errorbars!(ax_coef, coef_tbl.coef[order_idx], 1:length(term_order), xerr_lower, xerr_upper; direction=:x, color=:black)
  scatter!(ax_coef, coef_tbl.coef[order_idx], 1:length(term_order); color=:black)

  # Helper function to plot marginal effects
  function plot_marginal_effects!(ax, var_names)
    for var_name in var_names
      eff = predictor_effects[var_name]
      color = predictor_colors[var_name]
      β = predictor_coefs[var_name]

      band!(ax, eff[!, Symbol(var_name)], eff.lower, eff.upper; color=(color, 0.2))
      lines!(ax, eff[!, Symbol(var_name)], eff.response_left;
        color=color, linewidth=3,
        label="$(predictor_labels[var_name]) (β=$(round(β, digits=3)))")
    end
  end

  # Interval predictors plot
  ax_interval = Axis(f[1, 3]; ylabelvisible=false)
  vlines!(ax_interval, [0]; color=(:black, 0.4), linestyle=:dash)
  ylims!(ax_interval, 0, 1)
  ax_interval.xlabel = "Predictor (z)"
  ax_interval.title = "Interval predictors"

  plot_marginal_effects!(ax_interval, filter(x -> match(r"interval", x) !== nothing, predictor_vars))
  axislegend(ax_interval, position=:rb)

  # Occurrence predictors plot
  ax_occurrence = Axis(f[1, 2]; ylabelvisible=false)
  vlines!(ax_occurrence, [0]; color=(:black, 0.4), linestyle=:dash)
  ylims!(ax_occurrence, 0, 1)
  ax_occurrence.xlabel = "Predictor"
  ax_occurrence.title = "Occurrence predictors"

  plot_marginal_effects!(ax_occurrence, filter(x -> match(r"occurrence", x) !== nothing, predictor_vars))
  axislegend(ax_occurrence, position=:rb)

  f
end

function plot_choice_pred_curves(
  df::DataFrame;
  xcol::Symbol=:explorative_val,
  ycol::Symbol=:response_left,
  xlabel::String="Left - Right",
  ylabel::String="P(Left)",
  title="Choice Prediction Curves",
  caption=""
)
  glm_model(df) = glm(Term(ycol) ~ Term(xcol), df, Bernoulli(), LogitLink())

  df = dropmissing(df, [ycol, xcol])
  subset!(df, xcol => ByRow(x -> !isinf(x) && !isnan(x)))

  fig = Figure(size=(600, 400))

  # Create prediction grid
  x_pred = range(minimum(df[!, xcol]), maximum(df[!, xcol]), length=100)

  # Get individual curves for each participant/session
  individual_curves = combine(groupby(df, [:prolific_pid, :session])) do sdf
    if nrow(sdf) > 5  # Ensure enough data points
      model = glm_model(sdf)
      pred_df = DataFrame(xcol => x_pred)
      pred_df.predicted_prob = predict(model, pred_df)
      pred_df[!, :prolific_pid] .= sdf.prolific_pid[1]
      pred_df[!, :session] .= sdf.session[1]
      return pred_df
    else
      return DataFrame()  # Skip if insufficient data
    end
  end

  # Average the curves across participants
  avg_curve = combine(groupby(individual_curves, [xcol, :session])) do gdf
    DataFrame(avg_prob=mean(gdf.predicted_prob))
  end

  avg_curve_data = combine(groupby(df, [xcol, :session]), ycol => mean => :avg_prob)
  sort!(avg_curve_data, [:session, xcol])

  # Plot
  plt = (data(individual_curves) * mapping(xcol, :predicted_prob, color=:session, col=:session) * visual(Lines, alpha=0.2)) +
        (data(avg_curve) * mapping(xcol, :avg_prob, color=:session, col=:session) * visual(Lines, linewidth=3)) +
        (mapping(0.5) * visual(HLines, color=:gray, linestyle=:dash)) +
        (mapping(0) * visual(VLines, color=:gray, linestyle=:dash))

  draw!(fig[1, 1], plt,
    scales(Col=(; categories=["1" => "Session 1", "2" => "Session 2"])),
    axis=(; xlabel=xlabel, ylabel=ylabel))
  Label(fig[0, :], title, tellwidth=false)
  Label(fig[end+1, :], caption, halign=:right, tellwidth=false)
  fig
end

# Function to plot interaction effects of intervals by effort level
function plot_effort_interaction(model, data_df, predictor_var::String;
  xlabel::String=predictor_var,
  ylabel::String="P[Left]",
  color_palette=["lightskyblue1", "deepskyblue3", "midnightblue"])

  predictor_sym = Symbol(predictor_var)

  # Compute effects
  eff = effects(
    Dict(:current => unique(data_df.current),
      predictor_sym => range(extrema(data_df[!, predictor_sym])..., length=50)),
    model; invlink = identity, level=0.95, eff_col = :response_left
  )
  transform!(eff, [:response_left, :err, :lower, :upper] .=> ByRow(logistic), renamecols=false)

  # Create plot
  data(eff) *
  (
    mapping(
      predictor_sym => xlabel,
      :lower,
      :upper,
      color=:current => nonnumeric => "Required effort level",
      group=:current => nonnumeric => "Required effort level") *
    visual(Band; alpha=plot_config[:band_alpha]) +
    mapping(
      predictor_sym => xlabel,
      :response_left => ylabel,
      color=:current => nonnumeric => "Required effort level",
      group=:current => nonnumeric) *
    visual(Lines)
  ) |>
  draw(scales(Color=(; palette=color_palette)); axis=(; ylabel=ylabel))
end

end # module