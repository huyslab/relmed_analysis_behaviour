# ---
# title: "JC - Control Task Data Analysis"
# author: "Haoyang Lu"
# format:
#   html:
#     toc: true
#     toc-location: right-body
#     code-fold: true
#     embed-resources: true
# engine: julia
# ---

# ## Initialization and loading data

begin
  cd("/home/jovyan")
  include("$(pwd())/core/experiment-registry.jl")

  using DataFrames, TidierData, CairoMakie, AlgebraOfGraphics, Dates, CategoricalArrays, StatsBase, GLM, MixedModels, Random
  using LogExpFunctions: logistic, logit
  using StandardizedPredictors, Effects

  # Include data scripts
  include("$(pwd())/core/preprocess_data.jl")

  script_dir = dirname(@__FILE__)

  # Load configurations and theme
  include(joinpath(script_dir, "common", "config.jl"))
  include(joinpath(script_dir, "common", "utils.jl"))
  include(joinpath(pwd(), "projects", "control-task-analysis", "scripts", "common", "control_exploration_fn.jl"))

  experiment = NORMING
end

#+ Load data

# Load and preprocess data
begin
  dat = preprocess_project(experiment; force_download=false, delay_ms=65, use_manual_download=false)
end

begin
  control_task_data = subset(dat.control.control_task, :session => ByRow(==("wk0")))
  control_report_data = subset(dat.control.control_report, :session => ByRow(==("wk0")))
  @assert all(combine(groupby(control_task_data, [:module_start_time, :participant_id]), [:time_elapsed, :trial] => ((x, y) -> all(denserank(x) .== denserank(y))) => :in_order)[:, "in_order"]) "Trial numbers are created *incorrectly* in chronological order"
  nothing
end

#=
## What could predict participants' choices during exploration?

Several features might be used to predict participants' choices during exploration:
  - Interval from last seen
  - Interval from last (controlled) choice
  - Number of occurrence
  - Number of (controlled) choice
=#

begin
  ## Preprocess control task data to choice data
  explore_choice_df = @chain control_task_data begin
    filter(x -> x.trialphase .== "control_explore", _)
    select([:participant_id, :session, :trial, :left, :right, :response, :control_rule_used, :current, :trial_presses])
    transform([:left, :right, :response] => ByRow((left, right, resp) -> ismissing(resp) ? missing : ifelse(resp == "left", left, right)) => :choice)
    transform(:current => (x -> categorical(x; levels=[1, 2, 3], ordered=true)) => :current)
    groupby([:participant_id, :session])
    DataFrames.transform(
      :trial => (x -> 1:length(x)) => :trial_number
    )
    add_ship_onehot(_)
  end

  ## 1. Interval from last seen
  explore_by_interval = @chain explore_choice_df begin
    groupby([:participant_id, :session])
    transform(
      [
      ([:trial_number, color] =>
        ((t, occ) -> calc_choice_interval(t, occ, occ .== occ)) => color)
      for color in (:blue, :green, :yellow, :red)
    ]
    )
    add_explorative_measure(_, metric="interval")
    rename(:explorative_val => :diff_interval)
    select(Not([:blue, :green, :yellow, :red]))
  end

  ## 2. Interval from last (controlled) choice
  explore_by_choice_interval = @chain explore_choice_df begin
    groupby([:participant_id, :session])
    transform(
      [
      ([:trial_number, Symbol(color), :choice, :control_rule_used] =>
        ((t, occ, choice, r) -> calc_choice_interval(t, occ, .!ismissing.(r) .&& choice .== color .&& r .== "control")) => Symbol(color))
      for color in ("blue", "green", "yellow", "red")
    ]
    )
    add_explorative_measure(_, metric="interval")
    rename(:explorative_val => :diff_choice_interval)
    select(Not([:blue, :green, :yellow, :red]))
  end

  ## 3. Number of occurrence
  explore_by_occur = @chain explore_choice_df begin
    groupby([:participant_id, :session])
    transform(
      [:blue, :green, :yellow, :red] .=> cumsum, renamecols=false
    )
    add_explorative_measure(_, metric="occurrence")
    rename(:explorative_val => :diff_occurrence)
    select(Not([:blue, :green, :yellow, :red]))
  end

  ## 4. Number of (controlled) choice
  explore_by_choice_occur = @chain explore_choice_df begin
    transform(
      [
      ([:choice, :control_rule_used, Symbol(color)] =>
        ByRow((choice, rule, val) -> (!ismissing(choice) && rule == "control" && choice == color) ? val : 0) => Symbol(color))
      for color in ("blue", "green", "yellow", "red")
    ]
    )
    groupby([:participant_id, :session])
    transform([:blue, :green, :yellow, :red] .=> x -> lag(x; default=0), renamecols=false)
    groupby([:participant_id, :session])
    transform(
      [:blue, :green, :yellow, :red] .=> cumsum, renamecols=false
    )
    add_explorative_measure(_, metric="occurrence")
    rename(:explorative_val => :diff_choice_occurrence)
    select(Not([:blue, :green, :yellow, :red]))
  end

  ## Merge all features
  feature_keys = [:participant_id, :session, :trial_number]
  explore_features_df = reduce((left, right) -> leftjoin(left, right, on=feature_keys), [
    explore_choice_df,
    select(explore_by_interval, feature_keys..., :diff_interval),
    select(explore_by_choice_interval, feature_keys..., :diff_choice_interval),
    select(explore_by_occur, feature_keys..., :diff_occurrence),
    select(explore_by_choice_occur, feature_keys..., :diff_choice_occurrence)
  ])
end

#+

# ### How do these features predict choices?
begin
  ## Preprocess data for statistical modeling
  dropmissing!(explore_features_df, :response)
  transform!(explore_features_df, :response => ByRow(x -> x == "left" ? 1 : 0) => :response_left)

  ## Filter: only non-blue ship trials
  non_blue_df = filter(row -> !(row.left == "blue" || row.right == "blue"), explore_features_df)

  ## Standardize features
  transform!(non_blue_df, [:diff_interval, :diff_choice_interval, :diff_occurrence, :diff_choice_occurrence, :trial_number] .=> (x -> (x .- mean(x)) ./ std(x)) .=> (n -> Symbol(string(n), "_scaled")), renamecols=false)
end

begin
  ## GLMM for exploration choices (excluding blue ship trials), full model
  m0 = glmm(@formula(response_left ~ diff_interval_scaled + diff_choice_interval_scaled + diff_occurrence_scaled + diff_choice_occurrence_scaled + (diff_interval_scaled + diff_choice_interval_scaled + diff_occurrence_scaled + diff_choice_occurrence_scaled | participant_id)), non_blue_df, Bernoulli(), fast=false, progress=false)
end

let
  ## Visualize fixed-effect estimates and marginal predictions using Effects.jl
  coef_tbl = DataFrame(coeftable(m0))
  rename!(coef_tbl, Symbol("Coef.") => :coef, Symbol("Std. Error") => :se)
  coef_tbl.term = string.(coefnames(m0))
  coef_tbl.lower = coef_tbl.coef .- 1.96 .* coef_tbl.se
  coef_tbl.upper = coef_tbl.coef .+ 1.96 .* coef_tbl.se

  term_order = reverse(coef_tbl.term)
  order_idx = reverse(1:nrow(coef_tbl))

  predictor_labels = Dict(
    "diff_interval_scaled" => "Last seen interval (z)",
    "diff_choice_interval_scaled" => "Last controlled interval (z)",
    "diff_occurrence_scaled" => "Cumulative seen (z)",
    "diff_choice_occurrence_scaled" => "Cumulative controlled (z)",
    "(Intercept)" => "Intercept"
  )

  coef_ax_ticks = (1:length(term_order), [predictor_labels[i] for i in term_order])

  # Generate effects for all predictors using Effects.jl
  x_range = range(-2.5, 2.5; length=200)
  predictor_vars = ["diff_interval_scaled", "diff_choice_interval_scaled",
    "diff_occurrence_scaled", "diff_choice_occurrence_scaled"]

  # Compute effects for each predictor
  predictor_effects = Dict(
    var => effects(Dict(Symbol(var) => x_range), m0; invlink=logistic, level=0.95)
    for var in predictor_vars
  )

  # Extract coefficients for labels
  coef_names = string.(coefnames(m0))
  fixefs = collect(fixef(m0))
  predictor_coefs = Dict(
    var => fixefs[findfirst(==(var), coef_names)]
    for var in predictor_vars
  )

  line_colors = Dict(
    "diff_interval_scaled" => :royalblue3,
    "diff_choice_interval_scaled" => :darkorange2,
    "diff_occurrence_scaled" => :seagreen3,
    "diff_choice_occurrence_scaled" => :mediumpurple3,
  )

  # Helper function to plot marginal effects
  function plot_marginal_effects!(ax, var_names)
    for var_name in var_names
      eff = predictor_effects[var_name]
      color = line_colors[var_name]
      β = predictor_coefs[var_name]

      band!(ax, eff[!, Symbol(var_name)], eff.lower, eff.upper; color=(color, 0.2))
      lines!(ax, eff[!, Symbol(var_name)], eff.response_left;
        color=color, linewidth=3,
        label="$(predictor_labels[var_name]) (β=$(round(β, digits=3)))")
    end
  end

  m0_effect_fig = Figure(size=(1200, 300))

  # Fixed-effects coefficient plot
  ax_coef = Axis(m0_effect_fig[1, 1];
    title="Fixed-effects (log-odds)",
    xlabel="Estimate",
    yticks=coef_ax_ticks,
    yreversed=true)

  vlines!(ax_coef, [0]; color=:black, linestyle=:dash)
  xerr_lower = coef_tbl.coef[order_idx] .- coef_tbl.lower[order_idx]
  xerr_upper = coef_tbl.upper[order_idx] .- coef_tbl.coef[order_idx]
  errorbars!(ax_coef, coef_tbl.coef[order_idx], 1:length(term_order), xerr_lower, xerr_upper; direction=:x, color=:black)
  scatter!(ax_coef, coef_tbl.coef[order_idx], 1:length(term_order); color=:black)

  # Interval predictors plot
  ax_interval = Axis(m0_effect_fig[1, 3]; ylabelvisible=false)
  vlines!(ax_interval, [0]; color=(:black, 0.4), linestyle=:dash)
  ylims!(ax_interval, 0, 1)
  ax_interval.xlabel = "Predictor (z)"
  ax_interval.title = "Interval predictors"

  plot_marginal_effects!(ax_interval, ["diff_interval_scaled", "diff_choice_interval_scaled"])
  axislegend(ax_interval, position=:rb)

  # Occurrence predictors plot
  ax_occurrence = Axis(m0_effect_fig[1, 2]; ylabelvisible=false)
  vlines!(ax_occurrence, [0]; color=(:black, 0.4), linestyle=:dash)
  ylims!(ax_occurrence, 0, 1)
  ax_occurrence.xlabel = "Predictor"
  ax_occurrence.title = "Occurrence predictors"

  plot_marginal_effects!(ax_occurrence, ["diff_occurrence_scaled", "diff_choice_occurrence_scaled"])
  axislegend(ax_occurrence, position=:rb)

  m0_effect_fig
end

# ### What's the relationship between the last seen and last controlled intervals?
let
  summary_df = @chain non_blue_df begin
    groupby([:participant_id, :session, :diff_interval])
    combine(
      :diff_choice_interval => mean => :mean_diff_choice_interval
    )
    groupby([:session, :diff_interval])
    combine(
      :mean_diff_choice_interval => mean => :avg_diff_choice_interval,
      :mean_diff_choice_interval => (x -> std(x) / sqrt(length(x))) => :se_diff_choice_interval
    )
    sort(:diff_interval)
  end
  p_ind = data(non_blue_df) *
    mapping(:diff_interval, :diff_choice_interval) *
    (visual(Scatter; alpha=plot_config[:scatter_alpha]/10, markersize=plot_config[:small_markersize]))
  p_avg = data(summary_df) *
    mapping(:diff_interval, :avg_diff_choice_interval) *
    visual(Scatter; color=:darkorange, markersize=plot_config[:medium_markersize])
  draw(p_ind + p_avg; axis=(; xlabel="Last seen interval difference", ylabel="Last controlled interval difference"))
end

@time begin 
  m0_1 = glmm(@formula(response_left ~ diff_interval * diff_choice_interval + (diff_interval * diff_choice_interval | participant_id)), non_blue_df, Bernoulli(), fast=false, progress=false; contrasts = Dict(:diff_interval => ZScore(), :diff_choice_interval => ZScore()))
end

# ### Let's revise the model a bit to include the interaction
# Apparently they're correlated, and there is variance of controlled interval within each level of seen interval
let
  using RCall, JellyMe4
  @rput non_blue_df
  R"""
    library(tidyverse)
    library(afex)
    library(emmeans)
    set_sum_contrasts()
    m0_1_r <- glmer(response_left ~ diff_interval_scaled * diff_choice_interval_scaled + (diff_interval_scaled * diff_choice_interval_scaled | participant_id),
      data=non_blue_df,
      family=binomial, control=glmerControl(optimizer="bobyqa"))
    joint_tests(m0_1_r)
  """
  
  p_intervals = R"""
    theme_set(theme_light())
    theme_update(legend.position = "bottom", legend.direction = "horizontal")
    emmip(m0_1_r, diff_choice_interval_scaled ~ diff_interval_scaled, CIs = T, plotit = F, type = 'response', at = list(diff_choice_interval_scaled = c(-1, 0, 1), diff_interval_scaled = seq(-3, 3, by = 0.05))) |>
    ggplot() +
    geom_ribbon(aes(ymin = LCL, ymax = UCL, x = diff_interval_scaled, fill = factor(diff_choice_interval_scaled, ordered = T)), alpha = 0.1) +
    geom_line(aes(x = diff_interval_scaled, y = yvar, color = factor(diff_choice_interval_scaled, ordered = T)), size = 1) +
    labs(x = "Last seen interval (z)", y = "Predicted P[Left]", color = "Last controlled interval (z)", fill = "Last controlled interval (z)") +
    scale_color_brewer(palette = "Oranges", aesthetics = c("color", "fill"))
  """
  gg_show_save(p_intervals, "p_intervals_interact.pdf", width=4, height=3)
end

# ### Does the effect of these features change over time?
@time begin
  m1 = glmm(@formula(response_left ~ trial_number * (diff_interval * diff_choice_interval) + (trial_number * (diff_interval * diff_choice_interval) | participant_id)), non_blue_df, Bernoulli(), fast=false, progress=false; contrasts=Dict(:trial_number => ZScore(), :diff_interval => ZScore(), :diff_choice_interval => ZScore()))
end

let
  R"""
  m1_r <- glmer(response_left ~ trial_number_scaled * (diff_interval_scaled * diff_choice_interval_scaled) + (trial_number_scaled * (diff_interval_scaled * diff_choice_interval_scaled) | participant_id), non_blue_df, family = binomial, control=glmerControl(optimizer="nloptwrap", optCtrl = list(algorithm = "NLOPT_LN_BOBYQA"), calc.derivs = FALSE))
  joint_tests(m1_r)
  """

  p_time_interval_interact = R"""
  emmip(m1_r, trial_number_scaled ~ diff_interval_scaled, CIs = T, plotit = F, type = 'response', at = list(trial_number_scaled = c(-1, 0, 1), diff_interval_scaled = seq(-3, 3, by = 0.05))) |>
    ggplot() +
    geom_ribbon(aes(ymin = LCL, ymax = UCL, x = diff_interval_scaled, fill = factor(trial_number_scaled, ordered = T)), alpha = 0.1) +
    geom_line(aes(x = diff_interval_scaled, y = yvar, color = factor(trial_number_scaled, ordered = T)), size = 1) +
    labs(x = "Last seen interval (z)", y = "Predicted P[Left]", color = "Trial number (z)", fill = "Trial number (z)") +
    scale_color_brewer(palette = "Oranges", aesthetics = c("color", "fill"))
  """
  gg_show_save(p_time_interval_interact, "p_time_interval_interact.pdf", width=4, height=3)
end

# ### Does the effect of these features differ between required efforts?

# Julia version
let
  @time m2 = glmm(@formula(response_left ~ current * (diff_interval * diff_choice_interval) + (current * (diff_interval * diff_choice_interval) | participant_id)), non_blue_df, Bernoulli(), contrasts=Dict(:current => EffectsCoding(), :diff_interval => ZScore(), :diff_choice_interval => ZScore()), fast=false, progress=false)

  plot_effort_interaction(m2, non_blue_df, "diff_interval_scaled";
    xlabel="zSeen Interval[Left] - zSeen Interval[Right]")
  
  plot_effort_interaction(m2, non_blue_df, "diff_choice_interval_scaled";
    xlabel="zControlled Interval[Left] - zControlled Interval[Right]")
end

let
  @time R"""
  m2_r <- glmer(response_left ~ current * (diff_interval_scaled * diff_choice_interval_scaled) + (current * (diff_interval_scaled * diff_choice_interval_scaled) | participant_id), non_blue_df, family = binomial, control=glmerControl(optimizer="nloptwrap", optCtrl = list(algorithm = "NLOPT_LN_BOBYQA"), calc.derivs = FALSE))
  joint_tests(m2_r)
  """

  p_effort_seen_interval_interact = R"""
  emmip(m2_r, current ~ diff_interval_scaled, CIs = T, plotit = F, type = 'response', at = list(diff_interval_scaled = seq(-3, 3, by = 0.05))) |>
    ggplot() +
    geom_ribbon(aes(ymin = LCL, ymax = UCL, x = diff_interval_scaled, fill = factor(current, ordered = T)), alpha = 0.1) +
    geom_line(aes(x = diff_interval_scaled, y = yvar, color = factor(current, ordered = T)), size = 1) +
    labs(x = "Last seen interval (z)", y = "Predicted P[Left]", color = "Required effort level", fill = "Required effort level") +
    scale_color_brewer(palette = "Oranges", aesthetics = c("color", "fill"))
  """
  gg_show_save(p_effort_seen_interval_interact, "p_effort_seen_interval_interact.pdf", width=4, height=3)

  p_effort_choice_interval_interact = R"""
  emmip(m2_r, current ~ diff_choice_interval_scaled, CIs = T, plotit = F, type = 'response', at = list(diff_choice_interval_scaled = seq(-3, 3, by = 0.05))) |>
    ggplot() +
    geom_ribbon(aes(ymin = LCL, ymax = UCL, x = diff_choice_interval_scaled, fill = factor(current, ordered = T)), alpha = 0.1) +
    geom_line(aes(x = diff_choice_interval_scaled, y = yvar, color = factor(current, ordered = T)), size = 1) +
    labs(x = "Last controlled interval (z)", y = "Predicted P[Left]", color = "Required effort level", fill = "Required effort level") +
    scale_color_brewer(palette = "Oranges", aesthetics = c("color", "fill"))
  """
  gg_show_save(p_effort_choice_interval_interact, "p_effort_choice_interval_interact.pdf", width=4, height=3)
end

# ### Does trial number also modulate the required effort and interval interaction effects?
@time begin
  m3 = glmm(@formula(response_left ~ trial_number * current * (diff_interval * diff_choice_interval) + (trial_number * current * (diff_interval * diff_choice_interval) | participant_id)), non_blue_df, Bernoulli(), fast=false, progress=false; contrasts=Dict(:trial_number => ZScore(), :current => EffectsCoding(), :diff_interval => ZScore(), :diff_choice_interval => ZScore()))
end

# ### Color preference during exploration, over time
begin
  color_choice_df = @chain explore_choice_df begin
    dropmissing(:choice)
    transform(
      [
      ([:choice, :control_rule_used, Symbol(color)] =>
        ByRow((choice, rule, val) -> (!ismissing(choice) && choice == color) ? val : missing) => Symbol(color))
      for color in ("blue", "green", "yellow", "red")
    ]
    )
    select([:participant_id, :session, :trial_number, :blue, :green, :yellow, :red])
    stack([:blue, :green, :yellow, :red], variable_name=:color, value_name=:count)
    groupby([:session, :trial_number, :color])
    combine(:count => (x -> sum(skipmissing(x))) => :count)
    groupby([:session, :trial_number])
    transform(:count => (x -> x ./ sum(skipmissing(x))) => :prop)
    filter(x -> x.prop .> 0, _)
  end

  data(color_choice_df) *
  mapping(:trial_number, :prop, color=:color, group=:color, row=:session) *
  (visual(Scatter; alpha=0.3) + linear()) |>
  draw(scales(Color=(; palette=["blue" => "royalblue", "green" => "forestgreen", "yellow" => "goldenrod1", "red" => "salmon3"])))
end

## Is there a color preference during exploration?
begin
  color_choice_count = @chain explore_choice_df begin
    dropmissing(:choice)
    @count(participant_id, session, choice)
    groupby([:participant_id, :session])
    transform(:n => (x -> x ./ sum(x)) => :prop)
    filter(row -> row.session == "1", _)
  end

  m_full = lmm(@formula(prop ~ choice + (choice | participant_id)),
    color_choice_count,
    contrasts=Dict(:choice => DummyCoding(base="blue")))

  m_null = lmm(@formula(prop ~ 1 + (choice | participant_id)),
    color_choice_count,
    contrasts=Dict(:choice => DummyCoding(base="blue")))

  lrt = lrtest(m_null, m_full)
  println(lrt)
end

let
  data(color_choice_count) *
  mapping(:choice => "Color", :prop => "Proportion", color=:choice => "Color") *
  visual(RainClouds) |>
  draw(scales(Color=(; palette=["blue" => "royalblue", "green" => "forestgreen", "yellow" => "goldenrod1", "red" => "firebrick3"])))
end

# ### What about blue ship where they have known it in advance?
begin
  blue_choice_df = @chain explore_choice_df begin
    filter(row -> row.left == "blue" || row.right == "blue", _)
    transform(:choice => ByRow(x -> !ismissing(x) && x == "blue" ? 1 : 0) => :blue_choice)
    filter(row -> row.session == "1", _)
  end

  glmm(@formula(blue_choice ~ trial_number + (trial_number | participant_id)), blue_choice_df, Bernoulli(), contrasts=Dict(:trial_number => StandardizedPredictors.ZScore()), fast=false, progress=false)
end

let
  data(blue_choice_df) *
  mapping(:trial_number, :blue_choice) *
  (mapping(group=:participant_id) * visual(Lines; alpha=0.01) + linear()) |>
  draw(axis=(; ylabel="P[Blue]", xlabel="Trial number"))
end

# ### When people choose the known choice, how much effort do they choose to exert?
begin
  @chain explore_choice_df begin
    transform(:choice => ByRow(x -> !ismissing(x) && x == "blue" ? 1 : 0) => :blue_choice)
    transform(:control_rule_used => ByRow(x -> !ismissing(x) && x == "control" ? 1 : 0) => :control_choice)
    groupby([:participant_id, :session, :current, :blue_choice])
    combine(:control_choice => mean => :prop_control)
    groupby([:session, :current, :blue_choice])
    combine(:prop_control => mean => :prop_control)
    sort([:session, :current, :blue_choice])
  end

  control_blue_df = @chain explore_choice_df begin
    filter(row -> row.session == "1", _)
    filter(row -> row.left == "blue" || row.right == "blue", _)
    transform(:choice => ByRow(x -> !ismissing(x) && x == "blue" ? "blue" : "other") => :blue_choice)
    transform(:control_rule_used => ByRow(x -> !ismissing(x) && x == "control" ? 1 : 0) => :control_choice)
  end

  m_blue_control = glmm(@formula(control_choice ~ blue_choice * current + (blue_choice * current | participant_id)), control_blue_df, Bernoulli(), contrasts=Dict(:current => EffectsCoding(), :blue_choice => EffectsCoding(base="other")), fast=false, progress=false)
end

let
  eff_blue_control = effects(Dict(:current => unique(non_blue_df.current),
      :blue_choice => ["blue", "other"]),
    m_blue_control; invlink=logistic)
  eff_blue_control.current = categorical(eff_blue_control.current; levels=[1, 2, 3], ordered=true)
  data(eff_blue_control) *
  (
    mapping(:current, :control_choice, color=:blue_choice => "Choice", group=:blue_choice => "Choice") * (visual(Scatter) + mapping(:err) * visual(Errorbars)
    )) |>
  draw(axis=(; xlabel="Current", ylabel="P[Control]"))
end

# ### How close participants are to the required threshold to achieve control?
let
  effort_to_cutfoff_df = @chain explore_choice_df begin
    transform(:current => ByRow(function case_match(x)
      x == 1 && return 6
      x == 2 && return 12
      x == 3 && return 18
    end) => :control_cutoff)
    transform([:trial_presses, :control_cutoff] => ((x, y) -> x .- y) => :effort_to_cutoff)
  end

  p_effort_to_cutoff = Figure(size=(800, 600))

  p_ind = data(effort_to_cutfoff_df) *
          mapping(:trial_number, :effort_to_cutoff, group=:participant_id, col=:session, row=:current) *
          visual(Lines; alpha=0.05)

  p_avg = @chain effort_to_cutfoff_df begin
    groupby([:session, :current, :trial_number])
    combine(:effort_to_cutoff => mean => :mean_effort_to_cutoff)
    data(_) *
    mapping(:trial_number, :mean_effort_to_cutoff, col=:session, row=:current) *
    (visual(Lines; color=:darkorange, linewidth=2, linestyle=:dash))
  end

  reflines = mapping([0]) * visual(HLines; color=:purple, linestyle=:dot, linewidth=2)

  draw!(p_effort_to_cutoff, p_ind + p_avg + reflines, scales(Col=(; categories=["1" => "Session 1", "2" => "Session 2"])); axis=(; ylabel="Actual effort - Required", xlabel="Trial number"))

  p_effort_to_cutoff
end


#=
## How can exposure history predict participants' accuracy during prediction?

- Do cumulative counts of seeing and controlling a ship predict prediction accuracy?
- Do the same features explain confidence ratings, after accounting for experiment progress?
=#

# ### Do ship exposure counts forecast prediction accuracy?
begin
  colors = ["blue", "green", "red", "yellow"]

  function compute_prediction_features(task_df)
    feature_rows = NamedTuple{(:participant_id, :session, :trial_index, :ship, :seen_count, :controlled_count, :prediction_idx),
      Tuple{String,String,Int64,String,Int64,Int64,Int64}}[]

    for group_df in groupby(task_df, [:participant_id, :session])
      sdf = sort(group_df, :trial_index)
      pid = String(sdf.participant_id[1])
      sess = String(sdf.session[1])

      seen_counts = Dict{String,Int}(color => 0 for color in colors)
      control_counts = Dict{String,Int}(color => 0 for color in colors)
      prediction_counter = 0

      for row in eachrow(sdf)
        trial_idx = row.trial_index
        if ismissing(trial_idx)
          continue
        end
        trial_idx = Int(trial_idx)
        phase = row.trialphase

        if phase == "control_predict_homebase"
          ship_color = row.ship
          if !ismissing(ship_color)
            ship_str = String(ship_color)
            prediction_counter += 1
            push!(feature_rows, (participant_id=pid, session=sess, trial_index=trial_idx, ship=ship_str,
              seen_count=seen_counts[ship_str], controlled_count=control_counts[ship_str], prediction_idx=prediction_counter))
          end
        elseif phase == "control_explore"
          left_color = row.left
          right_color = row.right
          if !ismissing(left_color)
            seen_counts[String(left_color)] += 1
          end
          if !ismissing(right_color)
            seen_counts[String(right_color)] += 1
          end
          resp = row.response
          if resp isa String && (resp == "left" || resp == "right")
            chosen_color = row[Symbol(resp)]
            if !ismissing(chosen_color)
              chosen_str = String(chosen_color)
              if row.control_rule_used == "control"
                control_counts[chosen_str] += 1
              end
            end
          end
        end
      end
    end

    return DataFrame(feature_rows)
  end

  prediction_trials = @chain control_task_data begin
    filter(row -> row.trialphase == "control_predict_homebase", _)
    subset(:response => ByRow(x -> !ismissing(x)))
    select(:participant_id, :session, :trial_index, :trial, :ship, :correct)
    transform(:correct => ByRow(x -> x ? 1 : 0) => :correct_int)
  end

  relevant_task_rows = filter(row -> row.trialphase in ("control_explore", "control_predict_homebase"), control_task_data)
  prediction_features = compute_prediction_features(relevant_task_rows)

  prediction_trials = leftjoin(prediction_trials, prediction_features, on=[:participant_id, :session, :trial_index, :ship])

  confidence_df = @chain control_report_data begin
    filter(row -> row.trialphase == "control_confidence", _)
    select(:participant_id, :session, :trial, :response)
    rename(:response => :confidence)
  end

  prediction_trials = leftjoin(prediction_trials, confidence_df, on=[:participant_id, :session, :trial])

  function zscore_with_missing(vec)
    values = collect(skipmissing(vec))
    if isempty(values)
      return fill(missing, length(vec))
    end
    μ = mean(values)
    σ = std(values)
    if σ == 0
      return map(v -> ismissing(v) ? missing : 0.0, vec)
    end
    return map(v -> ismissing(v) ? missing : (v - μ) / σ, vec)
  end

  transform!(prediction_trials,
    [:seen_count, :controlled_count, :prediction_idx] .=> zscore_with_missing .=>
      [:seen_count_z, :controlled_count_z, :prediction_idx_z], renamecols=false)

  prediction_accuracy_df = dropmissing(prediction_trials, [:seen_count_z, :controlled_count_z, :prediction_idx_z])
  println("Prediction accuracy sample size: ", nrow(prediction_accuracy_df))
  m_accuracy = glmm(@formula(correct_int ~ seen_count_z + controlled_count_z + prediction_idx_z + (seen_count_z + controlled_count_z + prediction_idx_z | participant_id)), prediction_accuracy_df, Bernoulli(), fast=false, progress=false)
  println("Prediction accuracy model coefficients:")
  println(coeftable(m_accuracy))

  prediction_confidence_df = dropmissing(prediction_trials, [:seen_count_z, :controlled_count_z, :prediction_idx_z, :confidence])
  if !isempty(prediction_confidence_df)
    println("Prediction confidence sample size: ", nrow(prediction_confidence_df))
    transform!(prediction_confidence_df, :confidence => ByRow(Float64) => :confidence_float)
    m_confidence = lmm(@formula(confidence_float ~ seen_count_z + controlled_count_z + prediction_idx_z + (seen_count_z + controlled_count_z + prediction_idx_z | participant_id)), prediction_confidence_df)
    println("Prediction confidence model coefficients:")
    println(coeftable(m_confidence))
  end
end
