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
  cd("/home/jovyan") # Keep this if you need the root CWD
  
  # 1. Define where your script is and where the modules are
  script_dir = dirname(@__FILE__) # Gets ".../projects/control-task-analysis/scripts"
  module_dir = joinpath(script_dir, "common")

  # 2. Add the 'common' folder to Julia's search path
  if !(module_dir in LOAD_PATH)
      push!(LOAD_PATH, module_dir)
  end

  # 3. Load standard packages
  using DataFrames, TidierData, CairoMakie, AlgebraOfGraphics
  using Dates, CategoricalArrays, StatsBase, GLM, MixedModels, Random
  using LogExpFunctions: logistic, logit
  using StandardizedPredictors, Effects
  using Revise

  # 4. Now 'using' will find your files in 'common'
  # Note: The file names must match the module names (e.g., ControlUtils.jl defines module ControlUtils)
  using ControlUtils
  using ControlPlots
  using ControlAnalysis

  # Include other legacy scripts if needed
  include("$(pwd())/core/experiment-registry.jl")
  include("$(pwd())/core/preprocess_data.jl")
  include(joinpath(script_dir, "common", "config.jl")) 

  experiment = NORMING
end

#+ Load data

# Load and preprocess data
begin
  dat = preprocess_project(experiment; force_download=true, delay_ms=65, use_manual_download=false)
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
  explore_features_df = build_exploration_features(control_task_data)
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
  transform!(non_blue_df, [:diff_interval, :diff_choice_interval, :diff_occurrence, :diff_choice_occurrence, :trial_number] .=> zscore_with_missing .=> (n -> Symbol(string(n), "_z")), renamecols=false)
end

begin
  ## GLMM for exploration choices (excluding blue ship trials), full model
  m0 = glmm(@formula(response_left ~ diff_interval_z + diff_choice_interval_z + diff_occurrence_z + diff_choice_occurrence_z + (diff_interval_z + diff_choice_interval_z + diff_occurrence_z + diff_choice_occurrence_z | participant_id)), non_blue_df, Bernoulli(), fast=false, progress=false)
end

let
  predictor_vars = ["diff_interval_z", "diff_choice_interval_z",
    "diff_occurrence_z", "diff_choice_occurrence_z"]

  predictor_labels = Dict(
    "diff_interval_z" => "Last seen interval (z)",
    "diff_choice_interval_z" => "Last controlled interval (z)",
    "diff_occurrence_z" => "Cumulative seen (z)",
    "diff_choice_occurrence_z" => "Cumulative controlled (z)",
    "(Intercept)" => "Intercept"
  )

  predictor_colors = Dict(
    "diff_interval_z" => :royalblue3,
    "diff_choice_interval_z" => :darkorange2,
    "diff_occurrence_z" => :seagreen3,
    "diff_choice_occurrence_z" => :mediumpurple3,
  )

  plot_glmm_effects(m0, predictor_vars; predictor_labels, predictor_colors)
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
    m0_1_r <- glmer(response_left ~ diff_interval_z * diff_choice_interval_z + (diff_interval_z * diff_choice_interval_z | participant_id),
      data=non_blue_df,
      family=binomial, control=glmerControl(optimizer="bobyqa"))
    joint_tests(m0_1_r)
  """
  
  p_intervals = R"""
    theme_set(theme_light())
    theme_update(legend.position = "bottom", legend.direction = "horizontal")
    emmip(m0_1_r, diff_choice_interval_z ~ diff_interval_z, CIs = T, plotit = F, type = 'response', at = list(diff_choice_interval_z = c(-1, 0, 1), diff_interval_z = seq(-3, 3, by = 0.05))) |>
    ggplot() +
    geom_ribbon(aes(ymin = LCL, ymax = UCL, x = diff_interval_z, fill = factor(diff_choice_interval_z, ordered = T)), alpha = 0.1) +
    geom_line(aes(x = diff_interval_z, y = yvar, color = factor(diff_choice_interval_z, ordered = T)), size = 1) +
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
  m1_r <- glmer(response_left ~ trial_number_z * (diff_interval_z * diff_choice_interval_z) + (trial_number_z * (diff_interval_z * diff_choice_interval_z) | participant_id), non_blue_df, family = binomial, control=glmerControl(optimizer="nloptwrap", optCtrl = list(algorithm = "NLOPT_LN_BOBYQA"), calc.derivs = FALSE))
  joint_tests(m1_r)
  """

  p_time_interval_interact = R"""
  emmip(m1_r, trial_number_z ~ diff_interval_z, CIs = T, plotit = F, type = 'response', at = list(trial_number_z = c(-1, 0, 1), diff_interval_z = seq(-3, 3, by = 0.05))) |>
    ggplot() +
    geom_ribbon(aes(ymin = LCL, ymax = UCL, x = diff_interval_z, fill = factor(trial_number_z, ordered = T)), alpha = 0.1) +
    geom_line(aes(x = diff_interval_z, y = yvar, color = factor(trial_number_z, ordered = T)), size = 1) +
    labs(x = "Last seen interval (z)", y = "Predicted P[Left]", color = "Trial number (z)", fill = "Trial number (z)") +
    scale_color_brewer(palette = "Oranges", aesthetics = c("color", "fill"))
  """
  gg_show_save(p_time_interval_interact, "p_time_interval_interact.pdf", width=4, height=3)
end

# ### Does the effect of these features differ between required efforts?

# Julia version
let
  @time m2 = glmm(@formula(response_left ~ current * (diff_interval * diff_choice_interval) + (current * (diff_interval * diff_choice_interval) | participant_id)), non_blue_df, Bernoulli(), contrasts=Dict(:current => EffectsCoding(), :diff_interval => ZScore(), :diff_choice_interval => ZScore()), fast=false, progress=false)

  plot_effort_interaction(m2, non_blue_df, "diff_interval_z";
    xlabel="zSeen Interval[Left] - zSeen Interval[Right]")
  
  plot_effort_interaction(m2, non_blue_df, "diff_choice_interval_z";
    xlabel="zControlled Interval[Left] - zControlled Interval[Right]")
end

let
  @time R"""
  m2_r <- glmer(response_left ~ current * (diff_interval_z * diff_choice_interval_z) + (current * (diff_interval_z * diff_choice_interval_z) | participant_id), non_blue_df, family = binomial, control=glmerControl(optimizer="nloptwrap", optCtrl = list(algorithm = "NLOPT_LN_BOBYQA"), calc.derivs = FALSE))
  joint_tests(m2_r)
  """

  p_effort_seen_interval_interact = R"""
  emmip(m2_r, current ~ diff_interval_z, CIs = T, plotit = F, type = 'response', at = list(diff_interval_z = seq(-3, 3, by = 0.05))) |>
    ggplot() +
    geom_ribbon(aes(ymin = LCL, ymax = UCL, x = diff_interval_z, fill = factor(current, ordered = T)), alpha = 0.1) +
    geom_line(aes(x = diff_interval_z, y = yvar, color = factor(current, ordered = T)), size = 1) +
    labs(x = "Last seen interval (z)", y = "Predicted P[Left]", color = "Required effort level", fill = "Required effort level") +
    scale_color_brewer(palette = "Oranges", aesthetics = c("color", "fill"))
  """
  gg_show_save(p_effort_seen_interval_interact, "p_effort_seen_interval_interact.pdf", width=4, height=3)

  p_effort_choice_interval_interact = R"""
  emmip(m2_r, current ~ diff_choice_interval_z, CIs = T, plotit = F, type = 'response', at = list(diff_choice_interval_z = seq(-3, 3, by = 0.05))) |>
    ggplot() +
    geom_ribbon(aes(ymin = LCL, ymax = UCL, x = diff_choice_interval_z, fill = factor(current, ordered = T)), alpha = 0.1) +
    geom_line(aes(x = diff_choice_interval_z, y = yvar, color = factor(current, ordered = T)), size = 1) +
    labs(x = "Last controlled interval (z)", y = "Predicted P[Left]", color = "Required effort level", fill = "Required effort level") +
    scale_color_brewer(palette = "Oranges", aesthetics = c("color", "fill"))
  """
  gg_show_save(p_effort_choice_interval_interact, "p_effort_choice_interval_interact.pdf", width=4, height=3)
end

# ### Does trial number also modulate the required effort and interval interaction effects?
@time begin
  m3 = glmm(@formula(response_left ~ trial_number * current * diff_interval + (trial_number * current * diff_interval | participant_id)), non_blue_df, Bernoulli(), fast=true, progress=false; contrasts=Dict(:trial_number => ZScore(), :current => EffectsCoding(), :diff_interval => ZScore(), :diff_choice_interval => ZScore()))
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
