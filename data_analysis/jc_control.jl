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
  cd("/home/jovyan/")
  import Pkg
  ## activate the shared project environment
  Pkg.activate("relmed_environment")
  ## instantiate, i.e. make sure that all packages are downloaded
  Pkg.instantiate()
  using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, SHA
	using Tidier, GLM, MixedModels, PlutoUI, LaTeXStrings, ColorSchemes, HypothesisTests, ShiftedArrays
	using LogExpFunctions: logistic, logit
  include(joinpath(pwd(), "fetch_preprocess_data.jl"))
	include(joinpath(pwd(), "plotting_utils.jl"))
  include(joinpath(pwd(), "data_analysis", "control_exploration_fn.jl"))
  nothing
end

begin
  ## Set theme
  th = Theme(
    font="Helvetica",
    fontsize=16,
    Axis=(
      xgridvisible=false,
      ygridvisible=false,
      rightspinevisible=false,
      topspinevisible=false,
      xticklabelsize=14,
      yticklabelsize=14,
      spinewidth=1.5,
      xtickwidth=1.5,
      ytickwidth=1.5
    )
  )

  set_theme!(merge(theme_minimal(), th))
end

#+ Load data

begin
  _, _, _, _, _, raw_control_task_data, raw_control_report_data, jspsych_data = load_pilot9_data(; force_download=false)
  p_sum = summarize_participation(jspsych_data)
  p_no_double_take = exclude_double_takers(p_sum) |>
    x -> filter(x -> !ismissing(x.finished) && x.finished, x)
  nothing
end

begin
  control_task_data = semijoin(raw_control_task_data, p_no_double_take, on=:record_id)
  control_report_data = semijoin(raw_control_report_data, p_no_double_take, on=:record_id)
  @assert all(combine(groupby(control_task_data, :record_id), [:time_elapsed, :trial] => ((x, y) -> all(denserank(x) .== denserank(y))) => :in_order)[:, "in_order"]) "Trial numbers are created *incorrectly* in chronological order"
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
    select([:prolific_pid, :session, :trial, :left, :right, :response, :control_rule_used])
    transform([:left, :right, :response] => ByRow((left, right, resp) -> ismissing(resp) ? missing : ifelse(resp == "left", left, right)) => :choice)
    groupby([:prolific_pid, :session])
    DataFrames.transform(
      :trial => (x -> 1:length(x)) => :trial_number
    )
    add_ship_onehot(_)
    subset(:prolific_pid => ByRow(x -> x !==("670cf1a20d1fa15c58a175f7")))
  end

  ## 1. Interval from last seen
  explore_by_interval = @chain explore_choice_df begin
      groupby([:prolific_pid, :session])
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
      groupby([:prolific_pid, :session])
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
    groupby([:prolific_pid, :session])
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
    groupby([:prolific_pid, :session])
    transform([:blue, :green, :yellow, :red] .=> x -> lag(x; default = 0), renamecols=false)
    groupby([:prolific_pid, :session])
    transform(
        [:blue, :green, :yellow, :red] .=> cumsum, renamecols=false
      )
    add_explorative_measure(_, metric="occurrence")
    rename(:explorative_val => :diff_choice_occurrence)
    select(Not([:blue, :green, :yellow, :red]))
  end

  ## Merge all features
  feature_keys = [:prolific_pid, :session, :trial_number]
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
let
  ## Preprocess data for statistical modeling
  dropmissing!(explore_features_df, :response)
  transform!(explore_features_df, :response => ByRow(x -> x == "left" ? 1 : 0) => :response_left)
  transform!(explore_features_df, [:diff_interval, :diff_choice_interval, :diff_occurrence, :diff_choice_occurrence] .=> (x -> (x .- mean(x))./std(x)) .=> (n -> Symbol(string(n), "_scaled")), renamecols=false)

  ## Filter: 1. only the first session
  df = filter(row -> row.session == "1", explore_features_df)

  ## Filter: 2. only non-blue ship trials
  df = filter(row -> !(row.left == "blue" || row.right == "blue"), df)

  ## GLMM for exploration choices (excluding blue ship trials)
  glmm(@formula(response_left ~ diff_interval_scaled + diff_choice_interval_scaled + diff_occurrence_scaled + diff_choice_occurrence_scaled + (diff_interval_scaled + diff_choice_interval_scaled + diff_occurrence_scaled + diff_choice_occurrence_scaled | prolific_pid)), df, Bernoulli(), contrasts=Dict(:session => EffectsCoding()), fast=false, progress=false)
end
