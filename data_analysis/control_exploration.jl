begin
  cd("/home/jovyan/")
  import Pkg
  # activate the shared project environment
  Pkg.activate("relmed_environment")
  # instantiate, i.e. make sure that all packages are downloaded
  Pkg.instantiate()
  using Revise
  using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, SHA
	using Tidier, GLM, MixedModels, PlutoUI, LaTeXStrings, ColorSchemes, HypothesisTests, ShiftedArrays
	using LogExpFunctions: logistic, logit
  include(joinpath(pwd(), "fetch_preprocess_data.jl"))
	include(joinpath(pwd(), "plotting_utils.jl"))
  includet(joinpath(pwd(), "data_analysis", "control_exploration_fn.jl"))
  nothing
end

begin
  # Set theme
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

explore_choice_df = @chain control_task_data begin
  filter(x -> x.trialphase .== "control_explore", _)
  select([:prolific_pid, :session, :trial, :left, :right, :response, :control_rule_used])
  groupby([:prolific_pid, :session])
  DataFrames.transform(
    :trial => (x -> 1:length(x)) => :trial_number
  )
  add_ship_onehot(_)
  subset(:prolific_pid => ByRow(x -> x !==("670cf1a20d1fa15c58a175f7")))
end

# Summarize choices by the number of occurrence
explore_by_occur = @chain explore_choice_df begin
  groupby([:prolific_pid, :session])
  transform(
    [:blue, :green, :yellow, :red] .=> cumsum, renamecols=false
  )
  add_explorative_response(_, metric="occurrence")
  add_explorative_measure(_, metric="occurrence")
  transform(:response => ByRow(passmissing(x -> x == "left" ? 1 : 0)) => :response_left)
end

plot_explore_trend(
  explore_by_occur;
  xlabel="Trial",
  ylabel="P(Explorative)",
  xcol=:trial_number,
  ycol=:explorative_cat,
  title="Exploration Trend by # Occurrence",
  caption=""
)

## Test-retest reliability of exploration by occurrence (explorative trend)
plot_explore_trend_reliability(
  explore_by_occur;
  title = "Test-retest: Exploration by # Occurrence",
  caption = "Explore[0/0.5/1 | # Occur] ~ 1 + Trial",
)

plot_choice_pred_curves(
  explore_by_occur;
  title = "Choice Prediction Curves by # Occurrence Difference",
  xlabel = "Left(#Occur) - Right(#Occur)"
)

plot_bias_and_sensitivity_reliability(
  explore_by_occur;
  title = "Test-retest: Exploration by # Occurrence Difference",
)

# Summarize choices by the length of the interval
explore_by_interval = @chain explore_choice_df begin
    groupby([:prolific_pid, :session])
    transform(
      [:trial_number, :blue]   => ((t, x) -> calc_trial_interval(t, x)) => :blue,
      [:trial_number, :green]  => ((t, x) -> calc_trial_interval(t, x)) => :green,
      [:trial_number, :yellow] => ((t, x) -> calc_trial_interval(t, x)) => :yellow,
      [:trial_number, :red]    => ((t, x) -> calc_trial_interval(t, x)) => :red
    )
    add_explorative_response(_, metric="interval")
    add_explorative_measure(_, metric="interval")
    transform(:response => ByRow(passmissing(x -> x == "left" ? 1 : 0)) => :response_left)
end

plot_explore_trend(
  explore_by_interval;
  xlabel="Trial",
  ylabel="P(Explorative)",
  xcol=:trial_number,
  ycol=:explorative_cat,
  title="Exploration Trend by # Interval",
  caption=""
)

## Test-retest reliability of exploration by interval (explorative trend)
plot_explore_trend_reliability(
  explore_by_interval;
  title = "Test-retest: Exploration by # Interval",
  caption = "Explore[0/0.5/1 | # Interval] ~ 1 + Trial",
)

plot_choice_pred_curves(
  explore_by_interval;
  title = "Choice Prediction Curves by # Interval Difference",
  xlabel = "Left(#Interval) - Right(#Interval)"
)

plot_bias_and_sensitivity_reliability(
  explore_by_interval;
  title = "Test-retest: Exploration by # Interval Difference",
)

# Are these two measures of exploration correlated?
let
  x_df = select(explore_by_occur, :prolific_pid, :session, :trial_number, :explorative_cat => :explorative_occur)
  y_df = select(explore_by_interval, :prolific_pid, :session, :trial_number, :explorative_cat => :explorative_interval)

  # Combine the two dataframes
  combined_df = dropmissing(innerjoin(x_df, y_df, on=[:prolific_pid, :session, :trial_number]))

  # Average the explorative measures across trials for each participant and session
  combined_df = @chain combined_df begin
    transform(
      :trial_number => (x -> x .- mean(x)) => :trial_number
    )
    groupby([:prolific_pid, :session])
    combine(
      AsTable([:trial_number, :explorative_occur]) => (x -> [coef(lm(@formula(explorative_occur ~ trial_number), x))]) => [:β0_occur, :β_trial_occur],
      AsTable([:trial_number, :explorative_interval]) => (x -> [coef(lm(@formula(explorative_interval ~ trial_number), x))]) => [:β0_interval, :β_trial_interval]
    )
  end

  # Calculate correlation of avg exploration within each session
  session_avg_corrs = combine(groupby(combined_df, :session)) do df
    corr_test = CorrelationTest(df.β0_occur, df.β0_interval)
    DataFrame(
      correlation = corr_test.r,
      p_value = pvalue(corr_test),
      n = nrow(df)
    )
  end

  println("Correlations of avg exploration by session:")
  println(session_avg_corrs)

  # Calculate correlation of exploration trend within each session
  session_trend_corrs = combine(groupby(combined_df, :session)) do df
    corr_test = CorrelationTest(df.β_trial_occur, df.β_trial_interval)
    DataFrame(
      correlation = corr_test.r,
      p_value = pvalue(corr_test),
      n = nrow(df)
    )
  end

  println("Correlations of exploration trend by session:")
  println(session_trend_corrs)
end

# Summarize choices by the number of controlled choices
explore_nchoice_df = @chain explore_choice_df begin
  groupby([:prolific_pid, :session])
  transform(
    [:response, :left, :right] .=> lead => x -> "next_" * x,
  )
  transform(
    [:response, :control_rule_used, :left, :right, :blue, :green, :yellow, :red] => 
    ByRow((resp, rule, left, right, blue, green, yellow, red) -> begin
      if ismissing(resp)
        return (blue=0, green=0, yellow=0, red=0)
      end
      
      chosen_color = resp == "left" ? left : right
      controlled = !ismissing(rule) && rule == "control"

      return (
        blue = controlled && chosen_color == "blue" ? blue : 0,
        green = controlled && chosen_color == "green" ? green : 0,
        yellow = controlled && chosen_color == "yellow" ? yellow : 0,
        red = controlled && chosen_color == "red" ? red : 0
      )
    end) => [:blue, :green, :yellow, :red]
  )
  groupby([:prolific_pid, :session])
  transform([:blue, :green, :yellow, :red] .=> cumsum .=> [:blue, :green, :yellow, :red])
  add_explorative_response(_, metric="control")
  add_explorative_measure(_, metric="control")
  transform(:next_response => ByRow(passmissing(x -> x == "left" ? 1 : 0)) => :next_response_left)
  transform([:next_left, :next_right] => ByRow(passmissing((x, y) -> !(x == "blue" || y == "blue"))) => :has_no_blue)
end

plot_explore_trend(
  subset(explore_nchoice_df, :has_no_blue => ByRow(x -> x == true), skipmissing=true);
  xlabel="Trial",
  ylabel="P(Explorative)",
  xcol=:trial_number,
  ycol=:explorative_cat,
  title="Exploration Trend by # Controlled Choices",
  caption=""
)

## Test-retest reliability of exploration by interval (explorative trend)
plot_explore_trend_reliability(
  subset(explore_nchoice_df, :has_no_blue => ByRow(x -> x == true), skipmissing=true);
  title = "Test-retest: Exploration by # Controlled Choices",
  caption = "Explore[0/0.5/1 | # Controlled Choices] ~ 1 + Trial",
)

plot_choice_pred_curves(
  subset(explore_nchoice_df, :has_no_blue => ByRow(x -> x == true), skipmissing=true);
  title = "Choice Prediction Curves by # Controlled Choices",
  ycol = :next_response_left,
  xlabel = "Left(#Controlled) - Right(#Controlled)"
)

plot_bias_and_sensitivity_reliability(
  subset(explore_nchoice_df, :has_no_blue => ByRow(x -> x == true), skipmissing=true);
  ycol = :next_response_left,
  title = "Test-retest: Exploration by # Controlled Choices",
)

## Interval and/or Controlled info: which predicts choice better?

let
  explore_indices_df = @chain explore_nchoice_df begin
    groupby([:prolific_pid, :session])
    transform(:trial => lead => :trial)
    filter(x -> .!ismissing(x.has_no_blue) .&& x.has_no_blue .== true, _)
    select(:prolific_pid, :session, :trial, :explorative_val => :n_controlled, :next_response_left => :response_left)
    dropmissing()
  end
  interval_df = @chain explore_by_interval begin
    select(:prolific_pid, :session, :trial, :explorative_val => :interval, :response_left)
    dropmissing()
  end
  occurrence_df = @chain explore_by_occur begin
    select(:prolific_pid, :session, :trial, :explorative_val => :occurrence, :response_left)
    dropmissing()
  end
  leftjoin!(explore_indices_df, interval_df, on=[:prolific_pid, :session, :trial, :response_left])
  filter!(x -> !isinf(x.interval), explore_indices_df)
  transform!(explore_indices_df, [:n_controlled, :interval] .=> (x -> (x .- mean(x))./std(x)) .=> [:n_controlled_scaled, :interval_scaled])
  leftjoin!(explore_indices_df, occurrence_df, on=[:prolific_pid, :session, :trial, :response_left])
  transform!(explore_indices_df, :occurrence => (x -> (x .- mean(x))./std(x)) => :occurrence_scaled)

  # GLMM
  glmm(@formula(response_left ~ session * n_controlled_scaled * interval_scaled + (session * n_controlled_scaled * interval_scaled | prolific_pid)), explore_indices_df, Bernoulli(), contrasts=Dict(:session => EffectsCoding()), fast=true, progress=true)
end