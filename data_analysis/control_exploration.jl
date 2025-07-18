begin
  cd("/home/jovyan/")
  import Pkg
  # activate the shared project environment
  Pkg.activate("relmed_environment")
  # instantiate, i.e. make sure that all packages are downloaded
  Pkg.instantiate()
  using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, SHA
	using Tidier, GLM, MixedModels, PlutoUI, LaTeXStrings, ColorSchemes, HypothesisTests, ShiftedArrays
	using LogExpFunctions: logistic, logit
  include(joinpath(pwd(), "fetch_preprocess_data.jl"))
	include(joinpath(pwd(), "plotting_utils.jl"))
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
  nothing
end

function summarize_participation(data::DataFrame)

  participants = combine(groupby(data, [:prolific_pid, :session, :record_id, :exp_start_time]),
    :trialphase => (x -> "experiment_end_message" in x) => :finished,
    :trialphase => (x -> "kick-out" in x) => :kick_out,
    [:trial_type, :trialphase, :block, :n_stimuli] =>
      ((t, p, b, n) -> sum((t .== "PILT") .& (.!ismissing.(p) .&& p .!= "PILT_test") .& (typeof.(b) .== Int64) .& (n .== 2))) => :n_trial_PILT,
    [:block, :trial_type, :trialphase, :n_stimuli] =>
      ((x, t, p, n) -> length(unique(filter(y -> isa(y, Int64), x[(t.=="PILT").&(n.==2).&(.!ismissing.(p).&&p.!="PILT_test")])))) => :n_blocks_PILT,
    # :trialphase => (x -> sum(skipmissing(x .∈ Ref(["control_explore", "control_predict_homebase", "control_reward"])))) => :n_trial_control,
    # :trialPresses => (x -> mean(filter(y -> !ismissing(y), x))) =>  :max_trial_presses,
    :n_warnings => maximum => :n_warnings,
    :time_elapsed => (x -> maximum(x) / 1000 / 60) => :duration,
    :total_bonus => (x -> all(ismissing.(x)) ? missing : only(skipmissing(x))) => :bonus
    # :trialphase => (x -> sum(skipmissing(x .== "control_instruction_quiz_failure"), init=0)) => :n_quiz_failure
  )

  debrief = extract_debrief_responses(data)

  participants = leftjoin(participants, debrief,
    on=[:prolific_pid, :exp_start_time])

  return participants
end

function extract_debrief_responses(data::DataFrame)
  # Select trials
  debrief = filter(x -> !ismissing(x.trialphase) &&
                          occursin(r"(acceptability|debrief)", x.trialphase) &&
                          !(occursin("pre", x.trialphase)), data)


  # Select variables
  select!(debrief, [:prolific_pid, :exp_start_time, :trialphase, :response])

  # Long to wide
  debrief = unstack(
    debrief,
    [:prolific_pid, :exp_start_time],
    :trialphase,
    :response
  )


  # Parse JSON and make into DataFrame
  expected_keys = dropmissing(debrief)[1, Not([:prolific_pid, :exp_start_time])]
  expected_keys = Dict([c => collect(keys(JSON.parse(expected_keys[c])))
                        for c in names(expected_keys)])

  debrief_colnames = names(debrief[!, Not([:prolific_pid, :exp_start_time])])

  # Expand JSON strings with defaults for missing fields
  expanded = [
    DataFrame([
      # Parse JSON or use empty Dict if missing
      let parsed = ismissing(row[col]) ? Dict() : JSON.parse(row[col])
        # Fill missing keys with a default value (e.g., `missing`)
        Dict(key => get(parsed, key, missing) for key in expected_keys[col])
      end
      for row in eachrow(debrief)
    ])
    for col in debrief_colnames
  ]
  expanded = hcat(expanded...)

  # hcat together
  return hcat(debrief[!, Not(debrief_colnames)], expanded)
end

spearman_brown(
  r;
  n=2 # Number of splits
) = (n * r) / (1 + (n - 1) * r)

function add_ship_onehot(df)
  colors = ["blue", "green", "yellow", "red"]
  for color in colors
    df = DataFrames.transform(df, [:left, :right] => ByRow((l, r) -> (l == color || r == color) ? 1 : 0) => Symbol(color))
  end
  return df
end

function calc_trial_interval(trial_index, occurrence)
  interval = Vector{Union{AbstractFloat,Missing}}(missing, length(trial_index))
  idx = findall(occurrence .== 1)
  if !isempty(idx)
    prev_trial = trial_index[idx[1]]
    interval[idx[1]] = Inf  # or 0/missing if you prefer
    for i in 2:length(idx)
      curr_trial = trial_index[idx[i]]
      interval[idx[i]] = curr_trial - prev_trial
      prev_trial = curr_trial
    end
  end
  return interval
end

function add_explorative_response(df; metric::String="occurrence")
  if metric == "occurrence"
    return transform(df,
      [:response, :left, :right, :blue, :green, :yellow, :red] =>
        ByRow((resp, left, right, blue, green, yellow, red) -> begin
          if ismissing(resp)
            missing
          else
            left_val = left == "blue" ? blue : left == "green" ? green : left == "yellow" ? yellow : red
            right_val = right == "blue" ? blue : right == "green" ? green : right == "yellow" ? yellow : red

            if (resp == "left" && left_val < right_val) || (resp == "right" && right_val < left_val)
              1.0
            elseif left_val == right_val
              0.5
            else
              0.0
            end
          end
        end) => :explorative_cat
    )
  elseif metric == "interval"
    return transform(df,
      [:response, :left, :right, :blue, :green, :yellow, :red] =>
        ByRow((resp, left, right, blue, green, yellow, red) -> begin
          if ismissing(resp)
            missing
          else
            left_val = left == "blue" ? blue : left == "green" ? green : left == "yellow" ? yellow : red
            right_val = right == "blue" ? blue : right == "green" ? green : right == "yellow" ? yellow : red

            if (resp == "left" && left_val > right_val) || (resp == "right" && right_val > left_val)
              1.0
            elseif left_val == right_val
              0.5
            else
              0.0
            end
          end
        end) => :explorative_cat
    )
  elseif metric == "control"
    return transform(df,
      [:next_response, :next_left, :next_right, :blue, :green, :yellow, :red] =>
        ByRow((resp, left, right, blue, green, yellow, red) -> begin
          if ismissing(resp)
            missing
          else
            left_val = left == "blue" ? blue : left == "green" ? green : left == "yellow" ? yellow : red
            right_val = right == "blue" ? blue : right == "green" ? green : right == "yellow" ? yellow : red

            if (resp == "left" && left_val < right_val) || (resp == "right" && right_val < left_val)
              1.0
            elseif left_val == right_val
              0.5
            else
              0.0
            end
          end
        end) => :explorative_cat
    )
  else
    error("Unknown metric: $metric")
  end
end

function add_explorative_measure(df; metric::String="occurrence")
  if metric == "occurrence" || metric == "interval"
    return transform(df,
      [:response, :left, :right, :blue, :green, :yellow, :red] =>
        ByRow((resp, left, right, blue, green, yellow, red) -> begin
          if ismissing(resp)
            missing
          else
            left_val = left == "blue" ? blue : left == "green" ? green : left == "yellow" ? yellow : red
            right_val = right == "blue" ? blue : right == "green" ? green : right == "yellow" ? yellow : red
            left_val - right_val
          end
        end) => :explorative_val
    )
  elseif metric == "control"
    return transform(df,
      [:next_response, :next_left, :next_right, :blue, :green, :yellow, :red] =>
        ByRow((resp, left, right, blue, green, yellow, red) -> begin
          if ismissing(resp)
            missing
          else
            left_val = left == "blue" ? blue : left == "green" ? green : left == "yellow" ? yellow : red
            right_val = right == "blue" ? blue : right == "green" ? green : right == "yellow" ? yellow : red
            left_val - right_val
          end
        end) => :explorative_val
    )
  else
    error("Unknown metric: $metric")
  end
end

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

begin
  p_sum = summarize_participation(jspsych_data)
  p_no_double_take = exclude_double_takers(p_sum) |>
    x -> filter(x -> !ismissing(x.finished) && x.finished, x)
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
end

# Summarize choices by the number of occurrence
begin
	explore_by_occur = @chain explore_choice_df begin
		groupby([:prolific_pid, :session])
		transform(
			[:blue, :green, :yellow, :red] .=> cumsum, renamecols=false
		)
		add_explorative_response(_, metric="occurrence")
    add_explorative_measure(_, metric="occurrence")
    transform(:response => ByRow(passmissing(x -> x == "left" ? 1 : 0)) => :response_left)
	end
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

# Test-retest reliability of exploration by occurrence (explorative trend)
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
  subset(explore_by_occur, :prolific_pid => ByRow(x -> x !==("670cf1a20d1fa15c58a175f7")));
  title = "Test-retest: Exploration by # Occurrence Difference",
)


# Summarize choices by the length of the interval
begin
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

# Test-retest reliability of exploration by interval (explorative trend)
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
  subset(explore_by_interval, :prolific_pid => ByRow(x -> x !==("670cf1a20d1fa15c58a175f7")));
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
begin
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

# Test-retest reliability of exploration by interval (explorative trend)
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
  subset(subset(explore_nchoice_df, :has_no_blue => ByRow(x -> x == true), skipmissing=true), :prolific_pid => ByRow(x -> x !==("670cf1a20d1fa15c58a175f7")));
  ycol = :next_response_left,
  title = "Test-retest: Exploration by # Controlled Choices",
)