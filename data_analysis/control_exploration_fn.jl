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