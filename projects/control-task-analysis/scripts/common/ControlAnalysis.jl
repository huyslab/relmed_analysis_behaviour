module ControlAnalysis

using DataFrames, TidierData, CategoricalArrays, Statistics

# Export the main function so your script can use it
export build_exploration_features, compute_prediction_features, zscore_with_missing

"""
    build_exploration_features(control_task_data)

Takes the raw control task data, filters for the exploration phase,
calculates interval and occurrence metrics, and returns a joined feature DataFrame.
"""
function build_exploration_features(control_task_data)

  # 1. Base Cleaning & Transformation
  # (Refactoring lines 64-80)
  base_df = @chain control_task_data begin
    filter(x -> x.trialphase .== "control_explore", _)
    select([:participant_id, :session, :trial, :left, :right, :response, :control_rule_used, :current, :trial_presses])
    # Clean up choice column
    transform([:left, :right, :response] => ByRow((left, right, resp) ->
      ismissing(resp) ? missing : ifelse(resp == "left", left, right)) => :choice)
    # Order the 'current' variable
    transform(:current => (x -> categorical(x; levels=[1, 2, 3], ordered=true)) => :current)
    # Add trial numbers per participant/session
    groupby([:participant_id, :session])
    transform(:trial => (x -> 1:length(x)) => :trial_number)

    # NOTE: Ensure the helper 'add_ship_onehot' is defined in this module
    add_ship_onehot(_)
  end

  # 2. Interval from Last Seen
  # (Refactoring lines 84-95)
  feat_interval = @chain base_df begin
    groupby([:participant_id, :session])
    transform([
      ([:trial_number, Symbol(color)] => ((t, occ) ->
      # NOTE: Ensure 'calc_choice_interval' is defined in this module
        calc_choice_interval(t, occ, occ .== occ)) => Symbol(color))
      for color in (:blue, :green, :yellow, :red)
    ])
    # NOTE: Ensure 'add_explorative_measure' is defined in this module
    add_explorative_measure(_, metric="interval")
    rename(:explorative_val => :diff_interval)
    select(Not([:blue, :green, :yellow, :red]))
  end

  # 3. Interval from Last Controlled Choice
  # (Refactoring lines 98-109)
  feat_choice_interval = @chain base_df begin
    groupby([:participant_id, :session])
    transform([
      ([:trial_number, Symbol(color), :choice, :control_rule_used] =>
        ((t, occ, choice, r) -> calc_choice_interval(t, occ, .!ismissing.(r) .&& choice .== color .&& r .== "control")) => Symbol(color))
      for color in ("blue", "green", "yellow", "red")
    ])
    add_explorative_measure(_, metric="interval")
    rename(:explorative_val => :diff_choice_interval)
    select(Not([:blue, :green, :yellow, :red]))
  end

  # 4. Number of Occurrences (Cumulative)
  # (Refactoring lines 112-119)
  feat_occurrence = @chain base_df begin
    groupby([:participant_id, :session])
    transform([:blue, :green, :yellow, :red] .=> cumsum, renamecols=false)
    add_explorative_measure(_, metric="occurrence")
    rename(:explorative_val => :diff_occurrence)
    select(Not([:blue, :green, :yellow, :red]))
  end

  # 5. Number of Controlled Choices (Cumulative)
  # (Refactoring lines 122-135)
  feat_choice_occurrence = @chain base_df begin
    transform([
      ([:choice, :control_rule_used, Symbol(color)] =>
        ByRow((choice, rule, val) -> (!ismissing(choice) && rule == "control" && choice == color) ? val : 0) => Symbol(color))
      for color in ("blue", "green", "yellow", "red")
    ])
    groupby([:participant_id, :session])
    transform([:blue, :green, :yellow, :red] .=> x -> lag(x; default=0), renamecols=false)
    groupby([:participant_id, :session])
    transform([:blue, :green, :yellow, :red] .=> cumsum, renamecols=false)
    add_explorative_measure(_, metric="occurrence")
    rename(:explorative_val => :diff_choice_occurrence)
    select(Not([:blue, :green, :yellow, :red]))
  end

  # 6. Merge Everything
  # (Refactoring lines 138-146)
  feature_keys = [:participant_id, :session, :trial_number]

  final_df = reduce((left, right) -> leftjoin(left, right, on=feature_keys), [
    base_df,
    select(feat_interval, feature_keys..., :diff_interval),
    select(feat_choice_interval, feature_keys..., :diff_choice_interval),
    select(feat_occurrence, feature_keys..., :diff_occurrence),
    select(feat_choice_occurrence, feature_keys..., :diff_choice_occurrence)
  ])

  return final_df
end

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

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

function calc_choice_interval(trial_index, occurrence, resp)
  interval = Vector{Union{AbstractFloat,Missing}}(missing, length(trial_index))
  idx = findall(skipmissing(occurrence .== 1))
  prev_trial = 0
  if !isempty(idx)
    for i in eachindex(idx)
      curr_trial = trial_index[idx[i]]
      interval[idx[i]] = curr_trial - prev_trial
      prev_trial = !ismissing(resp[idx[i]]) && resp[idx[i]] == 1 ? curr_trial : prev_trial
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

end # module