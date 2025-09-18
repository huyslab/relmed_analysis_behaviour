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
	using StandardizedPredictors
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
    select([:prolific_pid, :session, :trial, :left, :right, :response, :control_rule_used, :current, :trial_presses])
    transform([:left, :right, :response] => ByRow((left, right, resp) -> ismissing(resp) ? missing : ifelse(resp == "left", left, right)) => :choice)
    transform(:current => (x -> categorical(x; levels=[1,2,3], ordered=true)) => :current)
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
begin
  ## Preprocess data for statistical modeling
  dropmissing!(explore_features_df, :response)
  transform!(explore_features_df, :response => ByRow(x -> x == "left" ? 1 : 0) => :response_left)

  ## Filter: 1. only the first session
  df = filter(row -> row.session == "1", explore_features_df)

  ## Filter: 2. only non-blue ship trials
  filter!(row -> !(row.left == "blue" || row.right == "blue"), df)

  ## Standardize features
  transform!(df, [:diff_interval, :diff_choice_interval, :diff_occurrence, :diff_choice_occurrence, :trial_number] .=> (x -> (x .- mean(x))./std(x)) .=> (n -> Symbol(string(n), "_scaled")), renamecols=false)

  ## GLMM for exploration choices (excluding blue ship trials), full model
  m0 = glmm(@formula(response_left ~ diff_interval_scaled + diff_choice_interval_scaled + diff_occurrence_scaled + diff_choice_occurrence_scaled + (diff_interval_scaled + diff_choice_interval_scaled + diff_occurrence_scaled + diff_choice_occurrence_scaled | prolific_pid)), df, Bernoulli(), fast=false, progress=false)

  ##TODO: coef plot of m0

  ## Only interval from last chosen
  m1 = glmm(@formula(response_left ~ diff_choice_interval_scaled + (diff_choice_interval_scaled | prolific_pid)), df, Bernoulli(), fast=false, progress=false)

  ## Only interval from last seen
  m2 = glmm(@formula(response_left ~ diff_interval_scaled + (diff_interval_scaled | prolific_pid)), df, Bernoulli(), fast=false, progress=false)

  ## Both intervals from last seen and last chosen
  m3 = glmm(@formula(response_left ~ diff_choice_interval_scaled + diff_interval_scaled + (diff_choice_interval_scaled + diff_interval_scaled | prolific_pid)), df, Bernoulli(), fast=false, progress=false)

  m3_1 = glmm(@formula(response_left ~ diff_choice_occurrence_scaled + diff_occurrence_scaled + (diff_choice_occurrence_scaled + diff_occurrence_scaled | prolific_pid)), df, Bernoulli(), fast=false, progress=false)
end

begin
  ## Compare m1 and m2 on information criteria, matching probability, and cross-validated accuracy
  matching_accuracy(truth_vals, probs) = mean(truth_vals .* probs .+ (1 .- truth_vals) .* (1 .- probs))
  threshold_accuracy(truth_vals, probs; threshold=0.5) = mean((probs .>= threshold) .== truth_vals)

  truth = df.response_left
  predictions = [fitted(m1), fitted(m2)]
  matching_probs = map(predictions) do probs
    matching_accuracy(truth, probs)
  end
  accuracy = map(predictions) do probs
    threshold_accuracy(truth, probs)
  end

  model_comparison_df = DataFrame(
    model = ["m1", "m2"],
    aic = [aic(m1), aic(m2)],
    bic = [bic(m1), bic(m2)],
    loglikelihood = [loglikelihood(m1), loglikelihood(m2)],
    matching_probability = matching_probs,
    accuracy = accuracy
  )

  ## Participant-level K-fold cross-validation using matching probability
  participants = unique(df.prolific_pid)
  nfolds = min(10, length(participants))
  cv_rows = DataFrame(fold=Int[], model=String[], matching_probability=Float64[], accuracy=Float64[])

  if nfolds > 1
    Random.seed!(1234)
    shuffled_ids = Random.shuffle(participants)
    fold_sets = [shuffled_ids[i:nfolds:end] for i in 1:nfolds]

    for (fold_idx, test_ids) in enumerate(fold_sets)
      train = subset(df, :prolific_pid => pid -> .!(pid .∈ Ref(test_ids)))
      test = subset(df, :prolific_pid => pid -> pid .∈ Ref(test_ids))
      if isempty(train) || isempty(test)
        continue
      end

      m1_cv = glmm(formula(m1), train, Bernoulli(), fast=false, progress=false)
      m2_cv = glmm(formula(m2), train, Bernoulli(), fast=false, progress=false)

      preds_m1 = predict(m1_cv, test; new_re_levels=:population)
      preds_m2 = predict(m2_cv, test; new_re_levels=:population)
      truth_test = test.response_left

      push!(cv_rows, (fold=fold_idx, model="m1", matching_probability=matching_accuracy(truth_test, preds_m1), accuracy=threshold_accuracy(truth_test, preds_m1)))
      push!(cv_rows, (fold=fold_idx, model="m2", matching_probability=matching_accuracy(truth_test, preds_m2), accuracy=threshold_accuracy(truth_test, preds_m2)))
    end

    if !isempty(cv_rows)
      cv_summary = combine(groupby(cv_rows, :model), [:matching_probability, :accuracy] .=> mean .=> [:cv_matching_probability, :cv_accuracy])
      model_comparison_df = leftjoin(model_comparison_df, cv_summary, on=:model)
    end
  end

  println(model_comparison_df)

  ## Display cross-validation results if desired
  ## if !isempty(cv_rows)
  ##   println(cv_rows)
  ## end
end

# ### Does the effect of these features change over time?
begin
  transform!(df, :trial_number => (x -> (x .- mean(x))./std(x)) => :trial_number_scaled)

  glmm(@formula(response_left ~ trial_number_scaled * (diff_interval_scaled + diff_choice_interval_scaled + diff_occurrence_scaled + diff_choice_occurrence_scaled) + (trial_number_scaled * (diff_interval_scaled + diff_choice_interval_scaled + diff_occurrence_scaled + diff_choice_occurrence_scaled) | prolific_pid)), df, Bernoulli(), fast=false, progress=false)

  glmm(@formula(response_left ~ trial_number_scaled * diff_interval_scaled + (trial_number_scaled * diff_interval_scaled | prolific_pid)), df, Bernoulli(), fast=false, progress=false)
end

# ### Does the effect of these features differ between required efforts?
begin
  m5 = glmm(@formula(response_left ~ current * diff_interval_scaled + (current * diff_interval_scaled | prolific_pid)), df, Bernoulli(), contrasts = Dict(:current => EffectsCoding()), fast=false, progress=false)
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
    select([:prolific_pid, :session, :trial_number, :blue, :green, :yellow, :red])
    stack([:blue, :green, :yellow, :red], variable_name=:color, value_name=:count)
    groupby([:session, :trial_number, :color])
    combine(:count => (x -> sum(skipmissing(x))) => :count)
    groupby([:session, :trial_number])
    transform(:count => (x -> x ./ sum(skipmissing(x))) => :prop)
    filter(x -> x.prop .> 0, _)
  end

  data(color_choice_df) *
  mapping(:trial_number, :prop, color=:color, group=:color, row=:session) *
  (visual(Scatter; alpha = 0.3) + linear()) |> 
  draw(scales(Color = (; palette = ["blue" => "royalblue", "green" => "forestgreen", "yellow" => "goldenrod1", "red" => "salmon3"])))
end

## Is there a color preference during exploration?
begin
  color_choice_count = @chain explore_choice_df begin
    dropmissing(:choice)
    @count(prolific_pid, session, choice)
    groupby([:prolific_pid, :session])
    transform(:n => (x -> x ./ sum(x)) => :prop)
  end
  
  m_full = lmm(@formula(prop ~ choice + (choice | prolific_pid)),
              color_choice_count,
              contrasts=Dict(:choice => DummyCoding(base="blue")))

  m_null = lmm(@formula(prop ~ 1 + (choice | prolific_pid)),
              color_choice_count,
              contrasts=Dict(:choice => DummyCoding(base="blue")))

  lrt = lrtest(m_null, m_full)
  println(lrt)
end

# ### What about blue ship where they have known it in advance?
begin
  blue_choice_df = @chain explore_choice_df begin
    filter(row -> row.left == "blue" || row.right == "blue", _)
    transform(:choice => ByRow(x -> !ismissing(x) && x == "blue" ? 1 : 0) => :blue_choice)
  end

  glmm(@formula(blue_choice ~ trial_number + (trial_number | prolific_pid)), blue_choice_df, Bernoulli(), contrasts=Dict(:trial_number => StandardizedPredictors.ZScore()), fast=false, progress=false)
end

# ### When people choose the known choice, how much effort do they choose to exert?
begin
  @chain explore_choice_df begin
    transform(:choice => ByRow(x -> !ismissing(x) && x == "blue" ? 1 : 0) => :blue_choice)
    transform(:control_rule_used => ByRow(x -> !ismissing(x) && x == "control" ? 1 : 0) => :control_choice)
    groupby([:prolific_pid, :session, :current, :blue_choice])
    combine(:control_choice => mean => :prop_control)
    groupby([:session, :current, :blue_choice])
    combine(:prop_control => mean => :prop_control)
    sort([:session, :current, :blue_choice])
  end

  control_blue_df = @chain explore_choice_df begin
    filter(row -> row.session == "1", _)
    filter(row -> row.left == "blue" || row.right == "blue", _)
    transform(:choice => ByRow(x -> !ismissing(x) && x == "blue" ? 1 : -1) => :blue_choice)
    transform(:control_rule_used => ByRow(x -> !ismissing(x) && x == "control" ? 1 : 0) => :control_choice)
  end

  glmm(@formula(control_choice ~ blue_choice * current + (blue_choice * current | prolific_pid)), control_blue_df, Bernoulli(), contrasts=Dict(:current => EffectsCoding()), fast=false, progress=false)
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

  p_effort_to_cutoff = Figure(size = (800, 600))

  p_ind = data(effort_to_cutfoff_df) *
  mapping(:trial_number, :effort_to_cutoff, group=:prolific_pid, col=:session, row=:current) *
  visual(Lines; alpha = 0.05)

  p_avg = @chain effort_to_cutfoff_df begin
    groupby([:session, :current, :trial_number])
    combine(:effort_to_cutoff => mean => :mean_effort_to_cutoff)
    data(_) *
    mapping(:trial_number, :mean_effort_to_cutoff, col=:session, row=:current) *
    (visual(Lines; color = :darkorange, linewidth=2, linestyle=:dash))
  end

  reflines = mapping([0]) * visual(HLines; color=:purple, linestyle=:dot, linewidth=2)

  draw!(p_effort_to_cutoff, p_ind + p_avg + reflines)

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
    feature_rows = NamedTuple{(:prolific_pid, :session, :trial_index, :ship, :seen_count, :controlled_count, :prediction_idx),
      Tuple{String, String, Int64, String, Int64, Int64, Int64}}[]

    for group_df in groupby(task_df, [:prolific_pid, :session])
      sdf = sort(group_df, :trial_index)
      pid = String(sdf.prolific_pid[1])
      sess = String(sdf.session[1])

      seen_counts = Dict{String, Int}(color => 0 for color in colors)
      control_counts = Dict{String, Int}(color => 0 for color in colors)
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
            push!(feature_rows, (prolific_pid=pid, session=sess, trial_index=trial_idx, ship=ship_str,
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
    select(:prolific_pid, :session, :trial_index, :trial, :ship, :correct)
    transform(:correct => ByRow(x -> x ? 1 : 0) => :correct_int)
  end

  relevant_task_rows = filter(row -> row.trialphase in ("control_explore", "control_predict_homebase"), control_task_data)
  prediction_features = compute_prediction_features(relevant_task_rows)

  prediction_trials = leftjoin(prediction_trials, prediction_features, on=[:prolific_pid, :session, :trial_index, :ship])

  confidence_df = @chain control_report_data begin
    filter(row -> row.trialphase == "control_confidence", _)
    select(:prolific_pid, :session, :trial, :response)
    rename(:response => :confidence)
  end

  prediction_trials = leftjoin(prediction_trials, confidence_df, on=[:prolific_pid, :session, :trial])

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
  m_accuracy = glmm(@formula(correct_int ~ seen_count_z + controlled_count_z + prediction_idx_z + (1 | prolific_pid)), prediction_accuracy_df, Bernoulli(), fast=false, progress=false)
  println("Prediction accuracy model coefficients:")
  println(coeftable(m_accuracy))

  prediction_confidence_df = dropmissing(prediction_trials, [:seen_count_z, :controlled_count_z, :prediction_idx_z, :confidence])
  if !isempty(prediction_confidence_df)
    println("Prediction confidence sample size: ", nrow(prediction_confidence_df))
    transform!(prediction_confidence_df, :confidence => ByRow(Float64) => :confidence_float)
    m_confidence = lmm(@formula(confidence_float ~ seen_count_z + controlled_count_z + prediction_idx_z + (1 | prolific_pid)), prediction_confidence_df)
    println("Prediction confidence model coefficients:")
    println(coeftable(m_confidence))
  end
end
