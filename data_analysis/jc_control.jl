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
	using StandardizedPredictors, Effects
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
end

let
  ## Visualize fixed-effect estimates and marginal predictions
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

  vc = Matrix(vcov(m0))
  coef_names = string.(coefnames(m0))
  fixefs = collect(fixef(m0))
  intercept_idx = findfirst(==("(Intercept)"), coef_names)
  intercept = fixefs[intercept_idx]
  x_vals = range(-2.5, 2.5; length=200)

  function prediction_curve(var_name::String)
    idx = findfirst(==(var_name), coef_names)
    β = fixefs[idx]
    η = intercept .+ β .* x_vals
    η_se = sqrt.(vc[intercept_idx, intercept_idx] .+ x_vals.^2 .* vc[idx, idx] .+ 2 .* x_vals .* vc[intercept_idx, idx])
    mean = logistic.(η)
    lower = logistic.(η .- 1.96 .* η_se)
    upper = logistic.(η .+ 1.96 .* η_se)
    return (; x_vals, mean, lower, upper, β)
  end

  line_colors = Dict(
    "diff_interval_scaled" => :royalblue3,
    "diff_choice_interval_scaled" => :darkorange2,
    "diff_occurrence_scaled" => :seagreen3,
    "diff_choice_occurrence_scaled" => :mediumpurple3,
  )

  m0_effect_fig = Figure(size = (1200, 300))

  ax_coef = Axis(m0_effect_fig[1, 1];
    title = "Fixed-effects (log-odds)",
    xlabel = "Estimate",
    yticks = coef_ax_ticks,
    yreversed = true)

  vlines!(ax_coef, [0]; color = :black, linestyle = :dash)
  xerr_lower = coef_tbl.coef[order_idx] .- coef_tbl.lower[order_idx]
  xerr_upper = coef_tbl.upper[order_idx] .- coef_tbl.coef[order_idx]
  errorbars!(ax_coef, coef_tbl.coef[order_idx], 1:length(term_order), xerr_lower, xerr_upper; direction = :x, color = :black)
  scatter!(ax_coef, coef_tbl.coef[order_idx], 1:length(term_order); color = :black)

  ax_interval = Axis(m0_effect_fig[1, 3]; ylabelvisible = false)
  vlines!(ax_interval, [0]; color = (:black, 0.4), linestyle = :dash)
  ylims!(ax_interval, 0, 1)
  ax_interval.xlabel = "Predictor (z)"
  ax_interval.title = "Interval predictors"

  for var_name in ["diff_interval_scaled", "diff_choice_interval_scaled"]
    curve = prediction_curve(var_name)
    color = line_colors[var_name]
    band!(ax_interval, curve.x_vals, curve.lower, curve.upper; color = (color, 0.2))
    lines!(ax_interval, curve.x_vals, curve.mean; color = color, linewidth = 3, label = "$(predictor_labels[var_name]) (β=$(round(curve.β, digits=3)))")
  end

  axislegend(ax_interval, position = :rb)

  ax_occurrence = Axis(m0_effect_fig[1, 2]; ylabelvisible = false)
  vlines!(ax_occurrence, [0]; color = (:black, 0.4), linestyle = :dash)
  ylims!(ax_occurrence, 0, 1)
  ax_occurrence.xlabel = "Predictor (z)"
  ax_occurrence.title = "Occurrence predictors"

  for var_name in ["diff_occurrence_scaled", "diff_choice_occurrence_scaled"]
    curve = prediction_curve(var_name)
    color = line_colors[var_name]
    band!(ax_occurrence, curve.x_vals, curve.lower, curve.upper; color = (color, 0.2))
    lines!(ax_occurrence, curve.x_vals, curve.mean; color = color, linewidth = 3, label = "$(predictor_labels[var_name]) (β=$(round(curve.β, digits=3)))")
  end

  axislegend(ax_occurrence, position = :rb)

  m0_effect_fig

end

begin
  ## Only interval from last chosen
  m1 = glmm(@formula(response_left ~ diff_choice_interval_scaled + (diff_choice_interval_scaled | prolific_pid)), df, Bernoulli(), fast=false, progress=false)

  ## Only interval from last seen
  m2 = glmm(@formula(response_left ~ diff_interval_scaled + (diff_interval_scaled | prolific_pid)), df, Bernoulli(), fast=false, progress=false)

  ## Both intervals from last seen and last chosen
  m3 = glmm(@formula(response_left ~ diff_choice_interval_scaled + diff_interval_scaled + (diff_choice_interval_scaled + diff_interval_scaled | prolific_pid)), df, Bernoulli(), fast=false, progress=false)

  m3_1 = glmm(@formula(response_left ~ diff_choice_occurrence_scaled + diff_occurrence_scaled + (diff_choice_occurrence_scaled + diff_occurrence_scaled | prolific_pid)), df, Bernoulli(), fast=false, progress=false)
end

let
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
      cv_se = combine(groupby(cv_rows, :model), [:matching_probability, :accuracy] .=> (x -> std(x) ./ sqrt(length(x))) .=> [:cv_matching_probability_se, :cv_accuracy_se])
      leftjoin!(model_comparison_df, cv_summary, on=:model)
      leftjoin!(model_comparison_df, cv_se, on=:model)
    end
  end

  model_comparison_df.model = ["Cumulative seen", "Last seen interval"]
  println(model_comparison_df)

  mod_cv_fig = Figure(size = (400, 600), figure_padding = (10, 60, 10, 10))
  p_match_prob = data(model_comparison_df) *
  mapping(:model => nonnumeric => "Model", :cv_matching_probability => "10-Fold matching probability") *
  (
    visual(Scatter) + 
    mapping(:cv_matching_probability_se) * visual(Errorbars)
  ) 
  draw!(mod_cv_fig[1,1], p_match_prob)

  p_accuracy = data(model_comparison_df) *
  mapping(:model => nonnumeric  => "Model", :cv_accuracy => "10-Fold accuracy") *
  (
    visual(Scatter) + 
    mapping(:cv_accuracy_se) * visual(Errorbars)
  ) 
  draw!(mod_cv_fig[2,1], p_accuracy)

  mod_cv_fig

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

let
  eff = effects(Dict(:current => unique(df.current), 
                    :diff_interval_scaled => range(extrema(df.diff_interval_scaled)..., length=50)), 
                m5; invlink=logistic)
  data(eff) *
  (
    mapping(
    :diff_interval_scaled => "zInterval[Left] - zInterval[Right]",
    :lower,
    :upper,
    color=:current => nonnumeric => "Required effort level", 
    group=:current => nonnumeric => "Required effort level") *
    visual(Band; alpha = 0.05) + 
    mapping(:diff_interval_scaled => "zInterval[Left] - zInterval[Right]", :response_left => "P[Left]", color=:current => nonnumeric => "Required effort level", group=:current => nonnumeric) *
    visual(Lines)
  ) |>
  draw(scales(Color = (; palette = ["lightskyblue1", "deepskyblue3", "midnightblue"])); axis=(; ylabel="P[Left]"))
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
    filter(row -> row.session == "1", _)
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

let
  data(color_choice_count) *
  mapping(:choice => "Color", :prop => "Proportion", color=:choice=>"Color") *
  visual(RainClouds) |>
  draw(scales(Color = (; palette = ["blue" => "royalblue", "green" => "forestgreen", "yellow" => "goldenrod1", "red" => "firebrick3"])))
end

# ### What about blue ship where they have known it in advance?
begin
  blue_choice_df = @chain explore_choice_df begin
    filter(row -> row.left == "blue" || row.right == "blue", _)
    transform(:choice => ByRow(x -> !ismissing(x) && x == "blue" ? 1 : 0) => :blue_choice)
    filter(row -> row.session == "1", _)
  end

  glmm(@formula(blue_choice ~ trial_number + (trial_number | prolific_pid)), blue_choice_df, Bernoulli(), contrasts=Dict(:trial_number => StandardizedPredictors.ZScore()), fast=false, progress=false)
end

let
  data(blue_choice_df) *
  mapping(:trial_number, :blue_choice) *
  (mapping(group=:prolific_pid) * visual(Lines; alpha = 0.01) + linear()) |>
  draw(axis=(; ylabel="P[Blue]", xlabel="Trial number"))
end
end
