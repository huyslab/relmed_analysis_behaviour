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

  set_theme!(th)
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

begin
	function add_ship_onehot(df)
			colors = ["blue", "green", "yellow", "red"]
			for color in colors
					df = DataFrames.transform(df, [:left, :right] => ByRow((l, r) -> (l == color || r == color) ? 1 : 0) => Symbol(color))
			end
			return df
	end

	explore_choice_df = @chain control_task_data begin
		filter(x -> x.trialphase .== "control_explore", _)
		select([:prolific_pid, :session, :trial, :left, :right, :response])
		groupby([:prolific_pid, :session])
		DataFrames.transform(
			:trial => (x -> 1:length(x)) => :trial_number
		)
		add_ship_onehot(_)
	end
end

function calc_trial_interval(trial_index, occurrence)
  interval = Vector{Union{AbstractFloat, Missing}}(missing, length(trial_index))
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

function add_explorative_response(df; metric::String = "occurrence")
  explorative_values = Vector{Union{Float64, Missing}}(undef, 0)
  if metric == "occurrence"
    for row in eachrow(df)
      if ismissing(row.response)
        push!(explorative_values, missing)
      elseif (row.response == "left" && row[row.left] < row[row.right]) || (row.response == "right" && row[row.right] < row[row.left])
        push!(explorative_values, 1.0)
      elseif row[row.right] == row[row.left]
        push!(explorative_values, 0.5)
      else
        push!(explorative_values, 0.0)
      end
    end
  elseif metric == "interval"
    for row in eachrow(df)
      if ismissing(row.response)
        push!(explorative_values, missing)
      elseif (row.response == "left" && row[row.left] > row[row.right]) || (row.response == "right" && row[row.right] > row[row.left])
        push!(explorative_values, 1.0)
      elseif row[row.right] == row[row.left]
        push!(explorative_values, 0.5)
      else
        push!(explorative_values, 0.0)
      end
    end
  elseif metric == "choice"
  elseif metric == "control"
  end
  df.explorative = explorative_values
  return df
end

# Summarize choices by the number of occurrence
begin
	explore_by_occur = @chain explore_choice_df begin
		groupby([:prolific_pid, :session])
		DataFrames.transform(
			[:blue, :green, :yellow, :red] .=> cumsum, renamecols=false
		)
		add_explorative_response(_, metric="occurrence")
	end
end

p_explore_occur_avg = @chain explore_by_occur begin
	dropmissing(_)
	@group_by(session, trial)
	@summarize(explorative = mean(explorative))
	@ungroup
	@arrange(session, trial)
	data(_) * mapping(:trial, :explorative, color=:session) * (visual(Scatter) + linear())
	draw(;axis=(;title = "Avg. exploration trend", xlabel = "Trial", ylabel = "Explorative? (by occurence)"), figure=(;size=(600, 400)))
end
display(p_explore_occur_avg)

p_explore_occur_ind_trend = @chain explore_by_occur begin
	dropmissing(_)
	@group_by(prolific_pid, session, trial_number)
	@summarize(explorative = mean(explorative))
	@ungroup
	@arrange(prolific_pid, session, trial_number)
	data(_) * mapping(:trial_number, :explorative, group=:prolific_pid, color=:session, col=:session) * (linear(interval=nothing))
	draw(;axis=(;xlabel = "Trial", ylabel = "Explorative? (by occurrence)"), figure=(;size=(600, 400)))
end
display(p_explore_occur_ind_trend)

# Test-retest reliability of exploration by occurrence (explorative trend)
let
	glm_coef(data) = coef(lm(@formula(explorative ~ trial_number), data))
	
	retest_df = @chain explore_by_occur begin
		transform(:trial_number => (x -> x .- mean(x)), renamecols=false)
		@drop_missing(explorative)
		groupby([:prolific_pid, :session])
		combine(AsTable([:explorative, :trial_number]) => (x -> [glm_coef(x)]) => [:β0, :β_trial])
	end

	fig=Figure(;size=(12, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=dropmissing(unstack(retest_df, [:prolific_pid], :session, :β0)),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Avg. exploration",
		correct_r=false
	)

	workshop_reliability_scatter!(
		fig[1,2];
		df=dropmissing(unstack(retest_df, [:prolific_pid], :session, :β_trial)),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Exploration trend",
		correct_r=false
	)
	Label(fig[0, :], "Test-retest: exploration by occurrence")
	display(fig)
end

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
  end
end

p_explore_int_avg = @chain explore_by_interval begin
	dropmissing([:explorative, :response])
	@group_by(session, trial)
	@summarize(explorative = mean(explorative))
	@ungroup
	@arrange(session, trial)
	data(_) * mapping(:trial, :explorative, color=:session) * (visual(Scatter) + linear())
	draw(;axis=(;title = "Avg. exploration trend", xlabel = "Trial", ylabel = "Explorative? (by interval)"), figure=(;size=(600, 400)))
end
display(p_explore_int_avg)

p_explore_int_ind_trend = @chain explore_by_interval begin
	dropmissing([:explorative, :response])
	@group_by(prolific_pid, session, trial_number)
	@summarize(explorative = mean(explorative))
	@ungroup
	@arrange(prolific_pid, session, trial_number)
	data(_) * mapping(:trial_number, :explorative, group=:prolific_pid, color=:session, col=:session) * (linear(interval=nothing))
	draw(axis=(;xlabel = "Trial", ylabel = "Explorative? (by interval)"), figure=(;size=(600, 400)))
end
display(p_explore_int_ind_trend)

# Test-retest reliability of exploration by interval (explorative trend)
let
	glm_coef(data) = coef(lm(@formula(explorative ~ trial_number), data))
	
	retest_df = @chain explore_by_interval begin
		transform(:trial_number => (x -> x .- mean(x)), renamecols=false)
		@drop_missing(explorative)
		groupby([:prolific_pid, :session])
		combine(AsTable([:explorative, :trial_number]) => (x -> [glm_coef(x)]) => [:β0, :β_trial])
	end

	fig=Figure(;size=(12, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=dropmissing(unstack(retest_df, [:prolific_pid], :session, :β0)),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Avg. exploration",
		correct_r=false
	)

	workshop_reliability_scatter!(
		fig[1,2];
		df=dropmissing(unstack(retest_df, [:prolific_pid], :session, :β_trial)),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Exploration trend",
		correct_r=false
	)
	Label(fig[0, :], "Test-retest: exploration by interval")
	display(fig)
end

# Are these two measures of exploration correlated?
let
  x_df = select(explore_by_occur, :prolific_pid, :session, :trial_number, :explorative => :explorative_occur)
  y_df = select(explore_by_interval, :prolific_pid, :session, :trial_number, :explorative => :explorative_interval)

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

# Summarize choices by the number of choices
begin
  explore_nchoice_df = @chain explore_choice_df begin
    groupby([:prolific_pid, :session])
    transform(
      [:response, :left, :right] .=> lead => x -> "next_" * x,
    )
    transform(
      [:response, :left, :right, :blue, :green, :yellow, :red] => 
      ByRow((resp, left, right, blue, green, yellow, red) -> begin
        if ismissing(resp)
          return (blue=0, green=0, yellow=0, red=0)
        end
        
        chosen_color = resp == "left" ? left : right
        unchosen_color = resp == "left" ? right : left
        
        return (
          blue = chosen_color == "blue" ? blue : (unchosen_color == "blue" ? 0 : blue),
          green = chosen_color == "green" ? green : (unchosen_color == "green" ? 0 : green),
          yellow = chosen_color == "yellow" ? yellow : (unchosen_color == "yellow" ? 0 : yellow),
          red = chosen_color == "red" ? red : (unchosen_color == "red" ? 0 : red)
        )
      end) => [:blue, :green, :yellow, :red]
    )
    groupby([:prolific_pid, :session])
    transform([:blue, :green, :yellow, :red] .=> cumsum .=> [:blue, :green, :yellow, :red])
    # add_explorative_response(_, metric="choice")
  end
end

