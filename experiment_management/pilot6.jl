### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 237a05f6-9e0e-11ef-2433-3bdaa51dbed4
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates
	using LogExpFunctions: logistic, logit
	using Tidier
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	nothing
end

# ╔═╡ 0d120e19-28c2-4a98-b873-366615a5f784
begin
	# Set theme
	inter_bold = assetpath(pwd() * "/fonts/Inter/Inter-Bold.ttf")
	
	th = Theme(
		font = "Helvetica",
		fontsize = 16,
		Axis = (
			xgridvisible = false,
			ygridvisible = false,
			rightspinevisible = false,
			topspinevisible = false,
			xticklabelsize = 14,
			yticklabelsize = 14,
			spinewidth = 1.5,
			xtickwidth = 1.5,
			ytickwidth = 1.5
		)
	)
	
	set_theme!(th)
end

# ╔═╡ d5811081-d5e2-4a6e-9fc9-9d70332cb338
md"""## Participant management"""

# ╔═╡ 36b348cc-a3bf-41e7-aac9-1f6d858304a2
begin
	# Load data
	PILT_data, test_data, vigour_data, post_vigour_test_data, PIT_data, WM_data,
		reversal_data, jspsych_data = load_pilot6_data(; force_download = false)
	nothing
end

# ╔═╡ d3a5d834-1c23-4882-adc9-c4d9571e7f71
begin
	include("questionnaire_utils.jl")
	prepare_questionnaire_data(jspsych_data; save_data=true)
end

# ╔═╡ 70257d73-ffa4-468a-bafc-694740847e06
vigour_data

# ╔═╡ 0874e16f-1a89-4103-aa56-ada2a5622f2f
data(vigour_data) *
mapping(:press_per_sec) *
AlgebraOfGraphics.density() |>
draw()

# ╔═╡ 3d8d98d0-069c-4ef4-a13d-80e6d582deb0
quantile(vigour_data.press_per_sec, [0.1, 0.5, 0.9]) .* 3

# ╔═╡ f85a72d4-0959-4579-9c22-ae9362da75e3
70.0284/32.8655

# ╔═╡ 36e9f8f1-9c35-45e0-8d6b-0578e5ab38a0
function lognormal_mean(x)
	sum(pdf(LogNormal(log(40), 0.3), collect(skipmissing(x))) .* collect(skipmissing(x)))/sum(pdf(LogNormal(log(40), 0.3), collect(skipmissing(x))))
end

# ╔═╡ 61774f1c-ce18-4a2d-b271-745c93d412a8
let 
	vigour_bonus = vigour_data |>
		x -> groupby(x, [:prolific_pid, :session]) |>
		x -> combine(x, :trial_reward => lognormal_mean => :vigour_bonus)
	pit_bonus = PIT_data |>
		x -> groupby(x, [:prolific_pid, :session]) |>
		x -> combine(x, :trial_reward => lognormal_mean => :pit_bonus)
	innerjoin(vigour_bonus, pit_bonus, on = [:prolific_pid, :session]) |>
		x -> @mutate(x, total_bonus = vigour_bonus + pit_bonus) |>
		x -> combine(x, :total_bonus => mean)
end

# ╔═╡ cb4f46a2-1e9b-4006-8893-6fc609bcdf52
md""" ## Sanity checks"""

# ╔═╡ 5d487d8d-d494-45a7-af32-7494f1fb70f2
md""" ### PILT"""

# ╔═╡ 2ff04c44-5f86-4617-9a13-6d4228dff359
let
	@assert sort(unique(PILT_data.response)) == sort(["right", "left", "noresp"]) "Unexpected values in response"
	
	@assert all(PILT_data.chosen_feedback .== ifelse.(
		PILT_data.response .== "right",
		PILT_data.feedback_right,
		ifelse.(
			PILT_data.response .== "left",
			PILT_data.feedback_left,
			minimum.(hcat.(PILT_data.feedback_left, PILT_data.feedback_right))
		)
	)) "`chosen_feedback` doens't match feedback and choice"


end

# ╔═╡ d0a2ba1e-8413-48f8-8bbc-542f3555a296
let
	# Clean data
	PILT_data_clean = exclude_PLT_sessions(PILT_data, required_n_blocks = 20)
	PILT_data_clean = filter(x -> x.response != "noresp", PILT_data_clean)

	# Sumarrize by participant and trial
	acc_curve = combine(
		groupby(PILT_data_clean, [:prolific_pid, :trial]),
		:response_optimal => mean => :acc
	)

	sort!(acc_curve, [:prolific_pid, :trial])

	# Summarize by trial
	acc_curve_sum = combine(
		groupby(acc_curve, :trial),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Plot
	mp = (data(acc_curve) * mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice",
		group = :prolific_pid,
		color = :prolific_pid
	) * visual(Lines, linewidth = 1, alpha = 0.7)) +
	(data(acc_curve_sum) * 
	mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice"
	) * visual(Lines, linewidth = 4))
	
	
	draw(mp; legend = (; show = false))
end

# ╔═╡ 2897a681-e8dd-4091-a2a0-bd3d4cd23209
md"""### Post-PILT test"""

# ╔═╡ 6244dd22-7c58-4e87-84ed-004b076bc4cb
test_data.response |> Set

# ╔═╡ 176c54de-e84c-45e5-872e-2471e575776d
let
	# Select post-PILT test
	test_data_clean = filter(x -> isa(x.block, Int64), test_data)

	@assert Set(test_data_clean.response) == 
	Set(["ArrowRight", "ArrowLeft", nothing]) "Unexpected values in respones: $(unique(test_data_clean.response))"

	# Remove missing values
	filter!(x -> !isnothing(x.response), test_data_clean)

	# Create magnitude high and low varaibles
	test_data_clean.magnitude_high = maximum.(eachrow((hcat(
		test_data_clean.magnitude_left, test_data_clean.magnitude_right))))

	test_data_clean.magnitude_low = minimum.(eachrow((hcat(
		test_data_clean.magnitude_left, test_data_clean.magnitude_right))))

	# Create high_chosen variable
	test_data_clean.high_chosen = ifelse.(
		test_data_clean.right_chosen,
		test_data_clean.magnitude_right .== test_data_clean.magnitude_high,
		test_data_clean.magnitude_left .== test_data_clean.magnitude_high
	)

	high_chosen_sum = combine(
		groupby(test_data_clean, :prolific_pid),
		:high_chosen => mean => :acc
	)

	@info "Proportion high magnitude chosen: 
		$(round(mean(high_chosen_sum.acc), digits = 2)), SE=$(round(sem(high_chosen_sum.acc), digits = 2))"

	# Summarize by participant and magnitude
	test_sum = combine(
		groupby(test_data_clean, [:prolific_pid, :magnitude_low, :magnitude_high]),
		:high_chosen => mean => :acc
	)

	test_sum_sum = combine(
		groupby(test_sum, [:magnitude_low, :magnitude_high]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	sort!(test_sum_sum, [:magnitude_low, :magnitude_high])

	mp = data(test_sum_sum) *
	mapping(
		:magnitude_high => nonnumeric => "High magntidue",
		:acc => "Prop. chosen high",
		:se,
		layout = :magnitude_low => nonnumeric
	) * (visual(Errorbars) + visual(ScatterLines))

	draw(mp)

end

# ╔═╡ 18956db1-4ad1-4881-a1e7-8362cf59f011
md"""### WM"""

# ╔═╡ 18e9fccd-cc0d-4e8f-9e02-9782a03093d7
let
	@assert sort(unique(WM_data.response)) == sort(["right", "middle", "left", "noresp"]) "Unexpected values in response"
	
	@assert all(WM_data.chosen_feedback .== ifelse.(
		WM_data.response .== "right",
		WM_data.feedback_right,
		ifelse.(
			WM_data.response .== "left",
			WM_data.feedback_left,
			ifelse.(
				WM_data.response .== "middle",
				WM_data.feedback_middle,
				minimum.(hcat.(WM_data.feedback_left, WM_data.feedback_right))
			)
		)
	)) "`chosen_feedback` doens't match feedback and choice"


end

# ╔═╡ 17666d61-f5fc-4a8d-9624-9ae79f3de6bb
let
	# Clean data
	WM_data_clean = exclude_PLT_sessions(WM_data, required_n_blocks = 10)
	WM_data_clean = filter(x -> x.response != "noresp", WM_data_clean)

	# Sumarrize by participant, trial, n_groups
	acc_curve = combine(
		groupby(WM_data_clean, [:prolific_pid, :trial, :n_groups]),
		:response_optimal => mean => :acc
	)

	# Summarize by trial and n_groups
	acc_curve_sum = combine(
		groupby(acc_curve, [:trial, :n_groups]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	acc_curve_sum.lb = acc_curve_sum.acc .- acc_curve_sum.se
	acc_curve_sum.ub = acc_curve_sum.acc .+ acc_curve_sum.se

	# Sort
	sort!(acc_curve_sum, [:n_groups, :trial])

	# Create figure
	f = Figure(size = (700, 350))

	# Create mapping
	mp1 = (data(acc_curve_sum) * (
		mapping(
		:trial => "Trial #",
		:lb,
		:ub,
		group = :n_groups => nonnumeric => "Set size",
		color = :n_groups => nonnumeric => "Set size"
	) * visual(Band, alpha = 0.5) +
		mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice",
		group = :n_groups => nonnumeric => "Set size",
		color = :n_groups => nonnumeric => "Set size"
	) * visual(Lines)))
	
	# Plot
	plt1 = draw!(f[1,1], mp1)
	legend!(f[0,1:2], plt1, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)


	# Create appearance variable
	sort!(WM_data_clean, [:prolific_pid, :session, :block, :trial])
	
	transform!(
		groupby(WM_data_clean, [:prolific_pid, :exp_start_time, :session, :block, :stimulus_group]),
		:trial => (x -> 1:length(x)) => :appearance
	)

	# Summarize by appearance
	app_curve = combine(
		groupby(WM_data_clean, [:prolific_pid, :appearance, :n_groups]),
		:response_optimal => mean => :acc
	)

	# Summarize by apperance and n_groups
	app_curve_sum = combine(
		groupby(app_curve, [:appearance, :n_groups]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	app_curve_sum.lb = app_curve_sum.acc .- app_curve_sum.se
	app_curve_sum.ub = app_curve_sum.acc .+ app_curve_sum.se

	# Sort
	sort!(app_curve_sum, [:n_groups, :appearance])

	# Create mapping
	mp2 = (data(app_curve_sum) * (
		mapping(
		:appearance => "Apperance #",
		:lb,
		:ub,
		group = :n_groups => nonnumeric => "Set size",
		color = :n_groups => nonnumeric => "Set size"
	) * visual(Band, alpha = 0.5) +
		mapping(
		:appearance => "Apperance #",
		:acc => "Prop. optimal choice",
		group = :n_groups => nonnumeric => "Set size",
		color = :n_groups => nonnumeric => "Set size"
	) * visual(Lines)))
	
	# Plot
	plt2 = draw!(f[1,2], mp2)

	f
end

# ╔═╡ 1d1d6d79-5807-487f-8b03-efb7d0898ae8
md"""### Reversal"""

# ╔═╡ e902cd57-f724-4c26-9bb5-1d03443fb191
let

	# Clean data
	reversal_data_clean = exclude_reversal_sessions(reversal_data; required_n_trials = 120)

	filter!(x -> !isnothing(x.response_optimal), reversal_data_clean)

	# Number trials leading to reversal
	DataFrames.transform!(
		groupby(reversal_data_clean, [:prolific_pid, :block]),
		:trial => (x -> x .- maximum(x) .- 1) => :trial_pre_reversal
	)

	# Count number of block
	DataFrames.transform!(
		groupby(reversal_data_clean, :prolific_pid),
		:block => (x -> maximum(x)) => :n_blocks
	)


	# Summarize accuracy pre reversal
	sum_pre = combine(
		groupby(
			filter(x -> (x.trial_pre_reversal > -4) && 
				(x.block < x.n_blocks) && (x.trial < 48), reversal_data_clean), 
			[:prolific_pid, :trial_pre_reversal]
		),
		:response_optimal => mean => :acc
	)

	sum_pre = combine(
		groupby(sum_pre, :trial_pre_reversal),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	rename!(sum_pre, :trial_pre_reversal => :trial)

	# Summarize post reversal
	sum_post = combine(
		groupby(
			filter(x -> x.trial < 10, reversal_data_clean),
			[:prolific_pid, :trial]
		),
		:response_optimal => mean => :acc
	)

	sum_post = combine(
		groupby(sum_post, :trial),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Concatenate pre and post
	sum_pre_post = vcat(sum_pre, sum_post)

	# Create group variable to break line plot
	sum_pre_post.group = sign.(sum_pre_post.trial)

	# Plot
	mp = data(sum_pre_post) *
		(
			mapping(
				:trial => "Trial relative to reversal",
				:acc  => "Prop. optimal choice",
				:se
			) * visual(Errorbars) +
			mapping(
				:trial => "Trial relative to reversal",
				:acc => "Prop. optimal choice"
			) * 
			visual(Scatter) +
			mapping(
				:trial => "Trial relative to reversal",
				:acc => "Prop. optimal choice",
				group = :group => nonnumeric 
			) * 
			visual(Lines)
		) +
		mapping([0]) * visual(VLines, color = :grey, linestyle = :dash)

	f = draw(mp; axis = (; xticks = -3:9))
end

# ╔═╡ 7559e78d-7bd8-4450-a215-d74a0b1d670a
md"""
### Vigour
"""

# ╔═╡ 7563e3f6-8fe2-41cc-8bdf-c05c86e3285e
begin
	filter!(x -> !(x.prolific_pid in ["671139a20b977d78ec2ac1e0", "6721ec463c2f6789d5b777b5", "62ae1ecc1bd29fdc6b14f6ea", "672c8f7bd981bf863dd16a98"]), vigour_data);
	transform!(vigour_data, [:trial_presses, :trial_duration] => ((x, y) -> x .* 1000 ./ y) => :press_per_sec);
	nothing;
end

# ╔═╡ 243e92bc-b2fb-4f76-9de3-08f8a2e4b25d
begin
	@chain vigour_data begin
		@filter(press_per_sec > 11)
		@count(prolific_pid)
	end
end

# ╔═╡ 0312ce5f-be36-4d9b-aee3-04497f846537
let
	n_miss_df = @chain vigour_data begin
		@mutate(pig = "Mag " * string(magnitude) * ", FR " * string(ratio))
		@group_by(prolific_pid, pig)
		@summarize(n_miss = sum(trial_presses == 0))
		@ungroup()
		@mutate(short_id = last.(prolific_pid, 5))
	end
	# Create the plot
	plot = data(n_miss_df) * 
		   mapping(:n_miss, layout=:pig) * 
		   histogram(bins=5)
	
	# Set up the axis
	axis = (
		xlabel = "# No-response trials",
		ylabel = "Count (# participants)"
	)
	
	# Draw the plot
	draw(plot; axis, figure=(;title="No-response trial distribution in Vigour task"))
end

# ╔═╡ 6f7acf24-dbdc-4919-badb-9fe58712eacd
let
	avg_df = @chain vigour_data begin
		@group_by(session, trial_number)
		@summarize(trial_presses = mean(trial_presses), se = mean(trial_presses)/sqrt(length(prolific_pid)))
		@ungroup
	end
	p = data(vigour_data) * mapping(:trial_number, :trial_presses, color=:session) * AlgebraOfGraphics.linear() + data(avg_df) * mapping(:trial_number, :trial_presses, color=:session) * visual(ScatterLines)
    draw(p)
end

# ╔═╡ 3d05e879-aa5c-4840-9f4f-ad35b8d9519a
let
	test_acc_df = @chain post_vigour_test_data begin
		@mutate(
			diff_rpp = (left_magnitude/left_ratio) - (right_magnitude/right_ratio),
			chose_left = Int(response === "ArrowLeft")
		)
		@group_by(prolific_pid, version)
		@select(diff_rpp, chose_left)
		@mutate(chose_left = chose_left * 2 - 1, truth = sign(diff_rpp))
		@mutate(acc = chose_left == truth)
		@summarize(acc = mean(acc))
		@ungroup
	end
	@info "Vigour acc: $(round(mean(test_acc_df.acc); digits=2))"
	data(test_acc_df) *
	mapping(:acc) *
	visual(Hist) |>
	draw(;axis=(;xlabel="Accuracy",ylabel="Count (#Participant)"))
end

# ╔═╡ 665aa690-4f37-4a31-b87e-3b4aee66b3b1
md"""
### PIT
"""

# ╔═╡ 43d5b727-9761-48e3-bbc6-89af0c4f3116
begin
	filter!(x -> !(x.prolific_pid in ["671139a20b977d78ec2ac1e0", "6721ec463c2f6789d5b777b5", "62ae1ecc1bd29fdc6b14f6ea", "672c8f7bd981bf863dd16a98"]), PIT_data);
	transform!(PIT_data, [:trial_presses, :trial_duration] => ((x, y) -> x .* 1000 ./ y) => :press_per_sec);
	nothing;
end

# ╔═╡ 89258a40-d4c6-4831-8cf3-d69d984c4f6e
let
	n_miss_df =  @chain PIT_data begin
		# @filter(coin != 0)
		@arrange(prolific_pid, magnitude, ratio)
		@mutate(pig = "Mag " * string(magnitude) * ", FR " * string(ratio))
		@group_by(prolific_pid, pig)
		@summarize(n_miss = sum(trial_presses == 0))
		@ungroup()
		@mutate(short_id = last.(prolific_pid, 5))
	end
	# Create the plot
	plot = data(n_miss_df) * 
		   mapping(:n_miss, layout=:pig) * 
		   histogram(bins=5)
	
	# Set up the axis
	axis = (
		xlabel = "# No-response trials",
		ylabel = "Count (# participants)"
	)
	
	# Draw the plot
	draw(plot; axis, figure=(;title="No-response trial distribution in PIT task"))
end

# ╔═╡ 8ad8c4be-3aaa-4f06-bc47-0123287b558c
let
	pav_eff = @chain PIT_data begin
			@filter(coin != 0)
			@mutate(valence = ifelse(coin > 0, "pos", "neg"))
			@group_by(prolific_pid, valence)
			@summarize(press_per_sec = mean(press_per_sec))
			@ungroup
			@pivot_wider(names_from = valence, values_from = press_per_sec)
			@mutate(diff = pos - neg)
	end
	@info "% ΔValence>1: $(round(mean(pav_eff.diff.>1)...;digits=2))"
	data(pav_eff) *
		mapping(:diff) *
		AlgebraOfGraphics.density() |>
	draw(axis=(;xlabel="ΔValence"))
end

# ╔═╡ ffd08086-f12c-4b8a-afb6-435c8729241e
let
	PIT_acc_df = @chain test_data begin
		@filter(block == "pavlovian")
		@group_by(same_valence)
		@summarize(acc = mean(skipmissing((magnitude_right > magnitude_left) == right_chosen)))
		@ungroup
	end
	@info "PIT acc for NOT in the same valence: $(round(PIT_acc_df.acc[PIT_acc_df.same_valence.==false][1]; digits=2))"
	@info "PIT acc for in the same valence: $(round(PIT_acc_df.acc[PIT_acc_df.same_valence.==true][1]; digits=2))"
	@chain test_data begin
		@filter(block == "pavlovian")
		@group_by(prolific_pid, exp_start_time, session, same_valence)
		@summarize(acc = mean(skipmissing((magnitude_right > magnitude_left) == right_chosen)))
		@ungroup
		data(_) * mapping(:same_valence => nonnumeric => "Same valence", :acc => "PIT test accuracy", color=:same_valence => nonnumeric => "Same valence", col=:session) * visual(RainClouds)
		draw()
	end
end

# ╔═╡ 70366452-db2e-4834-b612-59108841683f
let
	PIT_acc_df = @chain test_data begin
		@filter(block == "pavlovian")
		@mutate(valence = ifelse(magnitude_left * magnitude_right < 0, "Different", ifelse(magnitude_left > 0, "Positive", "Negative")))
		@group_by(valence)
		@summarize(acc = mean(skipmissing((magnitude_right > magnitude_left) == right_chosen)))
		@ungroup
	end
	@info "PIT acc for NOT in the same valence: $(round(PIT_acc_df.acc[PIT_acc_df.valence.=="Different"]...; digits=2))"
	@info "PIT acc for in the positive valence: $(round(PIT_acc_df.acc[PIT_acc_df.valence.=="Positive"]...; digits=2))"
	@info "PIT acc for in the negative valence: $(round(PIT_acc_df.acc[PIT_acc_df.valence.=="Negative"]...; digits=2))"

	p = @chain test_data begin
		@filter(block == "pavlovian")
		@mutate(valence = ifelse(magnitude_left * magnitude_right < 0, "Different", ifelse(magnitude_left > 0, "Positive", "Negative")))
		@mutate(correct = (magnitude_right .> magnitude_left) .== right_chosen)
		@filter(!ismissing(correct))
		@group_by(prolific_pid, valence)
		@summarize(acc = mean(correct))
		@ungroup
		data(_) * mapping(:valence, :acc, color=:valence => "Valence") * visual(RainClouds; clouds=hist, hist_bins = 20, plot_boxplots = false)
	end
	draw(p;axis=(;xlabel="Pavlovian valence", ylabel="PIT test accuracy"))
end

# ╔═╡ 16ced201-fb4c-4040-bba6-b29dcb8b4da9
let
	pav_eff = @chain PIT_data begin
			@filter(coin != 0)
			@mutate(valence = ifelse(coin > 0, "pos", "neg"))
			@group_by(prolific_pid, valence)
			@summarize(press_per_sec = mean(press_per_sec))
			@ungroup
			@pivot_wider(names_from = valence, values_from = press_per_sec)
			@mutate(diff = pos - neg)
	end
	acc_grp_df = @chain test_data begin
		@filter(block == "pavlovian")
		@mutate(valence = ifelse(magnitude_left * magnitude_right < 0, "Different", ifelse(magnitude_left > 0, "Positive", "Negative")))
		@group_by(prolific_pid,valence)
		@summarize(acc = mean(skipmissing((magnitude_right > magnitude_left) == right_chosen)))
		@ungroup()
	end
	acc_grp_df.acc_grp = cut(acc_grp_df.acc, [0.5, 0.75]; extend=true)
	data(@inner_join(pav_eff, acc_grp_df)) *
		mapping(:acc, :diff, col=:valence) * (visual(Scatter, alpha = 0.2) + AlgebraOfGraphics.linear()) |>
	draw(axis=(;xlabel="Post-test accuracy", ylabel="ΔValence"))
end

# ╔═╡ ce27b319-d728-46f5-aaf1-051fe252bf8b
function avg_presses_w_fn(vigour_data::DataFrame, x_var::Vector{Symbol}, y_var::Symbol, grp_var::Union{Symbol,Nothing}=nothing)
    # Define grouping columns
    group_cols = grp_var === nothing ? [:prolific_pid, x_var...] : [:prolific_pid, grp_var, x_var...]
    # Group and calculate mean presses for each participant
    grouped_data = groupby(vigour_data, Cols(group_cols...)) |>
                   x -> combine(x, y_var => mean => :mean_y) |>
                        x -> sort(x, Cols(group_cols...))
    # Calculate the average across all participants
    avg_w_data = @chain grouped_data begin
        @group_by(prolific_pid)
        @mutate(sub_mean = mean(mean_y))
        @ungroup
        @mutate(grand_mean = mean(mean_y))
        @mutate(mean_y_w = mean_y - sub_mean + grand_mean)
        groupby(Cols(grp_var === nothing ? x_var : [grp_var, x_var...]))
        @summarize(
            n = n(),
            avg_y = mean(mean_y),
            se_y = std(mean_y_w) / sqrt(length(prolific_pid)))
        @ungroup
    end
    return grouped_data, avg_w_data
end

# ╔═╡ 8f6d8e98-6d73-4913-a02d-97525176549a
let
	df = @chain PIT_data begin
		@arrange(prolific_pid, session, magnitude, ratio)
		@mutate(pig = "Mag " * string(magnitude) * ", FR " * string(ratio))
		@mutate(pig=categorical(pig,levels=["Mag 2, FR 16","Mag 2, FR 8","Mag 5, FR 8","Mag 1, FR 1"]))
	end
	grouped_data, avg_w_data = avg_presses_w_fn(df, [:session, :coin, :pig], :press_per_sec)
	p = data(avg_w_data) *
	mapping(:coin=>nonnumeric, :avg_y, col=:pig, row=:session) *
	(
	visual(Lines, linewidth=1, color=:gray) +
	visual(Errorbars, whiskerwidth=4) *
	mapping(:se_y, color=:coin => nonnumeric) +
	visual(Scatter) *
	mapping(color=:coin => nonnumeric)
	)
	p_ind = data(@mutate(grouped_data, avg_y = mean_y)) * mapping(:coin=>nonnumeric, :avg_y, col=:pig, row=:session, group=:prolific_pid) * visual(Lines, linewidth = 0.1, color=:gray80)
	draw(p_ind + p, scales(Color = (; palette=:PRGn_7)); axis=(;xlabel="Pavlovian stimuli (coin)", ylabel="Press/sec", width=150, height=150, xticklabelrotation=pi/4))
end

# ╔═╡ baf2480d-a7ea-4eba-9c11-808da603142e
let
	using ColorSchemes
	colors=ColorSchemes.PRGn_7.colors;
	colors[4]=colorant"rgb(210, 210, 210)";

	acc_grp_df = @chain test_data begin
		@filter(block == "pavlovian")
		@group_by(prolific_pid)
		@summarize(acc = mean(skipmissing((magnitude_right > magnitude_left) == right_chosen)))
		@ungroup()
	end
	# acc_quantile = quantile(acc_grp_df.acc, [0.25, 0.5, 0.75])
	# @info "Acc at each quantile: $([@sprintf("%.1f%%", v * 100) for v in acc_quantile])"
	acc_grp_df.acc_grp = cut(acc_grp_df.acc, [0.5, 0.75]; extend=true)

	grouped_data, avg_w_data = avg_presses_w_fn(innerjoin(PIT_data, acc_grp_df, on=[:prolific_pid]), [:coin, :acc_grp], :press_per_sec)
	
	p = data(avg_w_data) *
	mapping(:coin=>nonnumeric, :avg_y, col=:acc_grp) *
	(
		visual(Lines, linewidth=2, color=:gray75) +
		visual(Errorbars, whiskerwidth=4) *
		mapping(:se_y, color=:coin => nonnumeric => :"Coin value") +
		visual(Scatter, markersize=10) *
		mapping(color=:coin => nonnumeric => :"Coin value")
	)
	fig = Figure(;size=(8, 6) .* 144 ./ 2.54)
	p = draw!(fig[1,1], p, scales(Color = (; palette=colors)); axis=(;xlabel="Pavlovian stimuli (coin value)", ylabel="Press/sec", xticklabelrotation=pi/4))
	Label(fig[0,:], "Press Rates by Pavlovian Stimuli Across Test Accuracy", tellwidth = false)

	fig
end

# ╔═╡ 91f6a95c-4f2e-4213-8be5-3ca57861ed15
"""
    extract_debrief_responses(data::DataFrame) -> DataFrame

Extracts and processes debrief responses from the experimental data. It filters for debrief trials, then parses and expands JSON-formatted Likert scale and text responses into separate columns for each question.

# Arguments
- `data::DataFrame`: The raw experimental data containing participants' trial outcomes and responses, including debrief information.

# Returns
- A DataFrame with participants' debrief responses. The debrief Likert and text responses are parsed from JSON and expanded into separate columns.
"""
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

# ╔═╡ dc957d66-1219-4a97-be46-c6c5c189c8ba
"""
    summarize_participation(data::DataFrame) -> DataFrame

Summarizes participants' performance in a study based on their trial outcomes and responses, for the purpose of approving and paying bonuses.

This function processes experimental data, extracting key performance metrics such as whether the participant finished the experiment, whether they were kicked out, and their respective bonuses (PILT and vigour). It also computes the number of specific trial types and blocks completed, as well as warnings received. The output is a DataFrame with these aggregated values, merged with debrief responses for each participant.

# Arguments
- `data::DataFrame`: The raw experimental data containing participant performance, trial outcomes, and responses.

# Returns
- A summarized DataFrame with performance metrics for each participant, including bonuses and trial information.
"""
function summarize_participation(data::DataFrame)

	function extract_PILT_bonus(outcome)

		if all(ismissing.(outcome)) # Return missing if participant didn't complete
			return missing
		else # Parse JSON
			bonus = filter(x -> !ismissing(x), unique(outcome))[1]
			bonus = JSON.parse(bonus)[1] 
			return maximum([0., bonus])
		end

	end
	
	participants = combine(groupby(data, [:prolific_pid, :session, :record_id, :exp_start_time]),
		:trialphase => (x -> "experiment_end_message" in x) => :finished,
		:trialphase => (x -> "kick-out" in x) => :kick_out,
		:outcomes => extract_PILT_bonus => :PILT_bonus,
		:vigour_bonus => (x -> all(ismissing.(x)) ? missing : (filter(y -> !ismissing(y), unique(x))[1])) => :vigour_bonus,
		[:trial_type, :block, :n_stimuli] => 
			((t, b, n) -> sum((t .== "PILT") .& (typeof.(b) .== Int64) .& (n .== 2))) => :n_trial_PILT,
		[:trial_type, :block, :n_stimuli] => 
			((t, b, n) -> sum((t .== "PILT") .& (typeof.(b) .== Int64) .& (n .== 3))) => :n_trial_WM,
		[:block, :trial_type, :n_stimuli] => 
			((x, t, n) -> length(unique(filter(y -> isa(y, Int64), x[(t .== "PILT") .& (n .== 2)])))) => :n_blocks_PILT,
		[:block, :trial_type, :n_stimuli] => 
			((x, t, n) -> length(unique(filter(y -> isa(y, Int64), x[(t .== "PILT") .& (n .== 3)])))) => :n_blocks_WM,
		:trialphase => (x -> sum((.!ismissing.(x)) .&& (x .== "vigour_trial"))) => :n_trials_vigour,
		:trial_presses => (x -> mean(filter(y -> !ismissing(y), x))) => 
			:vigour_average_presses,
		:trialphase => (x -> sum((.!ismissing.(x)) .&& (x .== "pit_trial"))) => 
			:n_trials_pit,
		[:trialphase, :block] => 
			((t, b) -> length(unique(b[(.!ismissing.(t)) .&& (t .== "reversal")])) - 1) => :n_reversals,
		[:trialphase, :block] => 
			((t, b) -> length(b[(.!ismissing.(t)) .&& (t .== "reversal")])) => :n_trials_reversals,
		:n_warnings => maximum => :n_warnings
	)

	# Compute totla bonus
	insertcols!(participants, :n_trial_PILT, 
		:total_bonus => ifelse.(
			ismissing.(participants.vigour_bonus),
			fill(0., nrow(participants)),
			participants.vigour_bonus
		) .+ ifelse.(
			ismissing.(participants.PILT_bonus),
			fill(0., nrow(participants)),
			participants.PILT_bonus
		)
	)

	debrief = extract_debrief_responses(data)

	participants = leftjoin(participants, debrief, 
		on = [:prolific_pid, :exp_start_time])

	return participants
end

# ╔═╡ c6d0d8c2-2c26-4e9c-8c1b-a9b23d985971
begin
	p_sum = summarize_participation(jspsych_data)
	@info "# Valid data samples: $(sum(skipmissing(p_sum.finished)))"
end

# ╔═╡ f1e41618-42b2-4e30-a33c-a198aa86ac23
p_sum

# ╔═╡ 35fb49f6-1a00-407b-bb98-89115103a9ca
filter(x -> (!ismissing(x.finished)) & (x.finished .== true), p_sum) |>
	x -> combine(x, :vigour_bonus => mean)

# ╔═╡ d0aac275-814f-48d9-9c50-566928b88904
let 
	vigour_bonus = vigour_data |>
		x -> groupby(x, [:prolific_pid, :session]) |>
		x -> combine(x, :total_reward => (x -> (maximum(x)/100)) => :vigour_bonus)
	pit_bonus = PIT_data |>
		x -> groupby(x, [:prolific_pid, :session]) |>
		x -> combine(x, :total_reward => (x -> (maximum(x)/100)) => :pit_bonus)
	innerjoin(vigour_bonus, pit_bonus, on = [:prolific_pid, :session]) |>
		x -> @mutate(x, total_bonus = vigour_bonus + pit_bonus) |>
		x -> semijoin(x, filter(x -> !ismissing(x.finished) & (x.finished .== true), p_sum), on=:prolific_pid) |>
		x -> combine(x, :total_bonus => mean)
end

# ╔═╡ 6ca0676f-b107-4cc7-b0d2-32cc345dab0d
for r in eachrow(filter(x -> x.session == "2", p_sum))
	if r.total_bonus > 0.
		println(r.prolific_pid, ", ", round(r.total_bonus, digits = 2))
	end
end

# ╔═╡ 31792570-9a09-45df-90a6-287f1bd55929
let
	sess2_df = filter(x -> !ismissing(x.finished) & x.finished & (as_float(x.record_id) .<= 429), p_sum) |>
	x -> filter(x -> !(x.prolific_pid in ["67128e3eb49c9092672386df", "67128269bf70ae794e30a0f4", "67161aa3685133a16660929d", "nov07"]), x)
	
	for r in eachrow(sess2_df)
		println(r.prolific_pid, ", ")
	end
end

# ╔═╡ e3f88292-fdb9-4628-88ee-8d935f00a761
function plot_presses_vs_var(vigour_data::DataFrame; x_var::Union{Symbol, Pair{Symbol, typeof(AlgebraOfGraphics.nonnumeric)}}=:reward_per_press, y_var::Symbol=:trial_presses, grp_var::Union{Symbol,Nothing}=nothing, xlab::Union{String,Missing}=missing, ylab::Union{String,Missing}=missing, grplab::Union{String,Missing}=missing, combine::Bool=false)
    plain_x_var = isa(x_var, Pair) ? x_var.first : x_var
    grouped_data, avg_w_data = avg_presses_w_fn(vigour_data, [plain_x_var], y_var, grp_var)

	# Set up the legend title
    grplab_text = ismissing(grplab) ? uppercasefirst(join(split(string(grp_var), r"\P{L}+"), " ")) : grplab
	
    # Define mapping based on whether grp_var is provided
    individual_mapping = grp_var === nothing ?
                         mapping(x_var, :mean_y, group=:prolific_pid) :
                         mapping(x_var, :mean_y, color=grp_var => grplab_text, group=:prolific_pid)

    average_mapping = grp_var === nothing ?
                      mapping(x_var, :avg_y) :
                      mapping(x_var, :avg_y, color=grp_var => grplab_text)

    # Create the plot for individual participants
    individual_plot = data(grouped_data) *
                      individual_mapping *
                      visual(Lines, alpha=0.15, linewidth=1)

    # Create the plot for the average line
    if grp_var === nothing
        average_plot = data(avg_w_data) *
                       average_mapping * (
                           visual(Errorbars, whiskerwidth=4) *
                           mapping(:se_y) +
                           visual(ScatterLines, linewidth=2)) *
                       visual(color=:dodgerblue)
    else
        average_plot = data(avg_w_data) *
                       average_mapping * (
                           visual(Errorbars, whiskerwidth=4) *
                           mapping(:se_y, color=grp_var => grplab_text) +
                           visual(ScatterLines, linewidth=2))
    end

    # Combine the plots
    fig = Figure(
        size=(12.2, 7.6) .* 144 ./ 2.54, # 144 points per inch, then cm
    )

    # Set up the axis
    xlab_text = ismissing(xlab) ? uppercasefirst(join(split(string(x_var), r"\P{L}+"), " ")) : xlab
    ylab_text = ismissing(ylab) ? uppercasefirst(join(split(string(y_var), r"\P{L}+"), " ")) : ylab

    if combine
        axis = (;
            xlabel=xlab_text,
            ylabel=ylab_text,
        )
        final_plot = individual_plot + average_plot
        fig = draw(final_plot; axis=axis)
    else
        # Draw the plot
        fig_patch = fig[1, 1] = GridLayout()
        ax_left = Axis(fig_patch[1, 1], ylabel=ylab_text)
        ax_right = Axis(fig_patch[1, 2])
        Label(fig_patch[2, :], xlab_text)
        draw!(ax_left, individual_plot)
        f = draw!(ax_right, average_plot)
        legend!(fig_patch[1, 3], f)
        rowgap!(fig_patch, 5)
    end
    return fig
end

# ╔═╡ 814aec54-eb08-4627-9022-19f41bcdac9f
let
	two_sess_sub = combine(groupby(vigour_data, :prolific_pid), :session => length∘unique => :n_session) |>
x -> filter(:n_session => (==(2)), x)
	plot_presses_vs_var(@filter(semijoin(vigour_data, two_sess_sub, on=:prolific_pid), trial_number > 1); x_var=:reward_per_press, y_var=:press_per_sec, grp_var=:session, xlab="Reward/press", ylab = "Press/sec", combine=false)
end

# ╔═╡ a6794b95-fe5e-4010-b08b-f124bff94f9f
let
	common_rpp = unique(PIT_data.reward_per_press)
	instrumental_data = @chain PIT_data begin
		@filter(coin==0)
		@bind_rows(vigour_data)
		@mutate(trialphase=categorical(trialphase, levels=["vigour_trial", "pit_trial"], ordered=true))
		@mutate(trialphase=~recode(trialphase, "vigour_trial" => "Vigour", "pit_trial" => "PIT w/o coin"))
		@filter(reward_per_press in !!common_rpp)
	end
	plot_presses_vs_var(PIT_data; x_var=:reward_per_press, y_var=:press_per_sec, grp_var=:session, xlab="Reward/press", ylab = "Press/sec", combine=false)
end

# ╔═╡ Cell order:
# ╠═237a05f6-9e0e-11ef-2433-3bdaa51dbed4
# ╠═0d120e19-28c2-4a98-b873-366615a5f784
# ╟─d5811081-d5e2-4a6e-9fc9-9d70332cb338
# ╠═36b348cc-a3bf-41e7-aac9-1f6d858304a2
# ╠═c6d0d8c2-2c26-4e9c-8c1b-a9b23d985971
# ╠═70257d73-ffa4-468a-bafc-694740847e06
# ╠═0874e16f-1a89-4103-aa56-ada2a5622f2f
# ╠═3d8d98d0-069c-4ef4-a13d-80e6d582deb0
# ╠═f1e41618-42b2-4e30-a33c-a198aa86ac23
# ╠═35fb49f6-1a00-407b-bb98-89115103a9ca
# ╠═d0aac275-814f-48d9-9c50-566928b88904
# ╠═f85a72d4-0959-4579-9c22-ae9362da75e3
# ╠═36e9f8f1-9c35-45e0-8d6b-0578e5ab38a0
# ╠═61774f1c-ce18-4a2d-b271-745c93d412a8
# ╠═6ca0676f-b107-4cc7-b0d2-32cc345dab0d
# ╠═31792570-9a09-45df-90a6-287f1bd55929
# ╟─cb4f46a2-1e9b-4006-8893-6fc609bcdf52
# ╟─5d487d8d-d494-45a7-af32-7494f1fb70f2
# ╟─2ff04c44-5f86-4617-9a13-6d4228dff359
# ╟─d0a2ba1e-8413-48f8-8bbc-542f3555a296
# ╟─2897a681-e8dd-4091-a2a0-bd3d4cd23209
# ╠═6244dd22-7c58-4e87-84ed-004b076bc4cb
# ╟─176c54de-e84c-45e5-872e-2471e575776d
# ╟─18956db1-4ad1-4881-a1e7-8362cf59f011
# ╟─18e9fccd-cc0d-4e8f-9e02-9782a03093d7
# ╟─17666d61-f5fc-4a8d-9624-9ae79f3de6bb
# ╟─1d1d6d79-5807-487f-8b03-efb7d0898ae8
# ╟─e902cd57-f724-4c26-9bb5-1d03443fb191
# ╟─7559e78d-7bd8-4450-a215-d74a0b1d670a
# ╠═7563e3f6-8fe2-41cc-8bdf-c05c86e3285e
# ╠═243e92bc-b2fb-4f76-9de3-08f8a2e4b25d
# ╟─0312ce5f-be36-4d9b-aee3-04497f846537
# ╠═814aec54-eb08-4627-9022-19f41bcdac9f
# ╠═6f7acf24-dbdc-4919-badb-9fe58712eacd
# ╠═3d05e879-aa5c-4840-9f4f-ad35b8d9519a
# ╟─665aa690-4f37-4a31-b87e-3b4aee66b3b1
# ╠═43d5b727-9761-48e3-bbc6-89af0c4f3116
# ╠═89258a40-d4c6-4831-8cf3-d69d984c4f6e
# ╠═a6794b95-fe5e-4010-b08b-f124bff94f9f
# ╠═8ad8c4be-3aaa-4f06-bc47-0123287b558c
# ╠═8f6d8e98-6d73-4913-a02d-97525176549a
# ╟─ffd08086-f12c-4b8a-afb6-435c8729241e
# ╠═70366452-db2e-4834-b612-59108841683f
# ╟─baf2480d-a7ea-4eba-9c11-808da603142e
# ╠═16ced201-fb4c-4040-bba6-b29dcb8b4da9
# ╠═dc957d66-1219-4a97-be46-c6c5c189c8ba
# ╟─91f6a95c-4f2e-4213-8be5-3ca57861ed15
# ╟─ce27b319-d728-46f5-aaf1-051fe252bf8b
# ╟─e3f88292-fdb9-4628-88ee-8d935f00a761
# ╠═d3a5d834-1c23-4882-adc9-c4d9571e7f71
