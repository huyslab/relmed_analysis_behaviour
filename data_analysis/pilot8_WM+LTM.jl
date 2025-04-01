### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ d39433ea-0edd-11f0-01e1-89c989a532f3
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates
	using LogExpFunctions: logistic, logit
	using Tidier
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	nothing
end

# ╔═╡ 88470afd-7ced-45ca-8384-a558269e48a5
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

# ╔═╡ c00b3551-9d5c-4aa2-93e3-b3e20b9a7aba
# Load data
begin
	_, WM_data, LTM_data, WM_test_data, LTM_test_data, _, control_task_data, _, _ = load_pilot8_data(; force_download = false, return_version = "0.2")
	nothing
end

# ╔═╡ 9a500fea-07e6-4859-84d3-abf82e679be2
function equi_groups(x::AbstractVector; n::Int = 3, labels = ["Early", "Mid", "Late"])
	min_x, max_x = extrema(x)
	edges = range(min_x, max_x, length=n+1)[2:end]  # Define n equal-width bins
	return [findfirst(v ≤ edge for edge in edges) === nothing ? labels[end] : labels[findfirst(v ≤ edge for edge in edges)] for v in x]
end

# ╔═╡ 7dd18147-82eb-4f53-9a20-a0b9711721cc
# Recoding function
recoder = (x, edges, labels) -> ([findfirst(v ≤ edge for edge in edges) === nothing ? labels[end] : labels[findfirst(v ≤ edge for edge in edges)] for v in x])

# ╔═╡ fd49957d-14ad-41a0-b2a9-f5c5ef353f3d
function compute_delays(vec::AbstractVector)
    last_seen = Dict{Any, Int}()
    delays = zeros(Int, length(vec))

    for (i, val) in enumerate(vec)
        delays[i] = haskey(last_seen, val) ? (i - last_seen[val]) : 0
        last_seen[val] = i
    end

    return delays
end

# ╔═╡ f7e64bb7-244f-4803-bb41-d9a833eb4b7d
function clean_WM_LTM_data(
	df::AbstractDataFrame
)
	# Clean data
	data_clean = exclude_PLT_sessions(df, required_n_blocks = 1)

	# Sort
	sort!(
		data_clean,
		[:prolific_pid, :session, :block, :trial]
	)

	# Apperance number
	transform!(
		groupby(data_clean, [:prolific_pid, :exp_start_time, :session, :block, :stimulus_group]),
		:trial => (x -> 1:length(x)) => :appearance
	)

	# Compute delays
	DataFrames.transform!(
		groupby(
			data_clean,
			:prolific_pid
		),
		:stimulus_group => compute_delays => :delay,
	) 

	data_clean = filter(x -> x.response != "noresp", data_clean)

	# Previous correct
	DataFrames.transform!(
		groupby(
			data_clean,
			[:prolific_pid, :stimulus_group]
		),
		:response_optimal => lag => :previous_optimal,
	)

end

# ╔═╡ b64500f3-7d41-4f38-bac7-7cdd1a0b9302
# Clean and prepare data, and combine
data_clean  = let
	WM_data_clean, LTM_data_clean = clean_WM_LTM_data.([WM_data, LTM_data])

	# Indicator variable
	WM_data_clean.task .= "WM"

	LTM_data_clean.task .= "LTM"

	data_clean = vcat(
		WM_data_clean,
		LTM_data_clean
	)
end

# ╔═╡ dfb624c9-8363-4c8e-b12e-03518b418bc2
let df = data_clean

	# Sumarrize by participant, trial, task
	acc_curve = combine(
		groupby(df, [:prolific_pid, :task, :trial]),
		:response_optimal => mean => :acc
	)

	# Summarize by trial and task
	acc_curve_sum = combine(
		groupby(acc_curve, [:task, :trial]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	acc_curve_sum.lb = acc_curve_sum.acc .- acc_curve_sum.se
	acc_curve_sum.ub = acc_curve_sum.acc .+ acc_curve_sum.se

	# Sort
	sort!(acc_curve_sum, [:task, :trial])

	# Create figure
	f = Figure(size = (700, 350))

	# Create mapping
	mp1 = (data(acc_curve_sum) * (
		mapping(
			:trial => "Trial #",
			:lb,
			:ub,
			color = :task
	) * visual(Band, alpha = 0.5) +
		mapping(
			:trial => "Trial #",
			:acc => "Prop. optimal choice",
			color = :task
	) * visual(Lines)))
	
	# Plot
	plt1 = draw!(f[1,1], mp1, axis = (; ylabel = "Prop. optimal choice"))
	legend!(f[0,1:2], plt1, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)

	
	# Summarize by appearance
	app_curve = combine(
		groupby(df, [:prolific_pid, :task, :appearance]),
		:response_optimal => mean => :acc
	)

	# Summarize by apperance and n_groups
	app_curve_sum = combine(
		groupby(app_curve, [:task, :appearance]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	app_curve_sum.lb = app_curve_sum.acc .- app_curve_sum.se
	app_curve_sum.ub = app_curve_sum.acc .+ app_curve_sum.se

	# Sort
	sort!(app_curve_sum, [:task, :appearance])

	# Create mapping
	mp2 = (data(app_curve_sum) * (
		mapping(
			:appearance => "Apperance #",
			:lb,
			:ub,
			color = :task
	) * visual(Band, alpha = 0.5) +
		mapping(
			:appearance => "Apperance #",
			:acc => "Prop. optimal choice",
			color = :task
	) * visual(Lines)))
	
	# Plot
	plt2 = draw!(f[1,2], mp2)

	f
end

# ╔═╡ 58df33c7-1713-448d-8fbf-75f167b1ad50
data_clean |> describe

# ╔═╡ b7546403-153e-43bc-a360-bbb562b0762c
let df = data_clean
	
	df.delay_bin = recoder(df.delay, [0, 1, 5, 10], ["0", "1", "2-5", "6-10"])
	
	# Summarize by participant
	app_curve = combine(
		groupby(df, [:prolific_pid, :task, :delay_bin, :appearance]),
		:response_optimal => mean => :acc
	)

	# Summarize across participants
	app_curve_sum = combine(
		groupby(app_curve, [:task, :delay_bin, :appearance]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	app_curve_sum.lb = app_curve_sum.acc .- app_curve_sum.se
	app_curve_sum.ub = app_curve_sum.acc .+ app_curve_sum.se

	# Sort
	sort!(app_curve_sum, [:task, :delay_bin, :appearance])

	# Create mapping
	mp2 = (data(app_curve_sum) * (
		mapping(
			:appearance => "Apperance #",
			:lb,
			:ub,
			color = :delay_bin  => "Delay",
			col = :task
	) * visual(Band, alpha = 0.5) +
		mapping(
			:appearance => "Apperance #",
			:acc => "Prop. optimal choice",
			color = :delay_bin  => "Delay",
			col = :task
	) * visual(Lines))) + (
		data(filter(x -> x.delay_bin == "0", app_curve_sum)) *
		(mapping(
			:appearance  => "Apperance #",
			:acc,
			:se,
			color = :delay_bin => "Delay",
			col = :task
		) * visual(Errorbars) +
		mapping(
			:appearance  => "Apperance #",
			:acc,
			color = :delay_bin  => "Delay",
			col = :task
		) * visual(Scatter))
	)

	f = Figure()

	plt = draw!(f[1,1], mp2)

	legend!(f[0,1], plt, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)

	f
end

# ╔═╡ 3f9001f2-9f20-43cf-8bac-f296b8e6129a
let df = data_clean,
	appearance_breaks = [1, 2, 4, 14, 20]
	appearance_labels = ["1", "2", "3-4", "5-14", "15-20"]

	df.learning_phase = recoder(df.appearance, appearance_breaks, appearance_labels) 

	df.learning_phase = CategoricalArray(df.learning_phase, levels = appearance_labels)

	delay_sum = combine(
		groupby(
			df,
			[:prolific_pid, :task, :learning_phase, :delay]
		),
		:response_optimal => mean => :acc
	)

	delay_sum = combine(
		groupby(
			delay_sum,
			[:task, :delay, :learning_phase]
		),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	sort!(delay_sum, [:task, :delay, :learning_phase])

	mp = data(delay_sum) * (
		mapping(
			:delay, 
			:acc, 
			color = :learning_phase => nonnumeric => "Appearance #",
			col = :task
		) * visual(Lines) +
		mapping(
			:delay, 
			:acc, 
			color = :learning_phase => nonnumeric =>  "Appearance #",
			col = :task
		) * visual(Scatter) +
		mapping(
			:delay, 
			:acc, 
			:se, 
			color = :learning_phase  => nonnumeric =>  "Appearance #",
			col = :task
		) * visual(Errorbars)
	)

	f = Figure()

	plt = draw!(f[1,1], mp, axis = (; 
		ylabel = "Prop. optimal choice",
		xlabel = "Delay"
	))

	legend!(f[0,1], plt, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)
		
	f
	
end

# ╔═╡ 178bfb62-9c88-4f7a-9054-0ed58f852c7f
let df = data_clean

	# Divide into learning phases
	df.learning_phase = equi_groups(
		df.appearance;
		n=2,
		labels = ["1-10", "13-20"]
	)

	# Summarize by participant
	delay_sum = combine(
		groupby(
			df,
			[:prolific_pid, :task, :learning_phase, :previous_optimal, :delay]
		),
		:response_optimal => mean => :acc
	)

	# Summarize across participants
	delay_sum = combine(
		groupby(
			delay_sum,
			[:delay, :task, :learning_phase, :previous_optimal]
		),
		:acc => mean => :acc,
		:acc => sem => :se,
		:acc => length => :n
	)
	
	# Sort for plotting
	sort!(delay_sum, [:task, :delay, :learning_phase, :previous_optimal])

	# Filter
	filter!(x -> x.n > 10, delay_sum)

	# Label previous_optimal
	delay_sum.previous_optimal = passmissing(ifelse).(
		delay_sum.previous_optimal,
		fill("Correct", nrow(delay_sum)),
		fill("Error", nrow(delay_sum))
	)

	mp = data(delay_sum) * (
		mapping(
			:delay, 
			:acc, 
			color = :learning_phase  => "Apperance #",
			group = :previous_optimal => "Previous choice",
			col = :task
		) * visual(Lines) +
		mapping(
			:delay, 
			:acc, 
			color = :learning_phase  => "Apperance #",
			marker = :previous_optimal => "Previous choice",
			col = :task
		) * visual(Scatter) +
		mapping(
			:delay, 
			:acc, 
			:se, 
			color = :learning_phase => "Apperance #",
			group = :previous_optimal => "Previous choice",
			col = :task
		) * visual(Errorbars)
	)

	f = Figure()

	plt = draw!(f[1,1], mp, axis = (; 
		ylabel = "Prop. optimal choice",
		xlabel = "Delay"
	))

	legend!(f[0,1], plt, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left, nbanks = 2)
		
	f
	
end

# ╔═╡ Cell order:
# ╠═d39433ea-0edd-11f0-01e1-89c989a532f3
# ╠═88470afd-7ced-45ca-8384-a558269e48a5
# ╠═c00b3551-9d5c-4aa2-93e3-b3e20b9a7aba
# ╠═b64500f3-7d41-4f38-bac7-7cdd1a0b9302
# ╠═dfb624c9-8363-4c8e-b12e-03518b418bc2
# ╠═58df33c7-1713-448d-8fbf-75f167b1ad50
# ╠═b7546403-153e-43bc-a360-bbb562b0762c
# ╠═3f9001f2-9f20-43cf-8bac-f296b8e6129a
# ╠═178bfb62-9c88-4f7a-9054-0ed58f852c7f
# ╠═9a500fea-07e6-4859-84d3-abf82e679be2
# ╠═7dd18147-82eb-4f53-9a20-a0b9711721cc
# ╠═f7e64bb7-244f-4803-bb41-d9a833eb4b7d
# ╠═fd49957d-14ad-41a0-b2a9-f5c5ef353f3d
