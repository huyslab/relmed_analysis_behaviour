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

# ╔═╡ Cell order:
# ╠═d39433ea-0edd-11f0-01e1-89c989a532f3
# ╠═88470afd-7ced-45ca-8384-a558269e48a5
# ╠═c00b3551-9d5c-4aa2-93e3-b3e20b9a7aba
# ╠═b64500f3-7d41-4f38-bac7-7cdd1a0b9302
# ╠═dfb624c9-8363-4c8e-b12e-03518b418bc2
# ╠═f7e64bb7-244f-4803-bb41-d9a833eb4b7d
# ╠═fd49957d-14ad-41a0-b2a9-f5c5ef353f3d
