### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, Turing, SHA, HypothesisTests, Tidier
	using Tidier, GLM, MixedModels, PlutoUI, LaTeXStrings, ColorSchemes
	using LogExpFunctions: logistic, logit
	import OpenScienceFramework as OSF
	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/plotting_utils.jl")
	include("$(pwd())/stats_utils.jl")
	include("$(pwd())/model_utils.jl")
	include("$(pwd())/PILT_models.jl")
	nothing
end

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

## Plotting function ------------------
function plot_ecdf_by_session(
	df; 
	variable::Symbol = :response_optimal,
	xlabel="Prop. optimal choice", 
	ylabel="ECDF", 
	title="ECDF of accuracy by session",
	null_distribution::AbstractVector = Float64[],
	legend_position::Symbol = :lt
)
	sessions = unique(df.session)
	colors = Makie.wong_colors()[1:length(sessions)]
	fig = Figure(size = (700, 400))
	ax = Axis(
		fig[1, 1], 
		xlabel = xlabel, 
		ylabel = ylabel, 
		title = title,
		xautolimitmargin = (0., 0.),
		yautolimitmargin = (0., 0.),
		ygridvisible = true,
		yticks = 0.:0.2:1.0
	)

	for (i, sess) in enumerate(sessions)
		vals = sort(df[!, variable][df.session .== sess])
		n = length(vals)
		y = range(1/n, 1; length=n)
		lines!(ax, vals, y, color = colors[i], label = "Session $sess", linewidth = 2)

		# Quintiles
		quintiles = 0.2:0.2:0.8
		for q in quintiles
			idx = clamp(round(Int, q * n), 1, n)
			xq = vals[idx]
			yq = y[idx]
			# Vertical line from x-axis to curve
			lines!(ax, [xq, xq], [0, yq], color = colors[i], linestyle = :dash, linewidth = 1)
		end
	end

	
	# Plot of null distribution if provided
	if !isempty(null_distribution)
		null_pdf = Makie.KernelDensity.kde(null_distribution)
		scaled_density = null_pdf.density ./ (maximum(null_pdf.density) * 2)
		lines!(ax, null_pdf.x, scaled_density, color = :grey, linestyle = :dash, linewidth = 2, label = "Null distribution")

		q95 = quantile(null_distribution, 0.95)
        # Fill area under curve from q95 onwards
        idx = findall(x -> x ≥ q95, null_pdf.x)
        if !isempty(idx)
            x_fill = vcat(null_pdf.x[idx], last(null_pdf.x[idx]), first(null_pdf.x[idx]))
            y_fill = vcat(scaled_density[idx], 0.0, 0.0)
            poly!(ax, x_fill, y_fill, color = (:grey, 0.3), strokewidth = 0, label = "Unlikely under null")
        end
	end

	axislegend(ax, position = legend_position, framevisible = false)

	fig
end

# PILT --------------------------------
# Load data
begin
	PILT_data, test_data, _, _, jspsych_data = load_pilot9_data()
	nothing
end

# Clean
PILT_data_clean = let
	# Clean data
	PILT_data_clean = exclude_PLT_sessions(PILT_data, required_n_blocks = 21)
	filter!(x -> x.response != "noresp", PILT_data_clean)

	# Remove empty columns
	select!(
		PILT_data_clean,
		Not([:EV_right, :EV_left])
	)
end

# Compute null distribution
PILT_null_distribution = let n_seeds = 100000

	# Position of optimal choice
	optimal_choice = unique(select(
		filter(x -> (x.block < 16) && x.trial > 1, PILT_data_clean), # Filter down to size of trial1 sequennce
		:session, :block, :trial, :optimal_right))

	simulated_props = [mean((rand(nrow(optimal_choice)) .> .5) .== optimal_choice.optimal_right) for _ in 1:n_seeds]
	
end

# Summarize empirical accuracy and plot
let
	acc_sum = combine(
        groupby(
            filter(x -> x.trial > 1, PILT_data_clean),
            [:prolific_pid, :session]
        ),
        :response_optimal => mean => :response_optimal
    )

	plot_ecdf_by_session(acc_sum; 
		null_distribution = PILT_null_distribution,
		title = "PILT accuracy by session"
	)
end

# Reversal ------------------------------
# Load data
begin
	# Load data
	_, _, _, _, _, _, reversal_data, _ = load_pilot6_data()
	nothing
end

# Clean data
reversal_data_clean = let
	# Exclude sessions
	reversal_data_clean = exclude_reversal_sessions(reversal_data; required_n_trials = 150)

	# Sort
	sort!(reversal_data_clean, [:prolific_pid, :session, :block, :trial])

	# Cumulative trial number
	DataFrames.transform!(
		groupby(reversal_data_clean, [:prolific_pid, :session]),
		:trial => (x -> 1:length(x)) => :ctrial
	)

	# Exclude trials
	filter!(x -> !isnothing(x.response_optimal), reversal_data_clean)
end

# Null distribution
n_reversal_null, reversal_acc_null = let n_seeds = 100000
	
	# Load sequennce
	reversal_sequence = filter(x -> x.session == 1, DataFrame(CSV.File("data/trial1_eeg_reversal.csv")))

	criteria = unique(reversal_sequence[!, [:block, :criterion]]).criterion


	# Simulate and summarize n_reversals
	function simulate_reversals(
		rng::AbstractRNG; 
		statistic::String 
	)

		# Draw side bias
		side_bias = rand(rng, Beta(2, 2))

		choices = rand(rng, 150) .> side_bias # Random choices

		acc_counter = 0
		global_acc_counter = 0
		trial_counter = 1
		block_counter = 1
		block_data = reversal_sequence[reversal_sequence.block .== block_counter, :]
		for (i, c) in enumerate(choices)
			acc = c ? block_data.feedback_right[trial_counter] === 1. : block_data.feedback_left[trial_counter] == 1.
			acc_counter += acc

			# Don't count first trial
			if i > 1
				global_acc_counter += acc
			end 

			trial_counter += 1
			if acc_counter >= criteria[block_counter] || trial_counter > 80
				# Reversal
				acc_counter = 0
				block_counter += 1
				trial_counter = 1
				block_data = reversal_sequence[reversal_sequence.block .== block_counter, :]
			end
		end

		if statistic == "n_reversals"
			return block_counter - 1 # Number of reversals is number of blocks - 1
		end

		if statistic == "accuracy"
			return global_acc_counter / 149. # Accuracy is total correct choices divided by total trials
		end
		error("Unknown statistic: $statistic")
	end

	# Simulate multiple times
	rng = Xoshiro(0)
	n_reversal_null = [simulate_reversals(rng; statistic = "n_reversals") for _ in 1:n_seeds]
	reversal_acc_null = [simulate_reversals(rng; statistic = "accuracy") for _ in 1:n_seeds]

	n_reversal_null, reversal_acc_null
end

# Summarize empirical number of reversals and plot
let
	acc_sum = combine(
        groupby(
            filter(x -> x.trial > 1, reversal_data_clean),
            [:prolific_pid, :session]
        ),
        :block => (x -> maximum(x) - 1) => :n_reversals
    )

	plot_ecdf_by_session(acc_sum; 
		variable = :n_reversals,
		null_distribution = n_reversal_null,
		xlabel = "Number of reversals",
		title = "Number of reversals by session"
	)
end

let
	acc_sum = combine(
        groupby(
            filter(x -> x.trial > 1, reversal_data_clean),
            [:prolific_pid, :session]
        ),
        :response_optimal => mean => :response_optimal
    )

	plot_ecdf_by_session(acc_sum; 
		null_distribution = reversal_acc_null,
		xlabel = "Prop. optimal choice",
		title = "Reversal accuracy by session",
		legend_position = :lt
	)
end