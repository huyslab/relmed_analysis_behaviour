### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 62fb2c44-70f7-11ef-2499-9d1ed7c02f46
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using CairoMakie, Random, DataFrames, Distributions, StatsBase,
		ForwardDiff, LinearAlgebra, JLD2, FileIO, CSV, Dates, JSON, RCall, Turing, ParetoSmooth, MCMCDiagnosticTools, Printf, Combinatorics, JuMP, HiGHS, Cbc
	using LogExpFunctions: logistic, logit

	Turing.setprogress!(false)

	include("$(pwd())/PILT_models.jl")
	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/stats_utils.jl")
	include("$(pwd())/plotting_utils.jl")
	include("$(pwd())/single_p_QL.jl")
	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/generate_experimental_sequences/FI_sequences_utils.jl")
	nothing
end

# ╔═╡ a4db20c5-a30b-454f-a04b-ffbe25459e07
aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

# ╔═╡ 2b7e29ee-742c-4afd-95b9-602b3e42a6f2
function map_data_to_single_p_QL(
	data::AbstractDataFrame
)

	tdata = dropmissing(data)

	return (;
		block = collect(tdata.block),
		choice = tdata.choice,
		outcomes = hcat(tdata.feedback_suboptimal, tdata.feedback_optimal)
	)

end

# ╔═╡ b76a5feb-e86d-46fb-8c92-8dc662248f6b
let
	FI_seqs = [compute_save_FIs_for_all_seqs(;
		n_trials = 10,
		n_confusing = nc,
		fifty_high = fh
	) for nc in [1, 2] for fh in [true, false]]

	FIs = [x[1] for x in FI_seqs]

	common_seqs = [x[2] for x in FI_seqs]

	magn_seqs = [x[3] for x in FI_seqs]

	optimize_FI_distribution(
		n_wanted = [1, 1, 16, 17],
		FIs = FIs,
		common_seqs = common_seqs,
		magn_seqs = magn_seqs,
		ω_FI = 0.1,
		filename = "results/exp_sequences/eeg_pilot_FI_opt.jld2"
	)
end

# ╔═╡ 1e9e4f9c-6a61-4f0d-a17b-08ba5875616f
let n_trials = 10,
	n_confusing = 2,
	FI_res = 6

	# All possible sequences of confusing feedback
	sequences = collect(
		multiset_permutations(
			vcat(
				fill(false, n_confusing), 
				fill(true, n_trials - n_confusing)
			),
			n_trials
		)
	)

	# Compute FIs ---------------
	lk = ReentrantLock()
	FIs = fill(fill(-99., FI_res, FI_res), length(sequences))
	
	# Preallocate
	Threads.@threads for i in eachindex(sequences)
		thisFI = sum_FI_for_feedback_sequence(;
				task = sequence_to_task_df(;
					feedback_common = sequences[i],
					feedback_magnitude_high = fill(1., n_trials),
					feedback_magnitude_low = fill(0.01, n_trials)
				),
				ρ_vals = range(1., 10., length = FI_res),
				a_vals = range(-1.5, 1.5, length = FI_res),
				initV = aao,
				across_summary_method = identity,
				n_blocks = 200
			)

		lock(lk) do
			FIs[i] = thisFI
		end
	end

	FIs = zscore_avg_matrices(FIs)

	# Sort
	sorted_FIs = sort(FIs)
	sorted_seq = hcat(sequences[sortperm(FIs)]...) 

	# Plot heatmap
	f = Figure()

	ax = Axis(
		f[1,1],
		xlabel = "Sequence",
		ylabel = "FI",
		yticks = 1:2:length(sequences),
		ytickformat = 
			values -> ["$(round(sorted_FIs[round(Int64, v)], digits = 2))" 
				for v in values]
	)

	hidexdecorations!(ax, label = false)

	heatmap!(ax, sorted_seq)
	
	f
		
end

# ╔═╡ 5dbe7bc7-f348-4efa-91f3-17a1261a4e78
# Look at FI across parameter range for one sequence
let n_trials = 10,
	FI_res = 10

	# Create task dataframe
	task = sequence_to_task_df(;
		feedback_common = shuffled_fill([true, false], n_trials),
		feedback_magnitude_high = fill(1., n_trials),
		feedback_magnitude_low = shuffled_fill([0.01, 0.5], n_trials)
	)

	# Compute FI for each parameter combination
	FIs = sum_FI_for_feedback_sequence(;
		task = task,
		ρ_vals = range(0., 10., length = FI_res),
		a_vals = range(-2., 2., length = FI_res),
		initV = aao,
		across_summary_method = identity
	)

	println(mean(FIs))

	# Plot
	f = Figure()

	ax = Axis(
		f[1,1],
		aspect = 1
	)

	heatmap!(
		ax,
		range(0., 10., length = 15),
		range(-2., 2., length = 15),
		FIs
	)

	ax_d = Axis(
		f[1,2]
	)

	hist!(
		ax_d,
		vec(FIs)
	)

	f
end

# ╔═╡ Cell order:
# ╠═62fb2c44-70f7-11ef-2499-9d1ed7c02f46
# ╠═a4db20c5-a30b-454f-a04b-ffbe25459e07
# ╠═2b7e29ee-742c-4afd-95b9-602b3e42a6f2
# ╠═b76a5feb-e86d-46fb-8c92-8dc662248f6b
# ╠═1e9e4f9c-6a61-4f0d-a17b-08ba5875616f
# ╠═5dbe7bc7-f348-4efa-91f3-17a1261a4e78
