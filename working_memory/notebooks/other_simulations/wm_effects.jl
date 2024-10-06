### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ d2bb17a8-48fc-4b85-b36a-b022d54b316f
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using CairoMakie, Random, DataFrames, Distributions, StatsBase,
		ForwardDiff, LinearAlgebra, JLD2, FileIO, CSV, Dates, JSON, RCall, Turing, ParetoSmooth, MCMCDiagnosticTools, Printf
	using LogExpFunctions: logistic, logit

	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/simulate.jl")
	include("$(pwd())/turing_models.jl")
	include("$(pwd())/plotting_utils.jl")
end

# ╔═╡ 8c71424c-42ee-4cea-a397-b830dcc97db5
include("$(pwd())/turing_models.jl")

# ╔═╡ caa2114a-5007-4df4-81bc-500730586645
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

# ╔═╡ 7aee5c28-6c55-11ef-2d79-35681430eadc
# Load and clean data
begin
	PLT_data = load_PLT_data()

	PLT_data = exclude_PLT_sessions(PLT_data)

	DataFrames.transform!(groupby(PLT_data, [:prolific_pid, :session, :block]),
		:isOptimal => count_consecutive_ones => :consecutiveOptimal
	)

	PLT_data = exclude_PLT_trials(PLT_data)

	nothing
end

# ╔═╡ 5a8bb43c-efb0-48a1-9459-4148067ba768
begin
	(f_rl, rl_chce) = let
		pilot_data = prepare_for_fit(PLT_data)
		aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
		
		optimize_multiple_single_p_QL(
			pilot_data[1],
			estimate = "MAP",
			initV = aao,
			include_true = false
		)
	end
	rl_chce = rename!(rl_chce, :predicted_choice => :RL_predictions)
	describe(f_rl)
end

# ╔═╡ 25a9b34e-c739-4134-a948-9a2118df4e4a
begin
	(f_rlwm, rlwm_chce) = let
		pilot_data = prepare_for_fit(PLT_data)
		aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
		optimize_multiple_single_p_QL(
			pilot_data[1],
			estimate = "MAP",
			initV = aao,
			include_true = false,
			model = RLWM_ss,
			set_size = fill(2, maximum(pilot_data[1].block)),
			initial_params = [mean(truncated(Normal(0., 2.), lower = 0.)), 0.5, 0.5, 0.5, mean(truncated(Normal(2., 4.), lower = 1.))],
			parameters = [:ρ , :a, :F_wm, :W, :C],
			transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w0),
			priors = Dict(
		        :ρ => truncated(Normal(0., 1.), lower = 0.),
		        :a => Normal(0., 0.5),
		        :F_wm => Normal(0., 0.5),
		        :W => Normal(0., 0.5),
		        :C => truncated(Normal(3., 2.), lower = 1.)
		    )
		)
	end
	rename!(rlwm_chce, :predicted_choice => :RLWM_predictions)
	describe(f_rlwm)
end

# ╔═╡ d3eb8298-7458-4917-8a83-7a00597d6df7
begin
	(f_rlwm_pmst, rlwm_pmst_chce) = let
		pilot_data = prepare_for_fit(PLT_data)
		aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
		optimize_multiple_single_p_QL(
			pilot_data[1],
			estimate = "MAP",
			initV = aao,
			include_true = false,
			model = RLWM_pmst,
			set_size = fill(2, maximum(pilot_data[1].block)),
			initial_params = [mean(truncated(Normal(0., 2.), lower = 0.)), 0.5, 0.5, mean(truncated(Normal(4., 2.), lower = 1.))],
			parameters = [:ρ , :a, :W, :C],
			transformed = Dict(:a => :α, :W => :w0),
			priors = Dict(
		        :ρ => truncated(Normal(0., 1.), lower = 0.),
		        :a => Normal(0., 0.5),
		        :W => Normal(0., 0.5),
		        :C => truncated(Normal(3., 2.), lower = 1.)
		    )
		)
	end
	df_preds = let
		rename!(rlwm_pmst_chce, :predicted_choice => :RLWM_p_predictions)
		df1 = leftjoin(
			rl_chce[!, [:PID, :trial, :block, :valence, :RL_predictions]], rlwm_chce[!, [:PID, :trial, :block, :valence, :RLWM_predictions]],
			on = [:PID, :trial, :block, :valence]
		)
		df = leftjoin(df1, rlwm_pmst_chce, on = [:PID, :trial, :block, :valence])
		df.true_choice = ifelse.(df.true_choice, 1, 0)
		stack(df, [:true_choice, :RL_predictions, :RLWM_predictions, :RLWM_p_predictions])
	end

	describe(f_rlwm_pmst)
end

# ╔═╡ 9017be6c-5292-4c47-99b6-0aa2873511bf
let
    f = Figure(size = (800, 1400))
	
	plot_prior_accuracy!(
		GridLayout(f[1,1]),
		df_preds;
		group = :variable,
		pid_col = :PID,
		acc_col = :value,
		legend = true,
		legend_rows = 1,
		colors = Makie.colorschemes[:seaborn_pastel6],
		error_band = "PI"
	)
	Label(f[0,1], "All blocks", fontsize = 24, font = :bold)
	
	plot_prior_accuracy!(
		GridLayout(f[3,1]),
		filter(x -> x.valence > 0, df_preds);
		group = :variable,
		pid_col = :PID,
		acc_col = :value,
		legend = true,
		legend_rows = 1,
		colors = Makie.colorschemes[:seaborn_pastel6],
		error_band = "PI"
	)
	Label(f[2,1], "Reward blocks", fontsize = 24, font = :bold)

	plot_prior_accuracy!(
		GridLayout(f[5,1]),
		filter(x -> x.valence < 0, df_preds);
		group = :variable,
		pid_col = :PID,
		acc_col = :value,
		legend = true,
		legend_rows = 1,
		colors = Makie.colorschemes[:seaborn_pastel6],
		error_band = "PI"
	)
	Label(f[4,1], "Punishment blocks", fontsize = 24, font = :bold)
	colsize!(f.layout, 1, Relative(1))
	f
end

# ╔═╡ 49a0b75f-b70c-47bb-83ff-fbd82de9c03b
# ╠═╡ disabled = true
#=╠═╡
begin
	no_early_data = filter(x -> !x.early_stop, PLT_data)

	no_early_data.confusing = ifelse.(
		no_early_data.optimalRight .== 1, 
		no_early_data.outcomeRight .< no_early_data.outcomeLeft,
		no_early_data.outcomeLeft .< no_early_data.outcomeRight
	)

	DataFrames.transform!(groupby(no_early_data, [:prolific_pid, :session, :block]),
		:confusing => (x -> string(findall(x))) => :confusing_sequence
	)

	sequences = unique(no_early_data.confusing_sequence)
	sequences = DataFrame(
		confusing_sequence = sequences,
		sequence = 1:length(sequences)
	)

	no_early_data = leftjoin(no_early_data, sequences, on = :confusing_sequence)

	function plot_by_sequence(
		data::DataFrame;
		group::Union{Symbol, Missing}=missing,
		acc_col::Symbol=:isOptimal
	)
		by_sequence = groupby(data, :sequence)
	
		f_sequences = Figure(size = (800, 1400))
		axs = []
		for (i, gdf) in enumerate(by_sequence)
		
			r = div(i - 1, 5) + 1
			c = rem(i - 1, 5) + 1

			if ismissing(group)
			    gdf[!, :grp] .= 1
			else
			    gdf[!, :grp] .= gdf[!, group]
			end
			
			ax = plot_prior_accuracy!(
				GridLayout(f_sequences[r, c]), 
				gdf; 
				group = :grp,
				linewidth = 1.,
				pid_col = :prolific_pid,
				acc_col = acc_col,
				legend = false,
				colors = Makie.colorschemes[:seaborn_pastel6],
				error_band = "SE"
			)
	
			if !("Int64[]" in gdf.confusing_sequence)
				vlines!(
					ax,
					filter(x -> x <= 13, eval(Meta.parse(unique(gdf.confusing_sequence)[1])) .+ 1),
					color = :red,
					linewidth = 1
				)
			end

			#vlines!(ax, [6], linestyle = :dash, color = :grey)
			
			hideydecorations!(ax, ticks = c != 1, ticklabels = c != 1)
			hidexdecorations!(ax, 
				ticks = length(by_sequence) >= (5 * r + c), 
				ticklabels = length(by_sequence) >= (5 * r + c))
			hidespines!(ax)
	
			push!(axs, ax)
			
		end
	
		linkyaxes!(axs...)
		
		f_sequences
	end

	f_sequences = plot_by_sequence(no_early_data)

	f_sequences
end
  ╠═╡ =#

# ╔═╡ ee3f229f-b97e-45ce-a184-132bd9ea7a34
# ╠═╡ disabled = true
#=╠═╡
let
	avg_acc = combine(
		groupby(no_early_data, [:prolific_pid, :session, :block]),
		:isOptimal => mean => :overall_acc
	)

	avg_acc = combine(
		groupby(avg_acc, [:prolific_pid, :session]),
		:overall_acc => mean => :overall_acc
	)

	avg_acc.overall_acc_grp = avg_acc.overall_acc .> median(avg_acc.overall_acc)

	no_early_data = leftjoin(no_early_data, avg_acc, 
		on = [:prolific_pid, :session],
		order = :left
	)

	f_seuqnce_overall_acc = plot_by_sequence(
		no_early_data; 
		group = :overall_acc_grp
	)

	Legend(
		f_seuqnce_overall_acc[0, 1:5],
		[LineElement(color = c, linewidth = 3) for c in Makie.colorschemes[:seaborn_pastel6][[2, 1]]],
		["Best performing half", "Worst performing half"],
		framevisible = false,
		orientation = :horizontal
	)

	rowsize!(f_seuqnce_overall_acc.layout, 0, Relative(0.01))

	f_seuqnce_overall_acc

end
  ╠═╡ =#

# ╔═╡ 99261bdb-3984-4c08-b2d3-e5dfbd50a4b4
md"""
### Worst performing half: comparison of model predictions
"""

# ╔═╡ 4cbb9f67-d395-4203-9813-c99037420654
# ╠═╡ disabled = true
#=╠═╡
let
	#no_early_data = filter(x -> !x.early_stop, PLT_data)
	avg_acc = combine(
		groupby(no_early_data, [:prolific_pid, :session, :block]),
		:isOptimal => mean => :overall_acc
	)

	avg_acc = combine(
		groupby(avg_acc, [:prolific_pid, :session]),
		:overall_acc => mean => :overall_acc
	)

	avg_acc.overall_acc_grp = avg_acc.overall_acc .> median(avg_acc.overall_acc)

	no_early_data = leftjoin(no_early_data, avg_acc, 
		on = [:prolific_pid, :session],
		order = :left
	)

	worst_half = filter(x -> !x.overall_acc_grp, no_early_data)
	pids = prepare_for_fit(PLT_data)[2]
	worst_half = innerjoin(worst_half, pids[!, [:prolific_pid, :PID]], on = :prolific_pid)
	worst_half.true_choice = ifelse.(worst_half.isOptimal, 1, 0)

	preds = leftjoin(rl_chce[!, [:PID, :trial, :block, :valence, :RL_predictions]], rlwm_chce[!, [:PID, :trial, :block, :valence, :RLWM_predictions]], on = [:PID, :trial, :block, :valence])

	preds_worst = leftjoin(worst_half, preds, on = [:PID, :trial, :block, :valence])
	preds_worst = stack(preds_worst, [:true_choice, :RL_predictions, :RLWM_predictions])

	f_sequence_preds = plot_by_sequence(
		preds_worst; 
		group = :variable,
		acc_col = :value
	)

	Legend(
		f_sequence_preds[0, 1:5],
		[LineElement(color = c, linewidth = 3) for c in Makie.colorschemes[:seaborn_pastel6][[3, 2, 1]]],
		["true accuracy", "RL predictions", "RLWM predictions"],
		framevisible = false,
		orientation = :horizontal
	)

	rowsize!(f_sequence_preds.layout, 0, Relative(0.01))

	f_sequence_preds
end
  ╠═╡ =#

# ╔═╡ 603c51b6-1200-4898-8119-9452d63b4836
md"""
### Best performing half: comparison of model predictions
"""

# ╔═╡ 635f85b4-7040-4cf8-bf4b-9a0cec9912d8
# ╠═╡ disabled = true
#=╠═╡
let
	#no_early_data = filter(x -> !x.early_stop, PLT_data)
	avg_acc = combine(
		groupby(no_early_data, [:prolific_pid, :session, :block]),
		:isOptimal => mean => :overall_acc
	)

	avg_acc = combine(
		groupby(avg_acc, [:prolific_pid, :session]),
		:overall_acc => mean => :overall_acc
	)

	avg_acc.overall_acc_grp = avg_acc.overall_acc .> median(avg_acc.overall_acc)

	no_early_data = leftjoin(no_early_data, avg_acc, 
		on = [:prolific_pid, :session],
		order = :left
	)

	best_half = filter(x -> x.overall_acc_grp, no_early_data)
	pids = prepare_for_fit(PLT_data)[2]
	best_half = innerjoin(best_half, pids[!, [:prolific_pid, :PID]], on = :prolific_pid)
	best_half.true_choice = ifelse.(best_half.isOptimal, 1, 0)

	preds = leftjoin(rl_chce[!, [:PID, :trial, :block, :valence, :RL_predictions]], rlwm_chce[!, [:PID, :trial, :block, :valence, :RLWM_predictions]], on = [:PID, :trial, :block, :valence])

	preds_best = leftjoin(best_half, preds, on = [:PID, :trial, :block, :valence])
	preds_best = stack(preds_best, [:true_choice, :RL_predictions, :RLWM_predictions])

	f_sequence_preds = plot_by_sequence(
		preds_best; 
		group = :variable,
		acc_col = :value
	)

	Legend(
		f_sequence_preds[0, 1:5],
		[LineElement(color = c, linewidth = 3) for c in Makie.colorschemes[:seaborn_pastel6][[3, 2, 1]]],
		["true accuracy", "RL predictions", "RLWM predictions"],
		framevisible = false,
		orientation = :horizontal
	)

	rowsize!(f_sequence_preds.layout, 0, Relative(0.01))

	f_sequence_preds
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═d2bb17a8-48fc-4b85-b36a-b022d54b316f
# ╠═caa2114a-5007-4df4-81bc-500730586645
# ╠═7aee5c28-6c55-11ef-2d79-35681430eadc
# ╠═5a8bb43c-efb0-48a1-9459-4148067ba768
# ╠═25a9b34e-c739-4134-a948-9a2118df4e4a
# ╠═8c71424c-42ee-4cea-a397-b830dcc97db5
# ╠═d3eb8298-7458-4917-8a83-7a00597d6df7
# ╠═9017be6c-5292-4c47-99b6-0aa2873511bf
# ╠═49a0b75f-b70c-47bb-83ff-fbd82de9c03b
# ╠═ee3f229f-b97e-45ce-a184-132bd9ea7a34
# ╟─99261bdb-3984-4c08-b2d3-e5dfbd50a4b4
# ╠═4cbb9f67-d395-4203-9813-c99037420654
# ╟─603c51b6-1200-4898-8119-9452d63b4836
# ╠═635f85b4-7040-4cf8-bf4b-9a0cec9912d8
