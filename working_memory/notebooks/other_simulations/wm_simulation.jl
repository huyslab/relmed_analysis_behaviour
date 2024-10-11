### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 772362f4-75c0-11ef-25de-55ffd5418ed2
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
	
	Turing.setprogress!(false)
	
	include("$(pwd())/simulate.jl")
	include("$(pwd())/turing_models.jl")
	include("$(pwd())/plotting_utils.jl")
end

# ╔═╡ bde4f8c5-f7fe-4b5c-87e5-2bb393698a25
include("$(pwd())/turing_models.jl")

# ╔═╡ 4eb03947-90d6-4281-9d63-372f76bb4900
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

# ╔═╡ 24289c6f-4974-41ad-822b-6f3df77f7cb4
begin
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
	prior_sample = simulate_from_prior(
		100;
		model = RL_ss,
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5)
		),
		initial = aao,
		transformed = Dict(:a => :α),
		structure = (
            n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = [2, 4, 6]
		),
		gq = true,
		random_seed = 123
	)
	describe(prior_sample)
end

# ╔═╡ f99c4eb3-f018-4d3f-995c-88765ad6633d
let
	f = plot_prior_predictive_by_valence(
		prior_sample,
		[:Q_optimal, :Q_suboptimal];
		group = :set_size,
		legend = true,
		legend_title = "Set size",
		fig_size = (1000, 1000),
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ f743bbd4-44aa-437e-910a-6c4830f08ef4
begin
	prior_sample_wm = simulate_from_prior(
		100;
		model = RLWM_ss,
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:F_wm => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		initial = aao,
		transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w0),
		structure = (
            n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = [2, 4, 6]
		),
		gq = true,
		random_seed = 123
	)
	describe(prior_sample_wm)
end

# ╔═╡ e9a9cecb-c5c6-422d-b089-b1bb0c11e79f
let
	f = plot_prior_predictive_by_valence(
		prior_sample_wm,
		[:Q_optimal, :Q_suboptimal];
		W_cols = [:W_optimal, :W_suboptimal],
		group = :set_size,
		ylab = ("Q-value", "W-value"),
		legend = true,
		legend_title = "Set size",
		fig_size = (1000, 1000),
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ f1918ce1-9435-4757-b970-68e795a49557
begin
	prior_sample_pmst = simulate_from_prior(
		100;
		model = RLWM_pmst,
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		initial = aao,
		transformed = Dict(:a => :α, :W => :w0),
		structure = (
			n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = [2, 4, 6]
		),
		gq = true,
		random_seed = 123
	)
	describe(prior_sample_pmst)
end

# ╔═╡ acf1ba59-c202-4924-a58b-6ba3d6f7f338
let
	f = plot_prior_predictive_by_valence(
		prior_sample_pmst,
		[:Q_optimal, :Q_suboptimal];
		W_cols = [:W_optimal, :W_suboptimal],
		group = :set_size,
		ylab = ("Q-value", "W-value"),
		legend = true,
		legend_title = "Set size",
		fig_size = (1000, 1000),
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ faf9b5b6-737d-403c-b613-aa7558cf67ba
begin
	prior_sample_pmst2 = simulate_from_prior(
		100;
		model = RLWM_pmst_mk2,
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		),
		initial = aao,
		transformed = Dict(:a => :α, :W => :w0),
		structure = (
			n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = [2, 4, 6]
		),
		gq = true,
		random_seed = 123
	)
	describe(prior_sample_pmst2)
end

# ╔═╡ bab8c088-f8c7-4eb9-8cfa-0583c660eef1
let
	f = plot_prior_predictive_by_valence(
		prior_sample_pmst2,
		[:Q_optimal, :Q_suboptimal];
		W_cols = [:W_optimal, :W_suboptimal],
		group = :set_size,
		ylab = ("Q-value", "W-value"),
		legend = true,
		legend_title = "Set size",
		fig_size = (1000, 1000),
		colors = Makie.colorschemes[:seaborn_pastel6]
	)	
	f
end

# ╔═╡ bc443d65-78b1-42bc-b9ce-35a7ed713c51


# ╔═╡ e6b933c4-c92d-47f4-a28c-a53c20ede1e6
let
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
	f_pmst = optimization_calibration(
		prior_sample_pmst2,
		optimize_multiple,
		estimate = "MAP",
		model = RLWM_pmst_mk2,
		initial = aao, 
		initial_params = [mean(truncated(Normal(0., 2.), lower = 0.)), 0.5, 0.5, 0.5, 0.5, mean(truncated(Normal(4., 2.), lower = 1.))],
		parameters = [:ρ, :a, :W, :C],
		transformed = Dict(:a => :α, :W => :w0),
		priors = Dict(
			:ρ => truncated(Normal(0., 1.), lower = 0.),
			:a => Normal(0., 0.5),
			:W => Normal(0., 0.5),
			:C => truncated(Normal(3., 2.), lower = 1.)
		)
	)

	f_pmst
end

# ╔═╡ d4cb39eb-c881-4d0d-baaa-e19159be2780
md"""
### Simulate performance on different sequences
"""

# ╔═╡ 70626be0-27c5-419d-a27e-9fd2230f941c
begin
	prior_samples_seq = let 
		prior_samples = Vector{DataFrame}(undef, 20)
		
		for i in 1:20
			strct = create_random_task(;
				n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = [2, 4, 6]
		    )
			rl = simulate_from_prior(
				10;
				model = RL_ss,
				priors = Dict(
					:ρ => Dirac(3.),
					:a => Dirac(0.4)
				),
				initial = aao,
				transformed = Dict(:a => :α),
				fixed_struct = strct,
				gq = true,
				random_seed = 123
			)
			insertcols!(rl, :model => "RL")
			# rlwm = simulate_from_prior(
			# 	10;
			# 	model = RLWM_ss,
			# 	priors = Dict(
			# 		:ρ => Dirac(3.),
			# 		:a => Dirac(0.4),
			# 		:F_wm => Normal(0., 0.5),
			# 		:W => Normal(0., 0.5),
			# 		:C => truncated(Normal(3., 2.), lower = 1.)
			# 	),
			# 	initial = aao,
			# 	transformed = Dict(:a => :α, :F_wm => :φ_wm, :W => :w0),
			# 	fixed_struct = strct,
			# 	gq = true,
			# 	random_seed = 123
			# )
			# insertcols!(rlwm, :model => "RLWM")
			# pmst = simulate_from_prior(
			# 	10;
			# 	model = RLWM_pmst,
			# 	priors = Dict(
			# 		:ρ => Dirac(3.),
			# 		:a => Dirac(0.4),
			# 		:W => Normal(0., 0.5),
			# 		:C => truncated(Normal(3., 2.), lower = 1.)
			# 	),
			# 	initial = aao,
			# 	transformed = Dict(:a => :α, :W => :w0),
			# 	fixed_struct = strct,
			# 	gq = true,
			# 	random_seed = 123
			# )
			# insertcols!(pmst, :model => "RLWM-p")
			pmst_sg = simulate_from_prior(
				10;
				model = RLWM_pmst_mk2,
				priors = Dict(
					:ρ => Dirac(3.),
					:a => Dirac(0.4),
					:W => Normal(0., 0.5),
					:C => truncated(Normal(3., 2.), lower = 1.)
				),
				initial = aao,
				transformed = Dict(:a => :α, :W => :w0),
				fixed_struct = strct,
				gq = true,
				random_seed = 123
			)
			insertcols!(pmst_sg, :model => "RLWM-psg")
			prior_samples[i] = vcat(rl, pmst_sg; cols=:union)
		    prior_samples[i].PID .+= (i-1) * 10
		end

		vcat(prior_samples...)
	end
	describe(prior_samples_seq)
end

# ╔═╡ 2a0ecaee-f6c3-48b1-8e4b-eceb623d892d
begin
	samples = prior_samples_seq
	samples.confusing = samples.feedback_optimal .< samples.feedback_suboptimal
	
	DataFrames.transform!(groupby(samples, :PID),
		:confusing => (x -> string(findall(x))) => :confusing_sequence
	)

	sequences = unique(samples.confusing_sequence)
	sequences = DataFrame(
		confusing_sequence = sequences,
		sequence = 1:length(sequences)
	)

	samples = leftjoin(samples, sequences, on = :confusing_sequence)

	function plot_by_sequence(
		data::DataFrame;
		group::Union{Symbol, Missing}=:model,
		acc_col::Symbol=:choice
	)
		by_sequence = groupby(data, :sequence)
	
		f_sequences = Figure(size = (1200, 900))
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
				f_sequences[r, c], 
				gdf; 
				group = :grp,
				linewidth = 1.,
				pid_col = :PID,
				acc_col = acc_col,
				legend = i==5 ? true : false,
				legend_pos = :bottom,
				legend_rows = 2,
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

	f_sequences = plot_by_sequence(samples)

	f_sequences
end

# ╔═╡ Cell order:
# ╠═772362f4-75c0-11ef-25de-55ffd5418ed2
# ╠═4eb03947-90d6-4281-9d63-372f76bb4900
# ╠═24289c6f-4974-41ad-822b-6f3df77f7cb4
# ╠═f99c4eb3-f018-4d3f-995c-88765ad6633d
# ╠═f743bbd4-44aa-437e-910a-6c4830f08ef4
# ╠═e9a9cecb-c5c6-422d-b089-b1bb0c11e79f
# ╠═f1918ce1-9435-4757-b970-68e795a49557
# ╠═acf1ba59-c202-4924-a58b-6ba3d6f7f338
# ╠═bde4f8c5-f7fe-4b5c-87e5-2bb393698a25
# ╠═faf9b5b6-737d-403c-b613-aa7558cf67ba
# ╠═bab8c088-f8c7-4eb9-8cfa-0583c660eef1
# ╠═bc443d65-78b1-42bc-b9ce-35a7ed713c51
# ╠═e6b933c4-c92d-47f4-a28c-a53c20ede1e6
# ╟─d4cb39eb-c881-4d0d-baaa-e19159be2780
# ╠═70626be0-27c5-419d-a27e-9fd2230f941c
# ╠═2a0ecaee-f6c3-48b1-8e4b-eceb623d892d
