### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 172dcebc-9d1f-11ef-069b-cb4f7a52ef0a
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

	include("$(pwd())/fetch_preprocess_data.jl")
	
	include("$(pwd())/working_memory/simulate.jl")
	include("$(pwd())/working_memory/RL+RLWM_models.jl")
	include("$(pwd())/working_memory/plotting_utils.jl")

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

# ╔═╡ dc996131-1760-4d86-bf3a-adf51df3e7fb
include("$(pwd())/working_memory/RL+RLWM_models.jl")

# ╔═╡ 8eeab37a-3fdf-4103-b9e0-564b779a3493
begin
	PILT_data, _, _, _ = load_pilot4x_data()
	filter!(x -> x.version == "4.1", PILT_data)
	PILT_data = exclude_PLT_sessions(PILT_data, required_n_blocks = 18)
	PILT_data_clean = filter(x -> x.choice != "noresp", PILT_data)
	df = prepare_for_fit(PILT_data_clean; pilot4=true)[1]
end

# ╔═╡ 3cbde056-055a-4249-bc2f-6d9572add543
begin
	function true_vs_preds(df1::DataFrame, df2::DataFrame; og_df::DataFrame)
		df1 = leftjoin(df1, og_df; on = [:PID, :block, :valence, :trial])
		df2 = leftjoin(df2, og_df; on = [:PID, :block, :valence, :trial])
		
		# df1.true_choice, df1.predicted_choice = Int.(df1.true_choice), Int.(df1.predicted_choice)
		# df2.true_, df2.pred_ = Int.(df2.true_choice), Int.(df2.predicted_choice)
		
		df1 = stack(df1, [:true_choice, :predicted_choice])
		df1.chce_ss = df1.variable .* string.(df1.set_size)
		
		df2 = stack(df2, [:true_choice, :predicted_choice])
		df2.chce_ss = df2.variable .* string.(df2.set_size)

		return df1, df2
	end
	
	function plot_true_vs_preds(choice_df1::DataFrame, choice_df2::DataFrame)
		f = Figure(size = (1000, 800))
		plot_prior_accuracy!(
			f[1,1], filter(x -> x.set_size == 2, choice_df1);
			group = :variable, pid_col = :PID, acc_col = :value, error_band = "se", legend = true, legend_pos = :bottom, legend_rows = 2, 
			title = "Single (set size = 2)", colors = Makie.colorschemes[:Reds_3]
		)
		plot_prior_accuracy!(
			f[1,2], filter(x -> x.set_size == 2, choice_df2);
			group = :variable,
		    pid_col = :PID, acc_col = :value, error_band = "se",legend = true,
			legend_pos = :bottom, legend_rows = 2,
			title = "Reciprocal (set size = 2)", colors = Makie.colorschemes[:Reds_3]
		)
		plot_prior_accuracy!(
			f[2,1], filter(x -> x.set_size == 6, choice_df1);
			group = :variable, 
			pid_col = :PID, acc_col = :value, error_band = "se", legend = true, legend_pos = :bottom, legend_rows = 2, 
			title = "Single (set size = 6)", colors = Makie.colorschemes[:Blues_3]
		)
		plot_prior_accuracy!(
			f[2,2], filter(x -> x.set_size == 6, choice_df2);
			group = :variable, 
		    pid_col = :PID, acc_col = :value, error_band = "se",legend = true,
			legend_pos = :bottom, legend_rows = 2,
			title = "Reciprocal (set size = 6)", colors = Makie.colorschemes[:Blues_3]
		)
		plot_prior_accuracy!(
			f[3,1], filter(x -> x.set_size == 14, choice_df1);
			group = :variable, 
			pid_col = :PID, acc_col = :value, error_band = "se", legend = true, legend_pos = :bottom, legend_rows = 2, 
			title = "Single (set size = 14)", colors = Makie.colorschemes[:Oranges_3]
		)
		plot_prior_accuracy!(
			f[3,2], filter(x -> x.set_size == 14, choice_df2);
			group = :variable, 
		    pid_col = :PID, acc_col = :value, error_band = "se",legend = true,
			legend_pos = :bottom, legend_rows = 2,
			title = "Reciprocal (set size = 14)", colors = Makie.colorschemes[:Oranges_3]
		)
		return f
	end
end

# ╔═╡ 0d7e8671-b1ba-49ed-bb90-006704921c58
md"
#### 1. Palimpsest with *overall* capacity + *no averaging*
"

# ╔═╡ 2287ab8e-35ff-4e92-905b-3538f8cc9c97
begin
	# single update model
	aos_pmst_wm_ests, aos_pmst_choices, aos_pmst_covs = optimize_multiple(
		df;
		model = WM_all_outc_pmst_sgd_sum,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 4.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)
	# reciprocal update model
	aosr_pmst_wm_ests, aosr_pmst_choices, aosr_pmst_covs = optimize_multiple(
		df;
		model = WM_all_outc_pmst_sgd_recip,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 4.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)
	nothing
end

# ╔═╡ 94688c7f-9881-46c2-852a-90cdd2e3b553
let
	f = Figure(size = (800, 300))
	
	# Define parameters
	ρ1, C1, ll1 = aos_pmst_wm_ests.ρ, aos_pmst_wm_ests.C, aos_pmst_wm_ests.loglike
	ρ2, C2, ll2 = aosr_pmst_wm_ests.ρ, aosr_pmst_wm_ests.C, aosr_pmst_wm_ests.loglike

	# Set labels
	labs1 = (xlabel = "ρ", ylabel = "C", zlabel = "log-likelihood", title = "Single")
	labs2 = (xlabel = "ρ", ylabel = "C", zlabel = "log-likelihood", title = "Reciprocal")

	# Plot
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, ρ1, C1, ll1)
	scatter!(ax2, ρ2, C2, ll2)
	f
end

# ╔═╡ 93d85962-1db2-455c-a4b2-1fe79cc68c9e
let
	choice_df1, choice_df2 =
		true_vs_preds(aos_pmst_choices, aosr_pmst_choices; og_df = df)

	plot_true_vs_preds(choice_df1, choice_df2)
end

# ╔═╡ 996bdcbd-e720-4d26-a438-d9cf4e490281
md"
#### 2. Palimpsest with *overall* capacity + *averaging*
"

# ╔═╡ 9973390c-7255-4656-9490-b19a25c00f38
begin
	aom_pmst_wm_ests, aom_pmst_choices, aom_pmst_covs = optimize_multiple(
		df;
		model = WM_all_outc_pmst_sgd,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 4.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)
	aomr_pmst_wm_ests, aomr_pmst_choices, aomr_pmst_covs = optimize_multiple(
		df;
		model = WM_all_outc_pmst_sgd_recip,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 4.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)
	nothing
end

# ╔═╡ 13cd8844-6ba4-441b-a04e-8ec4da189750
let
	f = Figure(size = (800, 300))
	
	# Define parameters
	ρ1, C1, ll1 = aom_pmst_wm_ests.ρ, aom_pmst_wm_ests.C, aom_pmst_wm_ests.loglike
	ρ2, C2, ll2 = aomr_pmst_wm_ests.ρ, aomr_pmst_wm_ests.C, aomr_pmst_wm_ests.loglike

	# Set labels
	labs1 = (xlabel = "ρ", ylabel = "C", zlabel = "log-likelihood", title = "Single")
	labs2 = (xlabel = "ρ", ylabel = "C", zlabel = "log-likelihood", title = "Reciprocal")

	# Plot
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, ρ1, C1, ll1)
	scatter!(ax2, ρ2, C2, ll2)
	f
end

# ╔═╡ 7c29acd4-3311-42a9-a196-b9c751ad18c5
let
	choice_df1, choice_df2 =
		true_vs_preds(aom_pmst_choices, aomr_pmst_choices; og_df = df)

	plot_true_vs_preds(choice_df1, choice_df2)
end

# ╔═╡ 3f8e00ef-69b1-4443-be9c-32601d182cbd
md"
#### 3. Palimpsest with *stimulus-specific* capacity + *no averaging*
"

# ╔═╡ d2caaa2e-4b60-473d-8b3e-9571db1fa748
begin
	sss_pmst_wm_ests, sss_pmst_choices, sss_pmst_covs = optimize_multiple(
		df;
		model = WM_pmst_sgd_sum,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 4.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)
	sssr_pmst_wm_ests, sssr_pmst_choices, sssr_pmst_covs = optimize_multiple(
		df;
		model = WM_pmst_sgd_sum_recip,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 4.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)
	nothing
end

# ╔═╡ bfc60aac-1200-457a-accb-5345a1f5229e
let
	f = Figure(size = (800, 300))
	
	# Define parameters
	ρ1, C1, ll1 = sss_pmst_wm_ests.ρ, sss_pmst_wm_ests.C, sss_pmst_wm_ests.loglike
	ρ2, C2, ll2 = sssr_pmst_wm_ests.ρ, sssr_pmst_wm_ests.C, sssr_pmst_wm_ests.loglike

	# Set labels
	labs1 = (xlabel = "ρ", ylabel = "C", zlabel = "log-likelihood", title = "Single")
	labs2 = (xlabel = "ρ", ylabel = "C", zlabel = "log-likelihood", title = "Reciprocal")

	# Plot
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, ρ1, C1, ll1)
	scatter!(ax2, ρ2, C2, ll2)
	f
end

# ╔═╡ e29b93d7-ebf4-44ac-bc9f-84836dc39ae4
let
	choice_df1, choice_df2 =
		true_vs_preds(sss_pmst_choices, sssr_pmst_choices; og_df = df)

	plot_true_vs_preds(choice_df1, choice_df2)
end

# ╔═╡ d8e625a0-8363-4260-bcba-46c102684322
md"
#### 4. Palimpsest with *stimulus-specific* capacity + *averaging*
"

# ╔═╡ 0366af29-422f-46df-a0ff-4b194a4d3d92
begin
	ssm_pmst_wm_ests, ssm_pmst_choices, ssm_pmst_covs = optimize_multiple(
		df;
		model = WM_pmst_sgd,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 4.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)
	ssmr_pmst_wm_ests, ssmr_pmst_choices, ssmr_pmst_covs = optimize_multiple(
		df;
		model = WM_pmst_sgd_recip,
		estimate = "MAP",
		include_true = false,
		priors = Dict(
			:ρ => truncated(Normal(0., 4.), lower = 0.),
			:C => truncated(Normal(5., 3.), lower = 1.)
		),
		parameters = [:ρ, :C]
	)
	nothing
end

# ╔═╡ 6ebccb38-9a1d-4853-814f-71162b24af87
let
	f = Figure(size = (800, 300))
	
	# Define parameters
	ρ1, C1, ll1 = ssm_pmst_wm_ests.ρ, ssm_pmst_wm_ests.C, ssm_pmst_wm_ests.loglike
	ρ2, C2, ll2 = ssmr_pmst_wm_ests.ρ, ssmr_pmst_wm_ests.C, ssmr_pmst_wm_ests.loglike

	# Set labels
	labs1 = (xlabel = "ρ", ylabel = "C", zlabel = "log-likelihood", title = "Single")
	labs2 = (xlabel = "ρ", ylabel = "C", zlabel = "log-likelihood", title = "Reciprocal")

	# Plot
	ax1, ax2 = Axis3(f[1,1]; labs1...), Axis3(f[1,2]; labs2...)
	scatter!(ax1, ρ1, C1, ll1)
	scatter!(ax2, ρ2, C2, ll2)
	f
end

# ╔═╡ 7c18d92c-66be-4d98-a00b-590a95b71dc0
let
	choice_df1, choice_df2 =
		true_vs_preds(ssm_pmst_choices, ssmr_pmst_choices; og_df = df)

	plot_true_vs_preds(choice_df1, choice_df2)
end

# ╔═╡ Cell order:
# ╠═172dcebc-9d1f-11ef-069b-cb4f7a52ef0a
# ╠═dc996131-1760-4d86-bf3a-adf51df3e7fb
# ╠═8eeab37a-3fdf-4103-b9e0-564b779a3493
# ╠═3cbde056-055a-4249-bc2f-6d9572add543
# ╟─0d7e8671-b1ba-49ed-bb90-006704921c58
# ╠═2287ab8e-35ff-4e92-905b-3538f8cc9c97
# ╠═94688c7f-9881-46c2-852a-90cdd2e3b553
# ╠═93d85962-1db2-455c-a4b2-1fe79cc68c9e
# ╟─996bdcbd-e720-4d26-a438-d9cf4e490281
# ╠═9973390c-7255-4656-9490-b19a25c00f38
# ╠═13cd8844-6ba4-441b-a04e-8ec4da189750
# ╠═7c29acd4-3311-42a9-a196-b9c751ad18c5
# ╟─3f8e00ef-69b1-4443-be9c-32601d182cbd
# ╠═d2caaa2e-4b60-473d-8b3e-9571db1fa748
# ╠═bfc60aac-1200-457a-accb-5345a1f5229e
# ╠═e29b93d7-ebf4-44ac-bc9f-84836dc39ae4
# ╟─d8e625a0-8363-4260-bcba-46c102684322
# ╠═0366af29-422f-46df-a0ff-4b194a4d3d92
# ╠═6ebccb38-9a1d-4853-814f-71162b24af87
# ╠═7c18d92c-66be-4d98-a00b-590a95b71dc0
