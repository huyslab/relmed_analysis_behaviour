### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ fbe53102-74e9-11ef-2669-5fc149d6aee8
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, Distributions, StatsBase,
		CSV, Turing
	using LogExpFunctions: logistic, logit

	include("$(pwd())/PILT_models.jl")
	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/stats_utils.jl")
	include("$(pwd())/single_p_QL.jl")
	nothing
end

# ╔═╡ 610792a4-7f8f-48b5-8062-a600e094f0c1
# Paremters for running
begin
	input_file = "scripts_for_eeg/0011_export.csv"
	output_file = "scripts_for_eeg/0011_qvals.csv"
	plot_things = true
end

# ╔═╡ 6c040618-650b-4ab1-b64d-16be6d98e71a
begin
	if plot_things
		using AlgebraOfGraphics, CairoMakie
		
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
end

# ╔═╡ 332c204f-2534-4048-a4f6-2509b2ad8831
# Model priors
begin
	prior_ρ = truncated(Normal(0., 5.), lower = 0.)
	prior_a = Normal()
end

# ╔═╡ 56075d24-1a2c-4531-b6f2-ad2a3683dfaa
aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

# ╔═╡ 927f1dae-f91f-4c8b-8ee6-09e12a81811b
# Load data
begin
	data = DataFrame(CSV.File(input_file))

	is_multi_set_size = "n_pairs" in names(data) && maximum(data.n_pairs) > 1

	# Renmae varaibels
	for v in ["outcome_optimal", "outcome_suboptima"]
		if v in names(data)
			rename!(
				data,
				v => Symbol(replace(v, "outcome" => "feedback")),
			)
		end
	end

	# Add trial variable
	if "trial" ∉ names(data)
		DataFrames.transform!(
			groupby(data, :block),
			:choice => (x -> 1:length(x)) => :trial
		)
	end
end

# ╔═╡ 6b1937b9-30bb-44c6-8e00-ab284f3c1cc1
# Sort by stimulus pair rather than block for naive fitting
if is_multi_set_size
	@info "Treating each pair as an independent block"
	forfit = let
		remapped = copy(data)
	
		# Make sure sorted
		sort!(
			remapped,
			[:block, :trial]
		)
	
		# Renumber trials
		DataFrames.transform!(
			groupby(remapped, [:block, :pair]),
			:trial => (x -> 1:length(x)) => :trial,
			:trial => identity => :old_trial
		)

		# Treat pair number as block
		remapped.old_block = remapped.block
		remapped.block = remapped.cpair
	
		# Sort by new block
		sort!(
			remapped, 
			[:block, :trial]
		)
	
		# Make sure sorted correctly
		@assert all(combine(groupby(remapped, :block), :trial => issorted => :sorted).sorted)
	
		remapped
	end
else
	forfit = data	
end

# ╔═╡ fa80c3dd-a3fa-44d8-96b9-b46c5f3933ad
begin
	# Fit data
	fit = optimize_single_p_QL(
		forfit,
		initV = aao,
		prior_ρ = prior_ρ,
		prior_a = prior_a
	)

	ρ_est = fit.values[:ρ]

	a_est = fit.values[:a]
end

# ╔═╡ c95e03cb-fd17-4a0f-8c0c-eaeb56e68398
@info "Estimated parameter values: ρ=$(round(ρ_est, digits = 2)), α=$(round(a2α(a_est), digits = 2))"

# ╔═╡ 601b7221-0099-4769-8f7b-d963449d9fd7
if plot_things
	ρ_bootstraps, a_bootstraps = let n_bootstraps = 1000,
		random_seed = 0
	
		rng = Xoshiro(random_seed)
	
		# Preallocate
		ρ_ests = fill(-99., n_bootstraps)
		a_ests = fill(-99., n_bootstraps)
	
		# Run over bootstraps
		for i in 1:n_bootstraps
	
			# Sample blocks with replacement
			block_sample = sample(rng, unique(forfit.block), maximum(forfit.block))
	
			# Subset data
			this_dat = vcat([
				DataFrames.transform!(filter(x -> x.block == b, forfit), 
				:block => ByRow(x -> i) => :block,
				:block => identity => :original_block
				) 
				for (i, b) in enumerate(block_sample)]...)
		
			# Fit data
			fit = optimize_single_p_QL(
				this_dat,
				initV = aao,
				prior_ρ = prior_ρ,
				prior_a = prior_a
			)
	
			# Push results
			ρ_ests[i] = fit.values[:ρ]
			a_ests[i] = fit.values[:a]
		end
	
		ρ_ests, a_ests
	
	end
end

# ╔═╡ e571cd8c-33db-4bb1-9886-483f486c7b43
let
	f = Figure(size = (700, 250))

	# Reward sensitivity histogram
	mp1 = mapping(ρ_bootstraps => "ρ reward sensitivity") * visual(Hist, bins = 50) +
		mapping([ρ_est]) * visual(VLines, color = :blue, linestyle = :dash)

	draw!(f[1,1], mp1, axis = (; 
		limits = (0, maximum([10, maximum(ρ_bootstraps)]), nothing, nothing)
	))

	# Learning rate histogram
	mp2 = mapping(a_bootstraps => a2α => "α learning rate") * visual(Hist, bins = 50) +
		mapping([a2α(a_est)]) * visual(VLines, color = :blue, linestyle = :dash)

	draw!(f[1,2], mp2, axis = (; limits = (0., 1., nothing, nothing)))

	# Bivariate distribution
	mp3 = mapping(
			a_bootstraps => a2α => "α learning rate", 
			ρ_bootstraps => "ρ reward sensitivity"
		) * visual(Scatter, markersize = 4, color = :grey) +
		mapping([a2α(a_est)], [ρ_est]) * visual(Scatter, markersize = 15, color = :blue, marker = :+)

	draw!(f[1,3], mp3)

	Label(
		f[0,:],
		"Bootstrap distribution of parameters"
	)
	
	f

end

# ╔═╡ 09281096-1102-4f79-bb6a-f6a1cf488d0c
# Get Q values
begin
	# Compute Q values
	Qs = single_p_QL(
		block = forfit.block,
		choice = forfit.choice,
		outcomes = hcat(
			forfit.feedback_suboptimal,
			forfit.feedback_optimal
		),
		initV = fill(aao, 1, 2),
		prior_ρ = Dirac(ρ_est),
		prior_a = Dirac(a_est)
	)()

	# Add block and trial
	Qs_df = DataFrame(
		block = forfit.block,
		trial = forfit.trial,
		choice = forfit.choice,
		Q_suboptimal = Qs[:, 1],
		Q_optimal = Qs[:, 2],
		PE = ifelse.(
			forfit.choice .== 1,
			forfit.feedback_optimal .* ρ_est .- Qs[:, 2],
			forfit.feedback_suboptimal .* ρ_est .- Qs[:, 1]
		),
		rho = fill(ρ_est, nrow(data)),
		a = fill(a_est, nrow(data)),
		alpha = fill(a2α(a_est), nrow(data))
	)

	# Remap trial and block numbers
	if is_multi_set_size
		pre_nrow = nrow(Qs_df)
		
		Qs_df = leftjoin(
			Qs_df,
			forfit[!, [:block, :trial, :n_pairs, :pair, :cpair, :old_block, :old_trial]],
			on = [:block, :trial],
			order = :left
		)

		Qs_df.block = Qs_df.old_block
		Qs_df.trial = Qs_df.old_trial
		select!(Qs_df, Not([:old_block, :old_trial]))

		@assert nrow(Qs_df) == pre_nrow "Problem in remapping trials and blocks"

		# Resort
		sort!(Qs_df, [:block, :trial])

		@assert Qs_df.cpair == data.cpair "Problem in remapping trials and blocks"
	end

	Qs_df
end

# ╔═╡ b49414a7-faff-447a-960c-213b04d03c6e
CSV.write(output_file, Qs_df)

# ╔═╡ Cell order:
# ╠═fbe53102-74e9-11ef-2669-5fc149d6aee8
# ╠═610792a4-7f8f-48b5-8062-a600e094f0c1
# ╠═332c204f-2534-4048-a4f6-2509b2ad8831
# ╠═6c040618-650b-4ab1-b64d-16be6d98e71a
# ╠═56075d24-1a2c-4531-b6f2-ad2a3683dfaa
# ╠═927f1dae-f91f-4c8b-8ee6-09e12a81811b
# ╠═6b1937b9-30bb-44c6-8e00-ab284f3c1cc1
# ╠═fa80c3dd-a3fa-44d8-96b9-b46c5f3933ad
# ╠═c95e03cb-fd17-4a0f-8c0c-eaeb56e68398
# ╠═601b7221-0099-4769-8f7b-d963449d9fd7
# ╠═e571cd8c-33db-4bb1-9886-483f486c7b43
# ╠═09281096-1102-4f79-bb6a-f6a1cf488d0c
# ╠═b49414a7-faff-447a-960c-213b04d03c6e
