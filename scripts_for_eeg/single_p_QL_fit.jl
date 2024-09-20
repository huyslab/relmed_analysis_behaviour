### A Pluto.jl notebook ###
# v0.19.43

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
begin
	input_file = ARGS[1]
	output_file = ARGS[2]
end

# ╔═╡ 332c204f-2534-4048-a4f6-2509b2ad8831
begin
	prior_ρ = truncated(Normal(0., 5.), lower = 0.)
	prior_a = Normal()
end

# ╔═╡ e8d8f415-d2f0-47ad-9d4a-b77b0e3e0315
# Load data
begin
	data = DataFrame(CSV.File(input_file))

	rename!(
		data,
		:outcome_optimal => :feedback_optimal,
		:outcome_suboptimal => :feedback_suboptimal
	)

	DataFrames.transform!(
		groupby(data, :block),
		:choice => (x -> 1:length(x)) => :trial
	)
end

# ╔═╡ 56075d24-1a2c-4531-b6f2-ad2a3683dfaa
aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

# ╔═╡ fa80c3dd-a3fa-44d8-96b9-b46c5f3933ad
begin
	# Fit data
	fit = optimize_single_p_QL(
		data,
		initV = aao,
		prior_ρ = prior_ρ,
		prior_a = prior_a
	)

	ρ_est = fit.values[:ρ]

	a_est = fit.values[:a]
end

# ╔═╡ c95e03cb-fd17-4a0f-8c0c-eaeb56e68398
@info "Estimated parameter values: ρ=$(round(ρ_est, digits = 2)), α=$(round(a2α(a_est), digits = 2))"

# ╔═╡ 09281096-1102-4f79-bb6a-f6a1cf488d0c
# Get Q values
begin
	# Compute Q values
	Qs = single_p_QL(
		N = nrow(data),
		n_blocks = maximum(data.block),
		block = data.block,
		choice = data.choice,
		outcomes = hcat(
			data.feedback_suboptimal,
			data.feedback_optimal
		),
		initV = fill(aao, 1, 2),
		prior_ρ = Dirac(ρ_est),
		prior_a = Dirac(a_est)
	)()

	# Add block and trial
	Qs = DataFrame(
		block = data.block,
		trial = data.trial,
		Q_suboptimal = Qs[:, 1],
		Q_optimal = Qs[:, 2]
	)
end

# ╔═╡ b49414a7-faff-447a-960c-213b04d03c6e
CSV.write(output_file, Qs)

# ╔═╡ Cell order:
# ╠═fbe53102-74e9-11ef-2669-5fc149d6aee8
# ╠═610792a4-7f8f-48b5-8062-a600e094f0c1
# ╠═332c204f-2534-4048-a4f6-2509b2ad8831
# ╠═e8d8f415-d2f0-47ad-9d4a-b77b0e3e0315
# ╠═56075d24-1a2c-4531-b6f2-ad2a3683dfaa
# ╠═fa80c3dd-a3fa-44d8-96b9-b46c5f3933ad
# ╠═c95e03cb-fd17-4a0f-8c0c-eaeb56e68398
# ╠═09281096-1102-4f79-bb6a-f6a1cf488d0c
# ╠═b49414a7-faff-447a-960c-213b04d03c6e
