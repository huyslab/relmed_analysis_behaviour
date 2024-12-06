### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ b41e7252-a075-11ef-039c-f532a7fb0a94
# ╠═╡ show_logs = false
begin
	cd("/home/jovyan/")
	import Pkg
	# activate the shared project environment
	Pkg.activate("relmed_environment")
	# instantiate, i.e. make sure that all packages are downloaded
	Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, Turing, SHA, CSV
	using Tidier, GLM, MixedModels, PlutoUI, LaTeXStrings, ColorSchemes
	using LogExpFunctions: logistic, logit
	import OpenScienceFramework as OSF
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	include("stats_utils.jl")
	include("osf_utils.jl")
	include("vigour_utils.jl")
	Turing.setprogress!(false)
	nothing
end

# ╔═╡ 28b7224d-afb4-4474-b346-7ee353b6d3d3
TableOfContents(title="📚 Table of Contents", indent=true, depth=4, aside=true)

# ╔═╡ 1f8ca836-f2f7-4965-bf07-5656cf6c4ec6
md"""
Set theme globally
"""

# ╔═╡ 1c0196e9-de9a-4dfa-acb5-357c02821c5d
set_theme!(theme_minimal();
		font = "Helvetica",
		fontsize = 16);

# ╔═╡ 8255895b-a337-4c8a-a1a7-0983499f684e
md"""
## Press rate: Trial presses/sec
"""

# ╔═╡ 656429a5-008b-4e63-a1e1-061dfe19d191
md"""
Import your data here
"""

# ╔═╡ ae1cccae-0fd7-4b5b-8354-a6bba566e504
begin
	vigour_data = DataFrame(CSV.File("data/vigour_data.csv"))
end

# ╔═╡ d6a73b37-2079-4ed1-ac49-e7c596fc0997
md"""
## Model of press rate
"""

# ╔═╡ 6f11e67c-84b2-457d-9727-825e0631860b
md"""
### Press rate as a function of (log) reward rate
"""

# ╔═╡ 68985a71-98c4-485b-a800-643aea8b8a5e
let
	glm_predict(data) = predict(glm(@formula(trial_presses ~ log(reward_per_press)), data, Poisson(), LogLink(); offset=log.(data.dur)), data; offset=log.(data.dur))
	
	p = @chain vigour_data begin
		rename(:trial_duration => :dur)
		groupby([:PID, :session])
		DataFrames.transform(AsTable([:trial_presses, :reward_per_press, :dur]) => glm_predict => :pred)
		@mutate(pred = pred/dur)
		groupby(:reward_per_press)
		combine([:press_per_sec, :pred] .=> mean; renamecols=false)
		@pivot_longer(-reward_per_press)
		@arrange(reward_per_press)
		data(_) * (mapping(:reward_per_press, :value, color=:variable=>"", linestyle=:variable=>"") * visual(Lines))
	end
	fig = Figure(;size=(9, 6) .* 144 ./ 2.54)
	p = draw!(fig[1,1], p, scales(
		Color=(;palette=reverse(ColorSchemes.Paired_10[1:2]),categories=["pred"=>"Model pred.", "press_per_sec"=>"Data"]),
		LineStyle=(;palette=[:solid,:dash],categories=["pred"=>"Model pred.", "press_per_sec"=>"Data"])
	);axis=(;xlabel="Reward/press", ylabel="Press/sec"))
	legend!(fig[1,2], p
		#; halign=0.95, valign=0.05, tellheight=false, tellwidth=false
	)
	Label(fig[0,:], "log(Press rate) ~ β0 + β1 * log(Reward rate)")

	# Save
	filepaths = joinpath("results/EEG", "Vigour_reg_model_rpp.png")
	save(filepaths, fig; px_per_unit = 4)
	
	fig
end

# ╔═╡ cc7e08b3-e245-4483-8cac-086a673a2861
md"""
### Press rate as a function of (log) fixed ratio and reward magnitude
"""

# ╔═╡ 42e2e827-253d-4881-bfc9-65d206e6201d
let
	glm_predict(data) = predict(glm(@formula(trial_presses ~ log(ratio) + log(magnitude)), data, Poisson(), LogLink(); offset=log.(data.dur)), data; offset=log.(data.dur))
	
	p = @chain vigour_data begin
		rename(:trial_duration => :dur)
		DataFrames.transform([:ratio, :magnitude] .=> ByRow(as_float), renamecols=false)
		groupby([:PID, :session])
		DataFrames.transform(AsTable([:trial_presses, :ratio, :magnitude, :dur]) => glm_predict => :pred)
		@mutate(pred = pred/dur)
		groupby(:reward_per_press)
		combine([:press_per_sec, :pred] .=> mean; renamecols=false)
		@pivot_longer(-reward_per_press)
		@arrange(reward_per_press)
		data(_) * (mapping(:reward_per_press, :value, color=:variable=>"", linestyle=:variable=>"") * visual(Lines))
	end
	fig = Figure(;size=(9, 6) .* 144 ./ 2.54)
	p = draw!(fig[1,1], p, scales(
		Color=(;palette=reverse(ColorSchemes.Paired_10[1:2]),categories=["pred"=>"Model pred.", "press_per_sec"=>"Data"]),
		LineStyle=(;palette=[:solid,:dash],categories=["pred"=>"Model pred.", "press_per_sec"=>"Data"])
	);axis=(;xlabel="Reward/press", ylabel="Press/sec"))
	legend!(fig[1,2], p
		#; halign=0.95, valign=0.05, tellheight=false, tellwidth=false
	)
	Label(fig[0,:], "log(Press rate) ~ β0 + β1 * log(Fixed ratio) + β2 * log(Magnitude)")

	# Save
	filepaths = joinpath("results/EEG", "Vigour_reg_model_fr_n_rm.png")
	save(filepaths, fig; px_per_unit = 4)

	
	fig
end

# ╔═╡ b930ea7b-2922-4839-a167-a514b44039be
md"""
## Export single participant parameters
"""

# ╔═╡ a2e11ec7-f429-4b57-9367-a69f25132b7e
let
	fr_rm_coefs = let
		glm_coef(data) = coef(glm(@formula(trial_presses ~ log(ratio) + log(magnitude)), data, Poisson(), LogLink(); offset=log.(data.dur)))
		
		@chain vigour_data begin
			rename(:trial_duration => :dur)
			DataFrames.transform([:ratio, :magnitude] .=> ByRow(as_float), renamecols=false)
			groupby([:PID, :session])
			combine(AsTable([:trial_presses, :ratio, :magnitude, :dur]) => (x -> [glm_coef(x)]) => [:β0_frrm, :β_fr, :β_rm])
		end
	end

	rpp_coefs = let
		glm_coef(data) = coef(glm(@formula(trial_presses ~ log(reward_per_press)), data, Poisson(), LogLink(); offset=log.(data.dur)))
		
		@chain vigour_data begin
			rename(:trial_duration => :dur)
			groupby([:PID, :session])
			combine(AsTable([:trial_presses, :reward_per_press, :dur]) => (x -> [glm_coef(x)]) => [:β0_rpp, :β_rpp])
		end
	end
	leftjoin!(rpp_coefs, fr_rm_coefs, on = [:PID, :session])
	CSV.write("results/EEG/single_p_Vigour_reg_coef.csv", rpp_coefs)
end

# ╔═╡ Cell order:
# ╠═b41e7252-a075-11ef-039c-f532a7fb0a94
# ╠═28b7224d-afb4-4474-b346-7ee353b6d3d3
# ╟─1f8ca836-f2f7-4965-bf07-5656cf6c4ec6
# ╠═1c0196e9-de9a-4dfa-acb5-357c02821c5d
# ╟─8255895b-a337-4c8a-a1a7-0983499f684e
# ╟─656429a5-008b-4e63-a1e1-061dfe19d191
# ╠═ae1cccae-0fd7-4b5b-8354-a6bba566e504
# ╟─d6a73b37-2079-4ed1-ac49-e7c596fc0997
# ╟─6f11e67c-84b2-457d-9727-825e0631860b
# ╠═68985a71-98c4-485b-a800-643aea8b8a5e
# ╟─cc7e08b3-e245-4483-8cac-086a673a2861
# ╠═42e2e827-253d-4881-bfc9-65d206e6201d
# ╟─b930ea7b-2922-4839-a167-a514b44039be
# ╠═a2e11ec7-f429-4b57-9367-a69f25132b7e
