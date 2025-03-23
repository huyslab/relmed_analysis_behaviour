### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 237a05f6-9e0e-11ef-2433-3bdaa51dbed4
# ╠═╡ show_logs = false
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates
	using LogExpFunctions: logistic, logit
	using Tidier
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	nothing
end

# ╔═╡ 0d120e19-28c2-4a98-b873-366615a5f784
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

# ╔═╡ d5811081-d5e2-4a6e-9fc9-9d70332cb338
md"""## Participant management"""

# ╔═╡ d0869d70-34db-4253-9e57-80397d5a0b45
begin
	jspsych_json, records = get_REDCap_data("pilot8"; file_field = "jspsych_data", record_id_field = "participant_id");
end

# ╔═╡ caa18e41-83aa-4e6f-b19f-4e52dff54ede
begin
	jspsych_json7, records7 = get_REDCap_data("pilot7"; file_field = "file_data", record_id_field = "record_id");
end

# ╔═╡ 9ed86d22-5a1b-46fd-92b7-de0e2097bd1b
begin
	jspsych_data = REDCap_data_to_df(jspsych_json7, records7)
end

# ╔═╡ 869b0bfd-91fa-4999-afa1-aae2786fd556
unique(jspsych_data.trialphase)

# ╔═╡ Cell order:
# ╠═237a05f6-9e0e-11ef-2433-3bdaa51dbed4
# ╠═0d120e19-28c2-4a98-b873-366615a5f784
# ╟─d5811081-d5e2-4a6e-9fc9-9d70332cb338
# ╠═d0869d70-34db-4253-9e57-80397d5a0b45
# ╠═caa18e41-83aa-4e6f-b19f-4e52dff54ede
# ╠═9ed86d22-5a1b-46fd-92b7-de0e2097bd1b
# ╠═869b0bfd-91fa-4999-afa1-aae2786fd556
