### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ baba7ea8-9069-11ef-2bba-89fb74ddc46b
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates
	using LogExpFunctions: logistic, logit
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	nothing
end

# ╔═╡ eafd04fa-05ab-4f29-921a-63890e8c83a0
function load_pilot4_data()
	datafile = "data/pilot4.jld2"

	# Load data or download from REDCap
	if !isfile(datafile)
		jspsych_json, records = get_REDCap_data("pilot4"; file_field = "file_data")
	
		jspsych_data = REDCap_data_to_df(jspsych_json, records)

		remove_testing!(jspsych_data)

		JLD2.@save datafile jspsych_data
	else
		JLD2.@load datafile jspsych_data
	end

	# Exctract PILT
	PLT_data = prepare_PLT_data(jspsych_data)

	# Extract post-PILT test
	test_data = prepare_post_PILT_test_data(jspsych_data)

	# Exctract vigour
	vigour_data = prepare_vigour_data(jspsych_data) 

	# Extract post-vigour test
	post_vigour_test_data = prepare_post_vigour_test_data(jspsych_data)

	# Extract PIT
	PIT_data = prepare_PIT_data(jspsych_data)

	# Exctract reversal
	reversal_data = prepare_reversal_data(jspsych_data)

	return PLT_data, test_data, vigour_data, post_vigour_test_data, PIT_data, reversal_data, jspsych_data
end


# ╔═╡ 57ca3929-faa6-4a95-9e4d-6c1add13b121
PLT_data, test_data, vigour_data, post_vigour_test_data, pit_data, reversal_data, jspsych_data = load_pilot4_data()

# ╔═╡ Cell order:
# ╠═baba7ea8-9069-11ef-2bba-89fb74ddc46b
# ╠═eafd04fa-05ab-4f29-921a-63890e8c83a0
# ╠═57ca3929-faa6-4a95-9e4d-6c1add13b121
