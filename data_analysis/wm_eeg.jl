begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates
	using LogExpFunctions: logistic, logit
	using Tidier
	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/plotting_utils.jl")
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

# Load and preprocess data
eeg_wm = let

    # List all files in the directory
    eeg_dir = "data/WM_EEG_bhvdata"
    eeg_files = readdir(eeg_dir, join=true)

    # Keep only files that end with .csv
    filter!(f -> endswith(f, ".csv"), eeg_files)

    # Load all CSV files into a DataFrame
    eeg_wm = vcat(DataFrame.(CSV.File.(eeg_files; normalizenames = true))...; cols=:union)


    # Compute delays
    function compute_delays(vec::AbstractVector)
		last_seen = Dict{Any, Int}()
		delays = zeros(Int, length(vec))

		for (i, val) in enumerate(vec)
			delays[i] = haskey(last_seen, val) ? (i - last_seen[val]) : 0
			last_seen[val] = i
		end

		return delays
	end

    DataFrames.transform!(
        groupby(
            eeg_wm,
            :participant
        ),
        :stimulus_group => compute_delays => :delay,
    ) 

    # Remove non response trials
    filter!(x -> !ismissing(x.key_trial_keys) && x.key_trial_keys != "None", eeg_wm)

    # Compute choice optimal
    eeg_wm.chosen_side = (x -> x == "up" ? "middle" : x).(eeg_wm.key_trial_keys)
    eeg_wm.response_optimal = eeg_wm.chosen_side .== eeg_wm.optimal_side

    eeg_wm

end

# Plot accuracy curve
let

    # Summarize by participant and appearance
    acc_sum = combine(
        groupby(
            eeg_wm, [:participant, :appearance]
        ),
        :response_optimal => mean => :correct
    )

    # Summarize by appearance
    acc_sum_sum = combine(
        groupby(acc_sum, :appearance),
        :correct => mean => :correct,
        :correct => sem => :se
    )

    # Create figure
    f = Figure()

    mp = data(acc_sum) * mapping(
        :appearance,
        :correct,
        color = :participant,
    ) * visual(Lines, linewidth = 0.5, alpha = 0.5) +
    data(acc_sum_sum) * mapping(
        :appearance,
        :correct
    ) * visual(Lines, linewidth = 3, color = :black)


    plot!(f[1,1], mp)

    f
end
