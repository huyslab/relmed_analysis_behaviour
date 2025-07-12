begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, Distributions, StatsBase,
		CSV, Turing
	using LogExpFunctions: logistic, logit, softmax

	include("$(pwd())/PILT_models.jl")
	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/stats_utils.jl")
	include("$(pwd())/model_utils.jl")
	nothing
end

# Paremters for running
begin
	input_file = "data/WM_EEG_bhvdata/sub-0005_WM_2025-05-30_14h05.13.345.csv"
	output_file = "scripts_for_eeg/0005_WM_2025_qvals.csv"
	plot_things = true
end

begin
	if plot_things
		using AlgebraOfGraphics, CairoMakie
		
		# Set theme
		
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

# Auxiliary variables
begin
	# Model priors
    priors = Dict(
        :ρ => truncated(Normal(0., .5), lower = 0.01, upper = 10.),
        :a => truncated(Normal(0., 2.), lower = -5., upper = 5.)
    )


	wm_columns = Dict(
		"block" => :stimulus_group,
		"trial" => :appearance,
		"feedback_columns" => [:feedback_left, :feedback_middle, :feedback_right],
		"choice" => :choice
	)
end

# Load data
begin
	data = DataFrame(CSV.File(input_file; normalizenames = true))

	# Remove empty rows
	filter!(x -> !ismissing(x.trial) && (x.key_trial_keys ∈ ["left", "up", "right"]), data)

    # Sort by stimulus group and appearance
    sort!(data, [:stimulus_group, :appearance])

    # Compute choice variable
    data.choice = (x -> findfirst(x .== ["left", "up", "right"])).(
        data.key_trial_keys
    )

    # Drop missing
    dropmissing!(data, [:stimulus_group, :appearance, :choice, :feedback_left, :feedback_middle, :feedback_right])

    # Additional validation - check for NaN values
    @assert all(!isnan, data.feedback_left) "NaN values found in feedback_left"
    @assert all(!isnan, data.feedback_middle) "NaN values found in feedback_middle" 
    @assert all(!isnan, data.feedback_right) "NaN values found in feedback_right"
    @assert all(x -> x ∈ [1, 2, 3], data.choice) "Invalid choice values found"
    
end


begin

    # Fit data
	fit = optimize(
		unpack_single_p_QL_categorical(
            data;
            # Use the wm_columns for unpacking
			columns = wm_columns
		);
		model = single_p_QL_categorical,
		priors = priors,
        initV = mean([0.01, 0.9])
	)

	ρ_est = fit.values[:ρ]

	a_est = fit.values[:a]

    ρ_est, a_est
end

@info "Estimated parameter values: ρ=$(round(ρ_est, digits = 2)), α=$(round(a2α(a_est), digits = 2))"

if plot_things
	ρ_bootstraps, a_bootstraps = let n_bootstraps = 1000,
		random_seed = 0, data = copy(data)
	
		rng = Xoshiro(random_seed)
	
		# Preallocate
		ρ_ests = fill(-99., n_bootstraps)
		a_ests = fill(-99., n_bootstraps)
	
		# Run over bootstraps
		for i in 1:n_bootstraps
	
			# Sample blocks with replacement
			block_sample = sample(rng, unique(data.stimulus_group), length(unique(data.stimulus_group)))
            # println("Bootstrap sample: ", block_sample)
	
			# Subset data
			this_dat = vcat([
				DataFrames.transform(filter(x -> x.stimulus_group == b, data), 
				:stimulus_group => ByRow(x -> j) => :stimulus_group,
				:stimulus_group => identity => :original_stimulus_group
				) 
				for (j, b) in enumerate(block_sample)]...)

            # println("this_dat block counts: ", combine(groupby(this_dat, :stimulus_group), nrow))
            
            # Fit data
            try
                unpacked_data = unpack_single_p_QL_categorical(
                    this_dat;
                    columns = wm_columns
                )
                
                # Debug: check unpacked data            
                fit = optimize(
                    unpacked_data;
                    model = single_p_QL_categorical,
                    priors = priors,
                    initV = mean([0.01, 0.9])
                )

                # Push results
                ρ_ests[i] = fit.values[:ρ]
                a_ests[i] = fit.values[:a]

            catch e
                println("Error during optimization:")
                println(e)
                # Print more details about the data
                println("Data types:")
                println("stimulus_group: $(typeof(this_dat.stimulus_group)), unique values: $(unique(this_dat.stimulus_group))")
                println("choice: $(typeof(this_dat.choice)), unique values: $(unique(this_dat.choice))")
                rethrow(e)
            end
        
        end
	
		ρ_ests, a_ests
	
	end
end

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

# Get Q values
begin
	# Compute Q values
	Qs = single_p_QL_categorical(;
		unpack_single_p_QL_categorical(
			data;
			columns = wm_columns
		)...,
		priors = priors
	)()

	# Add block and trial
	Qs_df = DataFrame(
		stimulus_group = data.stimulus_group,
		appearance = data.appearance,
		choice = data.choice,
		Q_left = Qs[:, 1],
		Q_middle = Qs[:, 2],
        Q_right = Qs[:, 3],
		chosen_feedback = (r -> [r.feedback_left, r.feedback_middle, r.feedback_right][r.choice]).(eachrow(data)),
        rho = fill(ρ_est, nrow(data)),
		a = fill(a_est, nrow(data)),
		alpha = fill(a2α(a_est), nrow(data))
	)

    # Compute PE
    Qs_df.chosen_Q = (r -> [r.Q_left, r.Q_middle, r.Q_right][r.choice]).(eachrow(Qs_df))
    Qs_df.PE = Qs_df.chosen_feedback .* ρ_est .- Qs_df.chosen_Q


    # Combine with original data for trial numbers
    leftjoin!(
        Qs_df,
        select(data, :stimulus_group, :appearance, :trial),
        on = [:stimulus_group, :appearance]
    )

    sort!(Qs_df, :trial)

    @assert nrow(Qs_df) == nrow(data) "Mismatch in number of rows between Qs_df and data"

	Qs_df
end

CSV.write(output_file, Qs_df)
