begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates
	using LogExpFunctions: logistic, logit
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


# Function to compute delays since last appearance of each stimulus
# This function returns a vector of delays for each element in the input vector.
# If the element has not appeared before, the delay is 0.
function compute_delays(vec::AbstractVector)
	last_seen = Dict{Any, Int}()
	delays = zeros(Int, length(vec))

	for (i, val) in enumerate(vec)
		delays[i] = haskey(last_seen, val) ? (i - last_seen[val]) : 0
		last_seen[val] = i
	end

	return delays
end

# Load data
stimulus_sequence = let
	dense_sequence = DataFrame(CSV.File("generate_experimental_sequences/wm_stimulus_sequence_short_9stim.csv"))
	dense_sequence.sequence .= "dense"

	sparse_sequence = DataFrame(CSV.File("generate_experimental_sequences/pilot8_wm_stimulus_sequence.csv"))
	sparse_sequence.sequence .= "sparse"

	stimulus_sequence = vcat(dense_sequence, sparse_sequence)

	select!(
		stimulus_sequence,
		:sequence,
		:trial_ovl => :trial,
		:stimset => :stimulus_group
	)

	# Compute delays for each stimulus group
	transform!(
		groupby(stimulus_sequence, :sequence),
		:stimulus_group => compute_delays => :delay
	)

end

# Compute delay histograms per sequence
let
    f = Figure()
    ax = Axis(f[1, 1], 
        xlabel="Delay (trials)",
        ylabel="Probability",
        title="Distribution of Delays Between Stimulus Repetitions"
    )

    # Define consistent bins for both sequences
    max_delay = maximum(stimulus_sequence.delay)
    bins = 0:1:max_delay
    
    # Get data for each sequence
    dense_delays = stimulus_sequence[stimulus_sequence.sequence .== "dense", :delay]
    sparse_delays = stimulus_sequence[stimulus_sequence.sequence .== "sparse", :delay]
    
    # Get Wong colors
    colors = Makie.wong_colors()
    
    # Create histograms with transparency
    hist!(ax, dense_delays, bins=bins, normalization=:probability, 
          color=(colors[1], 0.6), label="dense")
    hist!(ax, sparse_delays, bins=bins, normalization=:probability, 
          color=(colors[2], 0.6), label="sparse")
    
    axislegend(ax)
    f
end

function compute_moving_unique_count(stimulus_groups::AbstractVector, window_size::Int=8)
    n = length(stimulus_groups)
    moving_counts = zeros(Int, n)
    
    for i in 1:n
        # Determine the start of the window (at least 1, at most window_size trials back)
        start_idx = max(1, i - window_size + 1)
        
        # Count unique stimulus groups in the window ending at current trial
        window_groups = stimulus_groups[start_idx:i]
        moving_counts[i] = length(unique(window_groups))
    end
    
    return moving_counts
end

transform!(
	groupby(stimulus_sequence, :sequence),
	:stimulus_group => compute_moving_unique_count => :moving_unique_count
)

let

	data(stimulus_sequence) * mapping(
		:trial,
		:moving_unique_count,
		color = :sequence
	) * visual(Lines) |> draw()
end

combine(
	groupby(stimulus_sequence, :sequence),
	:stimulus_group => (x -> length(unique(x))) => :n_groups
)