"""
RELMED - Task Simulation
========================
This module contains functions to simulate task performance for trial I of RELMED.
It includes simulations of random agents for the reversal learning task, probabilistic instrumental learning task (PILT),
and working memory task. Critical accuracy thresholds under the null hypothesis are calculated based on the simulations.
"""

using Random, DataFrames, Distributions

# Simulate and summarize n_reversals
"""
    simulate_reversals(; n_trials=150, reversal_sequence, criteria, bias, rng=Xoshiro(0), statistic="accuracy")

Simulate random reversal learning task performance with given response bias.
Returns either accuracy or number of reversals based on statistic parameter.
"""
function simulate_reversals(;
    n_trials::Int = 150,
    reversal_sequence::AbstractDataFrame,
    criteria::AbstractVector,
    bias::Float64,
    rng::AbstractRNG = Xoshiro(0),
    statistic::String = "accuracy"
)

    choices = rand(rng, n_trials) .< bias # Generate random binary choices based on rightward bias

    acc_counter = 0          # Track correct responses in current block
    global_acc_counter = 0   # Track total correct responses across all blocks
    trial_counter = 1        # Track trial number within current block
    block_counter = 1        # Track current block number
    block_data = reversal_sequence[reversal_sequence.block .== block_counter, :] # Get data for current block
    
    for (i, c) in enumerate(choices) # Iterate through each simulated choice
        # Determine if choice was correct based on feedback for this trial
        acc = c ? block_data.feedback_right[trial_counter] == 1. : block_data.feedback_left[trial_counter] == 1.
        acc_counter += acc        # Increment block accuracy counter
        global_acc_counter += acc # Increment global accuracy counter

        trial_counter += 1 # Move to next trial within block
        
        # Check if block should end (criterion met or max trials reached)
        if acc_counter >= criteria[block_counter] || trial_counter > 80
            # Reset counters for next block
            acc_counter = 0
            block_counter += 1
            trial_counter = 1
            # Load data for next block
            block_data = reversal_sequence[reversal_sequence.block .== block_counter, :]
        end
    end

    # Return requested statistic
    if statistic == "n_reversals"
        return block_counter - 1 # Number of reversals is number of blocks - 1
    end

    if statistic == "accuracy"
        return global_acc_counter / n_trials # Calculate overall accuracy proportion
    end
    error("Unknown statistic: $statistic") # Handle invalid statistic parameter
end

"""
    reversal_critical_under_null(response; session, n_samples=1000, seed=1234)

Calculate critical accuracy threshold (95th percentile) under the null hypothesis of random behaviour for reversal learning task.
"""
function reversal_critical_under_null(
    response::AbstractVector;
    reversal_sequence::AbstractDataFrame,
    session::String,
    n_samples::Int = 1000,
    seed::Int = 1234)

    # Set random number generator with specified seed for reproducibility
    rng = Xoshiro(seed)

    # Calculate observed rightward response bias from actual data
    bias = mean(response .== "right")

    # Filter sequence data for the specified session
    sequence = filter(x -> x.session == session, reversal_sequence)

    # Extract criterion values for each block in this session
    criteria = unique(sequence[!, [:block, :criterion]]).criterion
    
    # Run simulation n_samples times to build null distribution
    simulated_acc = [simulate_reversals(;
        n_trials = length(response),    # Use same number of trials as observed data
        reversal_sequence = sequence,   # Use session-specific sequence
        criteria = criteria,            # Use session-specific criteria
        bias = bias,                    # Use observed response bias
        rng = rng                       # Use seeded RNG
    ) for _ in 1:n_samples]

    # Return the 95th percentile as critical threshold
    return quantile(simulated_acc, 0.95)
end

"""
    simulate_PILT_block(; n_samples=1000, bias, optimal_right, rng=Xoshiro(0))

Simulate random choice on a PILT block with early stopping rule.
"""
function simulate_PILT_block(;
    n_samples::Int = 1000,
    bias::Float64,
    optimal_right::AbstractVector,
    rng::AbstractRNG = Xoshiro(0)
)
    n_trials = length(optimal_right) # Get number of trials in this block

    # Generate responses for each simulation based on rightward bias
    response_optimal = [(rand(rng, n_trials) .< bias) .== optimal_right for _ in 1:n_samples]

    # Apply early stopping rule: stop after 5 consecutive correct responses
    for (j, resp) in enumerate(response_optimal)
        for i in 5:(n_trials - 1) # Start checking from trial 5
            if sum(resp[(i-4):i]) == 5 # Check if last 5 responses are correct
                # Truncate response at early stopping point
                response_optimal[j] = resp[1:i]
                break # Exit inner loop once early stopping triggered
            end
        end
    end

    # Create iteration labels for each response across all simulations
    iteration = vcat([fill(i, length(resp)) for (i, resp) in enumerate(response_optimal)]...)

    # Return named tuple with iteration labels and concatenated responses
    return (iteration = iteration, response_optimal = vcat(response_optimal...))
end

"""
    PILT_critical_under_null(response; session, n_samples=1000, seed=1234)

Calculate critical accuracy threshold (95th percentile) under the null hypothesis of random choice for PILT task.
Returns missing for screening sessions, since the data is insufficient for analysis.
"""
function PILT_critical_under_null(
    response::AbstractVector;
    pilt_sequence::AbstractDataFrame,
    session::String,
    n_samples::Int = 1000,
    seed::Int = 1234)

    # Return missing for screening sessions (no analysis needed)
    if session == "screening"
        return missing
    end
	
    # Initialize random number generator with seed for reproducibility
    rng = Xoshiro(seed)

    # Calculate observed rightward response bias
    bias = mean(response .== "right")

    # Filter sequence data for the specified session
    sequence = filter(x -> x.session == session, pilt_sequence)
    
    # Simulate each block separately using groupby operation
    simulated_data = combine(
        groupby(sequence, :block), # Group by block
        # Apply simulation function to optimal_right column of each block
        :optimal_right => (x -> simulate_PILT_block(; bias = bias, optimal_right = x, rng = rng, n_samples = n_samples)) => AsTable
    )

    # Calculate mean accuracy for each simulation iteration
    simulated_acc = combine(
        groupby(simulated_data, :iteration), # Group by simulation iteration
        :response_optimal => mean => :acc     # Calculate mean accuracy per iteration
    )

    # Return 95th percentile of simulated accuracies as critical threshold
    return quantile(simulated_acc.acc, 0.95)
end

"""
    WM_critical_under_null(optimal_side, response; n_samples=1000, seed=1234)

Calculate critical accuracy threshold (95th percentile) under the null hypothesis of random choice in the working memory task.
Uses multinomial sampling for three-choice responses (left, middle, right).
"""
function WM_critical_under_null(
    optimal_side::AbstractVector,
    response::AbstractVector;
    n_samples::Int = 1000,
    seed::Int = 1234    
)

    # Initialize random number generator with seed
    rng = Xoshiro(seed)

    # Calculate response bias for each of the three choices
    bias = [
        mean(response .== "left"),   # Proportion of left responses
        mean(response .== "middle"), # Proportion of middle responses
        mean(response .== "right")   # Proportion of right responses
    ]

    # Ensure bias probabilities sum to 1 (valid probability distribution)
    @assert sum(bias) ≈ 1. "Bias vector must sum to 1"

    # Helper function to draw random response based on observed bias
    draw_trial = bias -> ["left", "middle", "right"][findfirst(1 .== rand(rng, Multinomial(1, bias)))]

    # Simulate accuracy for n_samples iterations
	simulated_acc = [mean([draw_trial(bias) for _ in 1:length(response)] .== optimal_side) for _ in 1:n_samples]

    # Return 95th percentile as critical threshold
    return quantile(simulated_acc, 0.95)
end