# This script contains function to test schemes for computing bonus payment to participants

"""
    least_british_coins(amount::Int) -> Vector{Int}

Compute the minimum number of British coins needed to make up a given amount of money.

# Arguments
- `amount::Int`: The total amount of money in pence. Must be a non-negative integer.

# Returns
- A vector of integers where each element represents the number of coins of a specific denomination used.
  The order of denominations in the vector corresponds to the order `[100, 50, 20, 10, 5, 2, 1]` (i.e., £1, 50p, 20p, 10p, 5p, 2p, and 1p).
"""
function least_british_coins(amount::Int)
    # Denominations of British coins in pence
    coins = [100, 50, 20, 10, 5, 2, 1]
    
    # Dictionary to store the count of each coin
    result = fill(0, length(coins))

    for (i, coin) in enumerate(coins)
        if amount >= coin
            result[i] = div(amount, coin)  # Number of this coin
            amount = amount % coin            # Remainder to be broken down
        end
    end

    return result
end

"""
    winning_per_acc_level(task::DataFrame, acc::Float64; n_shuffles::Int64 = 100) -> Int

Compute the amount of money a participant is expected to make on the PILT based on a given accuracy level.

# Arguments
- `task::DataFrame`: A DataFrame containing three columns:
  - `better_feedback::Vector{Float64}`: Winnings in pounds for correct answers.
  - `worse_feedback::Vector{Float64}`: Losses in pounds for incorrect answers.
  - `valence::Vector{Int}`: Indicator (-1 for negative feedback, 1 for positive feedback).

- `acc::Float64`: The accuracy level of the participant, as a decimal between 0 and 1.

- `n_shuffles::Int64`: (Optional) The number of shuffles to average over for a more accurate estimate. Default is 100.

# Returns
- An integer representing the expected amount of money in pence that the participant is expected to make, rounded to the nearest penny.
"""
function winning_per_acc_level(
	task::DataFrame,
	acc::Float64;
	n_shuffles::Int64 = 100 # How many shuffles to average over
)	

	n_trials = nrow(task)
	n_correct = round(Int64, acc * n_trials)

	# Compute how many pounds are in safe to begin with
	prelim_safe = -sum(filter(x -> x.valence == -1, task).worse_feedback)

	winnings = 0.

	for _ in 1:n_shuffles
		# Draw outcomes
		outcomes = ifelse.(
			shuffle(vcat(
				fill(true, n_correct),
				fill(false, n_trials - n_correct)
			)),
			task.better_feedback,
			task.worse_feedback
		)

		winnings += sum(outcomes) / n_shuffles

	end

	return round(Int64, (winnings + prelim_safe) * 100)
end

"""
    bonus_by_acc_plot(task::DataFrame; n_shuffles::Int64 = 500) -> Figure

Generate a plot showing the expected distribution of coins won in the PILT based on varying accuracy levels, with conversion to the largest value coins possible.

# Arguments
- `task::DataFrame`: A DataFrame containing columns `better_feedback`, `worse_feedback`, and `valence`, which represent the feedback values and their associated valence in the PILT.

- `n_shuffles::Int64`: (Optional) The number of shuffles to average over for estimating expected winnings. Default is 500.

# Returns
- A `Figure` object with two plots:
  1. A stacked bar plot showing the distribution of coin counts across different accuracy levels, as well as expected bonus amount.
  2. A band plot showing the proportion of each type of coin across different accuracy levels.
"""
function bonus_by_acc_plot(
	task::DataFrame;
	n_shuffles::Int64 = 500 # How many shuffles to average over
)

	coin_vals = [100, 50, 20, 10, 5, 2, 1]
	
	# Values of acc to evaluate
	accs = range(0.5, 1., length = 50)

	# Winnigs per acc
	winnings = (x -> winning_per_acc_level(task, x, n_shuffles = n_shuffles)).(accs)

	# Coins per acc
	coins = hcat(least_british_coins.(winnings)...)

	# Relative coins per acc
	rel_coins = coins ./ sum(coins, dims = 1)

	# Compute average reward
	avg_reward = sum(rel_coins .* coin_vals, dims = 1) |> vec

	f = Figure(size = (700, 350))

	# Make total of coins plot
	# Calculate the cumulative sums for stacking
	cumsum_coins = vcat(fill(0, 1, size(coins, 2)), cumsum(coins, dims = 1))

	ax_tot = Axis(f[1, 1], xlabel = "Accuracy", ylabel = "Coin counts")

	# Stacked bar plot
	for i in 1:size(coins, 1)
	    barplot!(ax_tot, 
			accs, 
			coins[i, :], 
			color = Makie.wong_colors()[i], 
			offset = cumsum_coins[i, :], 
			direction = :y,
			gap = 0.
		)
	end

	# Plot average reward
	lines!(
		ax_tot,
		accs,
		avg_reward,
		color = :white
	)


	# Make proportional coins plot
	cumsum_rel_coins = vcat(fill(0, 1, size(coins, 2)), cumsum(rel_coins, dims = 1))
	
	ax_prop = Axis(f[1, 2], xlabel = "Accuracy", ylabel = "Proportion of coins")
	

	for i in 1:(size(cumsum_rel_coins, 1) - 1)
	    band!(ax_prop, 
			accs, 
			cumsum_rel_coins[i, :], 
			cumsum_rel_coins[i+1, :], 
			color = Makie.wong_colors()[i]
		)
	end

	Legend(
		f[1, 3],
		vcat([PolyElement(color = c) for c in Makie.wong_colors()[1:size(coins, 1)]], [[PolyElement(color = :grey), LineElement(color = :white)]]),
		vcat(string.(coin_vals), ["Avg. reward"]),
		"Coin type",
		framevisible = false
	)

	Label(
		f[0,1:2],
		"PILT",
		font = :bold
	)


	f

end