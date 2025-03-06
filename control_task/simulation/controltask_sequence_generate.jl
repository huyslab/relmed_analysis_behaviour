import Pkg
Pkg.activate("relmed_environment")
Pkg.instantiate()
@info "Running with $(Threads.nthreads()) threads"

using Revise, Random, DataFrames, CairoMakie, Distributions, CSV, JSON, Combinatorics
CairoMakie.activate!(type="svg")
includet("ControlEffortTask.jl")
using .ControlEffortTask

pars = TaskParameters(
  n_trials=144,
  beta_true=[2, 5, 8] .* 3,
)

ship_name = ["green", "blue", "red", "yellow"]
island_name = ["banana", "coconut", "grape", "orange"]

# Set random seed for reproducibility
Random.seed!(123)

# Run batch experiments
stimuli_sequence = generate_stimuli(pars; mode=:factorial, shuffle=true)

# Exploration sequence
flattened_stimuli_sequence = map(stimuli_sequence) do s
  # Extract boat values
  left, right = s.boats
  # Create new NamedTuple with flattened structure
  return (; left=ship_name[left], right=ship_name[right], near=island_name[s.current_state], current=s.wind)
end

CSV.write("exploration_sequence.csv", DataFrame(flattened_stimuli_sequence))

open("exploration_sequence.json", "w") do io
  JSON.print(io, flattened_stimuli_sequence)
end

# Prediction sequence
function shuffle_with_no_identical_pairs(perms)
    # Convert to canonical form for comparison (sort each inner pair)
    function canonical_form(p)
        return sort([p[1], p[2]])
    end

    function shares_elements(p1, p2)
        return any(x -> x in p2, p1)
    end
    
    # Initial shuffle
    max_attempts = 10000000
    n_invalid = 0
    result = shuffle(copy(perms))
    
    for attempt in 1:max_attempts
        valid = true
        
        # Check for consecutive pairs that share elements
        for i in 1:(length(result)-1)
            if shares_elements(result[i], result[i+1])
                n_invalid += 1
                if n_invalid > (length(result) * 1/2)
                  valid = false
                  n_invalid = 0
                  shuffle!(result)
                  break
                end
            end
        end
        
        if valid
          @info "Found valid permutation after $(attempt) attempts"
          return result
        end
    end
    
    error("Could not find a valid permutation after $max_attempts attempts")
end

# Shuffle with constraint for Prediction sequence
perms = collect(permutations(1:pars.n_states, 2))
shuffled_perms = shuffle_with_no_identical_pairs(perms)
prediction_sequence = [];
for (i, perm) in enumerate(shuffled_perms)
  append!(prediction_sequence, [(;ship=ship_name[perm[1]]), (;ship=ship_name[perm[2]])])
end
CSV.write("prediction_sequence.csv", DataFrame(prediction_sequence))
open("prediction_sequence.json", "w") do io
  JSON.print(io, prediction_sequence)
end

# Reward sequence
target_list = [2, 3, 4, 1]
islands_df = allcombinations(DataFrame, target_island = 1:4, near_island = 1:4)
filter!([:target_island, :near_island] => (x, y) -> x != y, islands_df)
boats = map(collect(combinations(1:pars.n_states, 2))) do perm
  return (;left = perm[1], right = perm[2])
end
reward_df = crossjoin(islands_df, DataFrame(boats)) |>
  x -> transform(x, [:left, :right, :target_island] => ByRow((l, r, t) -> (findfirst(target_list .== t) ∈ [l, r])) => :ship_viable) |>
  x -> transform(x, [:left, :target_island] => ByRow((l, t) -> (findfirst(target_list .== t) == l)) => :left_viable) |>
  x -> transform(x, [:right, :target_island] => ByRow((r, t) -> (findfirst(target_list .== t) == r)) => :right_viable) |>
  x -> transform(x, [:near_island, :target_island] => ByRow((n, t) -> (findfirst(target_list .== t) == n)) => :island_viable) |>
  x -> filter(x -> x.ship_viable || x.island_viable, x)
combine(groupby(reward_df, [:left_viable, :right_viable, :island_viable]), nrow)

# Only island viable
island_viable_df = filter(row -> !row.ship_viable && row.island_viable, reward_df)
# Only ship viable
ship_viable_df = filter(row -> row.ship_viable && !row.island_viable, reward_df)
unique_combinations = combine(
    groupby(ship_viable_df, [:target_island, :left, :right])
) do group_df
    # Sample one random row from this group
    return group_df[rand(1:nrow(group_df)), :]
end

reward_sequence = vcat(island_viable_df, unique_combinations)
combine(groupby(reward_sequence, [:left_viable, :right_viable, :island_viable]), nrow)
shuffle!(reward_sequence)

# Convert to vector of NamedTuples for saving
reward_sequence_tuples = map(eachrow(reward_sequence)) do row
  return (
    target=island_name[row.target_island], 
    near=island_name[row.near_island], 
    left=ship_name[row.left], 
    right=ship_name[row.right]
  )
end

# Save to files
CSV.write("reward_sequence.csv", DataFrame(reward_sequence_tuples))
open("reward_sequence.json", "w") do io
  JSON.print(io, reward_sequence_tuples)
end