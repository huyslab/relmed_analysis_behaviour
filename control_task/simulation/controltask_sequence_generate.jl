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
  beta_true=[2, 4, 6] .* 3,
)

ship_name = ["green", "blue", "red", "yellow"]
island_name = ["banana", "coconut", "grape", "orange"]

# Set random seed for reproducibility
Random.seed!(0)

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
function shuffle_with_no_shared(perms)
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
        if n_invalid > (length(result) * 1 / 2)
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
shuffled_perms = shuffle_with_no_shared(perms)
prediction_sequence = [];
for (i, perm) in enumerate(shuffled_perms)
  append!(prediction_sequence, [(; ship=ship_name[perm[1]]), (; ship=ship_name[perm[2]])])
end
CSV.write("prediction_sequence.csv", DataFrame(prediction_sequence))
open("prediction_sequence.json", "w") do io
  JSON.print(io, prediction_sequence)
end

# Reward sequence
target_list = [2, 3, 4, 1]
islands_df = allcombinations(DataFrame, target_island=1:4, near_island=1:4)
filter!([:target_island, :near_island] => (x, y) -> x != y, islands_df)
boats = map(collect(permutations(1:pars.n_states, 2))) do perm
  return (; left=perm[1], right=perm[2])
end
reward_df = crossjoin(islands_df, DataFrame(boats)) |>
  x -> transform(x, [:left, :right, :target_island] => ByRow((l, r, t) -> (findfirst(target_list .== t) ∈ [l, r])) => :ship_viable) |>
  x -> transform(x, [:left, :target_island] => ByRow((l, t) -> (findfirst(target_list .== t) == l)) => :left_viable) |>
  x -> transform(x, [:right, :target_island] => ByRow((r, t) -> (findfirst(target_list .== t) == r)) => :right_viable) |>
  x -> transform(x, [:near_island, :target_island] => ByRow((n, t) -> (findfirst(target_list .== t) == n)) => :island_viable) |>
  x -> filter(x -> x.ship_viable || x.island_viable, x)
filter!(x -> !(x.target_island != 3 && (x.left == 2 || x.right == 2)), reward_df)
# filter!(x -> x.target_island != 3, reward_df)
combine(groupby(reward_df, [:left_viable, :right_viable, :island_viable]), nrow)

# Only island viable
island_viable_df = filter(row -> !row.ship_viable && row.island_viable, reward_df)
# Both island and ship viable
both_viable_df = filter(row -> row.ship_viable && row.island_viable, reward_df)
# Only ship viable
ship_viable_df = filter(row -> row.ship_viable && !row.island_viable, reward_df)
unique_combinations = combine(
  groupby(ship_viable_df, [:target_island, :left, :right])
) do group_df
  # Sample one random row from this group
  return group_df[rand(1:nrow(group_df)), :]
end

# # Replace some trials with both viable trials
# # Result in a fewer number of trials but would have unbalanced number of target island trials, especially with more 3 (grape islands)
# # Maybe consider removing all the trials with target island 3 (6 of them), and them swap three in total, one in each target island group
# # Use unique_combinations instead of ship_viable_df will result in only half of the trials (12)
# to_replace = combine(groupby(ship_viable_df, [:target_island, :left_viable, :right_viable])) do group_df
#   return group_df[sample(1:nrow(group_df), 1, replace=false), :]
# end
# reward_sequence = vcat(
#   semijoin(both_viable_df, to_replace, on=[:target_island, :left, :right]),
#   antijoin(ship_viable_df, to_replace, on=[:target_island, :near_island, :left, :right])
# )

# Or add both extra 6 viable trials to the reward sequence
to_add = filter(row -> row.target_island != 3, both_viable_df) |>
  x -> combine(groupby(x, [:target_island, :left_viable, :right_viable])) do group_df
    return group_df[rand(1:nrow(group_df)), :]
  end

reward_sequence = vcat(
  to_add,
  unique_combinations
)

combine(groupby(reward_sequence, [:left_viable, :right_viable, :island_viable]), nrow)
transform!(groupby(reward_sequence, [:target_island, :left_viable, :right_viable]), :target_island => (x -> sample(repeat(1:3, Int(length(x) / 3)), length(x), replace=false)) => :current)
# while nrow(combine(groupby(reward_sequence, [:target_island, :current]), nrow)) != 9
#   transform!(groupby(reward_sequence, [:left_viable, :right_viable]), :target_island => (x -> sample(repeat(1:3, Int(length(x) / 3)), length(x), replace=false)) => :current)
# end
combine(groupby(reward_sequence, [:target_island, :current]), nrow)
# Check for consecutive target islands and reshuffle if needed
function has_consecutive_targets(seq, n=3)
  for i in 1:(nrow(seq)-n+1)
    if length(unique(seq[i:(i+n-1), :target_island])) == 1
      return true
    end
  end
  return false
end

# Reshuffle until no consecutive target islands
max_attempts = 1000
attempt = 0
while has_consecutive_targets(reward_sequence) && attempt < max_attempts
  shuffle!(reward_sequence)
  global attempt += 1
end

if attempt == max_attempts
  @warn "Could not find sequence without consecutive targets after $max_attempts attempts"
else
  @info "Found valid sequence after $attempt reshuffles"
end

# Convert to vector of NamedTuples for saving
reward_sequence_tuples = map(eachrow(reward_sequence)) do row
  return (
    target=island_name[row.target_island],
    near=island_name[row.near_island],
    left=ship_name[row.left],
    right=ship_name[row.right],
    current=row.current,
    island_viable=row.island_viable,
    left_viable=row.left_viable,
    right_viable=row.right_viable
  )
end

# Save to files
CSV.write("reward_sequence.csv", DataFrame(reward_sequence_tuples))
open("reward_sequence.json", "w") do io
  JSON.print(io, reward_sequence_tuples)
end