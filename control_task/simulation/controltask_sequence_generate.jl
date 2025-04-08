import Pkg
Pkg.activate("relmed_environment")
Pkg.instantiate()
@info "Running with $(Threads.nthreads()) threads"

using Revise, Random, DataFrames, CairoMakie, Distributions, CSV, JSON, Combinatorics, StatsBase
CairoMakie.activate!(type="svg")
includet("ControlEffortTask.jl")
using .ControlEffortTask

pars = TaskParameters(
  n_trials=72,
  beta_true=[2, 4, 6] .* 3,
)

ship_name = ["red", "yellow", "green", "blue"]
island_name = ["banana", "coconut", "grape", "orange"]

# Set random seed for reproducibility
Random.seed!(0)

# Reduce exploration trials
stimuli_list = []
for current_state in 1:pars.n_states
  for wind in 1:3
    for (boat1, boat2) in combinations(1:pars.n_states, 2)
      push!(stimuli_list, (; current_state, boat1, boat2, wind))
    end
  end
end

attempt = 0
max_attempts = 10000000
swapped_stimuli_list = Vector(undef, length(stimuli_list))
while attempt < max_attempts
  copy!(swapped_stimuli_list, stimuli_list)
  # Check if the number of boats is balanced
  swap_idx = sample(1:72, 36, replace=false)
  for idx in swap_idx
    s = stimuli_list[idx]
    swapped_stimuli_list[idx] = (; s.current_state, boat1=s.boat2, boat2=s.boat1, s.wind)
  end
  if all(counts(DataFrame(swapped_stimuli_list).boat1) .== 18)
    @info "Found balanced swap between ships after $(attempt) attempts"
    break
  end

  global attempt += 1

end

if attempt == max_attempts
  error("Could not find a balanced swap after $max_attempts attempts")
end

# Keep shuffling until no consecutive items have the same boats and no more than two consecutive elements have the same wind
attempt = 0
max_attempts = 10000000
while attempt < max_attempts
  shuffle!(swapped_stimuli_list)

  # Check if any consecutive items have the same boats
  has_consecutive_same_boats = false
  for i in 1:length(swapped_stimuli_list)-1
    if swapped_stimuli_list[i].boat1 == swapped_stimuli_list[i+1].boat1 &&
       swapped_stimuli_list[i].boat2 == swapped_stimuli_list[i+1].boat2
      has_consecutive_same_boats = true
      break
    end
  end

  # Check if more than two consecutive elements have the same wind
  has_too_many_consecutive_same_wind = false
  for i in 1:length(swapped_stimuli_list)-2
    if swapped_stimuli_list[i].wind == swapped_stimuli_list[i+1].wind &&
       swapped_stimuli_list[i].wind == swapped_stimuli_list[i+2].wind
      has_too_many_consecutive_same_wind = true
      break
    end
  end

  # If both conditions are satisfied, we're done
  if !has_consecutive_same_boats && !has_too_many_consecutive_same_wind
    @info "Found valid sequence with no consecutive identical boats and no more than two consecutive same wind after $(attempt) attempts"
    break
  end

  global attempt += 1
end

if attempt == max_attempts
  error("Could not find a valid sequence without consecutive identical boats and with wind constraints after $max_attempts attempts")
end

# Exploration sequence
flattened_stimuli_sequence = map(swapped_stimuli_list) do s
  # Create new NamedTuple with flattened structure
  return (; left=ship_name[s.boat1], right=ship_name[s.boat2], near=island_name[s.current_state], current=s.wind)
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
# Since we've told pariticpants that the home base of blue ship (2) is grape (3), so if the target island is not grape but the left or right ship is blue, it tells them the other ship's home base
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
  filter(row -> row.target_island != 3, both_viable_df),
  ship_viable_df
)

combine(groupby(reward_sequence, [:left_viable, :right_viable, :island_viable]), nrow)
transform!(groupby(reward_sequence, [:target_island, :left_viable, :right_viable]), :target_island => (x -> sample(repeat(1:3, Int(length(x) / 3)), length(x), replace=false)) => :current)
# Add reward amount column (50p for half, £1 for half) within each target_island and current combination
transform!(groupby(reward_sequence, [:target_island, :current])) do group
  n = nrow(group)
  # Create array with half 50p and half £1
  rewards = fill("50p", n)
  # Randomly select half the indices to be £1
  one_pound_indices = sample(1:n, floor(Int, n/2), replace=false)
  rewards[one_pound_indices] .= "£1"
  return (; reward_amount = rewards)
end
# while nrow(combine(groupby(reward_sequence, [:target_island, :current]), nrow)) != 9
#   transform!(groupby(reward_sequence, [:left_viable, :right_viable]), :target_island => (x -> sample(repeat(1:3, Int(length(x) / 3)), length(x), replace=false)) => :current)
# end

transform!(reward_sequence, [:left_viable, :right, :left] => ((x, y, z) -> ifelse.(x, y, z)) => :wrong_option)
transform!(reward_sequence, [:wrong_option, :near_island] => ByRow((x, y) -> x == y) => :wrong_match_homebase)
combine(groupby(reward_sequence, [:target_island, :current]), nrow)
# Check for consecutive target islands and reshuffle if needed
function has_consecutive_targets(seq, var=:target_island, n=3)
  for i in 1:(nrow(seq)-n+1)
    if length(unique(seq[i:(i+n-1), var])) == 1
      return true
    end
  end
  return false
end

# Reshuffle until no consecutive target islands
attempt = 0
max_attempts = 10000000
while (has_consecutive_targets(reward_sequence, :target_island, 2) || reward_sequence.island_viable[1]) && attempt < max_attempts
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
    right_viable=row.right_viable,
    reward_amount=row.reward_amount
  )
end

# Save to files
CSV.write("reward_sequence.csv", DataFrame(reward_sequence_tuples))
open("reward_sequence.json", "w") do io
  JSON.print(io, reward_sequence_tuples)
end