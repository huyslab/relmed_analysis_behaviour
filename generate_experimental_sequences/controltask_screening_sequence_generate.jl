begin
  import Pkg
  Pkg.activate("relmed_environment")
  Pkg.instantiate()
  @info "Running with $(Threads.nthreads()) threads"

  using Revise, Random, DataFrames, CairoMakie, Distributions, CSV, JSON, Combinatorics, StatsBase
  includet("../control_task/simulation/ControlEffortTask.jl")
  using .ControlEffortTask
end

# Reduce exploration trials but creating the full combination of stimuli without considering the left/right order
begin
  pars = TaskParameters(
    n_trials=24,
    beta_true=[2, 4, 6] .* 3,
    n_states=3
  )

  ship_name = ["red", "green", "blue"]
  island_name = ["i1", "i2", "i3"]

  # Set random seed for reproducibility
  Random.seed!(0)

  stimuli_list = []
  for current_state in 1:pars.n_states
    for wind in 1:3
      for (boat1, boat2) in combinations(1:pars.n_states, 2)
        push!(stimuli_list, (; current_state, boat1, boat2, wind))
      end
    end
  end
end

# Randomly remove three rows where boat1 and boat2 are either (1,2) or (2,1), one for each wind value
# If we want full combinations, we can skip this part.
begin
  wind_to_remove = [1, 2, 3]  # We want to remove one row for each of these wind values
  indices_to_remove = []

  for w in wind_to_remove
    # Find eligible rows with boat combination (1,2) or (2,1) and the current wind value
    eligible_indices = findall(s -> ((s.boat1 == 1 && s.boat2 == 2) ||
                                     (s.boat1 == 2 && s.boat2 == 1)) &&
        s.wind == w, stimuli_list)

    # If eligible indices found, randomly select one to remove
    if !isempty(eligible_indices)
      push!(indices_to_remove, rand(eligible_indices))
    end
  end

  # Remove the selected rows
  if length(indices_to_remove) == 3
    deleteat!(stimuli_list, sort(indices_to_remove))
    @info "Removed $(length(indices_to_remove)) rows with boat combinations (1,2) or (2,1), one for each wind value 1, 2, and 3"
  else
    @warn "Could not find eligible rows to remove for all wind values. Only found $(length(indices_to_remove))"
  end
end

# Check if the number of boats is balanced between left/right
begin
  attempt = 0
  max_attempts = 10000000
  swapped_stimuli_list = Vector(undef, length(stimuli_list))
  while attempt < max_attempts
    copy!(swapped_stimuli_list, stimuli_list)
    # Check if the number of boats is balanced
    swap_idx = sample(1:24, 12, replace=false)
    for idx in swap_idx
      s = stimuli_list[idx]
      swapped_stimuli_list[idx] = (; s.current_state, boat1=s.boat2, boat2=s.boat1, s.wind)
    end
    if all(counts(DataFrame(swapped_stimuli_list).boat1) .== 8)
      @info "Found balanced swap between ships after $(attempt) attempts"
      break
    end

    global attempt += 1

  end

  if attempt == max_attempts
    error("Could not find a balanced swap after $max_attempts attempts")
  end
end

# Keep shuffling until no consecutive items have the same boats and no more than two consecutive elements have the same wind
begin
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
end

# Assemble exploration sequence and export
begin
  flattened_stimuli_sequence = map(swapped_stimuli_list) do s
    # Create new NamedTuple with flattened structure
    return (; left=ship_name[s.boat1], right=ship_name[s.boat2], near=island_name[s.current_state], current=s.wind)
  end

  CSV.write("screening_exploration_sequence.csv", DataFrame(flattened_stimuli_sequence))

  open("screening_exploration_sequence.json", "w") do io
    JSON.print(io, flattened_stimuli_sequence)
  end
end
