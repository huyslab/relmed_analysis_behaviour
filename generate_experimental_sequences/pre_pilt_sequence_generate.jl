import Pkg
Pkg.activate("relmed_environment")
Pkg.instantiate()

using Random
using DataFrames
using CSV, JSON

# Set random seed for reproducibility
Random.seed!(123)

# Create the initial vector and repeat each element 5 times
values = [-1, -0.5, -0.01, 0.01, 0.5, 1]
repeated_values = repeat(values, inner=5)

# Function to check if there are three consecutive same elements
function has_three_consecutive_same(arr)
  for i in 1:(length(arr)-2)
    if arr[i] == arr[i+1] && arr[i] == arr[i+2]
      return true
    end
  end
  return false
end

# Shuffle until no three consecutive elements are the same
shuffled_values = copy(repeated_values)
while has_three_consecutive_same(shuffled_values)
  shuffle!(shuffled_values)
end

prepilt_sequence = [];
foreach(shuffled_values) do value
  append!(prepilt_sequence, [(; pav_value = value)])
end

CSV.write("pre_pilt_sequence.csv", DataFrame(prepilt_sequence))
open("pre_pilt_sequence.json", "w") do io
  JSON.print(io, prepilt_sequence)
end