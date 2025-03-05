import Pkg
Pkg.activate("relmed_environment")
Pkg.instantiate()
@info "Running with $(Threads.nthreads()) threads"

using Revise, Random
includet("ControlEffortTask.jl")
using .ControlEffortTask

# Set random seed for reproducibility
Random.seed!(1234)

pars = TaskParameters(
     n_trials=144,
     beta_true=[2, 4, 8] .* 3,
)

# Run single experiment

dataset = generate_dataset(pars, mode=:factorial, shuffle=true)

results = run_experiment(pars, dataset; show_plots=true);

# Run batch experiments
stimuli_sequence = generate_stimuli(pars; mode=:factorial, shuffle=true)
results_list, datasets_list = run_batch_experiment(pars, stimuli_sequence; n_participants=25, show_plots=false);

# Plotting ---------------------------------------------------------------

using AlgebraOfGraphics, ColorSchemes, CairoMakie, DataFrames, StatsBase

set_theme!(theme_minimal())
colors = ColorSchemes.Paired_10.colors;
inch = 96
pt = 4 / 3
cm = inch / 2.54

fig = Figure(; figure_padding=0.1cm, size=(16cm, 8cm), fontsize=8pt)

subfig_r1 = GridLayout(1, 4)
colsize!(subfig_r1, 1, Fixed(4cm))
colsize!(subfig_r1, 2, Auto())
colsize!(subfig_r1, 3, Fixed(4cm))
colgap!(subfig_r1, 3, 0.1cm)
colsize!(subfig_r1, 4, Auto())

subfig_r2 = GridLayout(1, 4)
colsize!(subfig_r2, 1, Fixed(4cm))
colsize!(subfig_r2, 2, Fixed(4cm))
colsize!(subfig_r2, 3, Fixed(3cm))
colgap!(subfig_r2, 3, 0cm)
colsize!(subfig_r2, 4, Auto())

fig.layout[1, 1] = subfig_r1
fig.layout[2, 1] = subfig_r2

n_experiments = length(results_list)

# Control estimate accuracy
control_acc_df = DataFrame()
for (i, results) in enumerate(results_list)
     # We don't have dataset, so we can only estimate from control_estimates
     # High values in control_estimates[1,:] indicate high probability of control
     # High values in control_estimates[2,:] indicate high probability of uncontrol
     dataset = DataFrame(datasets_list[i])
     mean_control_acc = mean(results.control_estimates[1, dataset.controlled])
     mean_uncontrol_acc = mean(results.control_estimates[2, .!dataset.controlled])

     push!(control_acc_df, (experiment=i, condition="C=1", accuracy=mean_control_acc))
     push!(control_acc_df, (experiment=i, condition="C=0", accuracy=mean_uncontrol_acc))
end

p2 = data(control_acc_df) *
     mapping(:condition => presorted, :accuracy) *
     (mapping(color=:condition => presorted) *
      visual(RainClouds) +
      mapping(group=:experiment => nonnumeric) *
      visual(Lines, color=(:gray, 0.1)))
draw!(fig[1, 1][1, 2], p1; axis=(; title="Control events", xlabel="Condition", ylabel="Accuracy"))

# Transition matrix weights overlay
weights_array = zeros(size(results_list[1].weight_estimates)..., n_experiments)
for (i, results) in enumerate(results_list)
     weights_array[:, :, i] = results.weight_estimates
end

mean_weights = dropdims(mean(weights_array, dims=3), dims=3)
std_weights = dropdims(std(weights_array, dims=3), dims=3)

weights_df = DataFrame(mean_weights', :auto)
transform!(weights_df, eachindex => :x)
long_weights_df = stack(weights_df, Not(:x))
transform!(long_weights_df, :variable => ByRow(x -> parse(Int, match(r"[0-9]+", x).match)) => :y)

p3 = data(long_weights_df) *
     mapping(:y, :x, :value) *
     visual(Heatmap)

p3_leg = draw!(fig[1, 1][1, 3], p3, scales(Color=(; colormap=ColorSchemes.linear_blue_95_50_c20_n256));
     axis=(; title="Mean transition weights", xlabel="Trial", ylabel="Matrix weight"))
colorbar!(fig[1, 1][1, 4], p3_leg; label="Weight")

# Betas over time
betas_df = DataFrame()
beta_labels = ["Low", "Mid", "High"]

for (i, results) in enumerate(results_list)
     # We don't have dataset, so we can only estimate from control_estimates
     # High values in control_estimates[1,:] indicate high probability of control
     # High values in control_estimates[2,:] indicate high probability of uncontrol
     df = DataFrame(results.beta_estimates', :auto)
     df.experiment .= i
     transform!(df, eachindex => :trial)
     long_df = stack(df, Not([:trial, :experiment]))
     display(long_df)
     transform!(long_df, :variable => ByRow(x -> beta_labels[parse(Int, match(r"[0-9]+", x).match)]) => :labels)

     append!(betas_df, long_df)
end

p4 = data((; β=pars.beta_true, labels=beta_labels)) *
     mapping(:β, color=:labels => presorted => "") *
     visual(HLines, linestyle=:dash, label="Ground\ntruth") +
     data(betas_df) *
     mapping(:trial, :value => "β", color=:labels => presorted => "", group=:experiment => nonnumeric) *
     visual(Lines, alpha=0.1, label="Estimated") +
     data(combine(groupby(betas_df, [:trial, :labels]), :value => mean => :value)) *
     mapping(:trial, :value => "β", color=:labels => presorted => "") *
     visual(Lines, linewidth=2, label="Averaged response")
draw!(fig[2, 1][1, 1], p4, scales(Color=(; palette=:Reds_3)); axis=(; title="Betas over time", xlabel="Trial", ylabel="Estimated β"))

# Effort functions
estimated_beta_df = filter(x -> x.trial .== 144, betas_df)
transform!(estimated_beta_df, :variable => ByRow(x -> pars.beta_true[parse(Int, match(r"[0-9]+", x).match)]) => :true_beta)
select!(estimated_beta_df, :experiment, :value => :estimated_beta, :true_beta, :labels)
effort_df = crossjoin(
     DataFrame(effort=0:30),
     estimated_beta_df
)
transform!(effort_df, [:effort, :estimated_beta] => ByRow((x, y) -> 1 / (1 + exp(-2 * (x - y)))) => :p_control_estimated)
transform!(effort_df, [:effort, :true_beta] => ByRow((x, y) -> 1 / (1 + exp(-2 * (x - y)))) => :p_control_true)
avg_effort_df = combine(groupby(effort_df, [:effort, :labels]), :p_control_estimated => mean => :p_control_avg)

p5 = data(unique(effort_df, [:effort, :p_control_true, :labels])) *
     mapping(:effort, :p_control_true, color=:labels => presorted => "") *
     visual(Lines, linestyle=:dash, label="Ground\ntruth") +
     data(effort_df) *
     mapping(:effort, :p_control_estimated, color=:labels => presorted => "", group=:experiment => nonnumeric) *
     visual(Lines, alpha=0.1, linestyle=:solid, label="Estimated")
p5 = p5 + data(avg_effort_df) *
          mapping(:effort, :p_control_avg, color=:labels => presorted => "") *
          visual(Lines, linewidth=2.5, label="Averaged response")
p5_leg = draw!(fig[2, 1][1, 2], p5, scales(Color=(; palette=:Reds_3)); axis=(; title="Effort functions", xlabel="Effort", ylabel="P(Control)"))
legend!(fig[2, 1][1, 3], p5_leg; tellheight=false, tellwidth=false, padding=(0, 0, 0, 0), gridshalign=:left, patchsize=(10, 10), rowgap=0)

# Final beta estimates as raincloud
p6 = data((; β=pars.beta_true, labels=beta_labels)) *
     mapping(:β, color=:labels => presorted => "") *
     visual(HLines, linestyle=:dash, label="Ground\ntruth") +
     data(estimated_beta_df) *
     mapping(:labels => presorted, :estimated_beta, color=:labels => presorted) *
     visual(RainClouds)
draw!(fig[1, 1][1, 1], p6, scales(Color=(; palette=:Reds_3)); axis=(; title="Final beta estimates", xlabel="Current levels", ylabel="Estimated value"))

fig