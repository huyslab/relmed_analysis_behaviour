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
  n_trials = 144,
  beta_true = [2, 4, 8] .* 3,
  )

dataset = generate_dataset(pars, mode=:factorial, shuffle=true)

results = run_experiment(pars, dataset; show_plots=true);

# Plotting ---------------------------------------------------------------

using AlgebraOfGraphics, ColorSchemes, CairoMakie, DataFrames, StatsBase

set_theme!(theme_minimal())
colors = ColorSchemes.Paired_10.colors;
inch = 96
pt = 4/3
cm = inch / 2.54

fig = Figure(; figure_padding = 0.1cm, size = (16cm, 8cm), fontsize = 8pt)

subfig_r1 = GridLayout(1, 4)
colsize!(subfig_r1, 1, Fixed(4cm))
colsize!(subfig_r1, 2, Auto())
colsize!(subfig_r1, 3, Fixed(4cm))
colgap!(subfig_r1, 3, 0.1cm)
colsize!(subfig_r1, 4, Auto())

subfig_r2 = GridLayout(1, 4)
colsize!(subfig_r2, 1, Fixed(4cm))
colsize!(subfig_r2, 2, Fixed(3cm))
colgap!(subfig_r2, 2, 0cm)
colsize!(subfig_r2, 3, Auto())
colsize!(subfig_r2, 4, Fixed(4cm))

fig.layout[1, 1] = subfig_r1
fig.layout[2, 1] = subfig_r2

# Online control estimates
dataset = DataFrame(dataset)
dataset.trial = 1:nrow(dataset)
p_control_df = DataFrame(trial=1:nrow(dataset), p_control=results.control_estimates[1, :])

p1 = data(dataset) *
        mapping(:trial, :controlled => Int) *
        visual(Scatter, color=(colors[2], 0.9), markersize=4) +
    data(p_control_df) *
        mapping(:trial, :p_control) *
        visual(Lines, color=colors[1], linewidth=0.9)
draw!(fig[1,1][1,1], p1; axis=(; title = "Control inference", xlabel = "Trial", ylabel = "P(Control)"))

# Control estimate accuracy
control_acc_df = DataFrame(
    condition = ["C=1", "C=0"],
    accuracy = [mean(results.control_estimates[1, dataset.controlled]), mean(results.control_estimates[2, Not(dataset.controlled)])]
)

p2 = data(control_acc_df) *
        mapping(:condition => presorted, :accuracy, color=:condition => presorted) *
        visual(BarPlot, color=:condition, width = 0.8)
draw!(fig[1,1][1,2], p2; axis=(; title = "Control events", xlabel = "Condition", ylabel = "Accuracy"))

# Transition matrix weight
weights_df = DataFrame(results.weight_estimates, :auto)
transform!(weights_df, eachindex => :x)
long_weights_df = stack(weights_df, Not(:x))
transform!(long_weights_df, :variable => ByRow(x -> parse(Int, match(r"[0-9]+", x).match)) => :y)

p3 = data(long_weights_df) *
        mapping(:y, :x, :value) *
        visual(Heatmap)
p3_leg = draw!(fig[1,1][1,3], p3, scales(Color = (; colormap = ColorSchemes.linear_blue_95_50_c20_n256)); axis=(; title = "Transition weights over time", xlabel = "Trial", ylabel = "Matrix weight"))
colorbar!(fig[1,1][1,4], p3_leg; label = "Weight")

# Betas over time
betas_df = DataFrame(results.beta_estimates', :auto)
betas_df.trial = 1:nrow(betas_df)
long_betas_df = stack(betas_df, Not(:trial))
beta_labels = ["Low", "Mid", "High"]
transform!(long_betas_df, :variable => ByRow(x -> beta_labels[parse(Int, match(r"[0-9]+", x).match)]) => :labels)

p4 = data(long_betas_df) *
        mapping(:trial, :value => "β", color = :labels => presorted => "") *
        visual(Lines, label = "Estimated") +
        data((;β = pars.beta_true, labels = beta_labels)) *
        mapping(:β, color = :labels => presorted => "") *
        visual(HLines, linestyle=:dash, label = "Ground\ntruth")
draw!(fig[2,1][1,1], p4, scales(Color = (; palette = :Reds_3)); axis=(; title = "Betas over time", xlabel = "Trial", ylabel = "Estimated β"))

# Effort functions
effort_df = crossjoin(
  DataFrame(effort = 0:30),
  DataFrame(estimated_beta = results.beta_estimates[:, end], true_beta = pars.beta_true, label = beta_labels)
)
transform!(effort_df, [:effort, :estimated_beta] => ByRow((x, y) -> 1 / (1 + exp(-2 * (x - y)))) => :p_control_estimated)
transform!(effort_df, [:effort, :true_beta] => ByRow((x, y) -> 1 / (1 + exp(-2 * (x - y)))) => :p_control_true)

p5 = data(effort_df) *
        mapping(:effort, :p_control_true, color = :label => presorted => "") *
        visual(Lines, linestyle = :dash, label = "Ground\ntruth") +
    data(effort_df) *
        mapping(:effort, :p_control_estimated, color = :label => presorted => "") *
        visual(Lines, linestyle = :solid, label = "Estimated")
p5_leg = draw!(fig[2,1][1,2], p5, scales(Color = (; palette = :Reds_3)); axis=(; title = "Effort functions", xlabel = "Effort", ylabel = "P(Control)"))
legend!(fig[2,1][1,3], p5_leg; tellheight=false, tellwidth=false, padding=(0,0,0,0), gridshalign=:left, patchsize=(10, 10), rowgap = 0)

Box(fig[2,1][1,4]; visible = false)

fig