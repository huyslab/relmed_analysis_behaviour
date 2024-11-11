### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 05c22183-d42d-4d27-8b21-33f6842e0291
begin
  cd("/home/jovyan/")
  import Pkg
  # activate the shared project environment
  Pkg.activate("relmed_environment")
  # instantiate, i.e. make sure that all packages are downloaded
  Pkg.instantiate()
  using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, Dates, AlgebraOfGraphics, CairoMakie, Tidier, MixedModels, Printf
  include("fetch_preprocess_data.jl")
  set_theme!(theme_minimal()) # set the theme for the plots
end

# ╔═╡ 33e63f0d-8b59-46a5-af57-b45eb4b8b531
md"""
## Load data from REDCap or from local files
"""

# ╔═╡ be928c7e-363b-45cd-971b-4fad90335b06
md"""
### From REDCap
"""

# ╔═╡ 230df26d-e547-42d1-bb71-1bfa7a1dd25c
function load_pilot3_data()
  datafile = "data/pilot3.1.jld2"

  # Load data or download from REDCap
  if !isfile(datafile)
    jspsych_json, records = get_REDCap_data("pilot3.1"; file_field="file_data")

    jspsych_data = REDCap_data_to_df(jspsych_json, records)

    remove_testing!(jspsych_data)

    JLD2.@save datafile jspsych_data
  else
    JLD2.@load datafile jspsych_data
  end

  ### Completion by vigour bonus
  complete_sub = @chain jspsych_data begin
    @group_by(prolific_pid)
    @filter(!ismissing(vigour_bonus))
    @ungroup()
  end
  complete_jspsych_data = semijoin(jspsych_data, complete_sub, on=:prolific_pid)

  ### Vigour task here
  vigour_data = prepare_vigour_data(complete_jspsych_data) |>
                x -> exclude_vigour_trials(x, 36)

  ### Vigour test data
  vigour_test_data = prepare_post_vigour_test_data(complete_jspsych_data)

  return vigour_data, vigour_test_data, complete_jspsych_data
end

# ╔═╡ 485584ee-2efc-4ed1-b58d-2a73ec9c8ae1
raw_vigour_data, raw_vigour_test_data, jspsych_data = load_pilot3_data();

# ╔═╡ 7eafecc7-81c4-4702-b0cf-6f591cacb6cd
begin
  transform!(raw_vigour_data, :version => (x -> replace(x, missing => "3.1")) => :version)
  transform!(raw_vigour_test_data, :version => (x -> replace(x, missing => "3.1")) => :version)
end

# ╔═╡ e8568096-8344-43cf-94fa-22219b3a7c7b
begin
  n_miss_df = combine(groupby(raw_vigour_data, [:prolific_pid, :version]), :trial_presses => (x -> sum(x .== 0)) => :n_miss)
  many_miss = filter(x -> x.n_miss > 9, n_miss_df)
end

# ╔═╡ 45d95aea-7751-4f37-9558-de4c55a0881e
begin
  vigour_data = antijoin(raw_vigour_data, many_miss; on=:prolific_pid) |>
                x -> @filter(x, version === "3.1")
  vigour_test_data = antijoin(raw_vigour_test_data, many_miss; on=:prolific_pid) |>
                     x -> @filter(x, version === "3.1")
end

# ╔═╡ 5ae2870a-408f-4aac-8f12-5af081a250d8
@count(raw_vigour_data, prolific_pid, exp_start_time, version) |>
x -> @count(x, version)

# ╔═╡ bda488ea-faa3-4308-ac89-40b8481363bf
filter(x -> x.trial_presses .== 0, vigour_data)

# ╔═╡ ac1399c4-6a2f-446f-8687-059fa5da9473
md"""
## Vigour trial analyses
"""

# ╔═╡ 9cc0560b-5e3b-45bc-9ba8-605f8111840b
md"""
### Response time
"""

# ╔═╡ 2391eb75-db59-4156-99ae-abb8f1508037
begin
  let
    # Group and calculate mean presses for each participant
    grouped_data = vigour_data |>
                   x -> transform(x, :response_times => ByRow(safe_mean) => :avg_rt) |>
                        x -> groupby(x, [:prolific_pid, :reward_per_press]) |>
                             x -> combine(x, :avg_rt => median => :avg_rt)
    sort!(grouped_data, [:prolific_pid, :reward_per_press])

    # Calculate the average across all participants
    avg_data = combine(groupby(grouped_data, :reward_per_press), :avg_rt => (x -> median(skipmissing(x))) => :avg_rt)

    # Create the plot for individual participants
    individual_plot = data(grouped_data) *
                      mapping(
                        :reward_per_press => "Reward/press",
                        :avg_rt => "Response times (ms)",
                        group=:prolific_pid
                      ) *
                      visual(Lines, color=(:gray, 0.3), linewidth=1)

    # Create the plot for the average line
    average_plot = data(avg_data) *
                   mapping(
                     :reward_per_press => "Reward/press",
                     :avg_rt => "Response times (ms)"
                   ) *
                   visual(ScatterLines, color=:red, linewidth=2)

    # Combine the plots
    final_plot = individual_plot + average_plot
    fig = Figure()

    # Draw the individual lines plot
    draw!(fig[1, 1], individual_plot)

    # Draw the average line plot
    draw!(fig[1, 2], average_plot)

    # Add a overall title
    Label(fig[0, :], "Response times vs Reward/press")

    fig
  end
end

# ╔═╡ b9f4babc-8da0-40f2-b0d2-95f915989faf
md"""
### Number of key presses
"""

# ╔═╡ c4e51fb5-2833-4d48-b840-1f689afc9329
begin
  function avg_presses_w_fn(vigour_data::DataFrame, var::Vector{Symbol})
    # Group and calculate mean presses for each participant
    grouped_data = groupby(vigour_data, Cols(:prolific_pid, var)) |>
                   x -> combine(x, [:trial_presses, :trial_duration] => ((x, y) -> mean(x .* 1000 ./ y)) => :n_presses) |>
                        x -> sort(x, Cols(:prolific_pid, var))

    # Calculate the average across all participants
    avg_w_data = @chain grouped_data begin
      @group_by(prolific_pid)
      @mutate(sub_mean = mean(n_presses))
      @ungroup
      @mutate(grand_mean = mean(n_presses))
      @mutate(n_presses_w = n_presses - sub_mean + grand_mean)
      @group_by(!!var)
      @summarize(
        n = n(),
        avg_n_presses = mean(n_presses),
        se_n_presses = std(n_presses_w) / sqrt(length(prolific_pid)))
      @ungroup
    end
    return grouped_data, avg_w_data
  end

  function plot_presses_vs_var(vigour_data::DataFrame, var::Vector{Symbol}, combine::Bool=false, xlab::String=missing)
    grouped_data, avg_w_data = avg_presses_w_fn(vigour_data, var)
    # Create the plot for individual participants
    individual_plot = data(grouped_data) *
                      mapping(
                        var,
                        :n_presses,
                        group=:prolific_pid
                      ) *
                      visual(Lines, color=(:gray, 0.3), linewidth=1)

    # Create the plot for the average line
    average_plot = data(avg_w_data) *
                   mapping(
                     var,
                     :avg_n_presses
                   ) * (
                     visual(Errorbars, color=:red3, whiskerwidth=4) *
                     mapping(:se_n_presses) +
                     visual(ScatterLines, color=:red, linewidth=2))

    # Combine the plots
    fig = Figure()
    # Set up the axis
    xlab_text = ifelse(ismissing(xlab), uppercasefirst(join([string(v) for v in var], ", ")), xlab)
    if (combine)
      axis = (
        xlabel=xlab_text,
        ylabel="Press/sec",
      )
      final_plot = individual_plot + average_plot
      fig = draw(final_plot; axis)
    else
      # Draw the plot
      fig_patch = fig[1, 1] = GridLayout()
      ax_left = Axis(fig_patch[1, 1], ylabel="Press/sec")
      ax_right = Axis(fig_patch[1, 2])
      Label(fig_patch[2, :], xlab_text)
      draw!(ax_left, individual_plot)
      draw!(ax_right, average_plot)
      rowgap!(fig_patch, 5)
    end
    return fig
  end
end

# ╔═╡ 00f78d00-ca54-408a-a4a6-ad755566052a
plot_presses_vs_var(vigour_data, [:reward_per_press], false, "Reward/press")

# ╔═╡ 89d68509-bcf0-44aa-9ddc-a685131ce146
plot_presses_vs_var(vigour_data, [:reward_per_press], true, "Reward/press")

# ╔═╡ ef4c0f64-2f16-4700-8fc4-703a1b858c37
plot_presses_vs_var(vigour_data, [:ratio], false, "Fixed ratio")

# ╔═╡ eab61744-af48-4789-ba2f-a92d73527962
plot_presses_vs_var(vigour_data, [:ratio], true, "Fixed ratio")

# ╔═╡ c4e7d7ec-8da8-4a1f-8da3-0f2d29be6db7
plot_presses_vs_var(vigour_data, [:magnitude], false, "Reward magnitude")

# ╔═╡ 09a6f213-f70a-42c2-8eb7-8689d40d140b
plot_presses_vs_var(vigour_data, [:magnitude], true, "Reward magnitude")

# ╔═╡ 14d72296-7392-4375-9748-a8ee84d34701
CSV.write("data/pilot3.1_vigour_data.csv", vigour_data)

# ╔═╡ 1066a576-c7bf-42cc-8097-607c663dcdac
begin
  let
    # Group and calculate mean presses for each participant
    vigour_data.half = vigour_data.trial_number .> (maximum(vigour_data.trial_number) / 2)

    # grouped_data = combine(groupby(vigour_data, [:prolific_pid, :magnitude, :ratio]), :trial_presses => mean => :n_presses)
    # sort!(grouped_data, [:prolific_pid, :magnitude, :ratio])

    # # Calculate the average across all participants
    # avg_w_data = @chain grouped_data begin
    # 	@group_by(prolific_pid)
    # 	@mutate(sub_mean = mean(n_presses))
    # 	@ungroup
    # 	@mutate(grand_mean = mean(n_presses))
    # 	@mutate(n_presses_w = n_presses - sub_mean + grand_mean)
    # 	@group_by(magnitude, ratio)
    # 	@summarize(
    # 		n = n(),
    # 		avg_n_presses = mean(n_presses),
    # 		se_n_presses = std(n_presses_w)/sqrt(length(prolific_pid)))
    # 	@ungroup
    # end

    grouped_data, avg_w_data = avg_presses_w_fn(vigour_data, [:magnitude, :ratio])

    # Create the plot for the average line
    average_plot = data(avg_w_data) *
                   mapping(
                     :ratio,
                     :avg_n_presses,
                     color=:magnitude => nonnumeric => "Reward\nMagnitude",
                   ) * (
                     visual(ScatterLines, linewidth=2) +
                     visual(Errorbars, whiskerwidth=6) *
                     mapping(:se_n_presses))

    # Set up the axis
    axis = (
      xlabel="Fix Ratio",
      ylabel="Number of Presses",
      xticks=[1, 4, 8, 12, 16]
    )

    # Draw the plot

    fig = Figure(
      size=(12.2, 7.6) .* 144 ./ 2.54, # 72 points per inch, then cm
    )

    # Plot plot plot
    f = draw!(fig[1, 1], average_plot, scales(Color=(; palette=from_continuous(:viridis))); axis)
    legend!(fig[1, 2], f)
    fig
  end
end

# ╔═╡ 2b77a1c7-7ed4-479f-a377-e96b61676e38
# ╠═╡ show_logs = false
begin
  presses_mixed = fit(LinearMixedModel,
    @formula(trial_presses ~ magnitude * ratio +
                             (magnitude * ratio | prolific_pid)),
    vigour_data)
end

# ╔═╡ 128e1be1-191d-4095-bfa9-cd538b913181
md"""
### Response-missing trials
"""

# ╔═╡ 302e4f7a-fd09-48c3-89ac-a80a63841ee7
begin
  let
    # Create the plot
    plot = data(n_miss_df) *
           mapping(:n_miss,) *
           histogram(bins=20)

    # Set up the axis
    axis = (
      title="Some participants didn't respond in many trials",
      xlabel="# No-response trials",
      ylabel="Count (# participants)",
      yticks=0:10:50
    )

    # Draw the plot
    draw(plot; axis)
  end
end

# ╔═╡ b0c1d8fb-8f95-4a40-b760-6584022867ce
begin
  missed_trial_df = @chain raw_vigour_data begin
    @filter(trial_presses == 0)
    @group_by(prolific_pid, reward_per_press)
    @summarize(n_miss = n())
    @ungroup()
    @mutate(short_id = last.(prolific_pid, 5))
    @arrange(prolific_pid, reward_per_press)
  end
  data(missed_trial_df) *
  mapping(:reward_per_press, :n_miss, layout=:short_id) *
  visual(ScatterLines) |>
  x -> draw(x)
end

# ╔═╡ 9a8d96d7-13bd-4f70-9344-6569dfa39ab6
md"""
## Reliability analysis
"""

# ╔═╡ 16154073-51c3-4a37-b40f-8c4f6044702c
md"""
#### Split-half reliability: First-second
"""

# ╔═╡ 08fe8e09-51e5-4d54-a0c2-7c6ac8269a00
@count(vigour_data, prolific_pid)

# ╔═╡ b5c429ee-9f31-4815-98b3-2536952b6881
begin
  first_second_df = @chain vigour_data begin
    @mutate(press_per_sec = trial_presses .* 1000 ./ trial_duration)
    @group_by(prolific_pid, half)
    @summarize(n_presses = mean(press_per_sec))
    @ungroup
    @pivot_wider(names_from = half, values_from = n_presses)
    @rename(first = `false`, second = `true`)
  end
  ρ_first_second = cor(first_second_df.first, first_second_df.second)
  ρ_first_second = (2 * ρ_first_second) / (1 + ρ_first_second)
  first_second_plot =
    data(first_second_df) *
    mapping(:first, :second) *
    visual(Scatter) +
    data(DataFrame(intercept=0, slope=1)) *
    mapping(:intercept, :slope) *
    visual(ABLines, linestyle=:dash, color=:gray70)
  draw(
    first_second_plot,
    figure=(;
      title="Split-half reliability by first-second: " * @sprintf "%.2f" ρ_first_second
    )
  )
end

# ╔═╡ 00d6678d-4e26-4fe6-9937-b1055f78cad5
begin
  let
    df = @chain vigour_data begin
      @mutate(press_per_sec = trial_presses .* 1000 ./ trial_duration)
      @group_by(prolific_pid, half, magnitude, ratio)
      @summarize(n_presses = mean(press_per_sec))
      @ungroup
      @pivot_wider(names_from = half, values_from = n_presses)
      @rename(first = `false`, second = `true`)
      @mutate(magnitude = nonnumeric(magnitude),
        ratio = nonnumeric(ratio))
    end
    p = data(df) *
        mapping(:first, :second, col=:magnitude, row=:ratio) *
        visual(Scatter) +
        data(DataFrame(intercept=0, slope=1)) *
        mapping(:intercept, :slope) *
        visual(ABLines, linestyle=:dash, color=:gray70)
    draw(p)
  end
end

# ╔═╡ 50e8263e-a593-47d3-abc4-aceeeb68ba58
ρ_by_conds = @chain vigour_data begin
  @mutate(press_per_sec = trial_presses .* 1000 ./ trial_duration)
  @group_by(prolific_pid, half, magnitude, ratio)
  @summarize(n_presses = mean(skipmissing(press_per_sec)))
  @ungroup
  @pivot_wider(names_from = half, values_from = n_presses)
  @rename(first = `false`, second = `true`)
  @group_by(magnitude, ratio)
  @summarize(ρ = ~cor(first, second))
  @mutate(ρ = 2 * ρ / (1 + ρ))
  @ungroup
end

# ╔═╡ a2adae6f-d651-4f0c-89f3-38cf357c943e
md"""
### Split-half reliability: Even-odd
"""

# ╔═╡ 1adc75da-c6c4-4b11-81ff-227851a18076
begin
  even_odd_df = @chain vigour_data begin
    @mutate(press_per_sec = trial_presses .* 1000 ./ trial_duration)
    @mutate(even = trial_number % 2 === 0)
    @group_by(prolific_pid, even)
    @summarize(n_presses = mean(press_per_sec))
    @ungroup
    @pivot_wider(names_from = even, values_from = n_presses)
    @rename(odd = `false`, even = `true`)
  end
  ρ_even_odd = cor(even_odd_df.even, even_odd_df.odd)
  ρ_even_odd = (2 * ρ_even_odd) / (1 + ρ_even_odd)
  even_odd_plot =
    data(even_odd_df) *
    mapping(:even, :odd) *
    visual(Scatter) +
    data(DataFrame(intercept=0, slope=1)) *
    mapping(:intercept, :slope) *
    visual(ABLines, linestyle=:dash, color=:gray70)
  draw(
    even_odd_plot,
    figure=(;
      title="Split-half reliability by even-odd: " * @sprintf "%.2f" ρ_even_odd
    )
  )
end

# ╔═╡ 387e33da-31fd-4214-b819-bb4f2e43ab23
md"""
### Reliability of reward/press difference
"""

# ╔═╡ 46fdebe0-7a73-4d2a-8a31-1e07f1136081
begin
  rpp_dff_df = @chain vigour_data begin
    @mutate(press_per_sec = trial_presses .* 1000 ./ trial_duration)
    @arrange(prolific_pid, reward_per_press)
    @group_by(prolific_pid)
    @filter(reward_per_press != median(reward_per_press))
    @mutate(low_rpp = reward_per_press < median(reward_per_press),
      trial_number = row_number())
    @mutate(even = trial_number % 2 === 0)
    @ungroup
    @group_by(prolific_pid, even, low_rpp)
    @summarize(n_presses = mean(press_per_sec))
    @ungroup
    @pivot_wider(names_from = low_rpp, values_from = n_presses)
    @rename(high_rpp = `false`, low_rpp = `true`)
    @mutate(low_to_high_diff = low_rpp - high_rpp)
    @select(-ends_with("_rpp"))
    @pivot_wider(names_from = even, values_from = low_to_high_diff)
    @rename(odd = `false`, even = `true`)
  end
  ρ_rpp_diff = cor(rpp_dff_df.even, rpp_dff_df.odd)
  ρ_rpp_diff = (2 * ρ_rpp_diff) / (1 + ρ_rpp_diff)
  rpp_diff_plot =
    data(rpp_dff_df) *
    mapping(:even, :odd) *
    visual(Scatter) +
    data(DataFrame(intercept=0, slope=1)) *
    mapping(:intercept, :slope) *
    visual(ABLines, linestyle=:dash, color=:gray70)
  draw(
    rpp_diff_plot,
    figure=(;
      title="Split-half reliability of RPP difference by even-odd: " * @sprintf "%.2f" ρ_rpp_diff
    )
  )
end

# ╔═╡ a65b26bf-be9d-4a98-9130-2c4f870fb3b4
rpp_dff_df

# ╔═╡ 9a6438cc-d758-42a6-8c0b-36cb7dddacf9
md"""
## Post-vigour test
"""

# ╔═╡ 096ffb0d-cddb-445a-8887-d27dcd16ea47
begin
  diff_vigour_test_data = @chain vigour_test_data begin
    @mutate(
      diff_mag = left_magnitude - right_magnitude,
      diff_fr_rel = log2(left_ratio / right_ratio),
      diff_fr_abs = left_ratio - right_ratio,
      diff_rpp = (left_magnitude / left_ratio) - (right_magnitude / right_ratio),
      chose_left = Int(response === "ArrowLeft")
    )
    @mutate(across(starts_with("diff_"), zscore))
  end
end

# ╔═╡ 223d7ed5-8621-4f6a-905e-cfbd27849686
CSV.write("data/pilot3.1_vigour_test_data.csv", diff_vigour_test_data)

# ╔═╡ b3f76fad-a1c6-4fc0-afbf-7ce9736986e8
md"""
### Separate models for each effect

Since only RPP was balanced, perhaps it would be best to focus on and only on its effect.

#### RPP
"""

# ╔═╡ 7f3c680a-970b-483d-b3df-ed22b6dd8d81
# ╠═╡ show_logs = false
begin
  test_mixed_rpp = fit(GeneralizedLinearMixedModel,
    @formula(chose_left ~ diff_rpp +
                          (diff_rpp | prolific_pid)),
    diff_vigour_test_data, Binomial())
end

# ╔═╡ 341083df-98ee-4758-85a6-8a070b662b86
diff_vigour_test_data

# ╔═╡ 2f8f197a-77c1-48ac-b18a-d2e3a9006bfb
begin
  test_acc_df = @chain diff_vigour_test_data begin
    @group_by(prolific_pid)
    @select(diff_rpp, chose_left)
    @mutate(chose_left = chose_left * 2 - 1, truth = sign(diff_rpp))
    @mutate(acc = chose_left == truth)
    @summarize(acc = mean(acc))
  end
  @printf "%.2f" mean(test_acc_df.acc)
  data(test_acc_df) *
  mapping(:acc) *
  visual(Hist) |>
  draw()
end

# ╔═╡ 96bcba8a-e8a1-49c4-b60e-65aa322ca321
@chain diff_vigour_test_data begin
  @mutate(chose_left = chose_left * 2 - 1, truth = sign(diff_rpp))
  @mutate(correct = chose_left == truth)
  @mutate(easy = (left_ratio <= right_ratio && left_magnitude >= right_magnitude) || (left_ratio >= right_ratio && left_magnitude <= right_magnitude))
  @group_by(prolific_pid, easy)
  @summarize(acc = mean(correct))
  @ungroup
  @mutate(easy = ifelse(easy, "Easy pairs", "Hard pairs"))
  data(_) * mapping(:acc, col=:easy) * visual(Hist)
  draw(_, axis=(; xlabel="Accuracy"))
end

# ╔═╡ 6b8abbbb-bb91-49ae-8df0-aeebddb55a7f
@chain diff_vigour_test_data begin
  @mutate(chose_left = chose_left * 2 - 1, truth = sign(diff_rpp))
  @mutate(correct = chose_left == truth)
  @mutate(easy = (left_ratio <= right_ratio && left_magnitude >= right_magnitude) || (left_ratio >= right_ratio && left_magnitude <= right_magnitude))
  @group_by(prolific_pid, easy)
  @summarize(acc = mean(correct))
  @ungroup
  @mutate(easy = ifelse(easy, "Easy pairs", "Hard pairs"))
  @group_by(easy)
  @summarize(acc = mean(acc))
end

# ╔═╡ 779fac61-ef41-47ab-b461-ddb7fb930011
@chain diff_vigour_test_data begin
  @mutate(chose_left = chose_left * 2 - 1, truth = sign(diff_rpp))
  @mutate(correct = chose_left == truth)
  @mutate(easy = (left_ratio <= right_ratio && left_magnitude >= right_magnitude) || (left_ratio >= right_ratio && left_magnitude <= right_magnitude))
  @summarize(prop_easy = mean(easy))
end

# ╔═╡ c686ea0b-7654-408f-8d9c-cabbef803d70
@chain diff_vigour_test_data begin
  @mutate(chose_left = chose_left * 2 - 1, truth = sign(diff_rpp))
  @mutate(correct = chose_left == truth)
  @mutate(easy = (left_ratio <= right_ratio && left_magnitude >= right_magnitude) || (left_ratio >= right_ratio && left_magnitude <= right_magnitude))
  @filter(.!easy)
  data(_) *
  mapping(:diff_rpp) *
  visual(Hist)
  draw(_)
end

# ╔═╡ 24545992-c191-43c3-8741-22175650f580
begin
  diff_rpp_range = LinRange(minimum(diff_vigour_test_data.diff_rpp), maximum(diff_vigour_test_data.diff_rpp), 100)
  pred_effect_rpp = DataFrame(
    prolific_pid="New",
    chose_left=0.5,
    diff_rpp=diff_rpp_range
  )
  pred_effect_rpp.pred = predict(test_mixed_rpp, pred_effect_rpp; new_re_levels=:population, type=:response)
  rpp_pred_plot =
    data(diff_vigour_test_data) *
    mapping(:diff_rpp, :chose_left) *
    visual(Scatter, alpha=0.01) +
    data(pred_effect_rpp) *
    mapping(:diff_rpp, :pred) *
    visual(Lines, color=:royalblue)
  draw(rpp_pred_plot)
end

# ╔═╡ 54178f5c-6950-4aca-97e4-aeb56c176c74
md"""
#### Reward magnitude
"""

# ╔═╡ 74882f58-32bc-4eaa-b9e6-087390f8fe65
begin
  test_mixed_mag = fit(GeneralizedLinearMixedModel,
    @formula(chose_left ~ diff_mag +
                          (diff_mag | prolific_pid)),
    diff_vigour_test_data, Binomial())
end

# ╔═╡ e36d7568-2951-48e8-876b-a86b1a09c23a
begin
  diff_mag_range = LinRange(minimum(diff_vigour_test_data.diff_mag), maximum(diff_vigour_test_data.diff_mag), 100)
  pred_effect_mag = DataFrame(
    prolific_pid="New",
    chose_left=0.5,
    diff_mag=diff_mag_range
  )
  pred_effect_mag.pred = predict(test_mixed_mag, pred_effect_mag; new_re_levels=:population, type=:response)
  mag_pred_plot =
    data(diff_vigour_test_data) *
    mapping(:diff_mag, :chose_left) *
    visual(Scatter, alpha=0.01) +
    data(pred_effect_mag) *
    mapping(:diff_mag, :pred) *
    visual(Lines, color=:royalblue)
  draw(mag_pred_plot)
end

# ╔═╡ 664e3e14-c757-4cd1-9df4-3f530d7a33a7
md"""
#### Fixed ratio
"""

# ╔═╡ b58cd5f7-cd0e-4a73-adc5-cac8780bb0aa
# ╠═╡ show_logs = false
begin
  test_mixed_fr_abs = fit(GeneralizedLinearMixedModel,
    @formula(chose_left ~ diff_fr_abs_zscore +
                          (diff_fr_abs_zscore | prolific_pid)),
    diff_vigour_test_data, Binomial())
end

# ╔═╡ 8b96b42f-c2ce-408e-81dd-b413a12c6479
begin
  let
    diff_fr_zscore_range = LinRange(minimum(diff_vigour_test_data.diff_fr_abs_zscore), maximum(diff_vigour_test_data.diff_fr_abs_zscore), 100)

    diff_fr_range = std(diff_vigour_test_data.diff_fr_abs) * diff_fr_zscore_range .+ mean(diff_vigour_test_data.diff_fr_abs)

    pred_effect_fr = DataFrame(
      prolific_pid="New",
      chose_left=0.5,
      diff_fr_abs=diff_fr_range,
      diff_fr_abs_zscore=diff_fr_zscore_range
    )

    pred_effect_fr.pred = predict(test_mixed_fr_abs, pred_effect_fr; new_re_levels=:population, type=:response)

    fr_pred_plot =
      data(diff_vigour_test_data) *
      mapping(:diff_fr_abs, :chose_left) *
      visual(Scatter, alpha=0.01) +
      data(pred_effect_fr) *
      mapping(:diff_fr_abs, :pred) *
      visual(Lines, color=:royalblue)
    draw(fr_pred_plot)
  end
end

# ╔═╡ 25dc8353-d6f8-4503-bda3-6b288630b770
# ╠═╡ show_logs = false
begin
  test_mixed_fr_rel = fit(GeneralizedLinearMixedModel,
    @formula(chose_left ~ diff_fr_rel_zscore +
                          (diff_fr_rel_zscore | prolific_pid)),
    diff_vigour_test_data, Binomial())
end

# ╔═╡ eebe4ca4-d3ba-4bce-a8e9-310a53569019
begin
  let
    diff_fr_zscore_range = LinRange(minimum(diff_vigour_test_data.diff_fr_rel_zscore), maximum(diff_vigour_test_data.diff_fr_rel_zscore), 100)

    diff_fr_range = std(diff_vigour_test_data.diff_fr_rel) * diff_fr_zscore_range .+ mean(diff_vigour_test_data.diff_fr_rel)

    pred_effect_fr = DataFrame(
      prolific_pid="New",
      chose_left=0.5,
      diff_fr_rel=diff_fr_range,
      diff_fr_rel_zscore=diff_fr_zscore_range
    )

    pred_effect_fr.pred = predict(test_mixed_fr_rel, pred_effect_fr; new_re_levels=:population, type=:response)

    fr_pred_plot =
      data(diff_vigour_test_data) *
      mapping(:diff_fr_rel, :chose_left) *
      visual(Scatter, alpha=0.01) +
      data(pred_effect_fr) *
      mapping(:diff_fr_rel, :pred) *
      visual(Lines, color=:royalblue)
    draw(fr_pred_plot)
  end
end

# ╔═╡ 6a8d4aac-7c87-476c-8259-116d4dcf24aa
md"""
### Combined models
"""

# ╔═╡ 385ecb85-ea97-45e4-8766-b99969ab12a0
# ╠═╡ show_logs = false
begin
  test_mixed_1 = fit(GeneralizedLinearMixedModel,
    @formula(chose_left ~ diff_mag_zscore + diff_fr_rel_zscore + diff_rpp_zscore +
                          (diff_mag_zscore + diff_fr_rel_zscore + diff_rpp_zscore | prolific_pid)),
    diff_vigour_test_data, Binomial())
end

# ╔═╡ 7fe24aab-5cba-4413-835c-ad6a680a938d
# ╠═╡ show_logs = false
begin
  test_mixed_2 = fit(GeneralizedLinearMixedModel,
    @formula(chose_left ~ diff_mag_zscore + diff_fr_abs_zscore + diff_rpp_zscore +
                          (diff_mag_zscore + diff_fr_abs_zscore + diff_rpp_zscore | prolific_pid)),
    diff_vigour_test_data, Binomial())
end

# ╔═╡ 61bde716-bdfd-4a68-ac47-2c33c58e414c
bic(test_mixed_1)

# ╔═╡ f9325491-615e-4ba3-9607-dfc21fca228b
bic(test_mixed_2)

# ╔═╡ Cell order:
# ╠═05c22183-d42d-4d27-8b21-33f6842e0291
# ╟─33e63f0d-8b59-46a5-af57-b45eb4b8b531
# ╟─be928c7e-363b-45cd-971b-4fad90335b06
# ╠═230df26d-e547-42d1-bb71-1bfa7a1dd25c
# ╠═485584ee-2efc-4ed1-b58d-2a73ec9c8ae1
# ╠═7eafecc7-81c4-4702-b0cf-6f591cacb6cd
# ╠═e8568096-8344-43cf-94fa-22219b3a7c7b
# ╠═45d95aea-7751-4f37-9558-de4c55a0881e
# ╠═5ae2870a-408f-4aac-8f12-5af081a250d8
# ╠═bda488ea-faa3-4308-ac89-40b8481363bf
# ╟─ac1399c4-6a2f-446f-8687-059fa5da9473
# ╟─9cc0560b-5e3b-45bc-9ba8-605f8111840b
# ╠═2391eb75-db59-4156-99ae-abb8f1508037
# ╟─b9f4babc-8da0-40f2-b0d2-95f915989faf
# ╠═c4e51fb5-2833-4d48-b840-1f689afc9329
# ╠═00f78d00-ca54-408a-a4a6-ad755566052a
# ╠═89d68509-bcf0-44aa-9ddc-a685131ce146
# ╠═ef4c0f64-2f16-4700-8fc4-703a1b858c37
# ╠═eab61744-af48-4789-ba2f-a92d73527962
# ╠═c4e7d7ec-8da8-4a1f-8da3-0f2d29be6db7
# ╠═09a6f213-f70a-42c2-8eb7-8689d40d140b
# ╠═14d72296-7392-4375-9748-a8ee84d34701
# ╠═1066a576-c7bf-42cc-8097-607c663dcdac
# ╠═2b77a1c7-7ed4-479f-a377-e96b61676e38
# ╟─128e1be1-191d-4095-bfa9-cd538b913181
# ╠═302e4f7a-fd09-48c3-89ac-a80a63841ee7
# ╠═b0c1d8fb-8f95-4a40-b760-6584022867ce
# ╟─9a8d96d7-13bd-4f70-9344-6569dfa39ab6
# ╟─16154073-51c3-4a37-b40f-8c4f6044702c
# ╠═08fe8e09-51e5-4d54-a0c2-7c6ac8269a00
# ╠═b5c429ee-9f31-4815-98b3-2536952b6881
# ╠═00d6678d-4e26-4fe6-9937-b1055f78cad5
# ╠═50e8263e-a593-47d3-abc4-aceeeb68ba58
# ╟─a2adae6f-d651-4f0c-89f3-38cf357c943e
# ╠═1adc75da-c6c4-4b11-81ff-227851a18076
# ╟─387e33da-31fd-4214-b819-bb4f2e43ab23
# ╠═46fdebe0-7a73-4d2a-8a31-1e07f1136081
# ╠═a65b26bf-be9d-4a98-9130-2c4f870fb3b4
# ╟─9a6438cc-d758-42a6-8c0b-36cb7dddacf9
# ╠═096ffb0d-cddb-445a-8887-d27dcd16ea47
# ╠═223d7ed5-8621-4f6a-905e-cfbd27849686
# ╟─b3f76fad-a1c6-4fc0-afbf-7ce9736986e8
# ╠═7f3c680a-970b-483d-b3df-ed22b6dd8d81
# ╠═341083df-98ee-4758-85a6-8a070b662b86
# ╠═2f8f197a-77c1-48ac-b18a-d2e3a9006bfb
# ╠═96bcba8a-e8a1-49c4-b60e-65aa322ca321
# ╠═6b8abbbb-bb91-49ae-8df0-aeebddb55a7f
# ╠═779fac61-ef41-47ab-b461-ddb7fb930011
# ╠═c686ea0b-7654-408f-8d9c-cabbef803d70
# ╠═24545992-c191-43c3-8741-22175650f580
# ╟─54178f5c-6950-4aca-97e4-aeb56c176c74
# ╠═74882f58-32bc-4eaa-b9e6-087390f8fe65
# ╠═e36d7568-2951-48e8-876b-a86b1a09c23a
# ╟─664e3e14-c757-4cd1-9df4-3f530d7a33a7
# ╠═b58cd5f7-cd0e-4a73-adc5-cac8780bb0aa
# ╠═8b96b42f-c2ce-408e-81dd-b413a12c6479
# ╠═25dc8353-d6f8-4503-bda3-6b288630b770
# ╠═eebe4ca4-d3ba-4bce-a8e9-310a53569019
# ╟─6a8d4aac-7c87-476c-8259-116d4dcf24aa
# ╠═385ecb85-ea97-45e4-8766-b99969ab12a0
# ╠═7fe24aab-5cba-4413-835c-ad6a680a938d
# ╠═61bde716-bdfd-4a68-ac47-2c33c58e414c
# ╠═f9325491-615e-4ba3-9607-dfc21fca228b
