"""
    avg_presses_w_fn(vigour_data::DataFrame, x_var::Vector{Symbol}, y_var::Symbol, grp_var::Union{Symbol,Nothing}=nothing)

Calculate the average number of presses weighted by participant for a given dataset.

# Arguments
- `vigour_data::DataFrame`: The input DataFrame containing the vigour data.
- `x_var::Vector{Symbol}`: A vector of symbols representing the independent variables.
- `y_var::Symbol`: A symbol representing the dependent variable.
- `grp_var::Union{Symbol,Nothing}`: An optional symbol representing the grouping variable. Defaults to `nothing`.

# Returns
- `grouped_data::DataFrame`: A DataFrame grouped by the specified columns with the mean of `y_var` calculated for each group.
- `avg_w_data::DataFrame`: A DataFrame with the average number of presses weighted by participant, including the number of observations (`n`), the average (`avg_y`), and the standard error (`se_y`).

# Details
1. Defines the grouping columns based on the presence of `grp_var`.
2. Groups the data by the specified columns and calculates the mean of `y_var` for each group.
3. Calculates the average across all participants, adjusting for participant-specific means.
4. Returns the grouped data and the weighted average data.
"""
function avg_presses_w_fn(vigour_data::DataFrame, x_var::Vector{Symbol}, y_var::Symbol, grp_var::Union{Symbol,Nothing}=nothing)
    # Define grouping columns
    group_cols = grp_var === nothing ? [:prolific_id, x_var...] : [:prolific_id, grp_var, x_var...]
    # Group and calculate mean presses for each participant
    grouped_data = groupby(vigour_data, Cols(group_cols...)) |>
                   x -> combine(x, y_var => mean => :mean_y) |>
                        x -> sort(x, Cols(group_cols...))
    # Calculate the average across all participants
    avg_w_data = @chain grouped_data begin
        @group_by(prolific_id)
        @mutate(sub_mean = mean(mean_y))
        @ungroup
        @mutate(grand_mean = mean(mean_y))
        @mutate(mean_y_w = mean_y - sub_mean + grand_mean)
        groupby(Cols(grp_var === nothing ? x_var : [grp_var, x_var...]))
        @summarize(
            n = n(),
            avg_y = mean(mean_y),
            se_y = std(mean_y_w) / sqrt(length(prolific_id)))
        @ungroup
    end
    return grouped_data, avg_w_data
end

"""
    plot_presses_vs_var(vigour_data::DataFrame; x_var::Symbol=:reward_per_press, y_var::Symbol=:trial_presses, grp_var::Union{Symbol,Nothing}=nothing, xlab::Union{String,Missing}=missing, ylab::Union{String,Missing}=missing, combine::Bool=false)

Plot the relationship between two variables in the vigour data, optionally grouped by a third variable.

# Arguments
- `vigour_data::DataFrame`: The data containing the vigour measurements.
- `x_var::Symbol`: The variable to be plotted on the x-axis (default: `:reward_per_press`).
- `y_var::Symbol`: The variable to be plotted on the y-axis (default: `:trial_presses`).
- `grp_var::Union{Symbol,Nothing}`: An optional grouping variable (default: `nothing`).
- `xlab::Union{String,Missing}`: An optional label for the x-axis (default: `missing`).
- `ylab::Union{String,Missing}`: An optional label for the y-axis (default: `missing`).
- `combine::Bool`: Whether to combine individual and average plots into one (default: `false`).

# Returns
- `fig`: The resulting plot figure.

# Description
This function generates a plot showing the relationship between `x_var` and `y_var` in the provided `vigour_data`. If `grp_var` is provided, the data will be grouped by this variable. The function creates two plots: one for individual participants and one for the average line. The plots can be combined into a single plot if `combine` is set to `true`.

The x-axis and y-axis labels can be customized using `xlab` and `ylab`. If these are not provided, the function will generate labels based on the variable names.

The function returns the generated plot figure.
"""
function plot_presses_vs_var(vigour_data::DataFrame; x_var::Union{Symbol, Pair{Symbol, typeof(AlgebraOfGraphics.nonnumeric)}}=:reward_per_press, y_var::Symbol=:trial_presses, grp_var::Union{Symbol,Nothing}=nothing, xlab::Union{String,Missing}=missing, ylab::Union{String,Missing}=missing, grplab::Union{String,Missing}=missing, combine::Bool=false)
    plain_x_var = isa(x_var, Pair) ? x_var.first : x_var
    grouped_data, avg_w_data = avg_presses_w_fn(vigour_data, [plain_x_var], y_var, grp_var)

	# Set up the legend title
    grplab_text = ismissing(grplab) ? uppercasefirst(join(split(string(grp_var), r"\P{L}+"), " ")) : grplab
	
    # Define mapping based on whether grp_var is provided
    individual_mapping = grp_var === nothing ?
                         mapping(x_var, :mean_y, group=:prolific_id) :
                         mapping(x_var, :mean_y, color=grp_var => grplab_text, group=:prolific_id)

    average_mapping = grp_var === nothing ?
                      mapping(x_var, :avg_y) :
                      mapping(x_var, :avg_y, color=grp_var => grplab_text)

    # Create the plot for individual participants
    individual_plot = data(grouped_data) *
                      individual_mapping *
                      visual(Lines, alpha=0.15, linewidth=1)

    # Create the plot for the average line
    if grp_var === nothing
        average_plot = data(avg_w_data) *
                       average_mapping * (
                           visual(Errorbars, whiskerwidth=4) *
                           mapping(:se_y) +
                           visual(ScatterLines, linewidth=2)) *
                       visual(color=:dodgerblue)
    else
        average_plot = data(avg_w_data) *
                       average_mapping * (
                           visual(Errorbars, whiskerwidth=4) *
                           mapping(:se_y, color=grp_var => grplab_text) +
                           visual(ScatterLines, linewidth=2))
    end

    # Combine the plots
    fig = Figure(
        size=(12.2, 7.6) .* 144 ./ 2.54, # 144 points per inch, then cm
    )

    # Set up the axis
    xlab_text = ismissing(xlab) ? uppercasefirst(join(split(string(x_var), r"\P{L}+"), " ")) : xlab
    ylab_text = ismissing(ylab) ? uppercasefirst(join(split(string(y_var), r"\P{L}+"), " ")) : ylab

    if combine
        axis = (;
            xlabel=xlab_text,
            ylabel=ylab_text,
        )
        final_plot = individual_plot + average_plot
        fig = draw(final_plot; axis=axis)
    else
        # Draw the plot
        fig_patch = fig[1, 1] = GridLayout()
        ax_left = Axis(fig_patch[1, 1], ylabel=ylab_text)
        ax_right = Axis(fig_patch[1, 2])
        Label(fig_patch[2, :], xlab_text)
        draw!(ax_left, individual_plot)
        f = draw!(ax_right, average_plot)
        legend!(fig_patch[1, 3], f)
        rowgap!(fig_patch, 5)
    end
    return fig
end