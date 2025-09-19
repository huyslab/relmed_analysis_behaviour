# Script to generate all figures and combine into markdown file

# Setup
begin
    cd("/home/jovyan")
    import Pkg
    # activate the shared project environment
    Pkg.activate("$(pwd())/environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
    using DataFrames, CairoMakie, Dates

    # Include data scripts
    include("$(pwd())/core/experiment-registry.jl")
    include("$(pwd())/core/preprocess_data.jl")

    script_dir = dirname(@__FILE__)

    # Load configurations and theme
    include(joinpath(script_dir, "config.jl"))

    # Include task-specific scripts
    include(joinpath(script_dir, "generate-figures-card-choosing.jl"))
    include(joinpath(script_dir, "generate-figures-reversal.jl"))

    # Create output directory if it doesn't exist
    result_dir = joinpath(script_dir, "results")
    isdir(result_dir) || mkpath(result_dir)
end

# Figure registry for markdown generation
figure_registry = Vector{NamedTuple{(:filename, :title), Tuple{String, String}}}()

# Save figure function
function save_fig(filename::String, f::Figure)
    save(joinpath(result_dir, filename * ".svg"), f)
    return f
end

# Register figure function
"""
    register_figure(filename::String, title::String)

Registers a figure for inclusion in the markdown dashboard.

- `filename`: The base filename for the figure (without extension). The function expects the filename without the `.svg` extension; the extension is added automatically when saving and referencing the figure.
- `title`: The display title for the figure, which will be used as the section heading and image alt text in the generated markdown dashboard.
"""
function register_figure(filename::String, title::String)
    push!(figure_registry, (filename = filename, title = title))
end

# Load and preprocess data
begin 
    (; PILT, PILT_test, WM, WM_test, reversal, delay_discounting, max_press) = preprocess_project(TRIAL1)
end

# Generate PILT learning curve by session
let PILT_main_sessions = filter(x -> x.session != "screening", PILT)
    
    f = Figure(size = (800, 600))
    plot_learning_curves_by_factor!(f, PILT_main_sessions; factor = :session)

    filename = "PILT_learning_curves_by_session"
    save_fig(filename, f)
    register_figure(filename, "PILT Learning Curves by Session")
end

# Generate WM learning curve by session
let WM_main_sessions = filter(x -> x.session != "screening", WM) |> prepare_WM_data;

    f1 = Figure(size = (800, 600))
    plot_learning_curves_by_factor!(
        f1,
        WM_main_sessions;
        factor = :session,
        xcol = :appearance,
        early_stopping_at = nothing)

    filename1 = "WM_learning_curves_by_session"
    save_fig(filename1, f1)
    register_figure(filename1, "Working Memory Learning Curves by Session")

    f2 = Figure(size = (800, 600))
    plot_learning_curve_by_delay_bins!(f2, WM_main_sessions; facet = :session, variability = :individuals)
    filename2 = "WM_learning_curves_by_delay_bins_and_session_individuals"
    save_fig(filename2, f2)
    register_figure(filename2, "Working Memory Learning Curves by Delay Bins and Session (Individual Participants)")

    f3 = Figure(size = (800, 600))
    plot_learning_curve_by_delay_bins!(f3, WM_main_sessions; facet = :session)
    filename3 = "WM_learning_curves_by_delay_bins_and_session_group"
    save_fig(filename3, f3)
    register_figure(filename3, "Working Memory Learning Curves by Delay Bins and Session (Group Average)")

end

<<<<<<< HEAD
# Generate reversal accuracy curve
let preproc_df = preprocess_reversal_data(reversal)

    f = Figure(size = (800, 600))

    plot_reversal_accuracy_curve!(f, preproc_df)
end
=======
# Generate markdown dashboard
"""
    Generates a markdown dashboard file (`dashboard.md`) containing all registered figures.

    Side effects:
        - Writes the markdown file to the path `dashboard.md` in the results directory (`result_dir`).
    Return value:
        - Returns nothing.
    Dependency:
        - The order of figures in the dashboard matches the order of entries in `figure_registry`.
"""
function generate_markdown_dashboard()
    # Create the markdown content
    markdown_content = """
# Behaviour Analysis Dashboard

Generated on: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))

This dashboard contains all the generated figures from the behaviour analysis.

"""

    # Add each figure to the markdown
    for (i, fig) in enumerate(figure_registry)
        markdown_content *= """
## $(i). $(fig.title)

![$(fig.title)]($(fig.filename).svg)

"""
    end

    # Add summary information
    markdown_content *= """

---

**Summary**: Generated $(length(figure_registry)) figures from the behaviour analysis pipeline.

**Figure files**: All figures are saved as SVG files in the `results/` directory.
"""

    # Write to markdown file
    markdown_file = joinpath(result_dir, "dashboard.md")
    open(markdown_file, "w") do file
        write(file, markdown_content)
    end
    
    println("✅ Dashboard markdown file generated: $markdown_file")
    println("📊 Included $(length(figure_registry)) figures")
    
    return markdown_file
end

# Generate the dashboard
generate_markdown_dashboard()
>>>>>>> figure_dashboard
