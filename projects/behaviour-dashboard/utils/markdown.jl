# Generate markdown dashboard
using PrettyTables

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
    markdown_file = joinpath(result_dir, "README.md")
    open(markdown_file, "w") do file
        write(file, markdown_content)
    end
    
    println("✅ Dashboard markdown file generated: $markdown_file")
    println("📊 Included $(length(figure_registry)) figures")
    
    return markdown_file
end

# Helper to append a wide table to README.md
function append_wide_table_to_readme(df::AbstractDataFrame; result_dir::String, title::String)
    md_file = joinpath(result_dir, "README.md")
    open(md_file, "a") do io
        println(io, "\n\n### $(title)\n")
        println(io, "<details><summary>Click to expand</summary>\n")
        println(io, "```text")        
        pretty_table(io, df)
        println(io, "```")
        println(io, "\n</details>")
    end
end