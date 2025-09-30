# Plotting configurations for behaviour dashboard

# Plotting theme
th = Theme(
    font = "Helvetica",
    fontsize = 16,
    Axis = (
        xgridvisible = false,
        ygridvisible = false,
        rightspinevisible = false,
        topspinevisible = false,
        xticklabelsize = 14,
        yticklabelsize = 14,
        spinewidth = 1.5,
        xtickwidth = 1.5,
        ytickwidth = 1.5
    )
)

set_theme!(th)

# Plotting configurations
plot_config = Dict(
    # Line widths
    :thin_linewidth => 1,            # Thin individual lines
    :thick_linewidth => 3,           # Thick group lines
    
    # Marker sizes
    :small_markersize => 4,          # Small individual markers
    :medium_markersize => 10,        # Medium markers
    :large_markersize => 20,         # Large group markers
    
    # Alpha/transparency values
    :individual_alpha => 0.7,        # Individual trajectory transparency
    :band_alpha => 0.5,             # Confidence band transparency
    :thin_alpha => 0.3,             # Thin line transparency
    :scatter_alpha => 0.5,          # Scatter plot transparency
    
    # Stroke widths  
    :stroke_width => 0.5,           # Marker stroke width
    
)

