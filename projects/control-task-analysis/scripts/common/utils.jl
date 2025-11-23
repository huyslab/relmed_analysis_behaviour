using RCall

"""
    gg_show_save(plot_obj; width=6, height=4, dpi=150)

Saves R plot to PNG and displays it in VSCode using standard Julia I/O.
Requires NO extra Julia packages (no ImageMagick/FileIO).
"""
function gg_show_save(plot_obj::RObject, filename::String; width=6, height=4, dpi=300, scale=1)
    # 1. Save the real PDF file (Persistent)
    # PDF support is built-in to R, so this works without svglite
    R"ggsave($filename, plot=$plot_obj, width=$width, height=$height, device='pdf', scale=$scale)"
    println("✅ PDF saved to: ", abspath(filename))

    # 2. Define temp file
    temp_file = tempname() * ".png"
    
    # 3. Save PNG using R's built-in device
    # 'type="cairo"' is safer for Docker if you have libcairo installed, 
    # otherwise remove it.
    R"ggsave($temp_file, plot=$plot_obj, width=$width, height=$height, dpi=$dpi, device='png')"
    
    # 3. Read raw bytes from disk
    # We don't decode the image; we just grab the binary data
    image_data = read(temp_file)
    
    # 4. Send raw bytes to VSCode with the correct label (MIME type)
    display("image/png", image_data)
    
    # 5. Cleanup
    rm(temp_file)
end