# Preprocess data from REDCap, dividing into tasks and preparing various variables

remove_testing!(data::DataFrame) = filter!(x -> (!occursin(r"haoyang|yaniv|tore|demo|simulate|debug", x.prolific_pid)) && (length(x.prolific_pid) > 10), data)

