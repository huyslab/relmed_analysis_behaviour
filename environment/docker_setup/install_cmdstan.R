# Install cmdstanr
options(Ncpus = 12)
install.packages("remotes")
remotes::install_version("cmdstanr", version="0.9.0", 
                         repos = c("https://stan-dev.r-universe.dev"))
library(cmdstanr)

dir.create(file.path("/home/jovyan/", ".cmdstanr"), recursive = TRUE)
install_cmdstan(version = "2.37.0", 
    cores = 11, 
    dir = file.path("/home/jovyan/", ".cmdstanr"),
    cpp_options = list("CXX" = "clang++"))
cmdstan_path()