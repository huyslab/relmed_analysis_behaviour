# Install cmdstanr
options(Ncpus = 12)
install.packages("remotes")
remotes::install_version("cmdstanr", version="0.9.0", 
                         repos = c("https://stan-dev.r-universe.dev"))
library(cmdstanr)

# Install to /opt instead of home directory
dir.create("/opt/cmdstan", recursive = TRUE)
install_cmdstan(version = "2.37.0", 
    cores = 11, 
    dir = "/opt/cmdstan",
    cpp_options = list("CXX" = "clang++"))
cmdstan_path()