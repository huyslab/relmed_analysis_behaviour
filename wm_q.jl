# cd("/home/jovyan")
import Pkg
# activate the shared project environment
Pkg.activate("$(pwd())/relmed_environment")
# instantiate, i.e. make sure that all packages are downloaded
Pkg.instantiate()

# using CairoMakie, Random, DataFrames, Distributions, StatsBase,
# ForwardDiff, LinearAlgebra, JLD2, FileIO, CSV, Dates, JSON, RCall, Turing, ParetoSmooth, MCMCDiagnosticTools, Printf
# using LogExpFunctions: logistic, logit
# include("$(pwd())/simulate.jl")
# include("$(pwd())/turing_models.jl")
# include("$(pwd())/plotting_utils.jl")
# 
# aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
# prior_sample_pmst = simulate_from_prior(
# 	1;
# 	model = RLWM_pmst,
# 	initial = aao,
# 	priors = Dict(
# 		:ρ => truncated(Normal(0., 1.), lower = 0.),
# 		:a => Normal(0., 0.5),
# 		:W => Normal(0., 0.5),
# 		:C => truncated(Normal(3., 2.), lower = 1.)
# 	),
# 	transformed = Dict(:a => :α, :W => :w0),
# 	structure = (
# 		n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = [2, 4, 6]
# 	),
# 	gq = true,
# 	random_seed = 123
# )
# describe(prior_sample_pmst)


using ForwardDiff, LinearAlgebra, LogExpFunctions, Random, Plots, Optim

#------------------------------------------
# generate some experiments 

Tblocks = 20; 
Setsizes = [1 2 3 4 5]; 
Nblocks = length(Setsizes); 
pairs = zeros(maximum(Setsizes)*Tblocks,Nblocks); 
acorr = deepcopy(pairs); 
block = deepcopy(pairs); 
setsi = deepcopy(pairs); 
for bl = 1:Nblocks
    #stimn = 2*Setsizes[bl] ; 
    #stimids = collect(1:stimn) .+ sum(2*[0 Setsizes[1:bl-1]']); 

    S = collect(1:Setsizes[bl])*ones(1,Tblocks); 
    pairs[1:Setsizes[bl]*Tblocks,bl] = copy(S[randperm(Setsizes[bl]*Tblocks)]); 
    block[1:Setsizes[bl]*Tblocks,bl] .= bl; 
    setsi[1:Setsizes[bl]*Tblocks,bl] .= Setsizes[bl]; 
    for k=1:Setsizes[bl]
        acorr[pairs[:,bl].==k,bl] .= (rand()>.5)+1; 
    end
end
block = block[pairs.!=0]; 
setsi = setsi[pairs.!=0]; 
acorr = acorr[pairs.!=0]; 
pairs = pairs[pairs.!=0]; 

setsi   = [Int(x) for x in setsi]; 
pairs   = [Int(x) for x in pairs]; 
block   = [Int(x) for x in block]; 
acorr   = [Int(x) for x in acorr]; 

D = Dict([("acorr",acorr),("pairs",pairs),("setsi",setsi),("block",block)]); 


#------------------------------------------
# load models 
include("llmodls.jl")

#------------------------------------------
# RW simulations 

# first check the likelihood surface behaves 
x = [1 -1]
a,r = llrw(x,D,Tblocks)
Da = deepcopy(D)
Da["actions"] = a
Da["rewards"] = r

ali = -5:.1:5;
bei = -5:.1:5; 
l = zeros(length(ali),length(bei))
for bi = 1:length(bei)
   for ai = 1:length(ali)
       y = [bei[bi] ali[ai] ]
       l[bi,ai] = llrw(y,Da,Tblocks)
   end
end
heatmap!(bei,ali,log.(l))

# do inference 
f(x) = llrw(x,Da,Tblocks)
xtrue = randn(2,100)
Da = deepcopy(D);
xest = zeros(2,100)
for ns=1:100
    a,r = llrw(xtrue[:,ns],D,Tblocks)
    Da["actions"] = a;
    Da["rewards"] = r;
    est = optimize(f,randn(2,1));
    xest[:,ns] = est.minimizer;
end
scatter!(xtrue[1,:],xest[1,:])

#------------------------------------------
# working memory simulations 

# first check the likelihood surface behaves 
x = [8 1 1 .5];
a,r = llwm(x,D,Tblocks);
Da = deepcopy(D);
Da["actions"] = a;
Da["rewards"] = r;

# just vary C for now 
y = copy(x); 
l = zeros(0)
ci = 1:.1:9
l = zeros(length(ci));
k=1; 
for c = ci; y[1] = c; l[k] = llwm(y,Da,Tblocks); k=k+1; end
plot(ci,l)

# do inference 
f(x) = llwm(x,Da,Tblocks)
est = optimize(f,randn(4,1));

xtrue = randn(4,100);
xtrue[1,:] = LinRange(1,15,100);
xtrue[2,:] .= 1;
xtrue[3,:] .= -.5;
xtrue[4,:] .= 0;
xest = zeros(4,100); 
for ns=1:100
    a,r = llwm(xtrue[:,ns],D,Tblocks); 
    Da["actions"] = a;
    Da["rewards"] = r;
    #est = optimize(f,[10.0 1 1 -1]);
    est = optimize(f,randn(4))
    xest[:,ns] = est.minimizer;
    print("\r",ns)
end
p = plot(layout = (2,2))
for k=1:4
    scatter!(p[k],xtrue[k,:],xest[k,:])
    ylims!(p[k],(-10,15))
    #plot!(p[k],[-5 15]',[-5 15]')
end
display(p)


# use automatic differentiation - weird bugs 
#func = TwiceDifferentiable(x -> llwm(x, Da,Tblocks), zeros(4); autodiff=:forward);;
#est = optimize(func,randn(4,1))



