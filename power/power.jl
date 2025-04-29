# # cd("/home/jovyan")
# import Pkg
# # activate the shared project environment
# Pkg.activate("$(pwd())/relmed_environment")
# # instantiate, i.e. make sure that all packages are downloaded
# Pkg.instantiate()

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

using ForwardDiff, LinearAlgebra, LogExpFunctions, Random, Plots, Optim, Statistics, NaNStatistics


#------------------------------------------
# load models 
# include("llmodls.jl")

#------------------------------------------
# RW simulations 

# first check the likelihood surface behaves 
# x = [1 -1]
# a,r,qt = llrw(x,D)
# Da = deepcopy(D)
# Da["actions"] = a
# Da["rewards"] = r
# 
# ali = -5:.1:5;
# bei = -5:.1:5; 
# l = zeros(length(ali),length(bei))
# for bi = 1:length(bei)
#    for ai = 1:length(ali)
#        y = [bei[bi] ali[ai] ]
#        l[bi,ai] = llrw(y,Da)
#    end
# end
# p1=heatmap(bei,ali,log(l))


# do inference 
# f(x) = llrw(x,Da)
# xtrue = randn(2,100)
# Da = deepcopy(D);
# xest = zeros(2,100)
# for ns=1:100
#     a,r = llrw(xtrue[:,ns],D)
#     Da["actions"] = a;
#     Da["rewards"] = r;
#     est = optimize(f,randn(2,1));
#     xest[:,ns] = est.minimizer;
# end
# p2=scatter(xtrue',xest')

# do inference 


function runsim(llrw,llrwdb,D,nExp,deltatrue,nSubj,nDelta)

    ll= zeros(2,nDelta,nExp,nSubj)
    A = zeros(T*nStim,nDelta,nExp,nSubj)
    xest  = zeros(3,nSubj,nDelta,nExp)
    xtruei = zeros(3,nSubj,nDelta,nExp)

    Threads.@threads for di = 1:nDelta
       for ne = 1:nExp
          xtrue = randn(3,nSubj)
          xtrue[1,:] = xtrue[1,:] 
          xtrue[2,:] = xtrue[2,:] .* 0 
          xtrue[3,:] = xtrue[3,:] .- 1 
          for ns = 1:nSubj
             xsim = xtrue[:,ns] 
             xsim[2] = deltatrue[di] 
             a,r = llrwdb(xsim,D)
             #a,r = llrwda(xsim,D)
             Da = Dict([("acorr",acorr),("stim",stim),("actions",a),("rewards",r)])
             A[:,di,ne,ns] = a;
             xtruei[:,ns,di,ne] = xsim

				 f(x) = llrw(x,Da)
             while true 
                 est = optimize(f,randn(2,1)[:],NewtonTrustRegion(),autodiff=:forward);
                 if ~isnan.(est.minimum);
                     ll[1,di,ne,ns] = est.minimum;
                     break;
                 end
             end
             #xest[:,ns] = est.minimizer;

				 fdb(x) = llrwdb(x,Da)
             while true 
                 est = optimize(fdb,randn(3,1)[:],NewtonTrustRegion(),autodiff=:forward);
                 if ~isnan.(est.minimum) & (est.minimum < -1.1*T*nStim*log(.5))
                     xest[:,ns,di,ne] = est.minimizer;
                     ll[2,di,ne,ns] = est.minimum;
                     break;
                 end
             end

				 # fda(x) = llrwda(x,Da)
             # while true 
             #     est = optimize(fda,randn(3,1)[:],NewtonTrustRegion(),autodiff=:forward);
             #     if ~isnan.(est.minimum) & (est.minimum < -1.1*T*nStim*log(.5))
             #         xest[:,ns,di,ne] = est.minimizer;
             #         ll[3,di,ne,ns] = est.minimum;
             #         break;
             #     end
             # end
          end
          print("delta=",di," nexp=",ne,"\n")
       end
    end
    return ll, A, xest, xtruei
end

#------------------------------------------
# generate some experiments 

T = 40; 
nStim = 10; 
stim = zeros(Int,T*nStim); 
acorr = deepcopy(pairs); 
for k=1:nStim
    stim[ (1:T) .+ (k-1)*T] .= k
end
acorr = mod.(stim,2) .+ 1
D = Dict([("acorr",acorr),("stim",stim)])

#------------------------------------------
# get models 
include("llmodls.jl")

#------------------------------------------
# define simulations 

nExp = 20		# number of experiments to run 
nSubj = 30		# number of subjects per experiment 
deltatrue = [-5 -3 -2 -1 -.5  0 .5 1 2 3 5]	# shifts in parameter (transformed absolute)
nDelta = length(deltatrue)

# nExp = 1
# nSubj = 150
# deltatrue = [-5 -3 -2 -1 -.5  0 .5 1 2 3 5]
# nDelta = length(deltatrue)

#------------------------------------------
# run actual simulations 

ll,A,xest,xtruei = runsim(llrw,llrwdb,D,nExp,deltatrue,nSubj,nDelta)

#------------------------------------------
# compute model differences in LL and BIC 

bic = deepcopy(ll)
bic[1,:,:,:] = 2*ll[1,:,:,:] .+ 2*log(T*nStim)
bic[2,:,:,:] = 2*ll[2,:,:,:] .+ 3*log(T*nStim)

dll  = mean(diff(ll ,dims=1),dims=4)[1,:,:,1]; 
dbic = mean(diff(bic,dims=1),dims=4)[1,:,:,1]; 

#------------------------------------------
# compute probabilities of identifying models 

dbicthreshold = 3 

pbic= zeros(nDelta,nSubj)
dbic_size = zeros(nDelta,nExp,nSubj)
for k=1:nSubj
	dbic_size[:,:,k] = mean(diff(bic[:,:,:,1:k],dims=1),dims=4)[1,:,:,1]; 
	pbic[:,k] = sum(dbic_size[:,:,k] .< dbicthreshold,dims=2) ./ nExp 
end

#------------------------------------------
# make a nice plot 

p1 = plot(mean(mean(A,dims=4),dims=3)[:,:,1,1].-1,label=deltatrue,legend=:outerright)
	plot!(T*nStim/2*[1.0;1.0],[0;1.0],lc=:black,label="")
	xlabel!("Trial")
	ylabel!("Action")

p2 = plot(deltatrue',mean(mean(xest[2,:,:,:],dims=1)[1,:,:],dims=2);yerror=mean(std(xest[2,:,:,:],dims=1)[1,:,:],dims=2),lc=:black,label="mean")
	  scatter!(deltatrue',mean(xest[2,:,:,:],dims=1)[1,:,:],label="",mc=:black)
     plot!(deltatrue',mean(median(xest[2,:,:,:],dims=1)[1,:,:],dims=2),lc=:blue,label="median")
	  scatter!(deltatrue',median(xest[2,:,:,:],dims=1)[1,:,:],label="",mc=:blue)
	  plot!(deltatrue',deltatrue',lc=:red,label="true")
	xlabel!("True change")
	ylabel!("Inferred change")

p3 = plot(deltatrue',dll,lc=:blue ); scatter!(deltatrue',dll,legend=false,mc=:blue,yguidefontsize=7, ylabel="LL and BIC mean difference",xlabel="Parameter change")
 plot!(deltatrue',dbic,lc=:red); scatter!(deltatrue',dbic,legend=false,mc=:red); 
 plot!(deltatrue',mean(dbic,dims=2),lw=3,lc=:black)

d0 = findall(iszero,deltatrue)[1][2]
p4 = plot(pbic[1:d0-1,:]',label=deltatrue[1:d0-1]',legend=:right)
	plot!( pbic[d0,:],label=deltatrue[d0],lc=:black,lw=2,yguidefontsize=7,ylabel="P(complex model BIC lower)")
	ylims!(0,1)
	xlabel!("# participants")
	title!("Increases")

p5 = plot( pbic[d0,:],label=deltatrue[d0],legend=false,lc=:black,lw=2)
   plot!(pbic[d0+1:end,:]',label=deltatrue[d0+1:end]',legend=:right)
	ylims!(0,1)
	xlabel!("# participants")
	title!("Decreases")

# pbic[2,:] =  sum(dll .< -3,dims=2) ./ nExp 

l = @layout [a b; c d e]
plot(p1,p2,p3,p4,p5,layout = l)

meanbeta = @sprintf("%.2g",1/(1+exp(-mean(xtruei[1,:,:,:])))) 
meanalpha = @sprintf("%.2g",1/(1+exp(-mean(xtruei[3,:,:,:])))) 
savefig(string("Alpha_",meanalpha,"-Beta_",meanbeta,".pdf"))


