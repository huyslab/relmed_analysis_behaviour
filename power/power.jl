using ForwardDiff, LinearAlgebra, LogExpFunctions, Random, Plots, Optim, Statistics, NaNStatistics

function runsim(llrw,llrwdb,llrwda,D,nExp,deltatrue,nSubj,nDelta)

    ll= zeros(3,nDelta,nExp,nSubj)
    A = zeros(T*nStim,nDelta,nExp,nSubj)
    xest  = zeros(3,nSubj,nDelta,nExp,3)
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

             #a,r = llrwdb(xsim,D)
             a,r = llrwda(xsim,D)
				 
             Da = Dict([("acorr",acorr),("stim",stim),("actions",a),("rewards",r)])
             A[:,di,ne,ns] = a;
             xtruei[:,ns,di,ne] = xsim

				 f(x) = llrw(x,Da)
             while true 
                 est = optimize(f,randn(2,1)[:],NewtonTrustRegion(),autodiff=:forward);
                 if ~isnan.(est.minimum);
                     xest[[1;3],ns,di,ne,1] = est.minimizer;
                     ll[1,di,ne,ns] = est.minimum;
                     break;
                 end
             end
             #xest[:,ns] = est.minimizer;

				 fdb(x) = llrwdb(x,Da)
             while true 
                 est = optimize(fdb,randn(3,1)[:],NewtonTrustRegion(),autodiff=:forward);
                 if ~isnan.(est.minimum) & (est.minimum < -1.1*T*nStim*log(.5))
                     xest[:,ns,di,ne,2] = est.minimizer;
                     ll[2,di,ne,ns] = est.minimum;
                     break;
                 end
             end

				 fda(x) = llrwda(x,Da)
             while true 
                 est = optimize(fda,randn(3,1)[:],NewtonTrustRegion(),autodiff=:forward);
                 if ~isnan.(est.minimum) & (est.minimum < -1.1*T*nStim*log(.5))
                     xest[:,ns,di,ne,3] = est.minimizer;
                     ll[3,di,ne,ns] = est.minimum;
                     break;
                 end
             end
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

nExp = 30		# number of experiments to run 
nSubj = 30		# number of subjects per experiment 
deltatrue = [-5 -3 -2 -1 -.5  0 .5 1 2 3 5]	# shifts in parameter (transformed absolute)
nDelta = length(deltatrue)

# nExp = 1
# nSubj = 150
# deltatrue = [-5 -3 -2 -1 -.5  0 .5 1 2 3 5]
# nDelta = length(deltatrue)

#------------------------------------------
# run actual simulations 

ll,A,xest,xtruei = runsim(llrw,llrwdb,llrwda,D,nExp,deltatrue,nSubj,nDelta)

#------------------------------------------
# compute model differences in LL and BIC 

bic = deepcopy(ll)
bic[1,:,:,:] = 2*ll[1,:,:,:] .+ 2*log(T*nStim)
bic[2,:,:,:] = 2*ll[2,:,:,:] .+ 3*log(T*nStim)
bic[3,:,:,:] = 2*ll[3,:,:,:] .+ 3*log(T*nStim)

#dll  = mean( ll[2,:,:,:] .-  ll[1,:,:,:],dims=4)[:,1,:]; 
#dbic = mean(bic[2,:,:,:] .- bic[1,:,:,:],dims=4)[:,1,:]; 

dll  = mean( ll[3,:,:,:] .-  ll[1,:,:,:],dims=4)[:,1,:]; 
dbic = mean(bic[3,:,:,:] .- bic[1,:,:,:],dims=4)[:,1,:]; 


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

#p2 = plot(deltatrue',mean(mean(xest[2,:,:,:,2],dims=1)[1,:,:],dims=2);yerror=mean(std(xest[2,:,:,:,2],dims=1)[1,:,:],dims=2),lc=:black,label="mean")
#	  scatter!(deltatrue',mean(xest[2,:,:,:,2],dims=1)[1,:,:],label="",mc=:black)
#     plot!(deltatrue',mean(median(xest[2,:,:,:,2],dims=1)[1,:,:],dims=2),lc=:blue,label="median")
#	  scatter!(deltatrue',median(xest[2,:,:,:,2],dims=1)[1,:,:],label="",mc=:blue)
#	  plot!(deltatrue',deltatrue',lc=:red,label="true")
#	xlabel!("True change")
#	ylabel!("Inferred change")

p2 = plot(deltatrue',mean(mean(xest[2,:,:,:,3],dims=1)[1,:,:],dims=2);yerror=mean(std(xest[2,:,:,:,3],dims=1)[1,:,:],dims=2),lc=:black,label="mean")
	  scatter!(deltatrue',mean(xest[2,:,:,:,3],dims=1)[1,:,:],label="",mc=:black)
     plot!(deltatrue',mean(median(xest[2,:,:,:,3],dims=1)[1,:,:],dims=2),lc=:blue,label="median")
	  scatter!(deltatrue',median(xest[2,:,:,:,3],dims=1)[1,:,:],label="",mc=:blue)
	  plot!(deltatrue',deltatrue',lc=:red,label="true")
	ylims!(4*minimum(deltatrue),4*maximum(deltatrue))
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
	title!("Decreases")

p5 = plot( pbic[d0,:],label=deltatrue[d0],legend=false,lc=:black,lw=2)
   plot!(pbic[d0+1:end,:]',label=deltatrue[d0+1:end]',legend=:right)
	ylims!(0,1)
	xlabel!("# participants")
	title!("Increases")

l = @layout [a b; c d e]
plot(p1,p2,p3,p4,p5,layout = l)

meanbeta = @sprintf("%.2g",1/(1+exp(-mean(xtruei[1,:,:,:])))) 
meanalpha = @sprintf("%.2g",1/(1+exp(-mean(xtruei[3,:,:,:])))) 
savefig(string("AlphaChanges_MeanAlpha_",meanalpha,"-MeanBeta_",meanbeta,".pdf"))


