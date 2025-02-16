using DataFrames
using LinearAlgebra, LogExpFunctions
using ForwardDiff, Optim
using LaTeXStrings, Plots #PlotlyJS
using Random, Distributions, StatsBase, GLM
using Debugger

#------------------------------------------
# generate some experiments 
#

nTrials = 100

nSamples = 24*40
alpha = 10;  # dirichlet variance for weights 
sigma = 1     # beta variance 

#betatrue = 1 .* [1 5 10 20] # true control parameters 
#betatrue = 1 .* [.5 5] # true control parameters 
betatrue = [5 10 20] # true control parameters 
nbeta = length(betatrue)

Tc = [4 3 2 1] # true controlled transition matrix 
Tu = [2 3 4 1] # true uncontrolled transition matrix 

# make all  possible transition matrices 
Tci = zeros(Int,4,4,24)
Ti = zeros(Int,24,4)
i = 1 
Titrue = 0
for j=1:4
   for k=1:3
		 for l=1:2
          jx = j
          kind = collect(1:4); deleteat!(kind,j); 
          kx = kind[k]
          lind = deepcopy(kind); deleteat!(lind,k)
          lx = lind[l]
          mind = deepcopy(lind); deleteat!(mind,l)
          mx = mind[1]
          Ti[i,:] = [jx kx lx mx]
          Tci[jx,1,i] = 1
          Tci[kx,2,i] = 1
          Tci[lx,3,i] = 1
          Tci[mx,4,i] = 1
          if Tc == [jx kx lx mx]
              global Titrue = i
          end
          global i=i+1
      end
   end
end

    s = zeros(Int, nTrials,2); 
    b = zeros(Int, nTrials,2); 
    a = zeros(Int, 1, nTrials); 
    n = zeros(Int, 1, nTrials); 
    c = zeros(Int, 1, nTrials); 
    e = zeros(Float64, 1, nTrials); 

    Id = zeros(Float64,4,4)
    Id[diagind(Id)] .= 1

    T = ones(4,4)/4; 
    X = zeros(nTrials,nbeta)

    pc = zeros(nTrials,2)
    betahat = zeros(nbeta,nTrials)
    what = zeros(24,nTrials)

    BX = zeros( nbeta,nSamples,nTrials)
    WX = zeros(24,nSamples,nTrials)
    XX = zeros(nSamples,nTrials)

    resampled = zeros(nTrials,)
    neff      = zeros(nTrials,)

function infer() 

    x = ones(nSamples,)/nSamples

    bx = .1*betatrue' .+ 10 .+ rand(Normal(0,1),nbeta,nSamples)
    wx = zeros(24,nSamples)
    for k=1:24
        foo = ones(24,)
        foo[k] = 10
        wx[:,(1:Int(nSamples/24)).+(k-1)*Int(nSamples/24)] = rand(Dirichlet(foo*100),Int(nSamples/24))
    end

    prx = zeros(nSamples,1)
    pcx = zeros(nSamples,2)

    for t=1:nTrials

        s[t,1] = rand(DiscreteUniform(1,4)) # island 
        b[t,:] = rand(DiscreteUniform(1,4),1,2) # boats 
        n[t] = rand(DiscreteUniform(1,3)) # wind strength 

       # generate random choices 

        a[t] = b[t,Int(rand()>.5)+1] # choice 
        e[t] = rand(DiscreteUniform(0,30))
        # e[t] = rand(DiscreteUniform(max(betatrue[n[t]+1]-5,0),betatrue[n[t]+1]+5))

        #X[t,1] = e[t]
        #X[t,2] = -n[t]
        # pctrue = 1 ./ (1 .+ exp(-X[t,:]'*betatrue')) 
        #
        #X[t,1] = e[t]
        #X[t,2] = -(n[t].==1)
        #X[t,3] = -(n[t].==2)
        #X[t,4] = -(n[t].==3)
        # pctrue = 1 ./ (1 .+ exp(-X[t,:]'*betatrue')) 
        
        pctrue = 1 ./ (1 .+ exp(-2*(e[t] - betatrue[n[t]]))) 

        c[t] = rand() < pctrue[1]

       if c[t]==1
          s[t,2] = Tc[a[t]]
       else
          s[t,2] = Tu[s[t,1]]
       end

       for k=1:nSamples
          # draw new sample
          # bxn = bx[:,k] .+ rand(Normal(0,sigma),4) # rand(MvNormal(bx[:,k],Id),1)
          #
          # draw only the betas which are relevant for this trial
          # and use a smaller noise for the scaling beta than the threshold betas
          # bxn = bx[:,k]
          # bxn[1] = bxn[1] + rand(Normal(0,sigma)) # rand(MvNormal(bx[:,k],Id),1)
          # bxn[n[t]+1] = bxn[n[t]+1] + rand(Normal(0,sigma)) # rand(MvNormal(bx[:,k],Id),1)
          # bxn[2] = bxn[2] + rand(Normal(0,sigma)) # rand(MvNormal(bx[:,k],Id),1)
          # 
          # draw only the betas which are relevant for this trial
          bxn = bx[:,k]
          bxn[n[t]] = bxn[n[t]] + rand(Normal(0,sigma)) # rand(MvNormal(bx[:,k],Id),1)

          wxn = rand(Dirichlet(wx[:,k]*alpha),1)
          wxn = wxn.+1e-50; wxn = wxn./sum(wxn); # avoid zeros 

          # transition probability 
          # pbxn = pdf(MvNormal(bx[:,k],Id),bxn)
          # pwxn = pdf(Dirichlet(wx[:,k]*alpha),wxn)
          # ptrans = pbxn*pwxn

          bx[:,k] = bxn
          wx[:,k] = wxn

          # evaluate likelihood of new observation 
          # p(r_t | w_t, beta_t, o_t) = \sum_c p(r_t|c_t,a_t,s_t,w_t) p(c_t|e_t,n_t,beta_t)
          # first evaluate p(c_t=1|e_t,n_t,beta_t)
          # pcgae = 1 ./ (1 .+ exp.(-X[t,:]'*bx[:,k]))
          pcgae = 1 ./ (1 .+ exp.(-2*(e[t] - bx[n[t],k])))

          # then do p(r_t|c_t=1,a_t,s_t,w_t) 
          # for k=1:4; T[:,k] = Tci[:,k,:]*wx[:,k]; end
          # prgc = T[s[t,2],a[t]]
          prgc = Tci[s[t,2],a[t],:]'*wx[:,k]

          # p(r_t|c_t=0,a_t,s_t,w_t) 
          prgu = Float64(Tu[s[t,1]].==s[t,2])

          prx[k] = prgc*pcgae + prgu*(1-pcgae)

          # update weight
          x[k] = x[k] * prx[k]

          # p(c_t|a_t,s_t,w_t,e_t,n_t,beta_t)
          pcx[k,:] = [prgc*pcgae prgu*(1-pcgae)]
          pcx[k,:] = pcx[k,:] / sum(pcx[k,:]);

       end
       WX[:,:,t] = wx
       BX[:,:,t] = bx

       # p1=plot(prx)
       # p2=plot(x)
       # pp=plot(p1,p2)
       # display(pp)
       # sleep(.2)

       # renormalize  particle weights 
       x = x ./sum(x)

       XX[:,t] = x

       # evalute mean variables 
       betahat[:,t] = bx*x
       what[:,t] = wx*x
       pc[t,:] = x'*pcx

       # resample
       neff[t] = 1 ./ sum(x.^2)
       if neff[t] < .7*nSamples
          presample = cumsum(x)
          bxo = deepcopy(bx)
          wxo = deepcopy(wx)
          for k=1:nSamples
             newsample = sum(rand() .> presample)+1
             bx[:,k] = bxo[:,newsample]
             wx[:,k] = wxo[:,newsample]
          end
          x .= 1/nSamples
          resampled[t] = 1
       end
    end
    print("\nResampled: ",sum(resampled)," times \n\n" )
   return betahat[:,end]
end

function myplots()

    fs = 7
    for k=1:4; T[:,k] = Tci[:,k,:]*what[:,end]; end

    # plot true control variable and estimated control variable 
    p1=plot(c[1,:],ls=:dash,lw=[1],lc=[ :blue])
    scatter!(c[1,:],mc=:blue)
    plot!(pc[:,1],ls=:solid,lw=[2],lc=[ :red],legend=false)
    title!("control inference",titlefontsize=fs)
    xlabel!("Trial")

    # fraction of control events (left) and no control (right) correction
    # identified 
    p3 = bar([mean(pc[findall(c[1,:].==1),1]); mean(pc[findall(c[1,:].==0),2])],legend=false)
    ylims!(0, 1)
    xticks!([1,2],["c=1","c=0"])
    title!("Control events",titlefontsize=fs)
    ylabel!("% correct")

    # plot estimate beta weights over time 
    p2=plot(resampled.*maximum(betahat),lc=RGB(.7,.7,.7))
    plot!(ones(nTrials-2,1)*betatrue,linecolors=[:blue :red :green :cyan],label=["b0" "b1" "b2" "b3"])
    plot!(betahat',ls=:dashdot,linecolors=[:blue :red :green :cyan],label=["binf0" "binf1" "binf2" "binf3"])
    plot!(legend=false)
    xlabel!("Trial")
    ylabel!("beta")
    title!("Betas over trials",titlefontsize=fs)


    # plot estimated transition matrix weights over time. The first (bottom) one is the diagonal 
    p4 = plot(heatmap(what))
    xlabel!("Trial")
    ylabel!("w")
    plot!([1 nTrials/2 nTrials]',[Titrue Titrue Titrue]',line=(1,:red,:dashdot),marker=(:circle),legend=false,)
    ylims!(.5,24.5)
    xlims!(.5,nTrials+.5)
    title!("Transition weights over time",titlefontsize=fs)

    # plot estimated effort functions at the end 
    xx = collect(0:30)
    #t1=1 ./ (1 .+ exp.(- (betatrue[1] .* xx .- betatrue[2])))
    #t2=1 ./ (1 .+ exp.(- (betatrue[1] .* xx .- betatrue[3])))
    #t3=1 ./ (1 .+ exp.(- (betatrue[1] .* xx .- betatrue[4])))
    #y1=1 ./ (1 .+ exp.(- (betahat[1,end] .* xx .- betahat[2,end])))
    #y2=1 ./ (1 .+ exp.(- (betahat[1,end] .* xx .- betahat[3,end])))
    #y3=1 ./ (1 .+ exp.(- (betahat[1,end] .* xx .- betahat[4,end])))
    t1=1 ./ (1 .+ exp.(- 2 .*(xx .- betatrue[1])))
    t2=1 ./ (1 .+ exp.(- 2 .*(xx .- betatrue[2])))
    t3=1 ./ (1 .+ exp.(- 2 .*(xx .- betatrue[3])))
    y1=1 ./ (1 .+ exp.(- 2 .*(xx .- betahat[1,end])))
    y2=1 ./ (1 .+ exp.(- 2 .*(xx .- betahat[2,end])))
    y3=1 ./ (1 .+ exp.(- 2 .*(xx .- betahat[3,end])))
    p5 = plot(xx,[t1 y1 t2 y2 t3 y3],lc=[ :red :red :green :green :cyan :cyan],ls=[:solid :dash :solid :dash :solid :dash])
    xlabel!("Effort")
    ylabel!("p(control)")
    title!("Betas over time",titlefontsize=fs)

    return p1,p2,p3,p4,p5

    plot(p1,p3,p2,p4,p5)
      Plots.scalefontsizes(.5)
end

#break_on(:error)
infer()
myplots()

# now run it over a few iterations to see how it tends to do on average, rather
# than on individual trials

niterations=25
betahat_it = zeros(nbeta,niterations)
for i=1:niterations
   print("iteration ",i,"\r")
   betahat_it[:,i] = infer()
   (p1,p2,p3,p4,p5) = myplots()
   display(plot(p1,p3,p2,p4,p5))
end
mb = mean(betahat_it,dims=2)
sb = std(betahat_it,dims=2)
p6=bar(betatrue')
scatter!(betahat_it,legend=false,mc=:grey,ms=2)
plot!(1:3,mb,yerr=sb)
scatter!(mb,mc=:red,ms=4)
xticks!(1:3)
xlabel!("Current level")
ylabel!("True/inferred")
(p1,p2,p3,p4,p5) = myplots()
plot(p1,p3,p2,p4,p5,p6)


