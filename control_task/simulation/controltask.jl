import Pkg
Pkg.activate("relmed_environment")
Pkg.instantiate()
using DataFrames
using LinearAlgebra, LogExpFunctions
using ForwardDiff, Optim
using LaTeXStrings, Plots #PlotlyJS
using Random, Distributions, StatsBase, GLM

#------------------------------------------
# generate some experiments 
#

nTrials = 100
alpha = 0.1;

Tu = [2 3 4 1]
Tc = [4 1 2 3]

Tci = zeros(Int, 4, 4, 24)
Ti = zeros(Int, 24, 4)
i = 1
for j = 1:4
    for k = 1:3
        for l = 1:2
            jx = j
            kind = collect(1:4)
            deleteat!(kind, j)
            kx = kind[k]
            lind = deepcopy(kind)
            deleteat!(lind, k)
            lx = lind[l]
            mind = deepcopy(lind)
            deleteat!(mind, l)
            mx = mind[1]
            Ti[i, :] = [jx kx lx mx]
            Tci[jx, 1, i] = 1
            Tci[kx, 2, i] = 1
            Tci[lx, 3, i] = 1
            Tci[mx, 4, i] = 1
            global i = i + 1
        end
    end
end



effth = 1 .* [1 5 10 20]
effthinf = zeros(nTrials + 1, 4)
effthinf[1, :] = effth

s = zeros(Int, nTrials, 2);
b = zeros(Int, nTrials, 2);
a = zeros(Int, 1, nTrials);
n = zeros(Int, 1, nTrials);
c = zeros(Int, 1, nTrials);
e = zeros(Float64, 1, nTrials);

tn = alpha * ones(24)

function l(beta, X, pc)
    sig = 1 ./ (1 .+ exp.(-X * beta))
    ll = pc[:, 1]' * log.(sig) + (1 .- pc[:, 1])' * log.(1 .- sig .+ 1e-100)
    return ll
end


T = zeros(4, 4);
X = zeros(nTrials, 4)
w = ones(nTrials, 24) / 24
for k = 1:4
    T[:, k] = Tci[:, k, :] * w[1, :]
end
pc = zeros(nTrials, 2)
cinf = zeros(1, nTrials)
for t = 1:nTrials

    s[t, 1] = rand(DiscreteUniform(1, 4)) # island 
    b[t, :] = rand(DiscreteUniform(1, 4), 1, 2) # boats 
    n[t] = rand(DiscreteUniform(1, 3)) # wind strength 

    # generate random choices 

    a[t] = b[t, Int(rand() > 0.5)+1] # choice 
    #e[t] = rand(Poisson(effth[n[t]]))  # effort 
    e[t] = rand(DiscreteUniform(0, 30))

    X[t, 1] = e[t]
    X[t, 2] = -(n[t] .== 1)
    X[t, 3] = -(n[t] .== 2)
    X[t, 4] = -(n[t] .== 3)

    pctrue = 1 ./ (1 .+ exp(-X[t, :]' * effth'))
    c[t] = rand() < pctrue[1]

    if c[t] == 1
        s[t, 2] = Tc[a[t]]
    else
        s[t, 2] = Tu[s[t, 1]]
    end


    #e[t] = rand(Exponential(1))  # effort 
    #a[t] = rand(Poisson(e[t]))   # number of actions

    # infer control 
    compdefault = Tu[s[t, 1]] .== s[t, 2]
    compcontrol = Ti[:, a[t]] .== s[t, 2]


    if ~compdefault # must have been in control 
        # transition matrices that lead to successor island
        pottrans = findall(Ti[:, a[t]] .== s[t, 2])
        tn[pottrans] = tn[pottrans] .+ 1
        imptrans = findall(Ti[:, a[t]] .!= s[t, 2])
        tn[imptrans] = tn[imptrans] .- 5
        tn[tn.<0] .= 0
        w[t, :] = tn ./ sum(tn)
        #elseif e[t]==0 # can't have been in control 
    else
        if t > 1
            w[t, :] = w[t-1, :]
        end
    end
    if t > 1 # could be either 
        pcgae = 1 / (1 + exp(-X[t, :]' * effthinf[t-1, :]))
        for k = 1:4
            T[:, k] = Tci[:, k, :] * w[t-1, :]
        end
        psgc = T[s[t, 2], a[t]]
        psgu = Float64(Tu[s[t, 1]] .== s[t, 2])
        pc[t, :] = [psgc * pcgae psgu * (1 - pcgae)]
        pc[t, :] = pc[t, :] ./ sum(pc[t, :])
        cinf[t] = pc[t, 1]

        lx(x) = -l(x, X[2:t, :], pc[2:t, :]) + 0.001 * x' * x
        bh = optimize(lx, effthinf[t-1, :])
        effthinf[t, :] = bh.minimizer

        # Data = DataFrame([pc[1:t,1] X[1:t,:]],:auto)
        # effthinf[t+1,:] = coef(glm(@formula(x1~0+x2+x3+x4+x5),Data,Binomial(),ProbitLink()))

    end


end

for k = 1:4
    T[:, k] = Tci[:, k, :] * w[end, :]
end

p1 = plot(c[1, :], ls=:dash, lw=1, lc=:blue)
scatter!(c[1, :], mc=:blue)
plot!(cinf', ls=:solid, lw=1, lc=:red)

p2 = plot(ones(nTrials - 2, 1) * effth, linecolors=[:blue :red :green :cyan], label=["b0" "b1" "b2" "b3"])
plot!(effthinf[2:end-1, :], ls=:dashdot, linecolors=[:blue :red :green :cyan], label=["binf0" "binf1" "binf2" "binf3"])
plot!(legend=:outerright)

p3 = plot(heatmap(w))
#p3 = plot(w,legend=false)

x = collect(0:30)
t1 = 1 ./ (1 .+ exp.(-(effth[1] .* x .- effth[2])))
t2 = 1 ./ (1 .+ exp.(-(effth[1] .* x .- effth[3])))
t3 = 1 ./ (1 .+ exp.(-(effth[1] .* x .- effth[4])))
y1 = 1 ./ (1 .+ exp.(-(effthinf[end-1, 1] .* x .- effthinf[end-1, 2])))
y2 = 1 ./ (1 .+ exp.(-(effthinf[end-1, 1] .* x .- effthinf[end-1, 3])))
y3 = 1 ./ (1 .+ exp.(-(effthinf[end-1, 1] .* x .- effthinf[end-1, 4])))
p4 = plot(x, [t1 y1 t2 y2 t3 y3], lc=[:red :red :green :green :cyan :cyan], ls=[:solid :dash :solid :dash :solid :dash])

plot(p1, p2, p3, p4)

