function llrw(x::AbstractVector{T},D::Any) where T

	beta = exp(x[1]); 
	alfa = 1/(1+exp(-x[2])); 

   s     = D["stim"]; 
   nStim = length(unique(stim))
   acorr = D["acorr"];

   if haskey(D,"actions")
       a = D["actions"]; 
       r = D["rewards"]; 
       genactions = 0 ; 
   else 
       a = Vector(undef,length(s)); 
       r = Vector(undef,length(s)); 
       genactions = 1 ; 
   end

	Tt = length(s); 
   Q = zeros(eltype(x[1]),2,nStim)
   l = 0.0

	for t = 1:Tt

		q = Q[:,s[t]]
		q = q .- maximum(q)
      lpa = beta*q 
		lpa = lpa .- log(sum(exp.(lpa)));

		if genactions==1
		   pa = exp.(lpa);
			a[t] = sum(cumsum([0 pa'],dims=2).<rand())
         if a[t] == acorr[t]; r[t] = 1.0; else r[t] = -1.0; end
         if rand()<.3; r[t] = -r[t]; end
		else
			l = l+lpa[a[t]]
		end

		Q[a[t],s[t]] = Q[a[t],s[t]] + alfa*(r[t] - Q[a[t],s[t]])

	end

   l = -l; 

   if genactions==1
       return a,r,qt
   else
       return l
   end

end


function llrwdb(x::AbstractVector{T},D::Any) where T

	beta1 = exp(x[1]); 
   beta2 = exp(x[1]+x[2]); 
	alfa = 1/(1+exp(-x[3])); 

   s     = D["stim"]; 
   nStim = length(unique(stim))
   acorr = D["acorr"];

   if haskey(D,"actions")
       a = D["actions"]; 
       r = D["rewards"]; 
       genactions = 0 ; 
   else 
       a = Vector{Int64}(undef,length(s)); 
       r = Vector{Float64}(undef,length(s)); 
       genactions = 1 ; 
   end

	Tt = length(s); 
   Q = zeros(eltype(x[1]),2,nStim)
   l = 0.0

	for t = 1:Tt
      if t<=Tt/2; beta=beta1; else beta=beta2;end

		q = Q[:,s[t]]
		q = q .- maximum(q)
      lpa = beta*q 
		lpa = lpa .- log(sum(exp.(lpa)));

		if genactions==1
		   pa = exp.(lpa);
			a[t] = sum(cumsum([0 pa'],dims=2).<rand())
         if a[t] == acorr[t]; r[t] = 1.0; else r[t] = -1.0; end
         if rand()<.3; r[t] = -r[t]; end
		else
			l = l+lpa[a[t]]
		end

		Q[a[t],s[t]] = Q[a[t],s[t]] + alfa*(r[t] - Q[a[t],s[t]])

	end

   l = -l; 

   if genactions==1
       return a,r
   else
       return l
   end

end


function llrwda(x::AbstractVector{T},D::Any) where T

	alfa1 = 1/(1+exp(-x[1])); 
	alfa2 = 1/(1+exp(-(x[1]+x[2]))); 
	beta = exp(x[3]); 

   s     = D["stim"]; 
   nStim = length(unique(stim))
   acorr = D["acorr"];

   if haskey(D,"actions")
       a = D["actions"]; 
       r = D["rewards"]; 
       genactions = 0 ; 
   else 
       a = Vector{Int64}(undef,length(s)); 
       r = Vector{Float64}(undef,length(s)); 
       genactions = 1 ; 
   end

	Tt = length(s); 
   Q = zeros(eltype(x[1]),2,nStim)
   l = 0.0

	for t = 1:Tt
      if t<=Tt/2; alfa=alfa1; else alfa=alfa2;end

		q = Q[:,s[t]]
		q = q .- maximum(q)
      lpa = beta*q 
		lpa = lpa .- log(sum(exp.(lpa)));

		if genactions==1
		   pa = exp.(lpa);
			a[t] = sum(cumsum([0 pa'],dims=2).<rand())
         if a[t] == acorr[t]; r[t] = 1.0; else r[t] = -1.0; end
         if rand()<.3; r[t] = -r[t]; end
		else
			l = l+lpa[a[t]]
		end

		Q[a[t],s[t]] = Q[a[t],s[t]] + alfa*(r[t] - Q[a[t],s[t]])

	end

   l = -l; 

   if genactions==1
       return a,r
   else
       return l
   end

end


