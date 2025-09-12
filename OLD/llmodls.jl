
function llrw(x,D,Tblocks)

	beta = exp(x[1]); 
	alfa = 1/(1+exp(-x[2])); 

   block = D["block"]; 
   pairs = D["pairs"]; 
   setsize = D["setsi"]; 

   if haskey(D,"actions")
       a = D["actions"]; 
       r = D["rewards"]; 
       genactions = 0 ; 
   else 
       a = Vector(undef,length(pairs)); 
       r = Vector(undef,length(pairs)); 
       genactions = 1 ; 
   end

	T = length(pairs); 

	Q = zeros(2,setsize[1])
   l = 0

	for t = 1:T

      if (t==1) || (block[t]!=block[t-1])
			Q = zeros(2,setsize[t])
		end

      lpa = beta*Q[:,pairs[t]] .- log(sum(exp.(beta*Q[:,pairs[t]])));
		pa = exp.(lpa);

		if genactions==1
			a[t] = sum(cumsum([0 pa'],dims=2).<rand())
         if a[t] == acorr[t]; r[t] = 1; else r[t] = -1; end
		else
			l = l+lpa[a[t]]
		end

		Q[a[t],pairs[t]] = Q[a[t],pairs[t]] + alfa*(r[t] - Q[a[t],pairs[t]])

	end

   l = -l; 

   if genactions==1
       return a,r
   else
       return l
   end

end


function llwm(x,D,Tblocks)
	C = x[1]; 
	beta = exp(x[2]); 
	alfa = 1 ./ (1 .+ exp(-x[3])); 
   w = 0; #1/(1+exp(-x[4]));  

   block = D["block"]; 
   pairs = D["pairs"]; 
   setsize = D["setsi"]; 

   if haskey(D,"actions")
       a = D["actions"]; 
       r = D["rewards"]; 
       genactions = 0 ; 
   else 
       a = Vector(undef,length(pairs)); 
       r = Vector(undef,length(pairs)); 
       genactions = 1 ; 
   end

	T = length(pairs); 


	Q = zeros(2,setsize[1])
   m = zeros(2,1)
	M = zeros(3,Tblocks)
   l = 0
   Tsetsize = sum(setsize.==setsize[1]);
   cw = 1 ./ (1 .+ exp.((collect(1:Tsetsize) .- C)/.6)); 
   sw = sum(cw);
   cw = cw/sw;

	for t = 1:T

      if (t==1) || (block[t]!=block[t-1])
			Q = zeros(2,setsize[t])
         Tsetsize = sum(setsize.==setsize[t]);
         M = zeros(3,Tsetsize)
         cw = 1 ./ (1 .+ exp.((collect(1:Tsetsize) .- C)/.6)); 
         sw = sum(cw);
         cw = cw/sw;
		end

      i1=M[1,:] .== 1 .&& M[2,:].== Float64(pairs[t]); 
      i2=M[1,:] .== 2 .&& M[2,:].== Float64(pairs[t]);  
      if any(i1); m[1] = M[3,i1]'*cw[i1]; else m[1] = 0; end
      if any(i2); m[2] = M[3,i2]'*cw[i2]; else m[2] = 0; end

		W = (1-w) * m + w * Q[:,pairs[t]];

		lpa = beta*W .- log(sum(exp.(beta*W)));
		pa = exp.(lpa);

		if genactions==1
			a[t] = sum(cumsum([0 pa'],dims=2).<rand())
         if a[t] == acorr[t]; r[t] = 1; else r[t] = 0; end
		else
			l = l+lpa[a[t]]
		end

		Q[a[t],pairs[t]] = Q[a[t],pairs[t]] + alfa* ( r[t] - Q[a[t],pairs[t]] )

		M = [[a[t]; pairs[t]; r[t];]  M[:,1:end-1]]

	end

   l = -l; 

   if genactions==1
       return a,r
   else
       return l
   end

end

return 
