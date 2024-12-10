@testset "FI" begin

    # Simplest coin-toss model for testing
    @model function test_bernoulli(;
        x,
        priors::Dict = Dict(
            :p => Beta(1, 1)
        )
    )
        p ~ priors[:p]

        for i in eachindex(x)
            x[i] ~ Bernoulli(p)
        end
    end

    # Simple unpacking function
    map_data_to_model(df::AbstractDataFrame) = (; x = df.x,)
    
    # Test one value
    dat = DataFrame(
        PID = [1], 
        x = [1],
        p = [1]
    )
    
    @test FI(
        data = dat,
        model = test_bernoulli,
        map_data_to_model = map_data_to_model,
        param_names = [:p],
        id_col = :PID
    ) == 1.0 # Expected FI value, since for x = 1 FI = 1 / p^2 for Bernoulli ll.

    # Test grouping
    dat = DataFrame(
        PID = [1, 2],
        x = [1, 0],
        p = [1., 0.]
    )

    @test FI(
        data = dat,
        model = test_bernoulli,
        map_data_to_model = map_data_to_model,
        param_names = [:p],
        id_col = :PID
    ) == 2.0 # Expected FI value, given independent handling of participants

    # Test insufficient data columns
    @test_throws ArgumentError FI(
        data = DataFrame(PID = [1], y = [2.0]),
        model = test_bernoulli,
        map_data_to_model = map_data_to_model,
        param_names = [:p]
    )

    # Test keyword argument passing ------------------------
    @model function test_normal(;
        x,
        σ::Float64 = 1.0,
        priors = Dict(
            :μ => Normal(0, 1)
        )
    )   
        μ ~ priors[:μ]

        for i in eachindex(x)
            x[i] ~ Normal(μ, σ)
        end
    end

    # Test one value
    dat = DataFrame(
        PID = [1], 
        x = [1],
        μ = [0]
    )
    
    # Without passing kwargs
    @test FI(
        data = dat,
        model = test_normal,
        map_data_to_model = map_data_to_model,
        param_names = [:μ],
        id_col = :PID
    ) == 1.0 # σ = 1.0, FI = 1 / σ ^ 2

    # With passing kwargs
    @test FI(
        data = dat,
        model = test_normal,
        map_data_to_model = map_data_to_model,
        param_names = [:μ],
        id_col = :PID,
        σ = 2.
    ) == 0.25 # σ = 2.0, FI = 1 / σ ^ 2


end