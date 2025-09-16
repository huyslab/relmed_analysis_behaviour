# Setup
begin
    cd("/home/jovyan")
    import Pkg
    # activate the shared project environment
    Pkg.activate("$(pwd())/environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
    using Test, Random, DataFrames, Distributions, StatsBase, LinearAlgebra, ForwardDiff, Turing, DynamicPPL
    Turing.setprogress!(false)
    include("$(pwd())/core/model_utils.jl")
end

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
        p = [1-eps()]
    )
    
    @test FI(
        data = dat,
        model = test_bernoulli,
        map_data_to_model = map_data_to_model,
        param_names = [:p],
        id_col = :PID
    ) ≈ 1.0 # Expected FI value, since for x = 1 FI = 1 / p^2 for Bernoulli ll.

    # Test grouping
    dat = DataFrame(
        PID = [1, 2],
        x = [1, 0],
        p = [1. - eps(), 0.]
    )

    @test FI(
        data = dat,
        model = test_bernoulli,
        map_data_to_model = map_data_to_model,
        param_names = [:p],
        id_col = :PID
    ) ≈ 2.0 # Expected FI value, given independent handling of participants

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

@testset "prior_sample" begin

    # Test model for prior sampling
    @model function test_simple_model(;
        x,
        priors::Dict = Dict(
            :p => Beta(2, 2)
        )
    )
        p ~ priors[:p]

        for i in eachindex(x)
            x[i] ~ Bernoulli(p)
        end
    end

    # Test with NamedTuple input - single sample
    @testset "NamedTuple input - single sample" begin
        task_data = (
            x = [missing, missing, missing],
        )
        
        priors = Dict(:p => Beta(2, 2))
        
        result = prior_sample(
            task_data;
            model = test_simple_model,
            n = 1,
            priors = priors,
            outcome_name = :x
        )
        
        # Should return a vector of length 3
        @test length(result) == 3
        @test typeof(result) <: Vector
        # Each element should be 0 or 1 (Bernoulli outcomes)
        @test all(x -> x in [0, 1], result)
    end

    # Test with NamedTuple input - multiple samples
    @testset "NamedTuple input - multiple samples" begin
        task_data = (
            x = [missing, missing],
        )
        
        priors = Dict(:p => Beta(2, 2))
        
        result = prior_sample(
            task_data;
            model = test_simple_model,
            n = 5,
            priors = priors,
            outcome_name = :x
        )

        # Should return a 2x5 matrix
        @test size(result) == (2, 5)
        @test typeof(result) <: Matrix
        # Each element should be 0 or 1 (Bernoulli outcomes)
        @test all(x -> x in [0, 1], result)
    end

    # Test with custom RNG for reproducibility
    @testset "Custom RNG for reproducibility" begin
        task_data = (
            x = [missing, missing, missing],
        )
        
        priors = Dict(:p => Beta(2, 2))
        rng = MersenneTwister(42)
        
        result1 = prior_sample(
            task_data;
            model = test_simple_model,
            n = 1,
            priors = priors,
            outcome_name = :x,
            rng = rng
        )
        
        # Reset RNG to same seed
        rng = MersenneTwister(42)
        result2 = prior_sample(
            task_data;
            model = test_simple_model,
            n = 1,
            priors = priors,
            outcome_name = :x,
            rng = rng
        )
        
        @test result1 == result2
    end

    # Test DataFrame input
    @testset "DataFrame input" begin
        # Simple unpack function
        function unpack_test_data(df::AbstractDataFrame)
            return (x = df.x,)
        end
        
        task_df = DataFrame(
            x = [1, 0, 1]  # Will be replaced with missing values
        )
        
        priors = Dict(:p => Beta(2, 2))
        
        result = prior_sample(
            task_df;
            model = test_simple_model,
            unpack_function = unpack_test_data,
            n = 1,
            priors = priors,
            outcome_name = :x
        )
        
        # Should return a vector of length 3
        @test length(result) == 3
        @test typeof(result) <: Vector
        @test all(x -> x in [0, 1], result)
    end

    # Test with model that has kwargs
    @model function test_model_with_kwargs(;
        x,
        σ::Float64 = 1.0,
        priors::Dict = Dict(
            :μ => Normal(0, 1)
        )
    )
        μ ~ priors[:μ]

        for i in eachindex(x)
            x[i] ~ Normal(μ, σ)
        end
    end

    @testset "Model with kwargs" begin
        task_data = (
            x = [missing, missing],
        )
        
        priors = Dict(:μ => Normal(0, 1))
        
        result = prior_sample(
            task_data;
            model = test_model_with_kwargs,
            n = 1,
            priors = priors,
            outcome_name = :x,
            σ = 2.0
        )
        
        @test length(result) == 2
        @test typeof(result) <: Vector
        @test all(x -> typeof(x) <: Real, result)
    end

    # Test error handling
    @testset "Error handling" begin
        task_data = (
            x = [missing, missing],
        )
        
        priors = Dict(:p => Beta(2, 2))
        
        # Test with invalid outcome_name
        @test_throws Exception prior_sample(
            task_data;
            model = test_simple_model,
            n = 1,
            priors = priors,
            outcome_name = :y  # This field doesn't exist
        )
    end

    # Test with empty outcome array
    @testset "Empty outcome array" begin
        task_data = (
            x = Missing[],
        )
        
        priors = Dict(:p => Beta(2, 2))
        
        result = prior_sample(
            task_data;
            model = test_simple_model,
            n = 1,
            priors = priors,
            outcome_name = :x
        )
        
        @test length(result) == 0
        @test typeof(result) <: Vector
    end

    # Test with different prior distributions
    @testset "Different prior distributions" begin
        @model function test_normal_model(;
            y,
            priors::Dict = Dict(
                :μ => Normal(0, 1),
                :σ => Exponential(1)
            )
        )
            μ ~ priors[:μ]
            σ ~ priors[:σ]

            for i in eachindex(y)
                y[i] ~ Normal(μ, σ)
            end
        end

        task_data = (
            y = [missing, missing, missing],
        )
        
        priors = Dict(
            :μ => Normal(10, 2),
            :σ => Exponential(0.5)
        )
        
        result = prior_sample(
            task_data;
            model = test_normal_model,
            n = 1,
            priors = priors,
            outcome_name = :y
        )
        
        @test length(result) == 3
        @test all(x -> typeof(x) <: Real, result)
    end

end

@testset "optimize" begin

    # Test model for optimization
    @model function test_bernoulli_opt(;
        x,
        priors::Dict = Dict(
            :p => Beta(2, 2)
        )
    )
        p ~ priors[:p]

        for i in eachindex(x)
            x[i] ~ Bernoulli(p)
        end
    end

    # Test MAP estimation with Bernoulli model
    @testset "MAP estimation - Bernoulli" begin
        # Generate some synthetic data where true p ≈ 0.7
        data = (
            x = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],  # 8/10 = 0.8
        )
        
        priors = Dict(:p => Beta(2, 2))  # Uniform-like prior
        
        result = optimize(
            data;
            model = test_bernoulli_opt,
            estimate = "MAP",
            priors = priors,
            n_starts = 3
        )
        
        # Check that result has the expected structure
        @test :p in names(result.values)[1]
        @test 0 < result.values[:p] < 1  # p should be a valid probability
        @test result.lp isa Real  # log-probability should be a real number
        @test !isnan(result.lp)   # and not NaN
        
        # For this data with Beta(2,2) prior, MAP should be around (8+2-1)/(10+2+2-2) = 9/12 = 0.75
        @test 0.6 < result.values[:p] < 0.9  # Allow some tolerance due to optimization
    end

    # Test MLE estimation with Bernoulli model
    @testset "MLE estimation - Bernoulli" begin
        data = (
            x = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],  # 8/10 = 0.8
        )
        
        priors = Dict(:p => Beta(2, 2))
        
        result = optimize(
            data;
            model = test_bernoulli_opt,
            estimate = "MLE",
            priors = priors,
            n_starts = 3
        )
        
        @test :p in names(result.values)[1]
        @test 0 < result.values[:p] < 1
        @test result.lp isa Real
        @test !isnan(result.lp)
        
        # MLE should be closer to sample mean (0.8) than MAP
        @test 0.7 < result.values[:p] < 0.9
    end

    # Test with Normal model
    @model function test_normal_opt(;
        y,
        priors::Dict = Dict(
            :μ => Normal(0, 10),
            :σ => Exponential(1)
        )
    )
        μ ~ priors[:μ]
        σ ~ priors[:σ]

        for i in eachindex(y)
            y[i] ~ Normal(μ, σ)
        end
    end

    @testset "MAP estimation - Normal" begin
        # Generate data with known parameters
        Random.seed!(123)
        true_μ = 5.0
        true_σ = 2.0
        data = (
            y = true_μ .+ true_σ .* randn(50),
        )
        
        priors = Dict(
            :μ => Normal(0, 10),
            :σ => Exponential(1)
        )
        
        result = optimize(
            data;
            model = test_normal_opt,
            estimate = "MAP",
            priors = priors,
            n_starts = 5
        )
        
        @test :μ in names(result.values)[1]
        @test :σ in names(result.values)[1]
        @test result.values[:σ] > 0  # σ must be positive
        @test result.lp isa Real
        @test !isnan(result.lp)
        
        # Should recover approximately the true parameters
        @test abs(result.values[:μ] - true_μ) < 1.0  # Within reasonable tolerance
        @test abs(result.values[:σ] - true_σ) < 1.0
    end

    # Test with model that has kwargs
    @model function test_normal_with_kwargs(;
        y,
        fixed_σ::Float64 = 1.0,
        priors::Dict = Dict(
            :μ => Normal(0, 1)
        )
    )
        μ ~ priors[:μ]

        for i in eachindex(y)
            y[i] ~ Normal(μ, fixed_σ)
        end
    end

    @testset "Optimization with kwargs" begin
        # Data centered around μ = 3.0
        data = (
            y = [2.8, 3.2, 2.9, 3.1, 3.0, 2.7, 3.3, 2.9, 3.1, 3.0],
        )
        
        priors = Dict(:μ => Normal(0, 5))
        
        result = optimize(
            data;
            model = test_normal_with_kwargs,
            estimate = "MAP",
            priors = priors,
            n_starts = 3,
            fixed_σ = 0.5  # Pass kwargs to model
        )
        
        @test :μ in names(result.values)[1]
        @test result.lp isa Real
        @test !isnan(result.lp)
        
        # Should recover approximately μ ≈ 3.0
        @test 2.5 < result.values[:μ] < 3.5
    end

    # Test multistart functionality
    @testset "Multistart functionality" begin
        data = (
            x = [1, 1, 0, 1, 0],
        )
        
        priors = Dict(:p => Beta(1, 1))  # Uniform prior
        
        # Test with different numbers of starts
        result_1_start = optimize(
            data;
            model = test_bernoulli_opt,
            estimate = "MAP",
            priors = priors,
            n_starts = 1
        )
        
        result_5_starts = optimize(
            data;
            model = test_bernoulli_opt,
            estimate = "MAP",
            priors = priors,
            n_starts = 5
        )
        
        # Both should find valid solutions
        @test :p in names(result_1_start.values)[1]
        @test :p in names(result_5_starts.values)[1]
        @test 0 < result_1_start.values[:p] < 1
        @test 0 < result_5_starts.values[:p] < 1
        
        # The 5-start version should have log-probability >= 1-start version
        # (due to better optimization)
        @test result_5_starts.lp >= result_1_start.lp - 1e-6  # Allow small numerical tolerance
    end

    # Test error handling
    @testset "Error handling" begin
        data = (
            x = [1, 0, 1],
        )
        
        priors = Dict(:p => Beta(2, 2))
        
        # Test with invalid estimate type
        @test_throws UndefVarError optimize(
            data;
            model = test_bernoulli_opt,
            estimate = "INVALID",
            priors = priors
        )
    end

    # Test with minimal data
    @testset "Minimal data" begin
        data = (
            x = [1],  # Single observation
        )
        
        priors = Dict(:p => Beta(2, 2))
        
        result = optimize(
            data;
            model = test_bernoulli_opt,
            estimate = "MAP",
            priors = priors,
            n_starts = 2
        )
        
        @test :p in names(result.values)[1]
        @test 0 < result.values[:p] < 1
        @test result.lp isa Real
        @test !isnan(result.lp)
    end

    # Test with strong priors vs weak priors
    @testset "Prior strength effects" begin
        data = (
            x = [1, 1, 1, 1, 1],  # All successes
        )
        
        # Strong prior favoring p = 0.3
        strong_priors = Dict(:p => Beta(3, 7))  # Mode at 0.25
        
        # Weak prior
        weak_priors = Dict(:p => Beta(1, 1))    # Uniform
        
        result_strong = optimize(
            data;
            model = test_bernoulli_opt,
            estimate = "MAP",
            priors = strong_priors,
            n_starts = 3
        )
        
        result_weak = optimize(
            data;
            model = test_bernoulli_opt,
            estimate = "MAP",
            priors = weak_priors,
            n_starts = 3
        )
        
        # Strong prior should pull estimate away from 1.0 more than weak prior
        @test result_strong.values[:p] < result_weak.values[:p]
        @test result_strong.values[:p] < 0.8  # Should be pulled down by prior
    end

end