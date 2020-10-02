### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 2b4707a4-f479-11ea-053b-6bd2dfd53ff1
begin
	using PlutoUI
	using Random
	using Plots
	
	plotlyjs()

	md"""
	A Bernoulli multi-armed bandit can be described as a tuple of ``\langle \mathcal A, \mathcal R\rangle`` , where
	
	- We have K  machines with reward probabilities, ``\{\mu_1,…,\mu_K\}``
	- At each time step ``t``, we take an action a on one slot machine and receive a reward ``r``
	- ``\mathcal A`` is a set of actions, each referring to the interaction with one slot machine. The value of action a is the expected reward, ``Q(a)=\mathbb E[r|a]=\mu``. If action at at the time step ``t`` is on the ``i``-th machine, then ``Q(a_t)=\mu_i``
	- ``\mathcal R`` is a reward function. In the case of Bernoulli bandit, we observe a reward r in a stochastic fashion. At the time step ``t``, ``r_t=\mathcal R(a_t)`` may return reward 1 with a probability ``Q(a_t)`` or ``0`` otherwise.
	
	The goal is to maximize the cumulative reward ``\sum_{t=1}^T r_t`` 
	"""
end

# ╔═╡ f17fca0c-f58f-11ea-1ffc-8b6d75934e7d
import Pkg; Pkg.activate("robustrl"); Pkg.add("ORCA")

# ╔═╡ 6b5b3096-f47a-11ea-36c4-875225cbd662
md"""
``K`` $(@bind K Slider(1:100, default=10, show_value=true))

Horizon ``T`` $(@bind T Slider( 1:(10^4), default=10, show_value=true))

seed  $(@bind seed Slider(1:100, default=10, show_value=true))

"""

# ╔═╡ cad10452-f47c-11ea-041a-67111ea81689
begin
	Random.seed!(seed)
	μ = Random.rand(Float64, K)
	scatter(μ, title="μ", legend=false, size=(400, 100), ylims=(0, 1))
end

# ╔═╡ 99fe4624-f47b-11ea-0707-effeb9a029c1
"""
T: horizon
K: number of arms
"""
random_policy(T, K, past_actions, past_rewards) = Random.rand(1:K)

# ╔═╡ bfdd1230-f47b-11ea-3e41-5f5787acf0ac
evaluate_policy = π -> begin
	Random.seed!(seed)
	rewards = Float64[]
	regrets = Float64[]
	actions = Int64[]
	for t=1:T
		at = π(T, K, actions, rewards)
		
		p = μ[at]
		#rt = p + Random.randn()
		rt = (Random.rand() < p) * 1.
		regt = maximum(μ) - rt

		push!(actions, at)
		push!(rewards, rt)
		push!(regrets, regt)
	end
	# Our loss function is the total regret we might have by not selecting the optimal action up to the time step T
	regrets, rewards, actions
end

# ╔═╡ a09f6572-f47f-11ea-299c-e13d0d8f5c25
plot_evaluation_policy = (regrets, rewards, actions) -> begin
	function movingaverage(X::Vector,numofele::Int)
    BackDelta = div(numofele,2) 
    ForwardDelta = isodd(numofele) ? div(numofele,2) : div(numofele,2) - 1
    len = length(X)
    Y = similar(X)
    for n = 1:len
        lo = max(1,n - BackDelta)
        hi = min(len,n + ForwardDelta)
        Y[n] = sum(X[lo:hi]) / (hi-lo)
    end
    return Y
	end
	#plot_reward = plot(cumsum(rewards), title="cum reward", label="reward", size=(400, 200),  legend=(0.1, 0.9)) #ylims=(0, 100))
	T = size(regrets, 1)
	plot_regret = Plots.plot(movingaverage(regrets, 30), title="cum regret", label="regret",
		size=(400, 200),  legend=(0.1, 0.9), line=3) 
	plot!(diff(sqrt.(1:T)), ls=:dashed, label="sqrt(t)", line=(:dot, 1))
	h = histogram(actions)
	#histogram(regrets)
	#ylims=(-7, 7))
	plot(plot_regret, h, layout=(2,1))
	#plot(plot_reward, plot_regret, layout=(2, 1))
end

# ╔═╡ 2b3d6c00-f47c-11ea-3ad6-43bd408d9070
begin
	regrets, rewards, actions = evaluate_policy(random_policy)
	plot_evaluation_policy(regrets, rewards, actions)
	title!("Random policy")
end

# ╔═╡ 534d3040-f47c-11ea-0a75-edd2d6d0d044
begin
	ε_greedy = ε -> (T, K, past_actions, past_rewards) -> begin
		if sort(unique(past_actions)) != 1:K || Random.rand() < ε
			return Random.rand(1:K)
		end
		N = [ sum(past_actions .== a) for a=1:K ]
		μ_hat = 1 ./ N .* [ sum(rt for (rt,at)=zip(past_rewards, past_actions) if at == a) for a=1:K ]
		argmax(μ_hat)
	end
	
	md"``\varepsilon``-greedy policy with ``\varepsilon`` =  $(@bind ε Slider(range(0, stop=1/sqrt(T), length=11), default=.05, show_value=true))"
	
	
end

# ╔═╡ cdfa94ba-f47f-11ea-23ca-1d68a54fde5e
begin
		plot_evaluation_policy(evaluate_policy(ε_greedy(ε))...)
		title!("Greedy policy")
end

# ╔═╡ 5ef0b212-f492-11ea-0e07-3ddcf6d3c05a
begin
	
	function ucb_policy(ucb_multiplier)
		(T, K, past_actions, past_rewards) -> begin
		# If we have not explored all the arms yet
		if sort(unique(past_actions)) != 1:K
			return Random.rand(1:K)
		end
		# current round
		t = size(past_actions, 1) + 1
		
		N = [ sum(past_actions .== a) for a=1:K ]
		μ_hat = 1 ./ N .* [ sum(rt for (rt,at)=zip(past_rewards, past_actions) if at == a) for a=1:K ]
		upper_bound = sqrt.( ucb_multiplier .* log(t) ./ N )
		argmax(μ_hat + upper_bound)
		end
	end
	md"UCB multiplier $(@bind ucb_mult Slider(range(0, stop=3, length=10), default=1, show_value=true))"
end

# ╔═╡ 089a05a0-f493-11ea-2712-31b7c19ccf89
begin
		plot_evaluation_policy(evaluate_policy(ucb_policy(ucb_mult))...)
		title!("UCB policy")
end

# ╔═╡ Cell order:
# ╠═f17fca0c-f58f-11ea-1ffc-8b6d75934e7d
# ╠═2b4707a4-f479-11ea-053b-6bd2dfd53ff1
# ╟─6b5b3096-f47a-11ea-36c4-875225cbd662
# ╠═cad10452-f47c-11ea-041a-67111ea81689
# ╟─99fe4624-f47b-11ea-0707-effeb9a029c1
# ╟─bfdd1230-f47b-11ea-3e41-5f5787acf0ac
# ╟─a09f6572-f47f-11ea-299c-e13d0d8f5c25
# ╠═2b3d6c00-f47c-11ea-3ad6-43bd408d9070
# ╠═534d3040-f47c-11ea-0a75-edd2d6d0d044
# ╟─cdfa94ba-f47f-11ea-23ca-1d68a54fde5e
# ╟─5ef0b212-f492-11ea-0e07-3ddcf6d3c05a
# ╠═089a05a0-f493-11ea-2712-31b7c19ccf89
