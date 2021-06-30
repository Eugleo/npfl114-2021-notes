# Lekce 13

Show how to incrementally update a running average (how to compute an average of $N$ numbers using the average of the first $N-1$ numbers). [5]

Describe multi-arm bandits and write down the $\varepsilon$-greedy algorithm for solving it. [5]

Define the Markov Decision Process, including the definition of the return. [5]

Define the value function, such that all expectations are over simple random variables (actions, states, rewards), not trajectories. [5]

Define the action-value function, such that all expectations are over simple random variables (actions, states, rewards), not trajectories. [5]

Express the value function using the action-value function, and express the action-value function using the value function. [5]

Define the optimal value function and the optimal action-value function. Then define optimal policy in such a way that its existence is guaranteed. [5]

Write down the Monte-Carlo on-policy every-visit $\varepsilon$-soft algorithm. [10]

Formulate the policy gradient theorem. [5]

Prove the part of the policy gradient theorem showing the value of $\nabla_\theta v_\pi (s)$. [10]

Assuming the policy gradient theorem, formulate the loss used by the REINFORCE algorithm and show how can its gradient be expressed as an expectation over states and actions. [5]

Write down the REINFORCE algorithm. [10]

Show that introducing baseline does not influence validity of the policy gradient theorem. [5]

Write down the REINFORCE with baseline algorithm. [10]