# DERL

Influence Maximization is one of the most researched areas in social network, that aims to find a subset of nodes in a network (seed nodes) that, when activated or influenced, would result in the maximum spread of information or influence throughout the network. Solving the influence maximization problem has practical implications in fields such as marketing, viral advertising, opinion formation, and even public health campaigns. By understanding the dynamics of influence within social networks and identifying key individuals who can act as influential spreaders, it becomes possible to design more effective strategies for information dissemination and behavior change in a targeted manner.

This problem is considered to be NP-hard, and therefore finding the optimal solution becomes computationally challenging and time-consuming as the network size grows. Various mathematical, Evolutionary Algorithms (EA) based and Machine Learning (ML) based models have been developed by researchers to tackle this problem and have produced appreciable results.

EAs, the go to models to solve the optimization problems are inspired from the Darwin’s theory of “Survival of the fittest”, where the algorithm starts from a random set of population (solution set) and improves them continuously across iterations in arriving at a better solution. Among the present EA models, Differential Evolution (DE) is one of the most powerful algorithms that uses a unique mutation technique and has proved itself in solving many optimization problems.

On the other hand - ML, a branch of Artificial Intelligence (AI) learns and analyses the patterns present in the data provided to produce results. Reinforcement Learning (RL), a part of ML paradigm involves training agents to make sequential decisions in an environment, with feedback provided through rewards or penalties.

This project aims at leveraging the strengths of EA and ML, by using an adaptive hybridization technique to combine the Differential Evolution and DQN-based Reinforcement Learning in solving the influence maximization problem present in social networks.
