import networkx as nx
import random
import numpy as np
import time
import matplotlib.pyplot as plt

start_time=time.time()
seed_set_size = 10

F = 0.01
CR = 0.5
max_iter = 10
pop_size = 100

graph_file = '/home/g_jeyakumar/Agash/data/Data/Dolphins/dataset.txt'
edges = np.loadtxt(graph_file)

G = nx.DiGraph()
for u, v in edges:
    u = int(u)
    v = int(v)
    p = 0.1
    G.add_edge(u, v, weight=p)

nodeNum = len(G.nodes())
seeds=set()

localInfluenceList = np.zeros(nodeNum)-1
oneHopInfluenceList = np.zeros(nodeNum)-1 

def getOneHopInfluence(node):
    try:
        if oneHopInfluenceList[node] >= 0:
            return oneHopInfluenceList[node]

        result = 1
        for c in G.successors(node):
            result += G[node][c]['weight']

        oneHopInfluenceList[node] = result
    except:
        result = 1
    return result

def getLocalInfluence(node):
    try:
        if localInfluenceList[node] >= 0:
            return localInfluenceList[node]
    except:
        return 1

    result = 1
    Cu = set(G.successors(node))
    for c in Cu:
        temp = getOneHopInfluence(c)
        Cc = set(G.successors(c))
        if node in Cc:      # if egde (c,node) exits
                temp = temp - G[c][node]['weight']
        temp = temp * G[node][c]['weight']
        result += temp
    localInfluenceList[node] = result
    return result

result=0

def getEpsilon(S):
    result = 0
    for s in S:
        Cs = set(G.successors(s))
        S1 = Cs - S
        for c in S1:
            Cc = set(G.successors(c)) 
            S2 = Cc & S
            result += (0.01 * len(S2))
    return result

def getInfluence(S):
    # Calculate the influence score of activating other nodes within two hops of the seed node
    influence = 0
    for s in S:
        influence += getLocalInfluence(s)

    # Eliminate the influence score of activating other seed nodes within one or two hops of the seed node
    influence -= getEpsilon(S)
    # Eliminate the influence score of activating other seed nodes within one or two hops of the seed node
    for s in S:
        Cs = set(G.successors(s))
        S1 = S & Cs
        for s1 in S1:
            influence -= G[s][s1]['weight'] * getOneHopInfluence(s1)
    return influence
influence=0

def fitness_function(nodeset):
    influence=0

    for node in nodeset:
        seeds.add(node)
        reward = getInfluence(seeds) - influence
        influence += reward
        
    for node in nodeset:
        try:
            seeds.remove(node)
        except:
            continue
        
    return influence

nodes_final = []

def create():

    nodes_num = list(range(1, len(G.nodes())))

    for each in nodes_num:
        if G.has_node(each):
            nodes_final.append(each)

    indi = random.sample(nodes_final,seed_set_size)
    return (indi)

def initialize_population(popSize):
    population = []

    for i in range(0, popSize):
        population.append(create())
        
    return population

def mutation(population, F):
    new_population = []
    for x in (population):

        a, b, c = random.sample(population, 3)
        v = [a[j] + F * (b[j] - c[j]) for j in range(len(x))]
        v = [round(num) for num in v]

        for i in range(len(v)):
            if G.has_node(v[i]):
                continue
            else:
                a = set(nodes_final)
                b = set(v)
                c = list(a-b)
                v[i] = random.choices(c)[0]
                
        new_population.append(v)

    return new_population

def crossover(population, mutated_population, CR):

    new_population = []

    for x in (population):

        u = x.copy()
        j = random.randint(0, len(x) - 1)

        for i in range(len(x)):

            if i == j or random.random() < CR:
                
                u[i] = mutated_population[random.randint(0, len(mutated_population) - 1)][i]

        new_population.append(u)

    return new_population

def rankRoutes(population):
    fitnessResults = []
    for i in range(0,len(population)):
        # fitnessResults[i] = fitness(population[i])
        temp=[]
        temp.append(population[i])
        temp.append(fitness_function(population[i]))
        fitnessResults.append(temp)
        temp=[]

    k = sorted(fitnessResults,key=lambda l:l[1], reverse=True)

    for i in range(0,len(k)):
        k[i]=k[i][0]

    return k

x_axis = [x for x in range(max_iter)]
scores=[]
arr=[]

def differential_evolution(pop_size, population, F, CR, max_iter):

    for each in population:
        if len(each)!=10:
            population.remove(each)
    
    print("Received number of pop", len(population))

    population = rankRoutes(population)
    counter = 0 
    best_solution = population[0]
    best_fitness = fitness_function(best_solution)

    num = round(0.1*pop_size)

    ranked = population[:num]
    
    for i in range(max_iter):

        # Perform the mutation
        mutated_population = mutation(population, F)
        # Perform the crossover
        crossed_population = crossover(population, mutated_population, CR)
        # Perform the selection

        ranked_routes = rankRoutes(crossed_population)
        counter+=len(population)

        if fitness_function(ranked_routes[0]) > best_fitness:
            best_solution = ranked_routes[0]
            best_fitness = fitness_function(ranked_routes[0])

        population = ranked_routes[: len(ranked_routes) - num]

        for each in ranked:
            population.append(each)
        
        ranked_routes = rankRoutes(population)
        score = fitness_function(ranked_routes[0])
        scores.append(score)
        
        values = ranked_routes[0]
        arr.append(values)

        ranked = rankRoutes(population)[:num]

    return best_solution, best_fitness, population

# ini = initialize_population(100)
# print(differential_evolution(100, ini, F = 0.01
# ,CR = 0.5,max_iter = 10))
