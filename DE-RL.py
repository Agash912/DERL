import numpy as np, os, time, sys, random
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import random
import networkx as nx
from collections import namedtuple
from torch.autograd import Variable
import random, pickle
import Replay_Memory
import DE_dolphins as DE

start_time=time.time()

Ff = 0.01
CR = 0.5
max_iter = 10
pop_size = 100

def to_tensor(ndarray, volatile=False, requires_grad=False):
    return Variable(torch.from_numpy(ndarray).float(), volatile=volatile, requires_grad=requires_grad)

class DQN(nn.Module):

    def __init__(self, dim):
        super(DQN, self).__init__()
        l1 = 64
        self.w1 = nn.Linear(dim, l1)
        self.w2 = nn.Linear(l1, 1)

    def forward(self, input):
        out = self.w1(input)
        out = F.relu_(out)
        out = self.w2(out)

        return out

class Agent:
    def __init__(self, env):
       
        self.batch_size = 512 # 设置批次大小  Set batch size
        self.use_done_mask = True
        self.pop_size = 100 # 设置种群大小  Set population size
        self.buffer_size = 10000 # 设置缓存池大小 Set the buffer pool size
        self.randomWalkTimes = 20 # 设置基于DQN随机选点次数  Set the number of random selections based on DQN
        self.learningTimes = 10 # 设置基于DRL技术加速DQN训练的次数 Set the number of times to accelerate DQN training based on DRL technology
        self.dim = env.dim # 设置DQN输入层维数  Set DQN input layer dimension
        self.env = env # 初始化影响传播环境  Initialize the influence propagation environment
        self.evalStep = env.maxSeedsNum  # 基于种子节点数设置DQN选点次数 Set the number of DQN selections based on the number of seed nodes
        # 初始化DQN种群  Initialize the DQN population
        self.pop = []
        for _ in range(self.pop_size):
            self.pop.append(DQN(self.dim))

        self.all_fitness = []

        for dqn in self.pop:
            dqn.eval()
        self.rl_agent = DQN(self.dim)

        self.gamma = 0.8  # 设置更新比例
        self.optim = Adam(self.rl_agent.parameters(), lr=0.001) # 设置学习器
        self.loss = nn.MSELoss() # 设置使用均方误差作为损失函数
        self.replay_buffer = Replay_Memory.ReplayMemory(self.buffer_size) # 初始化缓冲池
        self.num_games = 0
        self.num_frames = 0
        self.gen_frames = 0

    def add_experience(self, state, action, next_state, reward, done):
        reward =to_tensor(np.array([reward])).unsqueeze(0)
        if self.use_done_mask:
            done =to_tensor(np.array([done]).astype('uint8')).unsqueeze(0)

        self.replay_buffer.push(state, action, next_state, reward, done)

    def evaluate(self, net, store_transition=True):
        total_reward = 0.0
        state = self.env.reset()
        state = to_tensor(state).unsqueeze(0)
        done = False
        seeds = []
        while not done:
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            Qvalues = net.forward(state)
            Qvalues = Qvalues.reshape((Qvalues.numel(),))
            sorted, indices = torch.sort(Qvalues, descending=True)
            
            actionNum = 0

            for i in range(state.shape[1]):
                if state[0][indices[i]][0].item() == 1:  
                    actionNum += 1
                    actionInt = indices[i].item()
                    seeds.append(actionInt)
                    action = torch.tensor([actionInt])

                    next_state, reward, done = self.env.step(actionInt)  

                    next_state = to_tensor(next_state).unsqueeze(0)

                    total_reward += reward
                    if store_transition: self.add_experience(state.cpu(), action, next_state.cpu(), reward, done)
                    state = next_state

                    if actionNum == self.evalStep or done:
                        break

        if store_transition: self.num_games += 1

        return total_reward, seeds

    def randomWalk(self):
        total_reward = 0.0
        state = self.env.reset()
        state = to_tensor(state).unsqueeze(0)
        done = False
        actionList = [i for i in range(self.env.nodeNum)]
        actionIndex = 0
        random.shuffle(actionList)
        while not done:
            self.num_frames += 1
            self.gen_frames += 1
            actionInt = actionList[actionIndex]
            action = torch.tensor([actionInt])
            next_state, reward, done = self.env.step(actionInt)  
            next_state =to_tensor(next_state).unsqueeze(0)
            total_reward += reward
            self.add_experience(state.cpu(), action, next_state.cpu(), reward, done)
            state = next_state
            actionIndex += 1
        self.num_games += 1
        return total_reward

    def rl_to_evo(self, rl_net, evo_net):
        for target_param, param in zip(evo_net.parameters(), rl_net.parameters()):
            target_param.data.copy_(param.data)

    def evaluate_all(self):
        self.all_fitness = []
        t1 = time.time()
        for net in self.pop:
            fitness, _ = self.evaluate(net)
            self.all_fitness.append(fitness)
        best_train_fitness = max(self.all_fitness)
        # print("fitness_init:", best_train_fitness)
        t2 = time.time()
        # print("evaluate finished.    cost time:", t2 - t1)

    def train(self):
        self.gen_frames = 0
        ####################### EVOLUTION #####################

        t1 = time.time()

        best_train_fitness = max(self.all_fitness)
        
        self.pop, self.all_fitness = self.get_offspring(self.pop, self.all_fitness)
        t2 = time.time()
        # print("epoch finished.    cost time:", t2 - t1)
        fitness_best, _ = self.evaluate(self.pop[0], True)

        ####################### DRL Learning #####################
        # rl learning step
        t1 = time.time()
        sol=[]
        for _ in range(self.learningTimes):
            index = random.randint(len(self.pop) // 2, len(self.pop) - 1)
            # 获得最优DQN
            self.rl_to_evo(self.pop[0], self.rl_agent)
            if len(self.replay_buffer) > self.batch_size * 2:
                transitions = self.replay_buffer.sample(self.batch_size)
                batch = Replay_Memory.Transition(*zip(*transitions))
                self.update_parameters(batch)
                fitness, _ = self.evaluate(self.rl_agent, True)
                if fitness_best < fitness:
                  self.rl_to_evo(self.rl_agent, self.pop[index])
                  self.all_fitness[index] = fitness
            
            for i in self.pop[0:len(self.pop) // 10]:
                f,s=self.evaluate(i)
                sol.append(s)
        t2 = time.time()
        # print("learning finished.    cost time:", t2 - t1)
        return best_train_fitness, sum(self.all_fitness) / len(self.all_fitness), self.rl_agent, self.pop[
                                                                                                 0:len(self.pop) // 10],sol

    def update_parameters(self, batch):
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = None
        if self.use_done_mask: 
            done_batch = torch.cat(batch.done)
            state_batch.volatile = False
            next_state_batch.volatile = True
            action_batch.volatile = False

        currentList = torch.Tensor([])
        currentList = torch.unsqueeze(currentList, 1)
        targetList = torch.Tensor([])
        targetList = torch.unsqueeze(targetList, 1)
        # DQN Update
        for state, action, reward, next_state, done in zip(state_batch, action_batch, reward_batch, next_state_batch,
                                                           done_batch):
            target = torch.Tensor([reward])
            if not done:
                next_q_values = self.rl_agent.forward(next_state)
                pred, idx = next_q_values.max(0)
                target = reward + self.gamma * pred

            target_f = self.rl_agent.forward(state)

            current = target_f[action]
            current = torch.unsqueeze(current, 1)
            target = torch.unsqueeze(target, 1)
            currentList = torch.cat((currentList, current), 0)
            targetList = torch.cat((targetList, target), 0)

        self.optim.zero_grad()
        dt = self.loss(currentList, targetList)
        dt.backward()
        nn.utils.clip_grad_norm(self.rl_agent.parameters(), 10000)
        self.optim.step()

    def get_offspring(self, pop, fitness_evals):
        all_pop = []
        fitness = []
        offspring = []
        offspring_fitness = []
        for i in range(len(pop)):
            all_pop.append(pop[i])
            fitness.append(fitness_evals[i])
        index_rank = sorted(range(len(fitness)), key=fitness.__getitem__)
        index_rank.reverse()
        for i in range(len(pop) // 2):
            offspring.append(all_pop[index_rank[i]])
            offspring_fitness.append(fitness[index_rank[i]])

        randomNum = len(all_pop) - len(pop) // 2
        randomList = list(range(randomNum))
        random.shuffle(randomList)
        for i in range(len(pop) // 2, len(pop)):
            index = randomList[i - len(pop) // 2]
            offspring.append(all_pop[index])
            offspring_fitness.append(fitness[index])
            ...
        return offspring, offspring_fitness

    def showScore(self, score):
        out = ""
        for i in range(len(score)):
            out = out + str(score[i])
            out = out + "\t"
        # print(out)

class Env:

    def __init__(self, maxSeedNum):

            self.dim = 64 + 3 # Set DQN output dimension
            self.nodeNum = 62 # number of nodes
            self.maxSeedsNum = maxSeedNum # set the seed size
            # self.networkName = 'Human Protein' # network dataset name
            self.nameList = [r"/home/g_jeyakumar/Agash/data/Data/Dolphins/dataset.txt"] # Network dataset path
            self.localInfluenceList = np.zeros(self.nodeNum)-1 # Record the influence score of each seed node
            self.oneHopInfluenceList = np.zeros(self.nodeNum)-1 # Record the influence score of each seed node
            self.graphIndex = -1 # dataset marker
            self.embedInfo = self.getembedInfo()
            self.graph, self.posi_graph, self.edges = self.constrctGraph(np.loadtxt(self.nameList[0]))

    def constrctGraph(self,edges):
        graph = nx.DiGraph()
        posi_graph = nx.DiGraph()
        for u, v in edges:
            u = int(u)
            v = int(v)
            p = 0.1
            posi_graph.add_edge(u, v, weight=p)
            graph.add_edge(u, v, weight=p)

        return graph, posi_graph, np.array(graph.edges())

    def reset(self):
        self.seeds = set([])
        self.influence = 0
        return self.seeds2input(self.seeds)

    def step(self, node):
        state = None
        if node in self.seeds:
            # print("choose repeated node!!!!!!!!!!!!!")
            state = self.seeds2input(self.seeds)
            return state, 0, False

        self.seeds.add(node)
        reward = self.fitness_function(list(self.seeds)) - self.influence

        self.influence += reward

        isDone = False
        if len(self.seeds) == self.maxSeedsNum:
            isDone = True

        state = self.seeds2input(self.seeds)
        return state, reward, isDone

    def seeds2input(self,seeds):
        input = np.array(self.embedInfo)
        flagList = np.array([])
        degreeList = np.array([])
        posi_degreeList = np.array([])
        for i in range(self.nodeNum):
            try:
                k = self.graph.out_degree[i+1]
                degreeList = np.append(degreeList, self.graph.out_degree[i+1])
            except:
                degreeList = np.append(degreeList,[0])
            # print(degreeList)
            try:
                posi_degreeList = np.append(posi_degreeList, self.posi_graph.out_degree[i+1])
            except:
                posi_degreeList = np.append(posi_degreeList, 0)
            if i in seeds:
                flagList = np.append(flagList, 0)
            else:
                flagList = np.append(flagList, 1)

        flagList = flagList.reshape((self.nodeNum,1))
        degreeList = degreeList.reshape((self.nodeNum, 1))
        posi_degreeList = posi_degreeList.reshape((self.nodeNum, 1))
        
        input = np.hstack((degreeList, input))
        input = np.hstack((posi_degreeList, input))
        input = np.hstack((flagList,input))
        return input

    def getembedInfo(self):

        # print("graph name == ", self.networkName)
        # print("seed num == ", self.maxSeedsNum)
        embedInfo = np.loadtxt(r"/home/g_jeyakumar/Agash/data/Data/Dolphins/dolphins_data_embedded.txt")
        return embedInfo

    def getInfluence(self, S):
        # 计算种子节点两跳以内激活其他节点的影响分数  
        # Calculate the influence score of activating other nodes within two hops of the seed node
        influence = 0
        try:
            for s in S:
                influence += self.getLocalInfluence(s)

            # 剔除种子节点一两跳以内激活其他种子节点的影响分数 
            # Eliminate the influence score of seed nodes that activate other seed nodes within one or two hops
            influence -= self.getEpsilon(S)
            # 剔除种子节点一两跳以内激活其他种子节点的影响分数
            # Eliminate the influence score of seed nodes that activate other seed nodes within one or two hops
            for s in S:
                Cs = set(self.graph.successors(s))
                S1 = S & Cs
                for s1 in S1:
                    influence -= self.graph[s][s1]['weight'] * self.getOneHopInfluence(s1)
            return influence
        except:
            return influence
    

    def getLocalInfluence(self, node):
        if self.localInfluenceList[node] >= 0:
            return self.localInfluenceList[node]

        result = 1
        try:
            Cu = set(self.graph.successors(node))
            for c in Cu:
                temp = self.getOneHopInfluence(c)
                Cc = set(self.graph.successors(c))
                if node in Cc:      # if egde (c,node) exits
                    temp = temp - self.graph[c][node]['weight']
                temp = temp * self.graph[node][c]['weight']
                result += temp
            self.localInfluenceList[node] = result
            return result
        except: 
            return 1

    def getOneHopInfluence(self, node):
        
        try: 
            if self.oneHopInfluenceList[node] >= 0:
                return self.oneHopInfluenceList[node]

            result = 1
            for c in self.graph.successors(node):
                result += self.graph[node][c]['weight']

            self.oneHopInfluenceList[node] = result
            return result
        except:
            return 1

    def getEpsilon(self, S):
        result = 0
        try:
            for s in S:
                Cs = set(self.graph.successors(s))
                S1 = Cs - S
                for c in S1:
                    Cc = set(self.graph.successors(c)) 
                    S2 = Cc & S
                    result += (0.01 * len(S2))
                return result
        except:
            return result
    
    def fitness_function(self,nodeset):
        influence=0
        seeds=set()
        for node in nodeset:
            seeds.add(node)
            reward = self.getInfluence(seeds) - influence
            influence += reward

        for node in nodeset:
            seeds.remove(node)
            
        return influence
    
def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
 
    if (a_set & b_set):
        return(a_set & b_set)
    else:
        return {}

def run(maxSeedsNum):
    # Create Env
    t1 = time.time()
    # print("start===========================")

    env = Env(maxSeedsNum)
    seed = 1 #123
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create Agent
    agent = Agent(env)

    time_start = time.time()
    print("Training started...")
    maxList = np.array([])
    resultList = np.array([])
    timeList = np.array([])
    solution_list=set()
    current_DE_pop = DE.initialize_population(100)
    global_best_fitness = 0
    global_best_seeds =[]

    for iter in range(10):  # Generation
        
        if iter == 0:
            agent.evaluate_all()
        print("=================================================================================")
        best_train_fitness, average, rl_agent, elitePop,sol = agent.train()
        print('#Games:', agent.num_games, '#Frames:', agent.num_frames, ' Epoch_Max:', '%.2f' % best_train_fitness,
                ' Avg:', average)
        if iter==0:
            solution_list = {tuple(a) for a in sol}
        else:
            solution_list.update({tuple(b) for b in sol})
        # print(solution_list)    
        maxList = np.append(maxList, best_train_fitness)

        RL_fitness, RL_seeds = agent.evaluate(agent.pop[0])
        print("The length of current population is: ",len(current_DE_pop))

        DE_seeds, DE_fitness, next_pop = DE.differential_evolution(pop_size, current_DE_pop, Ff, CR, max_iter)
        print("Received DE fitness is",DE_fitness)
        current_DE_pop = next_pop
        print("The length of next population is: ",len(next_pop))

        cmn = list(common_member(DE_seeds,RL_seeds))

        for ii in cmn:
            DE_seeds.remove(ii)

        # print("RL SEEDS",RL_seeds)

        final_seeds = []

        for k in range(len(RL_seeds)):
            temp = []
            temp.append(RL_seeds[k])
            temp.append(env.fitness_function((temp)))
            final_seeds.append(temp)

        for k in range(len(DE_seeds)):
            temp = []
            temp.append(DE_seeds[k])
            temp.append(env.fitness_function((temp)))
            final_seeds.append(temp)
        
        DERL_seeds = sorted(final_seeds, key=lambda x: x[1], reverse=True)

        final_seeds = []

        for k in range(len(DERL_seeds)):
            final_seeds.append(DERL_seeds[k][0])
        
        final_seeds = final_seeds[:10]
        
        fitness = env.fitness_function(final_seeds)

        print("In ",iter," generation, the fitness value is",fitness,"for the seeds:", final_seeds)
        
        if fitness>=global_best_fitness:
            global_best_fitness = fitness
            global_best_seeds = final_seeds
            # current_DE_pop[0] = final_seeds

        # elif fitness < global_best_fitness:
        #     current_DE_pop[50] = global_best_seeds

        current_DE_pop[90] = global_best_seeds
        
    return global_best_seeds, global_best_fitness

global_best_seeds, global_best_fitness = run(10)

print("The overall best seed set obtained is: ", global_best_seeds)
print("The overall best fitness obtained is: ", global_best_fitness)
print("Total time taken: ",(time.time() - start_time),"seconds")
