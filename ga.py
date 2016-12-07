#! /usr/bin/python
import random, copy, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")
df.drop("Id", axis=1, inplace=True)
data = pd.get_dummies(df)
X, Y = data.iloc[:,1:].values, data["Score"].values
allY = {'Y':Y, 'log':np.log(Y + 1), 'sqrt':np.sqrt(Y), 'sq':Y**2}
RANGE_Y = range(0, 71)

n = int(X.shape[0] * 0.8)

dtrain = xgb.DMatrix(X[:n], label=Y[:n])
dtest = xgb.DMatrix(X[n:], label=Y[n:])

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

gen_objective = lambda : random.choice(['reg:linear', 'count:poisson'])
gen_max_depth = lambda : random.randint(1, 10)
gen_eta = lambda : random.random()
gen_gamma = lambda : random.randint(0, 10)
gen_funcs = [gen_objective, gen_max_depth, gen_eta, gen_gamma]
gen_splits = lambda : [-1] + sorted(random.sample(range(1, 20), 4) + random.sample(range(20, 70), 3)) + [75]
def gen_param():
    param = list()
    for func in gen_funcs:
        param.append(func())
    param.extend(gen_splits)
    return param

IND_SIZE=1
toolbox = base.Toolbox()
# toolbox.register("attr_objective", random.choice, ['reg:linear', 'count:poisson'])
# toolbox.register("attr_max_depth", random.randint, 1, 10)
# toolbox.register("attr_eta", random.random)
# toolbox.register("attr_gamma", random.randint, 0, 10)
toolbox.register('gen_param', gen_param)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.gen_param)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
def evaluate(indiv):
    param = {'objective': indiv[0],
        'max_depth': indiv[1],
        'eta': indiv[2],
        'gamma': indiv[3],
        'silent': 1,
        'nthread': 4,
        'save_period': 0,
        'eval_metric': 'rmse'}
    Y_bins = pd.cut(Y, indiv[4:])
    num_round = 1502
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=150, verbose_eval=50)
    print(bst.best_iteration, bst.best_ntree_limit, bst.best_score)
    return bst.best_score,

def mut_indiv(indiv, indiv_pb):
    for i, ele in enumerate(indiv):
        if random.random() < indiv_pb:
            indiv[i] = gen_funcs[i]()

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mut_indiv, indiv_pb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

NGEN = 100
CXPB, MUTPB = 0.5, 0.2
pop = toolbox.population(n=300)
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

fisrt_gen = copy.deepcopy(pop)
pickle.dump(fisrt_gen, open('pop.pkl', 'wb'))

all_gens = [fisrt_gen]

def get_best(pop):
    best, best_fitness = pop[1], pop[1].fitness.values
    for indiv in pop[1:]:
        if indiv.fitness.values < best_fitness:
            best_fitness = indiv.fitness.values
            best = indiv
    return best, best_fitness

for g in range(NGEN):
    print("-- Generation %i --" % g)
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop[:] = offspring
    all_gens.append(copy.deepcopy(pop))

    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)
