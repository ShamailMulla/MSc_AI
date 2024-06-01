#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random, operator, time, itertools, math, operator, string
from copy import deepcopy
from itertools import islice
import numpy as np

# useful for visualization
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from matplotlib import rc
rc("text", usetex=False)

import seaborn
seaborn.set(style='whitegrid')
seaborn.set_context('notebook')


# In[2]:


import password_fitness


def initialize_population(pop_size, individual_len):
    '''
    Method to create initial population
    '''
    population = []
    char_set = string.digits + string.ascii_uppercase + "_"

    for i in range(0, pop_size):
        individual = ''.join([random.choice(char_set) for _ in range(individual_len)])
        population.append(deepcopy(individual))

    return population


# In[5]:


def sort_population(population, student_password):
    fitnesses = password_fitness.get_normalised_fitness(population, student_password)
    sorted_dict = dict(sorted(fitnesses.items(), key=operator.itemgetter(1), reverse=True))
    
    return sorted_dict


# In[6]:


def stop_criterion(fitnesses):
    """
    Evaluate the fitness of all individuals & return if any individual has converged to the true password
    """
    if max(fitnesses) == 1:
        return True
    
    return False


# In[7]:


def crossover(candidates, offspring_pairs=3):
    """
    Applies crossing over on the selected individuals with a probability `prob`
    """
    children = []
    for index, indiv in enumerate(candidates):
        # If crossing over should be applied on this individual for the given crossover rate
        # print('\nindex',index)

        remaining = deepcopy(candidates)
        del remaining[index]
        # for each pair of parents create `offsprings` pair of children
        for i in range(offspring_pairs):
            other_idx = random.choice(range(len(remaining)))
            # We get its index in order to modify it directly
            other = remaining[other_idx]
            # print('Parents:',indiv,other)
            # Randomly choose a starting point to swap with the other individual
            # Keep the starting part of individual 1 for first offspring
            startingIdx = random.choice(range(1,len(indiv)-1))
            # print('Crossover at:',startingIdx)
            # print('indiv[:startingIdx]',indiv[:startingIdx])
            # print('other[startingIdx:]',other[startingIdx:])            
            offspring1 = indiv[:startingIdx]+other[startingIdx:]
            # print('offspring1',offspring1)    
            # print('other[startingIdx:]',other[startingIdx:])
            # print('indiv[:startingIdx]',indiv[:startingIdx])
            offspring2 = other[:startingIdx]+indiv[startingIdx:]
            # print('offspring2:', offspring2)
            children.append(offspring1)
            children.append(offspring2)
    
    return children


# In[8]:


# offsprings = crossover(['6_XIY4FM6O', 'CNQEW0DVP3','HLMO7OKBSH', '5C4_1UEURN', 'HLMO7OFM6O'],4)
# len(offsprings),set(offsprings)


# In[9]:


def mutate(candidates, prob):
    """
    Mutate provided individuals with `prob` probability
    """
    char_set = string.digits + string.ascii_uppercase + "_"
    mutated_candidates = []
    
    for index, indiv in enumerate(candidates):
        new_individual = ''
        for pos in range(len(indiv)):
            if random.random() <= prob:
                remaining_charset = char_set.replace(indiv[pos],'')
                new_individual += random.choice(remaining_charset)
            else:
                new_individual += indiv[pos]
        mutated_candidates.append(new_individual)                    

    return mutated_candidates


# In[10]:


def selection(population_fitnesses, pop_size):
    """
    Selecting the best offspring from the population
    """
    fittest = islice(population_fitnesses.items(), pop_size)

    return dict(fittest)


# In[11]:


def evolutionary_search(student_password, pop_size = 10, mut_prob=0.2, offspring_pairs=3, max_generations=1000):   
    population = initialize_population(pop_size, individual_len=10)
    population_fitnesses = sort_population(population, student_password)
    # print('\tBest initial fitness score:',list(population_fitnesses.values())[0])
    t = 0    
    
    while not stop_criterion(list(population_fitnesses.values())):
        if t == max_generations:
            return None, list(population_fitnesses.keys())[0]
        offsprings = crossover(list(population_fitnesses.keys()), offspring_pairs=offspring_pairs)
        offsprings = mutate(offsprings, prob=mut_prob)
        population_fitnesses = sort_population(offsprings, student_password)
        population_fitnesses = selection(population_fitnesses, pop_size)
        t = t+1
    
    # print('Final',population_fitnesses)
    return t, list(population_fitnesses.keys())[0]


# In[12]:

if __name__=='__main__':
    # Testing for population size 10
    # For each pair of parents create 3 pairs of children to maintain selection pressure ~7
    reproductions, passwords = [], []
    student_password = password_fitness.get_password('ec23678')
    for _ in range(500):
        t, password = evolutionary_search(student_password, pop_size=10, offspring_pairs=3)
        if t is not None:
            reproductions.append(t)
        passwords.append(password)
    reproductions = np.array(reproductions)
    print('Reproductions:\tMean:', reproductions.mean().round(4),'\tStandard deviation:', reproductions.std().round(4))
    print('Password/s:',set(passwords))
    
    
    # In[15]:
    
    
    # Hyperparameter tuning
    pop_sizes = [5, 10, 15, 20]
    offspring_pairs = [1, 2, 3, 4, 5]
    epochs = 500
    for p in pop_sizes:
        for o in offspring_pairs:
            print('\nPopulation size:',p,'\toffspring_pairs:',o)
            # Calculating selection pressure
            parents = initialize_population(p, individual_len=10)
            offsprings = crossover(parents, offspring_pairs=o)
            print('Number of offspring:',len(offsprings))
            print('Selection pressure:',len(offsprings)/len(parents))
            
            reproductions, passwords = [], []
            no_solution = 0
            
            for _ in range(epochs):
                t, password = evolutionary_search(student_password, pop_size=p, offspring_pairs=o)
                if t is not None:
                    reproductions.append(t)
                    passwords.append(password)
                else:
                    no_solution += 1
                    passwords.append(password)
            reproductions = np.array(reproductions)
            print('\tReproductions:\tMean:', reproductions.mean().round(4),'\tStandard deviation:', reproductions.std().round(4))
            print('\tFound no solutions %s of %s times'%(no_solution,epochs))
            print('\tGenerated password/s:',set(passwords))


# In[ ]:




