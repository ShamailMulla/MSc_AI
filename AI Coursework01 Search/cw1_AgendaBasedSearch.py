#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from collections import defaultdict
from queue import PriorityQueue



df = pd.read_csv('tubedata.csv', header=None)
df.columns = ['StartingStation', 'EndingStation', 'TubeLine', 'AverageTimeTaken', 'MainZone', 'SecondaryZone']
df.sample(10)



station_dict = defaultdict(list)
zone_dict = defaultdict(set)

# get data row by row
for index, row in df.iterrows():  
    start_station = row[0]
    end_station = row[1]
    tube_line = row[2]
    act_cost = int(row[3])

    zone1 = row[4]
    zone2 = row[5]

    # station dictionary of child station tuples (child_name, cost from parent to the child & tube line used)
    # {"Mile End": [("Stepney Green", 2), ("Wembley", 1)]}
    station_list = station_dict[start_station]
    station_list.append((end_station, act_cost, tube_line))

    # the following two lines add the other direction of the tube "step"
    station_list = station_dict[end_station]
    station_list.append((start_station, act_cost, tube_line))

    # we add the main zone
    zone_dict[start_station].add(zone1)
    # we add the secondary zone
    if zone2 != "0":
        zone_dict[start_station].add(zone2)
        # if the secondary zone is not 0 it's the main zone for the ending station
        zone_dict[end_station].add(zone2)
    else:
        # otherwise the main zone for the ending station is the same as for the starting station
        zone_dict[end_station].add(zone1)


# # 2.1 Implement DFS, BFS and UCS


def construct_path_from_root(node, root, verbose=False):
    """
    This function returns the path from root to destination with the cost
    """
    print('construct_path_from_root')
    path_to_root = []
    cost = 0
    cur_line = node['line']
    line_changes = 0
    
    while node['parent']:
        if verbose:
            print(node['name'],node['line'])
        cost += node['name'][1]
        path_to_root.append((node['name'][0],node['line']))
        node = node['parent']  
        if cur_line != node['line']:
            line_changes += 1
            cur_line = node['line']
    
    path_to_root.append((node['name'][0]))
    
    # reverse the list
    return {'path':list(reversed(path_to_root)), 'time_taken':cost, 'line_changes':line_changes}



def dfs(initial:str, goal:str, verbose=False):
    """
    This method performs depth first search and turns the path, cost and nodes explored
    """
    print('Running dfs for',initial,'to',goal)
    frontier = [{'name':(initial, 0), 'parent':None, 'line':''}]
    explored = {initial}
    explored_nodes = 0
    
    while frontier:            
        element = frontier.pop() # pop from stack (at the end)            
        if verbose:        print('\nExploring Element:',element)
        explored_nodes += 1
        
        if element['name'][0] == goal:
            if verbose:            print('Returning',element['name'],element['line'])
            return {'node':element, 'exploration_cost':explored_nodes}
        
        children = station_dict[element['name'][0]]
        # print('Children',children)
        for child in children:
            if verbose:            print('Child:',child)
            node = {'name': child, 'parent': element, 'line':child[-1]+' line'}
            if child[0] not in explored:
                # print('Exploring this child')
                frontier.append(node) # push child into end of the stack
                explored.add(child[0])      
        
    return {'node':None, 'exploration_cost':None}


if __name__=='__main__':
    # DFS TEST CASE 1: Running DFS to go from start to destination
    start, destination = 'Euston', 'Victoria'
    response = dfs(start, destination)
    # print('Response:',type(response))
    if response is not None:
        result = construct_path_from_root(response['node'], destination)
    else:
        result = [], None
        
    print('PATH:',result)
    print('Explored Nodes:',response['exploration_cost'])
    
    
    # In[7]:
    
    
    # DFS TEST CASE 2: Running DFS for to go in the opposite direction
    start, destination = 'Victoria', 'Euston'
    response = dfs(start, destination)
    # print('Response:',type(response))
    if response is not None:
        result = construct_path_from_root(response['node'], destination)
    else:
        result = [], None
        
    print('PATH:',result)
    print('Explored Nodes:',response['exploration_cost'])
    
    
    # In[8]:
    
    
    # DFS TEST CASE 3: Canada Water to Stratford
    start, destination = 'Canada Water', 'Stratford'
    response = dfs(start, destination)
    # print('Response:',type(response))
    if response is not None:
        result = construct_path_from_root(response['node'], destination)
    else:
        result = [], None
        
    print('PATH:',result)
    print('Explored Nodes:',response['exploration_cost'])
    
    
    # In[9]:
    
    
    # DFS TEST CASE 4: Wembley Park TO Baker Street
    start, destination = 'Wembley Park', 'Baker Street'
    response = dfs(start, destination)
    
    if response is not None:
        result = construct_path_from_root(response['node'], destination)
    else:
        result = [], None
        
    print('PATH:',result)
    print('Explored Nodes:',response['exploration_cost'])
    
    
    
def bfs(initial, goal, verbose=False):
    """
    This method performs breadth first search and turns the path, cost and nodes explored
    """
    print('Running bfs')
    frontier = [{'name':(initial, 0), 'parent':None, 'line':''}]
    explored = {initial}
    explored_nodes = 0
    
    while frontier:
        element = frontier.pop(0) # pop from front of queue
        explored_nodes += 1
        
        if element['name'][0] == goal:
            return {'node':element, 'exploration_cost':explored_nodes}
        
        children = station_dict[element['name'][0]]
        for child in children:
            node = {'name': child, 'parent': element, 'line':child[-1]}
            if child[0] not in explored:
                frontier.append(node) # put child into end of the queue
                explored.add(child[0])      
        
    return {'node':None, 'exploration_cost':None}


if __name__=='__main__':
    # BFS TEST CASE 1: Go from start to destination
    start, destination = 'Euston', 'Victoria'
    response = bfs(start, destination)
    # print('Response:',type(response))
    if response is not None:
        result = construct_path_from_root(response['node'], destination)
    else:
        result = [], None
        
    print('PATH:',result)
    print('Explored Nodes:',response['exploration_cost'])
    
    
    # In[12]:
    
    
    # BFS TEST CASE 2: Go in the opposite direction of test case 1
    start, destination = 'Victoria', 'Euston'
    response = bfs(start, destination)
    # print('Response:',type(response))
    if response is not None:
        result = construct_path_from_root(response['node'], destination)
    else:
        result = [], None
        
    print('PATH:',result)
    print('Explored Nodes:',response['exploration_cost'])
    
    
    # In[13]:
    
    
    # BFS TEST CASE 3: Canada Water to Stratford
    start, destination = 'Canada Water', 'Stratford'
    response = bfs(start, destination)
    # print('Response:',type(response))
    if response is not None:
        result = construct_path_from_root(response['node'], destination)
    else:
        result = [], None
        
    print('PATH:',result)
    print('Explored Nodes:',response['exploration_cost'])
    
    
    # In[14]:
    
    
    # BFS TEST CASE 4: Wembley Park TO Baker Street
    start, destination = 'Wembley Park', 'Baker Street'
    response = bfs(start, destination)
    
    if response is not None:
        result = construct_path_from_root(response['node'], destination)
    else:
        result = [], None
        
    print('PATH:',result)
    print('Explored Nodes:',response['exploration_cost'])


def ucs(initial, goal, verbose=False):
    print('Running UCS')
    frontier = PriorityQueue()
    explored = {initial}
    frontier.put((0, initial, [[], None])) # cost, node, (path, tube_line)
    explored_nodes = 0

    while not frontier.empty():
        if verbose:        print('\nCurrent tube line',tube_line)
        cost, element, station = frontier.get()
        path, tube_line = station[0], station[1]
        explored.add(element)
        explored_nodes += 1

        if element == goal:   
            cur_tube_line = path[0][-1]
            line_changes = 0
            for _, _, next_tube_line in path[1:]:
                if cur_tube_line != next_tube_line:
                    line_changes += 1
                    cur_tube_line = next_tube_line
            path = path+[(element,cost)]
                
            return {'path':path, 'time_taken':cost, 'exploration_cost':explored_nodes,'line_changes':line_changes}

        children = station_dict[element]
        for child in children:
            total_cost = child[1]+cost
            child_tube_line = child[-1]                        
            node = (total_cost, child[0], (path+[(element,cost,child_tube_line+' line')], child_tube_line)) # adding child_name and parent and tube line
            if child[0] not in explored:
                frontier.put(node)
                
    return {'node':None, 'time_taken':cost, 'exploration_cost':None, 'line_changes':None}


if __name__=='__main__':
    # UCS TEST CASE 1: Running UCS to go from start to destination
    start, destination = 'Euston', 'Victoria'
    response = ucs(start, destination)
    print('Response:',response)
    
    # UCS TEST CASE 2: Running UCS to go in reverse direction of test case 1
    start, destination = 'Victoria', 'Euston'
    response = ucs(start, destination)
    print('Response:',response)
    
    
    # UCS TEST CASE 3: Canada Water to Stratford
    start, destination = 'Canada Water', 'Stratford'
    response = ucs(start, destination)
        
    print('Response:',response)
    
    
    # UCS TEST CASE 4: Wembley Park to Baker Street
    start, destination = 'Wembley Park', 'Baker Street'
    response = ucs(start, destination)
    
    
# # 2.3 Extending the cost function
# Improve and implement the current UCS cost function to include the time to change lines at one station
    
    
def ucs_updated(initial, goal, verbose=False):
    print('Running UCS UPDATED')
    frontier = PriorityQueue()
    explored = {initial}
    frontier.put((0, initial, [[], None])) # cost, node, (path, tube_line)
    explored_nodes = 0

    while not frontier.empty():
        cost, element, station = frontier.get()
        path, tube_line = station[0], station[1]
        if verbose:        print('\nCurrent tube line',tube_line)
        explored.add(element)
        explored_nodes += 1

        if element == goal:
            cur_tube_line = path[0][-1]
            line_changes = 0
            for _, _, next_tube_line in path[1:]:
                if cur_tube_line != next_tube_line:
                    line_changes += 1
                    cur_tube_line = next_tube_line
            path = path+[(element,cost)]
                
            return {'path':path, 'time_taken':cost, 'exploration_cost':explored_nodes,'line_changes':line_changes}

        children = station_dict[element]
        for child in children:            
            child_tube_line = child[-1]    
            travel_cost = cost
            if tube_line is None:
                if verbose:                    print('Assigning initial tube line to',tube_line)
            else:
                if verbose:                print('Change line for',element,'-',child[0],'to',child_tube_line)
                if tube_line != child_tube_line: # add 1 minute to travel time to change lines
                    travel_cost += 1
            total_cost = child[1]+travel_cost
            node = (total_cost, child[0], (path+[(element, travel_cost, child_tube_line+' line')], child_tube_line)) # adding child_name and parent and tube line
            if child[0] not in explored:
                frontier.put(node)
                
    return {'node':None, 'time_taken':cost, 'exploration_cost':None, 'line_changes':None}


if __name__=='__main__':
    # UCS TEST CASE 1: Running UCS to go from start to destination
    start, destination = 'Euston', 'Victoria'
    response = ucs_updated(start, destination)
    print('Response:',response)
    
    
    # UCS TEST CASE 2: Running UCS to go in reverse direction of test case 1
    start, destination = 'Victoria', 'Euston'
    response = ucs_updated(start, destination)
    print('Response:',response)
    
    
    # UCS TEST CASE 3: Canada Water to Stratford
    start, destination = 'Canada Water', 'Stratford'
    response = ucs_updated(start, destination)
        
    print('Response:',response)
    
    
    # UCS TEST CASE 4: Wembley Park to Baker Street
    start, destination = 'Wembley Park', 'Baker Street'
    response = ucs_updated(start, destination)
        
    print('Response:',response)

    



