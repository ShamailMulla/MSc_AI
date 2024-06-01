#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
from collections import defaultdict
from queue import PriorityQueue
print('Last run:',np.datetime64('now'))




df = pd.read_csv('tubedata.csv', header=None)
df.columns = ['StartingStation', 'EndingStation', 'TubeLine', 'AverageTimeTaken', 'MainZone', 'SecondaryZone']
if __name__ == '__main__':
    df.sample(10)




station_dict = defaultdict(list)
station_tube_dict = defaultdict(list)
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
    # {"Mile End": [("Stepney Green", 2, "District"), ("Wembley Park", 1, "Metropolitan line")]}
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


# # 2.4 Heuristic search
# Given that you know the zone(s) a station is in, consider how you might use this information to focus
# the search in the right direction and implement your heuristic Best-First Search (BFS) (Note: not
# A* Search)



# Getting zones for starting and ending stations
starting_zone = df.StartingStation.map(zone_dict).values.tolist()
ending_zone = df.EndingStation.map(zone_dict).values.tolist()
for i in range(len(starting_zone)):
    zone_count = len(starting_zone[i])
    if zone_count == 1:
        z = list(starting_zone[i])[0]
    else:
        z = list(ending_zone[i])[0]
    starting_zone[i] = z
    
for i in range(len(ending_zone)):
    zone_count = len(ending_zone[i])
    if zone_count == 1:
        z = list(ending_zone[i])[0]
    else:
        z = list(starting_zone[i])[0]
    ending_zone[i] = z


df['starting_zone'] = starting_zone
df['ending_zone'] = ending_zone


# Data cleanup using tfl zones for the stations
zone_map = {'1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','a':'7','b':'8','c':'9','d':'10'}
df['MainZone'].replace(zone_map,inplace=True)
df['SecondaryZone'].replace(zone_map,inplace=True)
# df = df.astype({'MainZone':'int','SecondaryZone':'int'})

df['starting_zone'].replace(zone_map,inplace=True)
df['ending_zone'].replace(zone_map,inplace=True)
# df = df.astype({'starting_zone':'int','ending_zone':'int'})

zones = df['starting_zone'].unique()


#df[df['StartingStation']=='Chorleywood']
#
#
#df[df['MainZone']=='1'][df['SecondaryZone']=='2']
#
#
#df[df['MainZone']=='10']
#
#
#
#zone_dict['Notting Hill Gate']


def zone_travel_time():
    """
    This function calculates the average time taken to travel within the same zones
    """
    # Travelling within the same zone
    zone_travel_dict = {int(zone_i):{} for zone_i in zones}
    zone_data = df[df['starting_zone']==df['ending_zone']]
    no_zone_data = []
    for zone_i in zones:
        # Traveling in the same zone
        travel_times = zone_data[zone_data['starting_zone']==zone_i].AverageTimeTaken.values.tolist()
        if len(travel_times):
            zone_travel_dict[int(zone_i)]['min'] = min(travel_times)
            zone_travel_dict[int(zone_i)]['max'] = max(travel_times)
        else:
            zone_travel_dict[int(zone_i)]['min'] = zone_travel_dict[zone_i-1]['min']
            zone_travel_dict[int(zone_i)]['max'] = zone_travel_dict[zone_i-1]['max']

    return zone_travel_dict

travel_time = zone_travel_time()


def heuristic(node, goal, verbose=0):
    node_zone, goal_zone = list(zone_dict[node]), list(zone_dict[goal])
    node_zone = [int(zone_map[i]) for i in node_zone]    
    goal_zone = [int(zone_map[i]) for i in goal_zone]
    if verbose: print('INIT start zone',node_zone,'goal zone',goal_zone)
    # print('node_zone',node_zone,'goal_zone',goal_zone)
    # Resolving zones if there are multiple zones for the nodes
    # starting and/or ending point lies in 2 zones
    if len(node_zone) == 2 or len(goal_zone)==2:        
        if verbose: print('Node_zone or goal_zone may have secondary zones')
        node_zone.sort()
        goal_zone.sort()
        if len(node_zone) == 2 and len(goal_zone)==2:
            if verbose: print('Both node_zone and goal_zone have secondary zones')            
            # node_zone = [int(node_zone[0]),int(node_zone[1])]
            # goal_zone = [int(goal_zone[0]),int(goal_zone[1])]
            # if both points fall in 2 same zones - take minimum time to travel within the zone
            if node_zone == goal_zone:
                if verbose: print('Both zones are same',goal_zone)
                # if the primary and secondary zones are same for both nodes then take the minimum distance to travel in the zones
                return min(travel_time[goal_zone[0]]['min'],travel_time[goal_zone[1]]['min'])
            else:
                if verbose: print('Zones are differe',goal_zone)
                # getting the closer zones for starting and ending nodes
                min_diff = node_zone[0]-goal_zone[0]
                best_node = node_zone[0]
                best_goal = goal_zone[0]
                for z1 in node_zone:
                    for z2 in goal_zone:
                        diff = abs(z1-z2)
                        if diff<min_diff:
                            best_node = z1
                            best_goal = z2
                node_zone, goal_zone = best_node, best_goal
        elif len(node_zone) == 2:
            goal_zone = int(goal_zone[0])
            node_zone = [int(node_zone[0]),int(node_zone[1])]
            abs1 = abs(node_zone[0]-goal_zone)
            abs2 = abs(node_zone[1]-goal_zone)
            node_zone = node_zone[0] if abs1<abs2 else node_zone[1]
        else:
            node_zone = int(node_zone[0])
            goal_zone = [int(goal_zone[0]),int(goal_zone[1])]
            abs1 = abs(goal_zone[0]-node_zone)
            abs2 = abs(goal_zone[1]-node_zone)
            goal_zone = goal_zone[0] if abs1<abs2 else goal_zone[1]
    else:
        node_zone = int(node_zone[0])
        goal_zone = int(goal_zone[0])
        
    if verbose: print('PROCESSED start zone:',node_zone,'goal zone:',goal_zone)
    
    if node_zone == goal_zone:
        h_travel_time = travel_time[node_zone]['min']
    else:                   
        # Taking the minimum distance to travel in the 2 zones next to each other
        h_travel_time = travel_time[node_zone]['min']+travel_time[goal_zone]['min']
        if verbose: print('Initial:[node_zone][`min`]+[goal_zone][`min`]:=',travel_time[node_zone]['min'],travel_time[goal_zone]['min'],'=',h_travel_time)
        node_zone = int(node_zone)
        goal_zone = int(goal_zone)
        if abs(goal_zone-node_zone) > 1:
            # If the travel is through multiple zones
            zone1, zone2 = min(node_zone, goal_zone), max(node_zone, goal_zone)
            # Add max travel time to pass through each zone in between 
            zones_to_cross = range(zone1+1, zone2)
            for z in zones_to_cross:
                if verbose: print('zone',z,'time:',travel_time[z])
                h_travel_time += travel_time[z]['max']

                                    
    return h_travel_time



#if __name__ == '__main__':
#    # zone (1,2) to zone (1,2)
#    heuristic('Baker Street','Notting Hill Gate', verbose=True)
#
#    # zone 2 to 2
#    heuristic('Stepney Green','Mile End', verbose=True)
#
#    # zone 10 to zone 10
#    heuristic('Amersham','Chesham', verbose=True)
#
#    # zone (2,3) to zone 6
#    heuristic('Stratford','West Ruislip', verbose=True) 
#
#
#    # zone 6 to zone (2,3) 
#    heuristic('West Ruislip', 'Stratford', verbose=True)
#
#    # zone (2,3) to zone 1
#    heuristic('Stratford','Liverpool Street', verbose=True) 
#
#    # zone 1 to zone (2,3)
#    heuristic('Aldgate','Stratford', verbose=True) 
#
#    # zone (1,2) to zone 9
#    heuristic('Baker Street', 'Chorleywood', verbose=True)


def search(initial, goal, verbose=False):
    print('Running Best First Search')
    frontier = PriorityQueue()
    explored = {initial}
    frontier.put((0, initial, [[], 0, None])) # heuristic, node, (path, cost, tube_line)
    explored_nodes = 0

    while not frontier.empty():        
        _, element, station = frontier.get()
        path, tube_line = station[0], station[1]
        if verbose:        print('\nCurrent tube line',tube_line)
        explored.add(element)
        explored_nodes += 1

        if element == goal:
            cur_tube_line = path[0][-1]
            
            time_taken, line_changes = 0, 0
            for e, time, next_tube_line in path[1:]:
                time_taken += time
                if cur_tube_line != next_tube_line:
                    line_changes += 1
                    cur_tube_line = next_tube_line
            path = [(initial)]+path+[(e)]    
            return {'path':path, 'time_taken':time_taken, 'exploration_cost':explored_nodes,'line_changes':line_changes}

        children = station_dict[element]  
        
        for child in children: 
            child_tube_line = child[-1] 
            change_cost = 0
            if not tube_line: # start of the algorithm no tube line is selected
                if verbose:                    print('Assigning initial tube line to',tube_line)
            else:                
                if tube_line != child_tube_line: # add 1 minute to travel time to change lines
                    if verbose:                print('Change line for',element,'-',child[0],'to',child_tube_line)
                    change_cost = 1
                    
            total_cost = child[1]+change_cost
            heuristic_cost = heuristic(child[0],goal)
            
            node = (heuristic_cost, 
                    child[0], # name of the node to explore
                    (path+[(child[0], total_cost, child_tube_line+' line')], child_tube_line)) # adding child_name and parent and tube line
            
            if child[0] not in explored:
                # print('Putting node:',node)
                frontier.put(node)
                
    return {'node':None, 'time_taken':float('-inf'), 'exploration_cost':None, 'line_changes':None}




# Best first search TEST CASE 1: to go from start to destination
if __name__ == '__main__':    
    start, destination = 'Euston', 'Victoria'
    response = search(start, destination)
    print('Response:',response)


    # Best first search TEST CASE 2: to go in reverse direction of test case 1    
    start, destination = 'Victoria', 'Euston'
    response = search(start, destination)
    print('Response:',response)
    
    
    # Best First Search TEST CASE 3: Canada Water to Stratford
    start, destination = 'Canada Water', 'Stratford'
    response = search(start, destination)        
    print('Response:',response)
    
    # Best fisrt search TEST CASE 4: Wembley Park to Baker Street
    start, destination = 'Wembley Park', 'Baker Street'
    response = search(start, destination)        
    print('Response:',response)



