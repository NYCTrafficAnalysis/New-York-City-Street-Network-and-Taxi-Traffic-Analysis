import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import collections
import scipy as sp
from scipy import stats
from collections import Counter
import folium
G = nx.read_graphml("/Users/pangyanbei/Desktop/final_project/project_data/manhatten.graphml")

#load data from OSRM
train_osrm_1 = pd.read_csv('/Users/pangyanbei/Desktop/final_project/project_data/fastest_routes_train_part_1.csv')
train_osrm_2 = pd.read_csv('/Users/pangyanbei/Desktop/final_project/project_data/fastest_routes_train_part_2.csv')
train_osrm = pd.concat([train_osrm_1, train_osrm_2])
train_osrm_new=train_osrm[['id', 'total_distance','total_travel_time', 'number_of_steps','step_direction','step_location_list']]

#load data from the original train data set
train_df = pd.read_csv('/Users/pangyanbei/Desktop/final_project/project_data/train.csv')

#merge the two datasets
train = pd.merge(train_df, train_osrm_new, on = 'id', how = 'left')
train_df = train.copy()

#compute the difference of traveling time between OSRM datasets and original datasets
#based on time
train_df['pickup_datetime'] = pd.to_datetime(train_df.pickup_datetime)
train_df["actual_time_cost"]=train_df["trip_duration"]
train_df["estimated_time_cost"]=train_df["total_travel_time"]
train_df['month'] = train_df['pickup_datetime'].dt.month
def travel_time_difference():
    x_data=train_df['month']
    y_actual=train_df["actual_time_cost"].groupby(train_df['month']).mean()
    y_estimated=train_df["estimated_time_cost"].groupby(train_df['month']).mean()
    plt.figure()
    plt.plot(y_actual,color='r',linewidth=2,label="actual_time_cost")
    plt.plot(y_estimated,'--',color='g',linewidth=2,label="estimated_time_cost_by_OSRM")
    plt.xlabel('Month')
    plt.ylabel('Duration')
    ax=plt.gca()
    plt.legend(loc=5)
    ax.set_ylim([0,1200])
    #plt.show()

#build undirected graph for Manhattan street network
G_undirected = nx.Graph(G)

#create single node degree distribution 
def node_distribution_1():
    degree_dic = Counter(dict(G_undirected.degree()).values())
    degree_hist = pd.DataFrame({"degree": list(degree_dic.values()),
                            "Number of Nodes": list(degree_dic.keys())})
    plt.figure(figsize=(20,10))
    sns.barplot(y = 'degree', x = 'Number of Nodes', 
              data = degree_hist, 
              color = 'green')
    plt.xlabel('Node Degree', fontsize=30)
    plt.ylabel('Number of Nodes', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=20)
    #plt.show() 

#compute the diameter of the undirected graph
def diameter:
    return nx.diameter(G_undirected)

#compute the average length of the shortest path of the undirected graph
def average_shortest_path:
    return nx.average_shortest_path_length(G_undirected)

#create node degree distribution 
def node_distribution_2():
    degrees  = G_undirected.degree()
    dic_h = Counter(dict(degrees).values())
    dic = collections.OrderedDict(sorted(dic_h.items()))
    degree_hist = list(dic.values())
    degree_values =list(dic.keys())
    
    plt.figure(figsize=(20,15))
    ax=plt.gca()
    ax.set_xlim([0,7])
    plt.plot(degree_values, degree_hist, 'bv-')
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.title("Degree Distribution of Manhattan Street Network")
    #plt.show()

#create distribution for indegree and outdegree
def node_distribution_in_out():
    in_degrees  = G.in_degree() 
    in_h = Counter(dict(in_degrees).values())
    in_dic = collections.OrderedDict(sorted(in_h.items()))
    in_hist = list(in_dic.values())
    in_values =list(in_dic.keys())

    out_degrees  = G.out_degree() 
    out_h =  Counter(dict(out_degrees).values())
    out_dic = collections.OrderedDict(sorted(out_h.items()))
    out_hist = list(out_dic.values())
    out_values =list(out_dic.keys())

    #math for the poission line
    mu = 2.17
    sigma = sp.sqrt(mu)
    mu_plus_sigma = mu + sigma
    x = range(0,10)
    prob = stats.poisson.pmf(x, mu)*4426

    plt.figure(figsize=(12, 8)) 
    #plot out-degree distribution
    plt.loglog(out_values,out_hist,'blue')  
    #plot in-degree distribution
    plt.loglog(in_values,in_hist,'green')
    #plot poission line
    plt.plot(x, prob, "o-", color="orange")
    plt.legend(['In-degree','Out-degree','Poission'])
    plt.xlabel('Degree')
    plt.ylabel('Number of  nodes')
    plt.title('Manhatten Street Network')
    plt.xlim([0,2*10**2])
    plt.show()

#compute the node with the largest betweenness and mark it on the map
def largest_betweenness():
    betweenness =  nx.betweenness_centrality(G_undirected)
    max_node, max_bc = max(betweenness.items(), key=lambda x: x[1])
    #print(max_node, max_bc)
    lat=G.nodes[max_node]['y']
    lon=G.nodes[max_node]['x']
    name=G.nodes[max_node]['highway']
    data = pd.DataFrame({
    'lat':[lat],
    'lon':[lon],
    'name':[name]
    })
    m = folium.Map(location=[lat, lon], zoom_start=13)
    folium.Marker([lat, lon], popup=data.iloc[0]['name']).add_to(m)
    m.save('largest_bt.html')
    return G[max_node],G.nodes[max_node]



#compute shortest path
def get_shortest_path(node1, node2):
    node_G_1 = get_nearest_node(G, node1)
    node_G_2 = get_nearest_node(G, node2)
    route = nx.shortest_path(G, str(node_G_1), str(node_G_2))

    gsub = G.subgraph(route)
    s_len = sum([float(d['length']) for u, v, d in gsub.edges(data=True)])
    length_in_km=s_len/1000
    route_path=[]
    for r in route:
        lon=float(G.nodes[r]['x'])
        lat=float(G.nodes[r]['y'])
        route_node=(lat,lon)
        route_path.append(route_node)
    return(route,route_path,length_in_km)
  

def great_circle_vec(lat1, lng1, lat2, lng2, earth_radius=6371009):

    phi1 = np.deg2rad(90 - lat1)

    phi2 = np.deg2rad(90 - lat2)

    theta1 = np.deg2rad(lng1)
    theta2 = np.deg2rad(lng2)

    cos = (np.sin(phi1) * np.sin(phi2) * np.cos(theta1 - theta2) + np.cos(phi1) * np.cos(phi2))
    arc = np.arccos(cos)

    distance = arc * earth_radius
   
    return distance

#compute the nearest node in the graph
def get_nearest_node(G, point, return_dist=False):

    coords = np.array([[node, data['x'], data['y']] for node, data in G.nodes(data=True)])
    df = pd.DataFrame(coords, columns=['node', 'x', 'y']).set_index('node')
    df['reference_y'] = point[0]
    df['reference_x'] = point[1]

    distances = great_circle_vec(lat1=df['reference_y'],
                                 lng1=df['reference_x'],
                                 lat2=df['x'].astype('float'),
                                 lng2=df['y'].astype('float'))
  
    nearest_node = int(distances.idxmin())
  
    if return_dist:
        return nearest_node, distances.loc[nearest_node]
    else:
        return nearest_node

import folium
#draw shortest path():
def lat_mean(route_path):
    return float(sum(float(i) for i, j in route_path)) / max(len(route_path), 1)

def lon_mean(route_path):
    return float(sum(float(j) for i, j in route_path)) / max(len(route_path), 1)

#avg_lon=lon_mean(route_path)
#avg_lat=lat_mean(route_path)
#m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)

def set_map(node):
    node_data=train_df[train_df['id']==node]
    nodes=[(node_data['pickup_longitude'],node_data['pickup_latitude']),(node_data['dropoff_longitude'],node_data['dropoff_latitude'])]
    new_nodes=[]
    for node in nodes:    
        new_nodes.append(((list(node[0])[0]),list(node[1])[0]))
    avg_lon=lon_mean(new_nodes)
    avg_lat=lat_mean(new_nodes)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)
    return m

def draw_shortest_path_1(nodes,m):
    node1=nodes[0]
    node2=nodes[1]
    route,route_path,length_in_km=get_shortest_path(node1, node2)
    folium.PolyLine(route_path).add_to(m)

def draw_shortest_path(node,m):
    node_data=train_df[train_df['id']==node]
    #node_data
    shortest_nodes=[(node_data['pickup_longitude'],node_data['pickup_latitude']),(node_data['dropoff_longitude'],node_data['dropoff_latitude'])]
    new_shortest_nodes=[]
    for node in shortest_nodes:    
        new_shortest_nodes.append(((list(node[0])[0]),list(node[1])[0]))
    draw_shortest_path_1(new_shortest_nodes,m)
    
def draw_fastest_path(node,m):
    step_list=list(train_df[train_df['id']==node]['step_location_list'])[0].split("|")
    new_step_list=[]
    for data in step_list:
        data1=float(data.split(',')[0])
        data2=float(data.split(',')[1])
        new_step_list.append((data2,data1))
    folium.PolyLine(new_step_list, color='red').add_to(m)

#draw_fastest_path_on_map

node='id3323083'
m=set_map(node)
draw_fastest_path(node,m)
draw_shortest_path(node,m)
m.save("shortest_vs_fastest.html")
    