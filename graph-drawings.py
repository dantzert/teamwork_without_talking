import networkx as nx
import matplotlib.pyplot as plt 

# draw an explosion graph, no labels on edges or nodes
# make an adjacency matrix where one central node connects to the other 8 nodes, none of which connect to each other. undirected graph
G = nx.Graph()
G.add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9)])




# draw the graph
nx.draw(G, with_labels=False)

# show the plot
plt.show()

