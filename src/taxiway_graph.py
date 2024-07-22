import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math 
import pandas as pd

class TaxiwayGraph:
    def __init__(self, csv_file_path, image_path):
        # Read data from CSV file
        self.data = pd.read_csv(csv_file_path)
        self.data.set_index('Name', inplace=True)
        
        # Initialize lists for vertices and edges
        self.pos = []
        self.vertices = []
        self.edges = []
        
        # Process the data
        self.process_data()

        # Create the graph
        self.G = nx.Graph()
        self.G.add_nodes_from(self.vertices)
        for u, v, weight in self.edges:
            self.G.add_edge(u, v, weight=weight)

        # Set up the position dictionary for visualization
        self.pos_dict = {label: (x, y) for label, x, y in self.pos}
        self.ordered_pos_dict = {label: self.pos_dict[label] for label in self.vertices}
        
        # Load the airport map image
        self.image_raw = mpimg.imread(image_path)  # Load the airport map image
        self.image = np.flipud(self.image_raw)     # Flip the image for correct orientation

    def process_data(self):
        length = len(self.data)

        for i in range(length):
            name = self.data.index[i]
            x_value = int(self.data.loc[name, 'Longitude (x)'])
            y_value = int(self.data.loc[name, 'Latitude (y)'])
            self.pos.append((name, x_value, y_value))
            self.vertices.append(name)

        for i in range(length):
            name = self.data.index[i]
            connection = self.data.loc[name, 'Connection']
            connection = [c.strip() for c in connection.split(',') if c.strip()]  # Strip whitespace and filter out empty strings
            for conn in connection:
                if conn in self.vertices:
                    # Get position index for the connection
                    idx = self.vertices.index(conn)
                    # Compute the Euclidean distance between positions
                    weight = math.sqrt((self.pos[i][1] - self.pos[idx][1])**2 + (self.pos[i][2] - self.pos[idx][2])**2)
                    self.edges.append((name, conn, weight))
                else:
                    print(f"Warning: Connection '{conn}' not found in vertices.")

    def directed_expanded_graph(self):
        D = nx.DiGraph()
        # Add "v in" vertices to D for each vertex v in G
        for v in self.G.nodes():
            D.add_node(f"{v} in")

        # Add "u from v" vertices to D for each connected vertex pair (u, v) in G
        for u in self.G.nodes():
            for v in self.G.neighbors(u):
                D.add_node(f"{u} from {v}")

        # Add directed edges to D based on the original edges in G
        for u, v, attr in self.G.edges(data=True):
            taxiway_name = attr.get('name', '')

            for w in self.G.neighbors(u):
                if w != v:
                    D.add_edge(f"{u} in", f"{v} from {u}", name=taxiway_name)

            for w in self.G.neighbors(v):
                if w != u:
                    D.add_edge(f"{v} in", f"{u} from {v}", name=taxiway_name)
        return D

    def directed_graph(self, D):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(D, seed=0)
        nx.draw(D, pos, with_labels=True, node_size=5000, node_color='lightblue', edge_color='gray', font_size=10)
        edge_labels = nx.get_edge_attributes(D, 'name')
        nx.draw_networkx_edge_labels(D, pos, edge_labels=edge_labels)
        plt.title("Directed Expanded Graph")
        plt.show()
    
    def undirected_graph(self, image, pos):
        G = nx.Graph()
        G.add_nodes_from(self.vertices)
        G.add_weighted_edges_from(self.edges)

        # Draw the graph
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 5]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 5]

        # nodes and edges
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.gca().invert_yaxis()
        nx.draw_networkx_nodes(G, pos, node_size=200)
        nx.draw_networkx_edges(G, pos, edgelist=elarge, width=2)
        nx.draw_networkx_edges(G, pos, edgelist=esmall, width=2, alpha=0.5, edge_color="b", style="solid")
        nx.draw_networkx_labels(G, pos, font_size=5, font_family="sans-serif")   # node labels

        # edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}  # edge weight labels
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title("Undirected Graph")
        plt.show()

    def find_shortest_path_flight(self, start_vertex, end_vertex):
        if start_vertex not in self.G or end_vertex not in self.G:
            raise ValueError(f"One or both vertices '{start_vertex}' and '{end_vertex}' are not in the graph.")
        distance = nx.dijkstra_path(self.G, start_vertex, end_vertex, weight='weight')
        control_points = []
        for i in range(len(distance)):
            control_points.append(self.pos_dict[distance[i]])
        return distance, control_points
    
    def create_and_visualize_graph_2(self, start, finish, shortest_path_flight):
        G = nx.DiGraph()
        G.add_nodes_from(self.vertices)
        G.add_weighted_edges_from(self.edges)

        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 5]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 5]

        plt.figure(figsize=(12, 8))
        plt.imshow(self.image)
        plt.gca().invert_yaxis()
        nx.draw_networkx_nodes(G, self.pos_dict, node_size=200)
        nx.draw_networkx_edges(G, self.pos_dict, edgelist=elarge, width=2)
        nx.draw_networkx_edges(G, self.pos_dict, edgelist=esmall, width=2, alpha=0.5, edge_color="b", style="solid")
        nx.draw_networkx_labels(G, self.pos_dict, font_size=5, font_family="sans-serif")

        # Draw the shortest path nodes and edges
        path_nodes = shortest_path_flight
        path_edges = list(zip(shortest_path_flight, shortest_path_flight[1:]))

        nx.draw_networkx_nodes(G, self.pos_dict, nodelist=path_nodes, node_color='r')
        nx.draw_networkx_edges(G, self.pos_dict, edgelist=path_edges, width=2, edge_color="r", style="solid")

        plt.title(f"Shortest Path from {start} to {finish}")
        plt.show()


if __name__ == '__main__':
    # Define the path to the CSV file and the image file
    csv_file_path = r"../data/Position_of_Airport.csv"
    image_path = 'map.png'

    # Create the TaxiwayGraph object
    taxiway_graph = TaxiwayGraph(csv_file_path, image_path)

    # Generate the directed expanded graph
    D = taxiway_graph.directed_expanded_graph()

    # Define the start and end gates
    start_gate = 'Gate 1'
    end_runway = 'R10L'

    # Find the shortest path from the starting gate to the runway
    shortest_path_flight, control_points = taxiway_graph.find_shortest_path_flight(start_gate, end_runway)

    # Compute the cardinal spline
    tension = 0  # Adjust this parameter to control the tension of the spline
    detail_points, derivative_points = taxiway_graph.cardinal_spline(control_points, tension=tension)

    taxiway_graph.undirected_graph(taxiway_graph.image, taxiway_graph.ordered_pos_dict)
    taxiway_graph.directed_graph(D)
    taxiway_graph.create_and_visualize_graph_2(start_gate, end_runway, shortest_path_flight)
