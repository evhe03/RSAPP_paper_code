# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:30:34 2026

@author: Niklas
"""
import geopandas as gpd
import networkx as nx

#dowload from https://data.apps.fao.org/catalog/dataset/f2615a41-6383-4aa4-aa21-743330eb03ae/resource/b19c8382-8cff-4b70-8689-a2f9aeb9fe63?inner_span=True
# 1. Load your shapefile
df = gpd.read_file(r"D:\master\rsap\paper\test\basins\final_world_basins.shp")

# --- Task 1: Filter by Major Basin ---
# Selects all sub-basins belonging to the major basin 5035
major_basin_5035 = df[df['MAJ_BAS'] == 5035].copy()

# --- Task 2: Find the Full Upstream Catchment ---
def get_upstream_catchment(data, target_id):
    # Create a Directed Graph
    G = nx.DiGraph()
    
    # Add edges: Flow goes from sub_bas -> to_bas
    # We use zip to iterate efficiently
    edges = zip(data['SUB_BAS'], data['TO_BAS'])
    G.add_edges_from(edges)
    
    # To find everything leading INTO the target, 
    # we reverse the graph and use breadth-first search (BFS)
    # or use the predecessors function.
    reverse_G = G.reverse()
    
    # nx.descendants in a reversed graph gives all upstream nodes
    upstream_nodes = nx.descendants(reverse_G, target_id)
    
    # Include the target basin itself
    all_catchment_ids = list(upstream_nodes) + [target_id]
    
    # Filter the original GeoDataFrame for these IDs
    return data[data['SUB_BAS'].isin(all_catchment_ids)].copy()

# Execute for sub-basin 5035068
full_catchment = get_upstream_catchment(df, 5035068)

# --- Results & Export ---
print(f"Sub-basins in Major Basin 5035: {len(major_basin_5035)}")
print(f"Sub-basins draining into 5035068: {len(full_catchment)}")

full_catchment.plot()
# Optional: Save to new shapefiles
# major_basin_5035.to_file("major_5035.shp")
#full_catchment.to_file("catchment_5035068.shp")

