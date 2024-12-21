"""
Traveling Salesman Problem (TSP) Data Loader

This module provides functionality for loading and handling TSP problem data from TSPLIB format files.
It includes functions for reading problem specifications, city coordinates, optimal tours, and
calculating distances between cities.

Key features:
- Load TSP problems from TSPLIB format files
- Extract city coordinates and optimal tours
- Generate distance matrices
- Calculate tour distances
- Support for missing data handling with defaults

Author: Ran Zhang
License: MIT License

Copyright (c) 2024 Ran Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

import tsplib95
import numpy as np
import os

def load_tsp_from_files(files_path, problem_name):
    """
    Load TSP problem data from TSPLIB format files.

    Args:
        files_path (str): Path to directory containing TSP files
        problem_name (str): Name of the TSP problem (without extension)

    Returns:
        tuple: Contains:
            - cities_coordinates (list): List of [x,y] coordinates for each city
            - optimal_tour (list): List of city indices representing optimal tour
            - distance_matrix (ndarray): Matrix of distances between cities
    """
    tsp_file_path = os.path.join(files_path, f'{problem_name}.tsp')
    opt_tour_file_path = os.path.join(files_path, f'{problem_name}.opt.tour')

    # Load the TSP problem from .tsp file
    problem = tsplib95.load(tsp_file_path)

    # Get number of nodes/cities
    nodes = list(problem.get_nodes())
    n = len(nodes)

    # Extract city coordinates, falling back to defaults if not available
    cities_coordinates = []
    if hasattr(problem, 'node_coords'):
        cities_coordinates = list(problem.node_coords.values())
    elif hasattr(problem, 'display_data'):
        cities_coordinates = list(problem.display_data.values())
    else:
        print(f"Warning: Problem {problem_name} does not have coordinate data. Using default coordinates.")
        cities_coordinates = [[0, 0] for _ in range(n)]

    # Load optimal tour from .opt.tour file if available
    optimal_tour = []
    if os.path.exists(opt_tour_file_path):
        with open(opt_tour_file_path, 'r') as f:
            tour_data = f.read().splitlines()

        # Extract the optimal tour
        tour_section = False
        for line in tour_data:
            if line == 'TOUR_SECTION':
                tour_section = True
                continue
            if tour_section:
                if line == '-1':
                    break
                optimal_tour.append(int(line) - 1)  # Subtract 1 to convert to 0-based index

    # If no optimal tour found, use default sequential tour
    if not optimal_tour:
        print(f"Warning: Optimal tour file {opt_tour_file_path} not found or empty. Using default sequential tour.")
        optimal_tour = list(range(n))

    # Get distance matrix
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = problem.get_weight(nodes[i], nodes[j])
            else:
                distance_matrix[i][j] = 0

    return cities_coordinates, optimal_tour, distance_matrix

def calculate_distance(distance_matrix, tour):
    """
    Calculate the total distance of a tour.

    Args:
        distance_matrix (ndarray): Matrix of distances between cities
        tour (list): List of city indices representing a tour

    Returns:
        float: Total distance of the tour including return to start
    """
    return sum(distance_matrix[tour[i]][tour[i+1]] for i in range(len(tour)-1)) + distance_matrix[tour[-1]][tour[0]]

if __name__ == "__main__":
    files_path = 'ALL_tsp'
    problem_name = 'burma14'
    cities_coordinates, optimal_tour, distance_matrix = load_tsp_from_files(files_path, problem_name)
    print(cities_coordinates)
    print(optimal_tour)
