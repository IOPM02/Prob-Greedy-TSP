"""
Probabilistic Greedy Algorithm for Traveling Salesman Problem

This module implements a probabilistic greedy algorithm for solving the Traveling Salesman Problem (TSP).
It uses MTJ devices to generate random numbers following probability distributions for city selection.

Key features:
- Probabilistic city selection based on distances
- Temperature parameter (kbT) to control exploration vs exploitation
- MTJ device based random number generation
- Support for different starting cities
- Results analysis and visualization

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

import numpy as np
from load_tsp import load_tsp_from_files, calculate_distance
from load_pdftrng import pdf_trng
import datetime

def prob_tsp(distance_matrix, kbT, test_parameters, device_parameters, start_city):
    """
    Solve the Traveling Salesman Problem using a probabilistic approach.

    Args:
    distance_matrix (list): Matrix of distances between cities.
    kbT (float): Temperature parameter for probability calculation.
    test_parameters (dict): Parameters for the test setup.
    device_parameters (dict): Parameters for the device setup.
    start_city (int): Index of the starting city.

    Returns:
    tuple: The tour itself.
    """
    num_cities = len(distance_matrix)
    unvisited_cities = list(range(num_cities))
    tour = []

    current_city = start_city
    tour.append(current_city)
    unvisited_cities.remove(current_city)

    for day in range(num_cities - 2):
        probabilities = []
        for city in unvisited_cities:
            if day == num_cities - 3:  # Second to last day
                # Consider the total distance from current city to candidate city,
                # then to the last unvisited city, and finally back to the start
                last_city = (set(unvisited_cities) - {city}).pop()
                prob = np.exp(-(distance_matrix[current_city][city] +
                                distance_matrix[city][last_city] +
                                distance_matrix[last_city][start_city]) / kbT)
            else:
                prob = np.exp(-distance_matrix[current_city][city] / kbT)
            probabilities.append(prob)

        probabilities = np.array(probabilities)
        sum_probabilities = np.sum(probabilities)

        if sum_probabilities == 0:
            # If all probabilities are zero, choose the next city randomly
            next_city_index = np.random.randint(0, len(unvisited_cities))
        else:
            probabilities /= sum_probabilities
            next_city_index, _ = pdf_trng(probabilities, test_parameters, device_parameters)

        next_city = unvisited_cities[next_city_index]

        tour.append(next_city)
        current_city = next_city
        unvisited_cities.remove(next_city)

    # Add the last city
    last_city = unvisited_cities[0]
    tour.append(last_city)

    # Return to the starting point
    tour.append(start_city)

    return tour

def run_prob_tsp(distance_matrix, kbT, test_parameters, device_parameters, start_city=None):
    """
    Run the probabilistic TSP solver for a given problem.

    Args:
    distance_matrix (list): Matrix of distances between cities.
    kbT (float): Temperature parameter for probability calculation.
    test_parameters (dict): Parameters for the test setup.
    device_parameters (dict): Parameters for the device setup.
    start_city (int, optional): Index of the starting city. If None, try all cities as start.

    Returns:
    tuple: Best distance, best tour, and best starting city.
    """
    num_cities = len(distance_matrix)

    best_distance = float('inf')
    best_tour = None
    best_start = start_city

    if start_city is None:
        for start in range(num_cities):
            tour = prob_tsp(distance_matrix, kbT, test_parameters, device_parameters, start)
            distance = calculate_distance(distance_matrix, tour)
            if distance < best_distance:
                best_distance = distance
                best_tour = tour
                best_start = start
    else:
        best_tour = prob_tsp(distance_matrix, kbT, test_parameters, device_parameters, start_city)
        best_distance = calculate_distance(distance_matrix, best_tour)

    return best_distance, best_tour, best_start

def run_and_save_kbT_scan(distance_matrix, kbT_range, num_runs, test_parameters, device_parameters, problem_name):
    """
    Run TSP multiple times for each kbT value and save the results to a file.

    Args:
    distance_matrix (list): Matrix of distances between cities.
    kbT_range (numpy.array): Range of kbT values to scan.
    num_runs (int): Number of runs for each kbT value.
    test_parameters (dict): Parameters for the test setup.
    device_parameters (dict): Parameters for the device setup.
    problem_name (str): Name of the TSP problem.

    Returns:
    dict: Results dictionary containing kbT values and corresponding distances.
    """
    results = {}

    for kbT in kbT_range:
        print(f"Running kbT: {kbT}")
        kbT_results = []
        for _ in range(num_runs):
            best_distance, _, _ = run_prob_tsp(distance_matrix, kbT, test_parameters, device_parameters)
            kbT_results.append(best_distance)
        results[kbT] = kbT_results

    return results

def main():
    """
    Main function to run the probabilistic TSP solver.
    """
    files_path = 'ALL_tsp'
    problem_name = 'burma14'

    test_parameters = {
        'set_voltage': 0.0,
        'read_voltage': 0.01,
        'source_delay': 0.005,
        'reset_voltage': -0.55,
        'reset_mode': False,
    }

    device_parameters = {
        'devices': [0, 1, 2, 3],
        'device_channels': [
            {'name': 'device1', 'work_channel': 'PXI1Slot11_0', 'ground_channel': 'PXI1Slot11_1'},
            {'name': 'device2', 'work_channel': 'PXI1Slot11_2', 'ground_channel': 'PXI1Slot11_3'},
            {'name': 'device3', 'work_channel': 'PXI1Slot11_4', 'ground_channel': 'PXI1Slot11_5'},
            {'name': 'device4', 'work_channel': 'PXI1Slot11_6', 'ground_channel': 'PXI1Slot11_7'}
        ],
        'para_list': [
            [1,211.0973,-0.2810,0],
            [1,274.7053,-0.2922,0],
            [1,203.9509,-0.2896,0],
            [1,202.9646,-0.3024,0],
        ],
        'resistance_list': [
            [3369.5116558215936,9257.3770302027],
            [3395.971729948851,9482.958084898104],
            [3652.139347127319,10343.797870896431],
            [3474.951284206959,9661.238099840524],
        ]
    }

    kbT_range = np.arange(1, 200, 1)  # Adjust the range and number of points as needed
    # kbT_range = [1]
    num_runs = 10000  # Number of runs for each kbT value

    # Load TSP data once
    cities_coordinates, optimal_tour, distance_matrix = load_tsp_from_files(files_path, problem_name)
    optimal_distance = calculate_distance(distance_matrix, optimal_tour)

    # Run the scan and plot function
    run_and_save_kbT_scan(distance_matrix, kbT_range, num_runs, test_parameters, device_parameters, problem_name)

if __name__ == "__main__":
    main()