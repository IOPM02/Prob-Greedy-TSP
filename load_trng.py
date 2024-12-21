"""
True Random Number Generator (TRNG) Implementation using MTJ Devices

This module implements a true random number generator based on Magnetic Tunnel Junction (MTJ) devices.
It provides functions for device control, measurement, and probability analysis.

Key features:
- Device control and measurement via NI-DCPower
- Sigmoid probability mapping
- Magnetic angle calculations
- Resistance measurement and state determination
- Probability analysis and visualization

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
import nidcpower
import hightime
import matplotlib.pyplot as plt
import datetime
import random

def sigmoid(ctrl, para =[1, 135, -0.346, 0]):
    """Calculate sigmoid function for probability mapping"""
    return (para[3]+para[0]/(1+np.exp(-para[1]*(ctrl+para[2]))))

def insigmoid(out, para=[1, 135, -0.346, 0]):
    """Inverse sigmoid function"""
    epsilon = 1e-10
    if out >= para[0] + para[3]:
        out = para[0] + para[3] - epsilon
    elif out <= para[3]:
        out = para[3] + epsilon
    try:
        result = (-np.log(para[0]/(out-para[3])-1)/para[1]-para[2])
    except RuntimeWarning:
        print("RuntimeWarning: overflow encountered in scalar divide")
        result = float('inf')
    return result

def cal_theta(theta):
    """Calculate normalized theta angle in range [0, pi]"""
    if theta > np.pi:
        return 2 * np.pi - theta
    elif theta < 0:
        return -theta
    else:
        return theta

def cal_angle(sigma_0):
    """
    Calculate magnetic angles based on LLG equation

    Args:
        sigma_0: Initial magnetic field strength

    Returns:
        theta_lst: List of theta angles
        fai_lst: List of phi angles
    """
    fai_0 = (np.pi / 180) * 90
    fai = -(np.pi / 2)
    theta = np.pi / 2 + 0.00001

    # Define effective fields
    K = 0.1  # Anisotropy constant
    M = 1    # Magnetization
    H_K = 2 * K / M  # Anisotropy field
    H_T = 5 * 10 ** (-4)  # Thermal field
    alpha = 0.1  # Damping factor
    gamma = 1  # Gyromagnetic ratio
    d_t = 10 ** (-2)  # Time step

    # Solve LLG equation iteratively
    theta_lst = [theta]
    fai_lst = [fai]
    theta_T = np.pi * random.random()
    fai_T = 2 * np.pi * random.random()
    for i in range(10000):
        sigma = sigma_0 * np.array([np.cos(theta) * np.cos(fai_0 - fai),
                                    np.sin(fai_0 - fai),
                                    np.sin(theta) * np.cos(fai_0 - fai)])
        H_fai = H_K * np.sin(theta) * np.sin(fai) * np.cos(fai) - np.sin(fai) * H_T * np.sin(theta_T) * np.cos(fai_T) + \
                np.cos(fai) * H_T * np.sin(theta_T) * np.sin(fai_T)
        H_theta = H_K * np.sin(theta) * np.cos(theta) * np.sin(fai) * np.sin(fai) + \
                  2 * M * np.sin(theta) * np.cos(theta) + np.cos(theta) * np.cos(fai) * H_T * np.sin(theta_T) * np.cos(
            fai_T) + \
                  np.cos(theta) * np.sin(fai) * H_T * np.sin(theta_T) * np.sin(fai_T) - np.sin(theta) * H_T * np.cos(
            theta_T)
        d_theta = (1 / (1 + alpha ** 2)) * \
                  (gamma * (H_fai + alpha * H_theta) + (sigma[0] - alpha * sigma[1])) * d_t
        theta = cal_theta(theta + d_theta)
        theta_lst.append(theta)
        d_fai = (1 / ((1 + alpha ** 2) * np.sin(theta))) * \
                (-gamma * (H_theta - alpha * H_fai) + (sigma[1] + alpha * sigma[0])) * d_t
        fai = (fai + d_fai)
        fai_lst.append(fai)
    return theta_lst, fai_lst

def cal(sigma):
    """
    Calculate MTJ state based on magnetic angles

    Returns:
        state: Binary state (0 or 1)
        resistance: Corresponding resistance value
    """
    theta_lst, fai_lst = cal_angle(sigma)
    y = np.sin(fai_lst[-1]) * np.sin(theta_lst[-1])
    if y >0:
        return 1,1800000
    else:
        return 0,1000000

def run_single_point_test(test_parameters, device_parameter):
    """
    Run single point measurement on MTJ device

    Args:
        test_parameters: Dictionary containing test settings
        device_parameter: Dictionary containing device configuration

    Returns:
        results: List of measurement results
    """
    results = []
    reset_mode = test_parameters.get('reset_mode', False)

    try:
        resource_name = f"{device_parameter['channel']['work_channel'].replace('_', '/')}, {device_parameter['channel']['ground_channel'].replace('_', '/')}"

        with nidcpower.Session(resource_name=resource_name) as session:
            # Configure the session
            session.source_mode = nidcpower.SourceMode.SINGLE_POINT
            session.voltage_level_autorange = True
            session.current_limit = 0.0001
            session.current_limit_autorange = True
            session.source_delay = hightime.timedelta(seconds=test_parameters['source_delay'])
            session.measure_when = nidcpower.MeasureWhen.AUTOMATICALLY_AFTER_SOURCE_COMPLETE

            # Enable all channels
            session.output_enabled = True

            work_channel = device_parameter['channel']['work_channel'].replace('_', '/')
            ground_channel = device_parameter['channel']['ground_channel'].replace('_', '/')
            set_voltage = test_parameters['set_voltage']
            read_voltage = test_parameters['read_voltage']

            # Apply set voltage or reset voltage
            session.channels[work_channel].voltage_level = set_voltage if not reset_mode else test_parameters['reset_voltage']
            session.channels[ground_channel].voltage_level = 0

            # Initiate session to apply voltage
            with session.initiate():
                pass

            if not reset_mode:
                # Apply read voltage
                session.channels[work_channel].voltage_level = read_voltage

                # Perform read measurement
                with session.initiate():
                    measurement = session.channels[work_channel].fetch_multiple(count=1, timeout=hightime.timedelta(seconds=1))[0]
                    voltage = measurement.voltage
                    current = measurement.current
                    resistance = abs(voltage / current) if current != 0 else float('inf')

                    # Print and store read results
                    results.append((work_channel, set_voltage, resistance))

            # Disable all channels
            session.output_enabled = False

    except Exception as e:
        print(f"Error: {e}")

    return results


def cal_lab(volt, test_parameters, device_parameter):
    """
    Calculate MTJ state through lab measurement

    Args:
        volt: Applied voltage
        test_parameters: Test configuration
        device_parameter: Device configuration

    Returns:
        out: Binary state (0 or 1)
        R: Measured resistance
    """
    if volt > 1.0:
        volt = 1.0
    if volt < -1.0:
        volt = -1.0

    # Apply reset voltage without reading
    test_parameters['set_voltage'] = test_parameters['reset_voltage']
    test_parameters['reset_mode'] = True
    results = run_single_point_test(test_parameters, device_parameter)

    # Apply the set voltage and read the resistance
    test_parameters['set_voltage'] = volt
    test_parameters['reset_mode'] = False
    results = run_single_point_test(test_parameters, device_parameter)

    if results:
        R = results[0][2]
    else:
        R = float('inf')

    R_m = sum(device_parameter['resistance']) / 2.0
    if R > R_m:
        out = 1
    else:
        out = 0

    return out, R

def main():
    """Main function for testing and visualization"""
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

    def get_device_parameter(device_index):
        """Get parameters for specified device"""
        if device_index < 0 or device_index >= len(device_parameters['devices']):
            raise ValueError("Invalid device index")

        return {
            'device': device_parameters['devices'][device_index],
            'channel': device_parameters['device_channels'][device_index],
            'para': device_parameters['para_list'][device_index],
            'resistance': device_parameters['resistance_list'][device_index]
        }

    device_parameter = get_device_parameter(0)

    # Define voltage range
    voltage_range = np.arange(0.225, 0.350, 0.001)
    probabilities = []

    for voltage in voltage_range:
        ones_count = 0
        for _ in range(1000):
            out, _ = cal_lab(voltage, test_parameters, device_parameter)
            ones_count += out
        probability = ones_count / 1000
        probabilities.append(probability)

    # Calculate ideal sigmoid curve
    ideal_sigmoid = sigmoid(voltage_range, device_parameter['para'])

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(voltage_range, ideal_sigmoid, 'b-', label='Ideal Sigmoid')
    plt.scatter(voltage_range, probabilities, color='red', s=10, alpha=0.5, label='Experimental Data')
    plt.xlabel('Voltage')
    plt.ylabel('Probability of 1')
    plt.title('Probability of 1 vs Voltage')
    plt.legend()
    plt.grid(True)

    # Save the plot with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'probability_vs_voltage_{timestamp}.png')
    plt.show()

if __name__ == '__main__':
    main()
