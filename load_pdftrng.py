"""
Probability Distribution Function based True Random Number Generator (PDF-TRNG)

This module implements a probability distribution function based true random number generator
using MTJ devices. It provides functions for generating random numbers following various
probability distributions.

Key features:
- Support for multiple probability distributions (Gaussian, Chi-square, Exponential, etc.)
- Custom probability distribution function support
- MTJ device based random number generation
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
from scipy.stats import norm
from scipy.stats import chi2
from scipy.stats import expon
import matplotlib.pyplot as plt
import load_trng

class Func():
    """
    Class for handling custom probability distribution functions

    Attributes:
        f (str): Function expression with x as variable
        a (float): Lower bound of function domain
        b (float): Upper bound of function domain
    """

    def __init__(self,s_func,a,b):
        """Initialize function with expression and domain bounds"""
        self.f = s_func
        self.a = a
        self.b = b

    def input_func(self,in_x):
        """Evaluate function at given x value"""
        x = in_x
        return eval(self.f)

    def int_func(self,beg,end):
        """
        Numerically integrate function from beg to end

        Uses rectangular approximation with 50 intervals
        """
        n = 5*10**1
        delta_t = (end-beg)/n
        res = 0
        for i in range(n):
            res += self.input_func(beg+delta_t*i)*delta_t
        return res

    def ppf(self,p):
        """
        Calculate percent point function (inverse CDF)

        Args:
            p: Probability value between 0 and 1

        Returns:
            x value corresponding to probability p
        """
        ite_num = 10**3
        px = np.linspace(self.a,self.b,ite_num)
        for i in range(ite_num):
            if abs(self.int_func(self.a,px[i])/self.int_func(self.a,self.b) - p) < 10**(-3):
                return px[i]
        print('error! ite_num is small!')

    def cdf(self,v):
        """
        Calculate cumulative distribution function

        Args:
            v: x value to evaluate CDF at

        Returns:
            Probability P(X <= v)
        """
        return self.int_func(self.a,v)/self.int_func(self.a,self.b)

    def draw(self):
        """Plot probability density function"""
        plt.figure(figsize=(12,6))
        x1 = np.linspace(self.a,self.b,100)
        y1 = [0]*len(x1)
        all_area = self.int_func(self.a,self.b)
        for i in range(len(x1)):
            x = x1[i]
            y1[i] = eval(self.f)/all_area
        plt.plot(x1,y1)

def cal_P_new(limit,n,dis_num):
    """
    Calculate probability distribution based on selected distribution type

    Args:
        limit: Truncation limit for distributions
        n: Number of intervals
        dis_num: Distribution type selector
            1: Normal distribution
            2: Chi-square (df=50)
            3: Exponential
            4: Custom sin/exp function
            6: Uniform
            7: Single point mass
            8: Chi-square (df=8)
            9: Chi-square (df=2)

    Returns:
        Array of probabilities for each interval
    """
    if dis_num == 1:
        x0 = np.linspace(norm.ppf(limit), norm.ppf(1-limit), n+1)
        x = [0]*n
        p = [0]*n
        for i in range(n):
            x[i] = (x0[i]+x0[i+1])/2.0
            p[i] = (norm.cdf(x0[i+1])-norm.cdf(x0[i]))/(1-2*limit)
    elif dis_num == 2:
        df = 50 #chi2 parameter
        x0 = np.linspace(chi2.ppf(limit,df), chi2.ppf(1-limit,df), n+1)
        x = [0]*n
        p = [0]*n
        for i in range(n):
            x[i] = (x0[i]+x0[i+1])/2.0
            p[i] = (chi2.cdf(x0[i+1],df)-chi2.cdf(x0[i],df))/(1-2*limit)

    elif dis_num == 3:
        x0 = np.linspace(expon.ppf(limit), expon.ppf(1-limit), n+1)
        x = [0]*n
        p = [0]*n
        for i in range(n):
            x[i] = (x0[i]+x0[i+1])/2.0
            p[i] = (expon.cdf(x0[i+1])-expon.cdf(x0[i]))/(1-2*limit)
    elif dis_num == 4:
        s_func = ' abs(np.sin(x)*(0.01+np.exp(-x)))' #Function with x as variable
        a = np.pi
        b = 3*np.pi
        func = Func(s_func,a,b)
        x0 = np.linspace(func.ppf(limit), func.ppf(1-limit), n+1)
        x = [0]*n
        p = [0]*n
        for i in range(n):
            x[i] = (x0[i]+x0[i+1])/2.0
            p[i] = (func.cdf(x0[i+1])-func.cdf(x0[i]))/(1-2*limit)
    elif dis_num == 6:
        p = [1.0/n]*n
    elif dis_num == 7:
        p = [0]*n
        p[8] = 1.0
    elif dis_num == 8:
        df = 8 #chi2 parameter
        x0 = np.linspace(chi2.ppf(limit,df), chi2.ppf(1-limit,df), n+1)
        x = [0]*n
        p = [0]*n
        for i in range(n):
            x[i] = (x0[i]+x0[i+1])/2.0
            p[i] = (chi2.cdf(x0[i+1],df)-chi2.cdf(x0[i],df))/(1-2*limit)
    elif dis_num == 9:
        df = 2 #chi2 parameter
        x0 = np.linspace(chi2.ppf(limit,df), chi2.ppf(1-limit,df), n+1)
        x = [0]*n
        p = [0]*n
        for i in range(n):
            x[i] = (x0[i]+x0[i+1])/2.0
            p[i] = (chi2.cdf(x0[i+1],df)-chi2.cdf(x0[i],df))/(1-2*limit)
    else:
        print('Invalid distribution selection!')

    return p

def get_device_parameter(device_parameters, device_index):
    """
    Get parameters for specified device

    Args:
        device_parameters: Dictionary containing all device parameters
        device_index: Index of target device

    Returns:
        Dictionary with parameters for specified device
    """
    if device_index < 0 or device_index >= len(device_parameters['devices']):
        raise ValueError("Invalid device index")

    return {
        'device': device_parameters['devices'][device_index],
        'channel': device_parameters['device_channels'][device_index],
        'para': device_parameters['para_list'][device_index],
        'resistance': device_parameters['resistance_list'][device_index]
    }

def process_mtj(p_dis, test_parameters, device_parameters):
    """
    Process probability distribution through MTJ devices

    Args:
        p_dis: Target probability distribution
        test_parameters: Test configuration parameters
        device_parameters: Device configuration parameters

    Returns:
        out: Array of binary outputs
        R: Array of measured resistances
    """
    n = len(p_dis)  # Number of elements
    n_BerRN = int(np.ceil(np.log2(n)))  # Required number of Bernoulli RNs
    n_new = int(np.power(2, n_BerRN))

    # Normalize probability distribution
    probDis_sum = sum(p_dis)
    probDis_new = np.zeros(n_new)

    for i in range(n):
        probDis_new[i] = p_dis[i] / probDis_sum

    probDis_new = np.where(probDis_new == 0, 0.0001, probDis_new)
    probDis_new = np.where(probDis_new == 1, 0.9999, probDis_new)

    probDis_new_sum = sum(probDis_new)
    p = probDis_new / probDis_new_sum

    n_mtj = n_BerRN  # Number of MTJs

    position = 0
    size = n_new
    p_real = [0] * n_mtj
    p_real_nu = [0] * n_mtj
    p_real_de = [0] * n_mtj
    volt = [0] * n_mtj
    out = [0] * n_mtj
    R = [0] * n_mtj

    n_available = len(device_parameters['devices'])
    for i in range(n_mtj):
        size /= 2
        for j in range(int(position), int(position + size + size)):
            p_real_de[i] += p[j]
        for j in range(int(position + size), int(position + size + size)):
            p_real_nu[i] += p[j]
        p_real[i] = p_real_nu[i] / p_real_de[i]

        device_index = i % n_available
        device_parameter = get_device_parameter(device_parameters, device_index)
        volt[i] = load_trng.insigmoid(p_real[i], device_parameter['para'])

        out[i], R[i] = load_trng.cal_lab(volt[i], test_parameters, device_parameter, 'simulation')

        if out[i] == 1:
            position += size

    return out, R

def pdf_trng(p_dis, test_parameters, device_parameters):
    """
    Generate random number following target probability distribution

    Args:
        p_dis: Target probability distribution
        test_parameters: Test configuration parameters
        device_parameters: Device configuration parameters

    Returns:
        output_real: Generated random number
        R_lst: List of resistance measurements
    """
    n = len(p_dis)  # Number of elements
    n_BerRN = int(np.ceil(np.log2(n)))  # Required number of Bernoulli RNs
    output_real = n
    while output_real >= n:
        R_lst = []
        output, R = process_mtj(p_dis, test_parameters, device_parameters)
        output_real = 0
        for j in range(n_BerRN):
            output_real += output[j] * pow(2, n_BerRN - j - 1)
        R_lst.append(R)
    return output_real, R_lst


def main():
    """Main function for testing PDF-TRNG"""
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

    p_dis = [0.25, 0.25, 0.25, 0.25]

    num_runs = 10000
    results = []
    for _ in range(num_runs):
        out, _ = pdf_trng(p_dis, test_parameters, device_parameters)
        results.append(out)

    unique, counts = np.unique(results, return_counts=True)
    percentages = counts / num_runs * 100

    for value, percentage in zip(unique, percentages):
        print(f"{value}: {percentage:.2f}%")


if __name__ == '__main__':
    main()