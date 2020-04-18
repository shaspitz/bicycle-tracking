# This function utilizes the calibration data to estimate the measurement noise
#
# Authors: David Tondreau, Shawn Marshall-Spitzbart
# Date: 04/18/20

import numpy as np
import matplotlib.pyplot as plt

# Load Calibration data
experimentalRun = 0
experimentalData = np.genfromtxt('../data/run_{0:03d}.csv'.format(experimentalRun), delimiter=',')

