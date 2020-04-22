# This function utilizes the calibration data to estimate the measurement noise
#
# Authors: David Tondreau, Shawn Marshall-Spitzbart
# Date: 04/18/20

import numpy as np
import matplotlib.pyplot as plt

def normalpdf(mu, var, x):
    sigma = np.sqrt(var)
    f = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((x-mu)/sigma)**2)
    return f

# Load Calibration data
experimentalRun = 0
experimentalData = np.genfromtxt('../data/run_{0:03d}.csv'.format(experimentalRun), delimiter=',')

# Calibration data columns are organized as follows:
# 0 = Time	
# 1 = Steering Angle	
# 2 = Pedal Speed	
# 3 = X	
# 4 = Y	
# 5 = True X (last row only)
# 6 = True Y (last row only)	
# 7 = True Theta (last row only)

timeData = experimentalData[:,0]
xData, yData = experimentalData[:,3], experimentalData[:,4]
trueX, trueY, trueTheta = experimentalData[ -1,5], experimentalData[ -1,6], experimentalData[ -1,7]

xDataNums = np.array([x for x in xData if not(np.isnan(x))])
yDataNums = np.array([x for x in yData if not(np.isnan(x))])
# Parse the time data list for only those times in which we recieve x,y measurements
timeDataNums = np.array([t for i_t, t in enumerate(timeData) if not(np.isnan(xData[i_t]))])
deltaTimeDataNums = np.array([timeDataNums[i_t] - timeDataNums[i_t-1] for i_t, t in enumerate(timeDataNums)])
deltaTimeDataNums = deltaTimeDataNums[1:]

# Use the measurement model to determine x1 and y1
B = 0.8
x1DataNums = xDataNums - 1/2*B*np.cos(trueTheta)
y1DataNums = yDataNums - 1/2*B*np.sin(trueTheta)

# Calculate the mean and variance
x1Mean = np.mean(x1DataNums)
y1Mean = np.mean(y1DataNums)
x1Var = np.var(x1DataNums)
y1Var = np.var(y1DataNums)


print("X,Y Means: ")
print(x1Mean, y1Mean)
print("X,Y Variances: ")
print(x1Var,y1Var)

# Create functions for the pdf of x,y given the mean and var to anazlyze against the data
xpdf = np.linspace(x1Mean - 3*np.sqrt(x1Var), x1Mean + 3*np.sqrt(x1Var), 100)
ypdf = np.linspace(y1Mean - 3*np.sqrt(y1Var), y1Mean + 3*np.sqrt(y1Var), 100)

num_bins = 50

plt.figure(0)
plt.hist(x1DataNums, num_bins, facecolor='blue', edgecolor='black', alpha=1, density=True)
plt.axvline(x1Mean, color = 'r')
plt.plot(xpdf, np.array([normalpdf(x1Mean, x1Var, x) for x in xpdf]), color = 'r')
plt.axvline(trueX, color = 'g')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(r'Histogram of X - Calibration Data'
          + ' with bin size = ' + repr(2/num_bins), fontsize=10)

plt.figure(1)
plt.hist(y1DataNums, num_bins, facecolor='blue', edgecolor='black', alpha=1, density=True)
plt.axvline(y1Mean, color = 'r')
plt.plot(ypdf, np.array([normalpdf(y1Mean, y1Var, y) for y in ypdf]), color = 'r')
plt.axvline(trueY, color = 'g')
plt.xlabel('y')
plt.ylabel('f(y)')
plt.title(r'Histogram of Y - Calibration Data'
          + ' with bin size = ' + repr(2/num_bins), fontsize=10)

plt.figure(2)
plt.hist(deltaTimeDataNums, num_bins, facecolor='blue', edgecolor='black', alpha=1, density=True)
plt.xlabel('deltaTime')
plt.ylabel('f(deltaTime)')
plt.title(r'Histogram of deltaTime - Calibration Data'
          + ' with bin size = ' + repr(2/num_bins), fontsize=10)

# Estimate f(x(0)) by examining the inital measurements from all of the data

x0Data, y0Data = np.zeros((100,1)),np.zeros((100,1))
for data in range(0,100):
    experimentalRun = data
    experimentalData = np.genfromtxt('../data/run_{0:03d}.csv'.format(experimentalRun), delimiter=',')
    idx = 0
    x0, y0 = experimentalData[idx,3], experimentalData[idx,4]
    while np.isnan(x0):
        idx = idx + 1
        x0 = experimentalData[idx,3]
    idx = 0
    while np.isnan(y0):
        idx = idx + 1
        y0 = experimentalData[idx,4]
    x0Data[data], y0Data[data] = x0, y0

# Use the measurement model to determine x0 and y0
B = 0.8
theta0 = np.pi / 4
x0DataNums = x0Data - 1/2*B*np.cos(theta0)
y0DataNums = y0Data - 1/2*B*np.sin(theta0)

#x0DataNums = x0Data 
#y0DataNums = y0Data 

# Calculate the mean and variance
x0Mean = np.mean(x0DataNums)
y0Mean = np.mean(y0DataNums)
x0Var = np.var(x0DataNums)
y0Var = np.var(y0DataNums)


print("X0,Y0 Means: ")
print(x0Mean, y0Mean)
print("X0,Y0 Variances: ")
print(x0Var,y0Var)

# Create functions for the pdf of x,y given the mean and var to anazlyze against the data
x0pdf = np.linspace(x0Mean - 3*np.sqrt(x0Var), x0Mean + 3*np.sqrt(x0Var), 100)
y0pdf = np.linspace(y0Mean - 3*np.sqrt(y0Var), y0Mean + 3*np.sqrt(y0Var), 100)

num_bins = 20

plt.figure(3)
plt.hist(x0DataNums, num_bins, facecolor='blue', edgecolor='black', alpha=1, density=True)
plt.axvline(x0Mean, color = 'r')
plt.plot(x0pdf, np.array([normalpdf(x0Mean, x0Var, x) for x in x0pdf]), color = 'r')
plt.xlabel('x0')
plt.ylabel('f(x)')
plt.title(r'Histogram of X0'
          + ' with bin size = ' + repr(2/num_bins), fontsize=10)

plt.figure(4)
plt.hist(y0DataNums, num_bins, facecolor='blue', edgecolor='black', alpha=1, density=True)
plt.axvline(y0Mean, color = 'r')
plt.plot(y0pdf, np.array([normalpdf(y0Mean, y0Var, y) for y in y0pdf]), color = 'r')
plt.xlabel('y0')
plt.ylabel('f(y)')
plt.title(r'Histogram of Y0'
          + ' with bin size = ' + repr(2/num_bins), fontsize=10)

plt.show()
