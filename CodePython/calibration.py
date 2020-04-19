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

# Calculate the mean and variance
xMean = np.mean(xDataNums)
yMean = np.mean(yDataNums)
xVar = np.var(xDataNums)
yVar = np.var(yDataNums)

# Calculate the bias
xbias = trueX - xMean
ybias = trueY - yMean

print("X,Y Means: ")
print(xMean, yMean)
print("X,Y Variances: ")
print(xVar,yVar)
print("X,Y Bias: ")
print(xbias, ybias)

# Create functions for the pdf of x,y given the mean and var to anazlyze against the data
xpdf = np.linspace(xMean - 3*np.sqrt(xVar), xMean + 3*np.sqrt(xVar), 100)
ypdf = np.linspace(yMean - 3*np.sqrt(yVar), yMean + 3*np.sqrt(yVar), 100)

num_bins = 50

plt.figure(0)
plt.hist(xDataNums, num_bins, facecolor='blue', edgecolor='black', alpha=1, density=True)
plt.axvline(xMean, color = 'r')
plt.plot(xpdf, np.array([normalpdf(xMean, xVar, x) for x in xpdf]), color = 'r')
plt.axvline(trueX, color = 'g')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(r'Histogram of X - Calibration Data'
          + ' with bin size = ' + repr(2/num_bins), fontsize=10)

plt.figure(1)
plt.hist(yDataNums, num_bins, facecolor='blue', edgecolor='black', alpha=1, density=True)
plt.axvline(yMean, color = 'r')
plt.plot(ypdf, np.array([normalpdf(yMean, yVar, y) for y in ypdf]), color = 'r')
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

plt.show()
