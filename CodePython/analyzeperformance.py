
# This function is to analyze performance of difference estimators by estimating the error over a multitude of runs
#
# Authors: David Tondreau, Shawn Marshall-Spitzbart
# Date: 04/25/20


import numpy as np
import matplotlib.pyplot as plt
from estRun import estRun
from estInitialize import estInitialize

startingRun = 1
endingRun = 99
numRuns = endingRun - startingRun + 1

xErrorArr = np.zeros((1,numRuns))
yErrorArr = np.zeros((1,numRuns))
thErrorArr = np.zeros((1,numRuns))
scoreArr = np.zeros((1,numRuns))

for run in range(startingRun, endingRun+1):
    # Loop thru experimental runs
    experimentalRun = run

    print('Loading the data file #', experimentalRun)
    experimentalData = np.genfromtxt('../data/run_{0:03d}.csv'.format(experimentalRun), delimiter=',')

    #===============================================================================
    # Here, we run your estimator's initialization
    #===============================================================================
    internalState, studentNames, estimatorType = estInitialize()

    numDataPoints = experimentalData.shape[0]

    # Here we will store the estimated position and orientation, for later plotting:
    estimatedPosition_x = np.zeros([numDataPoints, ])
    estimatedPosition_y = np.zeros([numDataPoints, ])
    estimatedAngle = np.zeros([numDataPoints, ])

    print('Running the system #', experimentalRun)
    dt = experimentalData[1, 0] - experimentalData[0, 0]
    for k in range(numDataPoints):
        t = experimentalData[k, 0]
        gamma = experimentalData[k, 1]
        omega = experimentalData[k, 2]
        measx = experimentalData[k, 3]
        measy = experimentalData[k, 4]

        # Run the estimator:
        x, y, theta, internalState = estRun(t, dt, internalState, gamma, omega, (measx, measy))

        # Keep track:
        estimatedPosition_x[k] = x
        estimatedPosition_y[k] = y
        estimatedAngle[k] = theta


    print('Done running #' , experimentalRun)
    # Make sure the angle is in [-pi,pi]
    estimatedAngle = np.mod(estimatedAngle+np.pi, 2*np.pi)-np.pi

    # Modified from Main file to only use the final position and th estimate in determining estimator error
    posErr_x = estimatedPosition_x[-1] - experimentalData[-1, 5]
    posErr_y = estimatedPosition_y[-1] - experimentalData[-1, 6]
    angErr = np.mod(estimatedAngle[-1] - experimentalData[-1, 7]+np.pi, 2*np.pi)-np.pi

    ax = np.sum(np.abs(posErr_x))/numDataPoints
    ay = np.sum(np.abs(posErr_y))/numDataPoints
    ath = np.sum(np.abs(angErr))/numDataPoints
    score = ax + ay + ath

    xErrorArr[:, run - startingRun] = ax
    yErrorArr[:, run - startingRun] = ay
    thErrorArr[:, run - startingRun] = ath
    scoreArr[:, run - startingRun] = score

xMeanErr = np.mean(xErrorArr)
yMeanErr = np.mean(yErrorArr)
thMeanErr = np.mean(thErrorArr)
scoreMean = np.mean(scoreArr)

scoreWorst = np.max(scoreArr)
scoreBest = np.min(scoreArr)

if not np.isnan(score):
    # This is for evaluation by the instructors
    print('average error:')

    print('   pos x =', xMeanErr, 'm')
    print('   pos y =', yMeanErr, 'm')
    print('   angle =', thMeanErr, 'rad')

    # Our scalar score.
    print('average score:', scoreMean)

    # Our Best score.
    print('Best score:', scoreBest)

    # Our Worst score.
    print('Worst score:', scoreWorst)
