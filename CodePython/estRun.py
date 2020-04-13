import numpy as np
import scipy as sp
# NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)


def estRun(time, dt, internalStateIn, steeringAngle, pedalSpeed, measurement):
    # In this function you implement your estimator. The function arguments
    # are:
    #  time: current time in [s]
    #  dt: current time step [s]
    #  internalStateIn: the estimator internal state, definition up to you.
    #  steeringAngle: the steering angle of the bike, gamma, [rad]
    #  pedalSpeed: the rotational speed of the pedal, omega, [rad/s]
    #  measurement: the position measurement valid at the current time step
    #
    # Note: the measurement is a 2D vector, of x-y position measurement.
    #  The measurement sensor may fail to return data, in which case the
    #  measurement is given as NaN (not a number).
    #
    # The function has four outputs:
    #  est_x: your current best estimate for the bicycle's x-position
    #  est_y: your current best estimate for the bicycle's y-position
    #  est_theta: your current best estimate for the bicycle's rotation theta
    #  internalState: the estimator's internal state, in a format that can be understood by the next call to this function

    # Example code only, you'll want to heavily modify this.

    # 4/2/2020: PF since dynamics are highly nonlinear (asymptotic parts tangent)

    # this internal state needs to correspond to your init function:

    x = internalStateIn.x
    y = internalStateIn.y
    theta = internalStateIn.theta
    myColor = 'green'

    x = x + pedalSpeed
    y = y + pedalSpeed

    if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
        # have a valid measurement
        x = measurement[0]
        y = measurement[1]
        theta = theta + 1

    # We're unreliable about our favourite colour: 
    if myColor == 'green':
        myColor = 'red'
    else:
        myColor = 'green'


    #### OUTPUTS ####
    # Update the internal state (will be passed as an argument to the function
    # at next run), must obviously be compatible with the format of
    # internalStateIn:
    internalStateOut = internalStateIn
    '''
    internalStateOut = [x,
                        y,
                        theta,
                        myColor
                        ]
    '''

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x, y, theta, internalStateOut


