import numpy as np
import scipy as sp
# NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

# Will return to this class structure later


class InternalState():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.theta = np.pi/4
        self.B = 0.8
        self.r = 0.425

    def v(self, w):
        '''
        Speed of bicycle with pedaling speed, w, as input
        '''
        return self.r*5*w

    def q_update(self, u, dt):
        '''
        Nonlinear descretized state evolution function
        '''
        self.x = self.x + self.v(u[0])*np.cos(self.theta)*dt
        self.y = self.y + self.v(u[0])*np.sin(self.theta)*dt
        self.theta = self.theta + self.v(u[0])*np.tan(u[1])/self.B

    def get_state(self):
        return np.array([[self.x], [self.y], [self.theta]])


def estInitialize():
    # Fill in whatever initialization you'd like here. This function generates
    # the internal state of the estimator at time 0. You may do whatever you
    # like here, but you must return something that is in the format as may be
    # used by your run() function as the first returned variable.
    #
    # The second returned variable must be a list of student names.
    #
    # The third return variable must be a string with the estimator type

    # We make the internal state a list, with the first three elements the position
    # x, y; the angle theta; and our favorite color.
    x = 0
    y = 0
    theta = 0
    color = 'green'

    internalState = InternalState()

    # note that there is *absolutely no prescribed format* for this internal state.
    # You can put in it whatever you like. Probably, you'll want to keep the position
    # and angle, and probably you'll remove the color.
    '''
    internalState = [x,
                     y,
                     theta,
                     color
                     ]
    '''
    # replace these names with yours. Delete the second name if you are working alone.
    studentNames = ['Bart Simpson',
                    'Lisa Simpson']

    # replace this with the estimator type. Use one of the following options:
    #  'EKF' for Extended Kalman Filter
    #  'UKF' for Unscented Kalman Filter
    #  'PF' for Particle Filter
    #  'OTHER: XXX' if you're using something else, in which case please
    #                 replace "XXX" with a (very short) description
    estimatorType = 'EKF'

    return internalState, studentNames, estimatorType

