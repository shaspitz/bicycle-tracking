import numpy as np
import scipy as sp
# NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)


class InternalState():

    def __init__(self):

        # State x0
        self.x = 0
        self.y = 0
        self.theta = np.pi/4
        self.xlen = 3
        self.zlen = 2

        # Covariance matrix P0
        self.P = np.eye(self.xlen)

        # System parameters with uncertainity
        self.B = 0.8
        self.r = 0.425

        # Noise values
        self.V = np.eye(self.xlen)
        self.W = np.array([[1.088070104075678, 0],
                           [0, 2.9844723942433373]])

        # Matricies for EKF implementation (see A as method below)
        self.H = np.array([[1, 0, -1/2*self.B*np.sin(self.theta)],
                           [0, 1,  1/2*self.B*np.cos(self.theta)]])
        self.L = np.eye(self.xlen)
        self.M = np.eye(self.zlen)

    def v(self, w):
        '''
        Speed of bicycle with pedaling speed, w, as input
        '''
        return self.r*5*w

    def A(self, u, dt):
        '''
        Linearized A matrix for EKF implementation
        '''
        return np.array([[1, 0, -self.v(u[0])*dt*np.sin(self.theta)],
                         [0, 1,  self.v(u[0])*dt*np.cos(self.theta)],
                         [0, 0,  1]])

    def meas_model(self):
        '''
        Nonlinear measurement function, h (meas noise = 0)
        Returns measurment vector
        '''
        return np.array([[self.x + 1/2*self.B*np.cos(self.theta)],
                         [self.y + 1/2*self.B*np.sin(self.theta)]])

    def prior_update(self, u, dt):

        # Update variance
        self.P = self.A(u, dt) @ self.P @ self.A(u, dt).T + self.L @ self.V @ self.L.T

        # Update state using nonlinear function q(x, u, 0)
        self.x = self.x + self.v(u[0])*np.cos(self.theta)*dt
        self.y = self.y + self.v(u[0])*np.sin(self.theta)*dt
        self.theta = self.theta + self.v(u[0])*np.tan(u[1])/self.B

    def measurement_update(self, z):

        # Update state with measurement
        z = np.array([[z[0]], [z[1]]])
        K = self.P @ self.H.T @ np.linalg.inv(
            self.H @ self.P @ self.H.T + self.M @ self.W @ self.M.T)
        self.update_state(self.get_state() + K @ (z - self.meas_model()))
        self.P = (np.eye(self.xlen) - K @ self.H) @ self.P

    def get_state(self):
        '''
        Output state into 2D np array
        '''
        return np.array([[self.x], [self.y], [self.theta]])

    def update_state(self, state):
        '''
        Place 2D np array into respective states
        '''
        self.x, self.y, self.theta = state[0][0], state[1][0], state[2][0]


def estInitialize():
    # Fill in whatever initialization you'd like here. This function generates
    # the internal state of the estimator at time 0. You may do whatever you
    # like here, but you must return something that is in the format as may be
    # used by your run() function as the first returned variable.
    #
    # The second returned variable must be a list of student names.
    #
    # The third return variable must be a string with the estimator type

    internalState = InternalState()

    # note that there is *absolutely no prescribed format* for this internal
    # state. You can put in it whatever you like. Probably, you'll want to
    # keep the position and angle, and probably you'll remove the color.

    studentNames = ['David Tondreau', 'Shawn Marshall-Spitzbart']

    # replace this with the estimator type. Use one of the following options:
    #  'EKF' for Extended Kalman Filter
    #  'UKF' for Unscented Kalman Filter
    #  'PF' for Particle Filter
    #  'OTHER: XXX' if you're using something else, in which case please
    #                 replace "XXX" with a (very short) description
    estimatorType = 'EKF'

    return internalState, studentNames, estimatorType

