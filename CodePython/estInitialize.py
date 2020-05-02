import numpy as np
import scipy as sp
# NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)


class InternalState():

    def __init__(self):

        # Number of particles
        '''
        200 particles took about 10 sec per run on 16GB RAM, I7 processor.
        We need to stay under 30 seconds when submitting so this is reasonable.
        '''
        self.Np = 1000

        # Variances for process and measurement noises
        '''
        Note that the standard deviations for B and r are 1/3 times their
        manufacturer's suggested tolerance. We did this since most data
        in a normal distrubtion lies within 3 standard deviations of its
        mean. The performace of the PF with 0 variance in B and r was
        tested to be worse than this implementation over multiple trials.
        A uniform distrubtion for B and r variance also performed worse.
        '''
        self.V = np.diag([1, 1, (np.pi/12)**2, (1/3*0.8/10)**2, (1/3*0.425/20)**2])
        self.W = np.diag([1.088070104075678, 2.9844723942433373])

        # Initalize PF with particles sampled from pdf, f(x(0))
        self.x = np.random.normal(0, np.sqrt(7.0241800107377825), self.Np)
        self.y = np.random.normal(0, np.sqrt(15.04128926026523), self.Np)
        self.theta = np.random.normal(np.pi/4, np.sqrt(self.V[2][2]), self.Np)
        self.B = np.random.normal(0.8, np.sqrt(self.V[3][3]), self.Np)
        self.r = np.random.normal(0.425, np.sqrt(self.V[4][4]), self.Np)

        # State and measurement lengths
        self.xlen = 5
        self.zlen = 2

    def v(self, w, r):
        '''
        Speed of bicycle with pedaling speed, w, as input
        '''
        return r*5*w

    def meas_likelihood(self, xp_n, z):
        '''
        Need to update this with change of variables output
        (note xp_n is a 3 state group from self.get_state)
        '''
        # Convert measurement tuple into 2D np array
        z = np.array([[z[0]], [z[1]]])

        # Define change of variables expression h
        h = z - np.array([[xp_n[0] + 1/2*xp_n[3]*np.cos(xp_n[2])],
                          [xp_n[1] + 1/2*xp_n[3]*np.sin(xp_n[2])]])

        # Return normal pdf, f(w) evaluated at h(z, x)
        meas_likelihood = 1/((2*np.pi)**(self.xlen/2)*np.sqrt(
            np.linalg.det(self.W)))*np.exp(-1/2*h.T @ np.linalg.inv(self.W) @ h)

        return meas_likelihood

    def prior_update(self, u, dt):

        # sample process noise particles
        vk = np.zeros((self.xlen, self.Np))
        for x in range(self.xlen):
            vk[x] = np.random.normal(0, np.sqrt(self.V[x][x]), self.Np)

        # Simulate particles forward with noise using nl function q(x, u, vk)
        x_old = self.x
        y_old = self.y
        theta_old = self.theta
        B_old = self.B
        r_old = self.r

        self.x = x_old + self.v(u[0], r_old)*np.cos(theta_old)*dt + vk[0]
        self.y = y_old + self.v(u[0], r_old)*np.sin(theta_old)*dt + vk[1]
        self.theta = theta_old + self.v(u[0], r_old)*np.tan(u[1])/B_old + vk[2]
        self.B = B_old + vk[3]
        self.r = r_old + vk[4]

    def measurement_update(self, z):

        # Scale particles by meas likelihood and apply normalization const
        beta = np.array([self.meas_likelihood(xp_n,
                                              z) for xp_n in self.get_state()])
        alpha = np.sum(beta)
        beta = beta / alpha

        # Resampling
        beta_sum = np.cumsum(beta)
        xm = np.zeros((self.Np, self.xlen))
        for i in range(self.Np):
            r = np.random.uniform()
            # first occurance where beta_sum[n_index] >= r
            n_index = np.nonzero(beta_sum > r)[0][0]
            xm[i] = self.get_state()[n_index]

        xm = self.roughening(xm)
        self.update_state(xm)

    def roughening(self, xm):
        d = self.xlen
        K = 0.01
        for i in range(d):
            Ei = np.abs(np.max(xm[:, i]) - np.min(xm[:, i]))
            # Ei = np.max(np.array([
            #    np.abs(xm[idx, i] - xm[idx - 1, i]) for idx, x in enumerate(
            #        np.sort(xm[1:-1, i]))]))
            sigma_i = K * Ei * self.Np ** (-1 / d)
            xm[:, i] += np.random.normal(0, sigma_i, size=xm[:, i].shape)
        return xm

    def get_state(self):
        '''
        Output state into 2D np array
        (first iterator corresponds to each 5 state particle group)
        '''
        return np.array([[self.x[i], self.y[i],
                          self.theta[i], self.B[i],
                          self.r[i]] for i in range(self.Np)])

    def update_state(self, xm):
        '''
        Updates state with respective particle arrays
        '''
        self.x, self.y, self.theta = xm[:, 0], xm[:, 1], xm[:, 2]


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
    estimatorType = 'PF'

    return internalState, studentNames, estimatorType

