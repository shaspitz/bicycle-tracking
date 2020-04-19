# Nonlinear-State-Estimation-for-Bicycle-Movement
Our goal is to implement a state estimator to track the position and heading of a bicycle as it moves.

Current questions:
- Would adding modulus operator to theta value improve speed? avoiding need to put 24 rad into a np.cos function for example
- need to initialize x0 better, E[Initial state] = [0 0 pi/4], can we do better from data/is this good assumption?
- Can actually estimate B and R in state
- Truth point (black in plot) is at the last timestep
- How to best model process/meas noise (is additive good assumption)?
- Timing of measurements and how that relates to timestep (dt = 0.1 currently), time-invariant formulation of H??
- Calibration data and how that relates to noise numerics
