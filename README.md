# Nonlinear-State-Estimation-for-Bicycle-Movement
Our goal is to implement a state estimator to track the position and heading of a bicycle as it moves.

Current questions:
- How to best model process/meas noise (is additive good assumption)?
- Timing of measurements and how that relates to timestep (dt = 0.1 currently)
- E[Initial state] = [0 0 pi/4], can we do better from data
- Calibration data and how that relates to noise numerics
