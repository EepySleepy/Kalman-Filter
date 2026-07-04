import numpy as np
import matplotlib.pyplot as plt
from kf import KF

plt.ion()
plt.figure()

real_x = 0.0
real_v = 0.5
meas_variance = .1

kf = KF(initial_x = 0.0, initial_v = 1.0, accel_variance = .1)

DT = 0.1
NUM_STEPS = 1000
MEAS_EVERY_STEP = 20

mus = []
covs = []
real_vs=[]
real_xs = []

for step in range(NUM_STEPS):
    if step > 500:
        real_v *= 0.9

    covs.append(kf.cov)
    mus.append(kf.mean)

    real_x = real_x + DT * real_v

    kf.predict(dt = DT)
    if step != 0 and step % MEAS_EVERY_STEP ==0:
        kf.update(meas_value = real_x + np.random.randn() * np.sqrt(meas_variance), 
                  meas_variance= meas_variance)
    
    real_xs.append(real_x)
    real_vs.append(real_v)


plt.subplot(2,1,1)
plt.title('position')
plt.plot([mu[0] for mu in mus], 'r')
plt.plot(real_xs, 'b')
plt.plot([mu[0] - 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus,covs)], 'r-.')
plt.plot([mu[0] + 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus,covs)], 'r-.')

plt.subplot(2,1,2)
plt.title('velocity')
plt.plot(real_vs, 'b')
plt.plot([mu[1] for mu in mus], 'r')
plt.plot([mu[1] - 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus,covs)], 'r-.')
plt.plot([mu[1] + 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus,covs)], 'r-.')


plt.show()
plt.ginput(3)
