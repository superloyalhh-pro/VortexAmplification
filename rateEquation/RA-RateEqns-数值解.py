# Program to simulate laser-rate equations in Python
# Theory and equations sourced from:
# Title: Period doubling and deterministic chaos in continuously pumped regenerative amplifiers
# DOI: 10.1364/OPEX.12.001759
# URL: https://opg.optica.org/oe/abstract.cfm?uri=OE-12-8-1759

from scipy.integrate import ode
import matplotlib.pyplot as plt
import numpy as np

# Simualtion input  parameters
h = 6.6e-34         # Planck constant (J·s)
c = 3e8             # Speed of light (m/s)
Lambda = 1030e-9    # Wavelength of signal light (m)
g0 = 0.5              # Initial gain
L = 1.7             # length of cavity (m)
Tr = 2*L/c          # cavity round trip time (s)
tL = 0.3e-3         # fluorescence lifetime (s)
loss = 0.04         # All losses
sigma = 3e-20       # Stimulated emission cross section (cm^2)
w0 = 0.04           # waist radius in medium (cm)
Esat = (h*c/Lambda)*(np.pi*w0*w0)/sigma     # Saturation energy (J)
Eseed = 1e-9        # Seed pulse energy (J)

# Define equations to be solved
def Laser_rates(t, y):
    dy = np.zeros([2])
    dy[0] = (g0-y[0])/tL-y[0]*y[1]/(Esat*Tr)
    dy[1] = y[1]/Tr*(y[0]-loss)
    return dy

# Time and initial conditions
t0 = 0
tEnd = Tr * 120
dt = Tr  # Time constraints
y0 = [g0, Eseed]  # Initial conditions
Y = []
T = []  # Create empty lists

# Setup integrator with desired parameters
r = ode(Laser_rates).set_integrator('vode', method='bdf')
r.set_initial_value(y0, t0)

# Simualtion check
while r.successful() and r.t + dt < tEnd:
    r.integrate(r.t + dt)
    Y.append(r.y)  # Makes a list of 1d arrays
    T.append(r.t/Tr)

# Format output
Y = np.array(Y)  # Convert from list to 2d array
G = Y[:, 0]     # y[0] gain
E = Y[:, 1]*1e3     # y[1] pulse energy

# Plotting
f, axarr = plt.subplots(2)  # Two subplots, the axes array is 1-d
axarr[0].plot(T, G, 'g')
axarr[0].set_ylabel("gain (a.u.)")
axarr[0].set_xlabel("Round Trip")
axarr[1].plot(T, E, 'b')
axarr[1].set_ylabel("pulse energy (mJ)")
axarr[1].set_xlabel("Round Trip")
plt.show()