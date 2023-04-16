# Program to simulate laser-rate equations in Python
# Theory and equations sourced from:
# The solution computed after combining two literatures

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
sigma = 3e-20       # Stimulated emission cross section (cm^2)
w0 = 0.04           # waist radius in medium (cm)
Esat = (h*c/Lambda)*(np.pi*w0*w0)/sigma     # Saturation energy (J)
Eseed = 1e-9        # Seed pulse energy (J)

epsilons = Eseed
dt = Tr
ga = []
epsilon = []
index = 0
iterNum = 100
while index < iterNum:
    tao = index*dt
    ga.append((g0 + epsilons / Esat) / (1 + epsilons / Esat / g0 * np.exp(g0 * tao / Tr)))
    epsilon.append(Esat * (g0 + epsilons / Esat) / (1 + g0 / (epsilons / Esat) * np.exp(-g0 * tao / Tr)))
    index +=1
epsilon = np.array(epsilon)

print("Round trip time:{}s".format(Tr))
nums = np.linspace(0,iterNum, iterNum)
f, axarr = plt.subplots(2)  # Two subplots, the axes array is 1-d
axarr[0].plot(nums, ga, 'g')
axarr[0].set_ylabel("gain (a.u.)")
axarr[0].set_xlabel("Round Trip")
axarr[1].plot(nums, epsilon*1000, 'b')
axarr[1].set_ylabel("pulse energy (mJ)")
axarr[1].set_xlabel("Round Trip")
plt.show()