# Program to simulate laser-rate equations in Python
# Theory and equations sourced from:
# Title: Dynamics of high repetition rate regenerative amplifiers
# DOI: 10.1364/OE.15.009434
# URL: https://opg.optica.org/oe/fulltext.cfm?uri=oe-15-15-9434&id=139929

import matplotlib.pyplot as plt
import numpy as np

# Simualtion input  parameters
h = 6.6e-34       # Planck constant (J·s)
Lambda = 1030e-9  # Wavelength of signal light (m)
P = 2.0           # Optical intensity of Pump (W)
L = 1.7           # length of cavity (m)
La = 3            # active medium length (cm)
c = 3e8           # Speed of light (m/s)
T0 = 2*L/c        # cavity round trip time (s)
T1 = 0.3e-3       # fluorescence lifetime (s)
sigma = 3e-20     # Stimulated emission cross section (cm^2)
w0 = 0.04         # waist radius in medium (cm)
G0 = 1            # steady state small signal gain, g0~0 G0=exp(g0)~1
Esat = (h*c/Lambda)*(np.pi*w0*w0)/sigma     # Saturation energy (J)
Eseed = 1e-9      # Seed pulse energy (J)

g0 = 1
epsilons = Eseed/(Esat*G0)
dt = 1e-9
ga = []
epsilon = []
index = 0
iterNum = 300
while index < iterNum:
    tao = index*dt/T0
    ga.append((g0 + epsilons) / (1 + epsilons / g0 * np.exp(g0 * tao)))
    epsilon.append((g0 + epsilons) / (1 + g0 / epsilons * np.exp(-g0 * tao)))
    index +=1
epsilon = np.array(epsilon)*Esat*G0

print("Round trip time:{}s".format(T0))
# plt.figure(1)
# plt.plot(, ga)
# plt.figure(2)
# plt.plot(np.linspace(0,iterNum*dt, iterNum), epsilon)
# plt.show()
f, axarr = plt.subplots(2)  # Two subplots, the axes array is 1-d
axarr[0].plot(np.linspace(0,iterNum, iterNum), ga, 'g')
axarr[0].set_ylabel("gain (a.u.)")
axarr[0].set_xlabel("Round Trip")
axarr[1].plot(np.linspace(0,iterNum, iterNum), epsilon, 'b')
axarr[1].set_ylabel("pulse energy (mJ)")
axarr[1].set_xlabel("Round Trip")
plt.show()