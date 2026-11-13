
import numpy as np
import matplotlib.pyplot as plt

from physics.fog_model import (
    fog_attenuation,
    fog_transmission
)

# Visibility range in kilometers
visibility_values = np.linspace(0.1, 10, 300)

distance = 1.0
wavelength = 1550

attenuation_values = []
transmission_values = []

for visibility in visibility_values:

    attenuation = fog_attenuation(
        visibility,
        wavelength
    )

    transmission = fog_transmission(
        visibility,
        distance,
        wavelength
    )

    attenuation_values.append(attenuation)
    transmission_values.append(transmission)


# Plot 1: Fog attenuation
plt.figure(figsize=(9, 5))

plt.plot(
    visibility_values,
    attenuation_values
)

plt.xlabel("Visibility (km)")
plt.ylabel("Attenuation (dB/km)")
plt.title("Kim Fog Attenuation Model")

plt.grid(True)
plt.savefig("results/fog_attenuation.png", dpi=300)
plt.show()


# Plot 2: Optical power transmission
plt.figure(figsize=(9, 5))

plt.plot(
    visibility_values,
    transmission_values
)

plt.xlabel("Visibility (km)")
plt.ylabel("Power Transmission")
plt.title("FSO Power Transmission Under Fog")

plt.grid(True)
plt.savefig("results/fog_transmission.png", dpi=300)
plt.show()
