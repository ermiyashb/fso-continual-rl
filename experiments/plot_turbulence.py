
import numpy as np
import matplotlib.pyplot as plt

from physics.turbulence_model import gamma_gamma_fading


# System parameters
wavelength = 1550e-9
distance = 1000

# Three turbulence conditions
turbulence_levels = {
    "Weak": 1e-17,
    "Moderate": 1e-15,
    "Strong": 1e-13
}

plt.figure(figsize=(10, 6))

for name, cn2 in turbulence_levels.items():

    irradiance, sigma_r, alpha, beta = gamma_gamma_fading(
        cn2=cn2,
        distance=distance,
        wavelength=wavelength,
        samples=100000,
        seed=42
    )

    print("\nCondition:", name)
    print("Cn2:", cn2)
    print("Rytov variance:", sigma_r)
    print("Alpha:", alpha)
    print("Beta:", beta)
    print("Mean irradiance:", np.mean(irradiance))

    plt.hist(
        irradiance,
        bins=200,
        density=True,
        histtype="step",
        label=name
    )


plt.xlabel("Normalized Irradiance")
plt.ylabel("Probability Density")

plt.title("Gamma-Gamma Atmospheric Turbulence Model")

plt.legend()
plt.grid(True)

plt.xlim(0, 5)

plt.savefig(
    "results/gamma_gamma_distribution.png",
    dpi=300
)

plt.show()
