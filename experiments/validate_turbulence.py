
import numpy as np
import matplotlib.pyplot as plt

from physics.turbulence_model import gamma_gamma_fading


wavelength = 1550e-9
distance = 1000

turbulence_levels = {
    "Weak": 1e-17,
    "Moderate": 1e-15,
    "Strong": 1e-13
}

samples = 100000

results = {}

for name, cn2 in turbulence_levels.items():

    irradiance, sigma_r, alpha, beta = gamma_gamma_fading(
        cn2=cn2,
        distance=distance,
        wavelength=wavelength,
        samples=samples,
        seed=42
    )

    mean_value = np.mean(irradiance)
    std_value = np.std(irradiance)

    cv = std_value / mean_value

    # Probability that normalized irradiance
    # falls below 0.5
    outage = np.mean(irradiance < 0.5)

    results[name] = {
        "mean": mean_value,
        "std": std_value,
        "cv": cv,
        "outage": outage,
        "rytov": sigma_r
    }

    print("\n", name)
    print("Rytov variance:", sigma_r)
    print("Mean:", mean_value)
    print("Standard deviation:", std_value)
    print("Coefficient of variation:", cv)
    print("Outage probability:", outage)


# Plot standard deviation
names = list(results.keys())

std_values = [
    results[name]["std"] for name in names
]

plt.figure(figsize=(8, 5))
plt.bar(names, std_values)

plt.xlabel("Turbulence Level")
plt.ylabel("Standard Deviation")
plt.title("Intensity Fluctuation Under Different Turbulence Levels")

plt.grid(axis="y")
plt.savefig("results/turbulence_std.png", dpi=300)
plt.show()


# Plot outage probability
outage_values = [
    results[name]["outage"] for name in names
]

plt.figure(figsize=(8, 5))
plt.bar(names, outage_values)

plt.xlabel("Turbulence Level")
plt.ylabel("Outage Probability")
plt.title("Outage Probability Under Different Turbulence Levels")

plt.grid(axis="y")
plt.savefig("results/turbulence_outage.png", dpi=300)
plt.show()
