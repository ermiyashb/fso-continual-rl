
import numpy as np
import matplotlib.pyplot as plt

from physics.fso_channel import simulate_fso_channel


visibility_values = [10, 5, 2, 1, 0.5]

cn2_values = {
    "Weak turbulence": 1e-17,
    "Moderate turbulence": 1e-15,
    "Strong turbulence": 1e-13
}

samples = 5000

plt.figure(figsize=(10, 6))

for label, cn2 in cn2_values.items():

    mean_received_power_dbm = []

    for visibility in visibility_values:

        received_powers_watt = []

        for seed in range(samples):

            result = simulate_fso_channel(
                visibility=visibility,
                cn2=cn2,
                seed=seed
            )

            received_powers_watt.append(
                result["received_power_w"]
            )

        # First calculate mean power in Watts
        mean_power_watt = np.mean(received_powers_watt)

        # Then convert the mean power to dBm
        mean_power_dbm = (
            10 * np.log10(mean_power_watt) + 30
        )

        mean_received_power_dbm.append(
            mean_power_dbm
        )

    plt.plot(
        visibility_values,
        mean_received_power_dbm,
        marker="o",
        label=label
    )


plt.xlabel("Visibility (km)")
plt.ylabel("Mean Received Power (dBm)")

plt.title("Mean FSO Received Power Under Fog and Turbulence")

plt.legend()
plt.grid(True)

plt.savefig(
    "results/fso_channel_comparison.png",
    dpi=300
)

plt.show()
