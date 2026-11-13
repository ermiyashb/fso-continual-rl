
import numpy as np

from physics.fog_model import fog_transmission
from physics.turbulence_model import gamma_gamma_fading


def dbm_to_watt(power_dbm):
    """Convert optical power from dBm to Watts."""
    return 10 ** ((power_dbm - 30) / 10)


def watt_to_dbm(power_watt):
    """Convert power from Watts to dBm."""
    if power_watt <= 0:
        return -np.inf

    return 10 * np.log10(power_watt) + 30


def gaussian_beam_radius(w0, distance, wavelength):
    """Calculate Gaussian beam radius at receiver."""

    rayleigh_range = np.pi * w0**2 / wavelength

    beam_radius = w0 * np.sqrt(
        1 + (distance / rayleigh_range)**2
    )

    return beam_radius


def geometric_loss(
    receiver_radius,
    beam_radius
):
    """Calculate centered Gaussian beam capture fraction."""

    return 1 - np.exp(
        -2 * receiver_radius**2 / beam_radius**2
    )


def simulate_fso_channel(
    visibility,
    cn2,
    transmit_power_dbm=10,
    distance=1000,
    wavelength=1550e-9,
    beam_waist=0.01,
    receiver_radius=0.025,
    eta_tx=0.8,
    eta_rx=0.8,
    seed=None
):
    """
    Simulate a single FSO channel realization.

    Distance: meters
    Wavelength: meters
    Visibility: kilometers
    cn2: m^(-2/3)
    """

    # Convert transmitter power to Watts
    pt = dbm_to_watt(transmit_power_dbm)

    # Fog attenuation
    h_fog = fog_transmission(
        visibility=visibility,
        distance=distance / 1000,
        wavelength=wavelength * 1e9
    )

    # Gaussian beam propagation
    beam_radius = gaussian_beam_radius(
        beam_waist,
        distance,
        wavelength
    )

    # Geometric loss
    h_geo = geometric_loss(
        receiver_radius,
        beam_radius
    )

    # Atmospheric turbulence
    irradiance, sigma_r, alpha, beta = gamma_gamma_fading(
        cn2=cn2,
        distance=distance,
        wavelength=wavelength,
        samples=1,
        seed=seed
    )

    h_turb = irradiance[0]

    # Total received optical power
    pr = (
        pt
        * eta_tx
        * eta_rx
        * h_geo
        * h_fog
        * h_turb
    )

    return {
        "transmit_power_w": pt,
        "received_power_w": pr,
        "received_power_dbm": watt_to_dbm(pr),
        "fog_loss": h_fog,
        "geometric_loss": h_geo,
        "turbulence_gain": h_turb,
        "beam_radius_m": beam_radius,
        "rytov_variance": sigma_r,
        "alpha": alpha,
        "beta": beta
    }


if __name__ == "__main__":

    result = simulate_fso_channel(
        visibility=2.0,
        cn2=1e-15,
        seed=42
    )

    for key, value in result.items():
        print(f"{key}: {value}")
