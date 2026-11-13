
import numpy as np


def kim_q(visibility):
    """
    Calculate q parameter using Kim model.
    Visibility is in kilometers.
    """

    if visibility > 50:
        return 1.6

    elif visibility > 6:
        return 1.3

    elif visibility > 1:
        return 0.16 * visibility + 0.34

    elif visibility > 0.5:
        return visibility - 0.5

    else:
        return 0.0


def fog_attenuation(visibility, wavelength=1550):
    """
    Calculate fog attenuation coefficient.

    visibility: km
    wavelength: nm

    Returns:
        attenuation in dB/km
    """

    if visibility <= 0:
        raise ValueError("Visibility must be positive.")

    q = kim_q(visibility)

    gamma = (
        3.91 / visibility
    ) * (
        wavelength / 550
    ) ** (-q)

    return gamma


def fog_transmission(visibility, distance=1.0,
                     wavelength=1550):
    """
    Calculate fraction of optical power
    transmitted through fog.

    distance: km
    """

    gamma = fog_attenuation(
        visibility,
        wavelength
    )

    total_attenuation = gamma * distance

    transmission = 10 ** (
        -total_attenuation / 10
    )

    return transmission


if __name__ == "__main__":

    visibility = 2.0
    distance = 1.0

    gamma = fog_attenuation(visibility)

    transmission = fog_transmission(
        visibility,
        distance
    )

    print("Visibility:", visibility, "km")
    print("Attenuation:", gamma, "dB/km")
    print("Transmission:", transmission)
    print("Remaining power:", transmission * 100, "%")
