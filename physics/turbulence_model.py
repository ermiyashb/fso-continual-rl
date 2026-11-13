
import numpy as np


def rytov_variance(cn2, distance, wavelength):
    """
    Calculate Rytov variance for plane-wave propagation.

    cn2: refractive index structure parameter (m^(-2/3))
    distance: propagation distance (m)
    wavelength: optical wavelength (m)
    """

    k = 2 * np.pi / wavelength

    sigma_r_squared = (
        1.23
        * cn2
        * k ** (7 / 6)
        * distance ** (11 / 6)
    )

    return sigma_r_squared


def gamma_gamma_parameters(sigma_r_squared):
    """
    Calculate alpha and beta parameters
    of the Gamma-Gamma turbulence model.
    """

    if sigma_r_squared <= 0:
        raise ValueError("Rytov variance must be positive.")

    alpha_term = (
        0.49 * sigma_r_squared
        / (1 + 1.11 * sigma_r_squared ** (12 / 5))
        ** (7 / 6)
    )

    beta_term = (
        0.51 * sigma_r_squared
        / (1 + 0.69 * sigma_r_squared ** (12 / 5))
        ** (5 / 6)
    )

    alpha = 1 / np.expm1(alpha_term)
    beta = 1 / np.expm1(beta_term)

    return alpha, beta


def gamma_gamma_fading(cn2, distance, wavelength,
                       samples=10000, seed=None):
    """
    Generate normalized Gamma-Gamma irradiance samples.
    """

    sigma_r_squared = rytov_variance(
        cn2,
        distance,
        wavelength
    )

    alpha, beta = gamma_gamma_parameters(
        sigma_r_squared
    )

    rng = np.random.default_rng(seed)

    large_scale = rng.gamma(
        shape=alpha,
        scale=1 / alpha,
        size=samples
    )

    small_scale = rng.gamma(
        shape=beta,
        scale=1 / beta,
        size=samples
    )

    irradiance = large_scale * small_scale

    return irradiance, sigma_r_squared, alpha, beta


if __name__ == "__main__":

    wavelength = 1550e-9
    distance = 1000
    cn2 = 1e-15

    irradiance, sigma_r, alpha, beta = (
        gamma_gamma_fading(
            cn2,
            distance,
            wavelength
        )
    )

    print("Rytov variance:", sigma_r)
    print("Alpha:", alpha)
    print("Beta:", beta)
    print("Mean irradiance:", np.mean(irradiance))
    print("Std irradiance:", np.std(irradiance))
