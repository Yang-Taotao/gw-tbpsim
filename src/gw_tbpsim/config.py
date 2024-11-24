"""
Config file
===========

Configurations for:
    - F_BASE: Frequency settings for signal and reference.
    - MC_BASE: Chirp mass settings.
    - ETA_BASE: Symmetric mass ratio settings. 
    - PARAM_BASE: Initial param with m1, m2
        - [m1, m2, chi1, chi2, dist_mpc, tc, phic, inclination].
    - THETA_BASE: Initial param with mc, eta
        - [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination].
"""

# Initial CONST config
# =========================================================================== #
# Freq - min, max, step size
F_BASE = 24.0, 512.0, 0.5
# Mass - Chirp mass - min, max, step size
MC_BASE = 1.0, 21.0, 100
# MASS - Symmetric mass ratio - min, max, step size
ETA_BASE = 0.05, 0.25, 100
# Param base - m1, m2, chi1, chi2, dist_mpc, tc, phic, inclination
PARAM_BASE = [36.0, 29.0, 0.0, 0.0, 400.0, 0.0, 0.0, 0.0]
# Theta base - mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination
THETA_BASE = [28.0956, 0.2471, 0.0, 0.0, 400.0, 0.0, 0.0, 0.0]
# =========================================================================== #
