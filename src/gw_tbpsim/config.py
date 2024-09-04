"""
Config file
===========

Configurations for:
    - F_BASE: Frequency settings for signal and reference.
    - M_BASE: Component mass settings for mass entries, same for m1, m2.
    - PARAM_BASE: Initial param 
        - [m1, m2, chi1, chi2, dist_mpc, tc, phic, inclination].
    - THETA_BASE: Initial param
        - [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination].
"""

# Initial CONST config
# =========================================================================== #
# Freq - min, max, step size
F_BASE = 24.0, 512.0, 0.5
# Mass - min, max, step size
M_BASE = 1.0, 21.0, 100
# Param base - m1, m2, chi1, chi2, dist_mpc, tc, phic, inclination
PARAM_BASE = [36.0, 29.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0]
# Theta base - mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination
THETA_BASE = [28.0956, 0.2471, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0]
# =========================================================================== #
