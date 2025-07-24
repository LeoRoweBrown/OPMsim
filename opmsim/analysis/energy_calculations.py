import numpy as np
from rays import PolarRays
from dipole_source import DipoleSource

def calculate_efficiency(rays: PolarRays, ray_max_half_angle=(np.pi / 2)):
    relative_collection_efficieny = rays.total_power / rays.total_power_initial
    collection_efficiency = None
    emission_efficiency = None
    if hasattr(rays, 'half_sphere_power'):
        collection_efficiency = rays.total_power_initial_hemisphere / rays.total_power_initial
        # calculate "emission efficiency" by scaling half_sphere_power by actual solid angle
        solid_angle = 2 * np.pi * (1 - np.cos(ray_max_half_angle))
        emission_efficiency = rays.total_power_initial_hemisphere * (2 * np.pi / solid_angle) / rays.total_power

    return relative_collection_efficieny, collection_efficiency, emission_efficiency
