import numpy as np
from detector import Detector
import dipole_source
import optical_elements
import graphics
from matplotlib import pyplot as plt
import time

def count_lenses(elements):
    count = 0
    for i, element in enumerate(elements):
        if element.type == 'SineLens':
            count += 1
    return count

def trace_rays(system, source):
    elements = system['elements']
    max_radius=None
    if count_lenses(elements) % 2 == 0:
        is_curved_pupil = True
        max_radius=source.NA
        print("Curved pupil")
    else:
        is_curved_pupil = False
        print("Flat pupil")

    detector = Detector(curved=is_curved_pupil)

    rays = source.rays
    for i, element in enumerate(elements):
        rays = element.apply_matrix(rays)
    
    detector.detect_rays(rays)  # calculate the pupil field

    return detector