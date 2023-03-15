import numpy as np
import optical_systems
import dipole_source

def main():
    O1 = {'NA':1.2, 'n':1.33, 'f': 0.180/60, 'rotation':0}  # 60x
    # O1 = {'NA':0.5, 'n':1.33, 'f': 0.180/60, 'rotation':0}  # 60x

    O2 = {'NA':0.95, 'n':1, 'f': 0.180/50}  # 50x
    O3 = {'NA':0.6, 'n':1, 'f': 0.200/40, 'rotation':35}  # 40x
    # O3 = {'NA':1, 'n':1, 'f': 0.200/40, 'rotation':0}  # 40x
    TL1 = {'NA':1, 'n':1, 'f': 0.180}
    TL2 = {'NA':1, 'n':1, 'f': 0.162}
    # TL2 = {'NA':1, 'n':1, 'f': 0.180}

    TL3 = {'NA':1, 'n':1, 'f': 0.160}
    TLX = {'NA':1, 'n':1, 'f': 0.160/40}

    lenses = [O1, TL1, TL2, O2, O3]
    lenses = [O1, TL1, TL2, O2, O3, TL3]
    # lenses = [O1, O1]
    # lenses = [O1]


    #source.add_dipoles((0,0))
    #source.add_dipoles(np.pi/2,0)

    ## light-sheet
    title = "Intensity map of 1000-dipole source with photoselection in BFP of collection objective for polarised light-sheet (out-of-plane)"

    ## OPM
    # title = "Intensity map in BFP of collection objective for 35-degree oblique light-sheet"

    titles = ["X photoselection", "Y photoselection", "Z' photoselection", "Uniform"]

    opm_angle =35*np.pi/180

    source_x = dipole_source.DipoleSource(name='random dipoles')
    source_x.generate_dipoles(1000)#
    source_x.classical_photoselection((0,0))

    detector = optical_systems.objective_system(lenses, source_x, title=titles[0], ray_count=7000)
    fig = detector.figure