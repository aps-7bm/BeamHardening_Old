import BeamHardeningCorrection as bh
import numpy as np
import matplotlib.pyplot as plt

bh.fread_config_file()
print(dir(bh))
bh.fcompute_calibrations()

print(bh.angular_coeffs)
print(bh.centerline_coeffs)

bh.fsave_coeff_data()

def test_centerline():
    '''Test the ability of the code to accurately correct for beam hardening
     on the centerline of the beam.'''
    # Create a list of Fe thicknesses
    Fe_thicknesses = np.logspace(0, 4, 100)
    print(Fe_thicknesses)
    # Get the transmission of the beam through these
    bh.fread_config_file()
    bh.sample_material = bh.possible_materials['Fe']
    source_spectrum = bh.fread_source_data()
    if len(bh.filters):
        filtered_spectrum = bh.fapply_filters(bh.filters, source_spectrum)
    # For each thickness, compute the absorbed power in the scintillator
    detected_power = np.zeros_like(Fe_thicknesses)
    for i in range(Fe_thicknesses.size):
        sample_filtered_spectrum = bh.sample_material.fcompute_transmitted_spectrum(Fe_thicknesses[i],
                                                                              filtered_spectrum)
        detected_power[i] = bh.scintillator_material.fcompute_absorbed_power(bh.scintillator_thickness,
                                                                          sample_filtered_spectrum)
    ref_detected_power = bh.scintillator_material.fcompute_absorbed_power(bh.scintillator_thickness,
                                                                          filtered_spectrum)
    Fe_transmission = detected_power / ref_detected_power
    #Compute the coefficients for conversions
    bh.fcompute_calibrations()
    #Convert back to pathlength
    convert_pathlength = bh.fconvert_to_pathlength_center_only(Fe_transmission)
    plt.semilogx(Fe_thicknesses, convert_pathlength / Fe_thicknesses)
    plt.figure(2)
    plt.loglog(Fe_thicknesses, convert_pathlength)
    plt.show()

test_centerline()
