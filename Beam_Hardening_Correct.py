import h5py
import numpy as np
import os


#Set the file to be used to write the coefficients to file.
pwd = os.getcwd() + '/'
hdf_fname = pwd + 'Beam_Hardening_LUT.hdf5'

#Global variables for when we convert images
trans_fit_coeffs = None
angle_fit_coeffs = None

def fprepare_conversion(input_hdf_name = None):
    if input_hdf_name is None:
        input_hdf_name = hdf_fname
    with h5py.File(input_hdf_name,'r') as hdf_file:
        global trans_fit_coeffs
        trans_fit_coeffs = hdf_file['Fit_Coeff']
        global angle_fit_coeffs
        angle_fit_coeffs = hdf_file['Angular_Fits']

def fconvert_to_pathlength(input_trans):
    '''Corrects for the beam hardening, assuming we are in the ring plane.
    Input: transmission
    Output: sample pathlength in microns.
    '''
    return np.polyval(trans_fit_coeffs, np.log(input_trans))

def fcorrect_angular(pathlength_image,center_row,d_source_m,pixel_size_mm):
    '''Corrects for the angular dependence of the BM spectrum.
    First, use fconvert_data to get in terms of pathlength assuming we are
    in the ring plane.  Then, use this function to correct.
    '''
    angles = np.abs(np.arange(pathlength_image.shape[0]) - center_row)
    angles *= pixel_size_mm / d_source_m * 1e3
    correction_factor = np.polyval(angle_fit_coeffs,angles)
    return pathlength_image * correction_factor[:,None]