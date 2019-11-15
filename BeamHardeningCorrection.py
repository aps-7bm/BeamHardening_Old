'''Code to correct for beam hardening in filtered white beam imaging experiments.
This code creates an HDF5 file with coefficients to perform a polynomial fit
of transmission to correct for the effect of beam hardening for white beam
imaging and tomography.

Alan Kastengren, XSD, APS

Started: November 11, 2015

Edits: Apr 27, 2016. Dan & Katie edited fcompute_lookup_table_function to sort the 
       values going into the lookup table. Our version of interp1d was unhappy with
       non monotonic x values and it spat out all NaNs. Also added 'pwd' to allow
       files to be stored somewhere else.
    
        June 16, 2017: several edits to make the code more generally usable.
            * Change to different files, from xCrossSec, for more materials.

        January 30, 2019: set up for using a config file to avoid having to alter
            source code every time we run this code.
        
        September 23, 2019: set up Spectrum objects to simplify calls.

Usage:
* Run fread_config_file to load in configuration information.
* Run fcompute_calibrations to compute the polynomial fits for correcting
    the beam hardening
* Run either fcorrect_as_pathlength or fcorrect_as_transmission, as desired,
    to correct an image.

'''
import numpy as np
import scipy.interpolate
import scipy.integrate
import h5py
import os
from pathlib import Path, PurePath
from copy import deepcopy
from scipy.interpolate import InterpolatedUnivariateSpline

#Global variables we need for computing LUT
filters = {}
sample_material = None
scintillator_material = None
scintillator_thickness = None
ref_trans = None
data_path = None
source_data_file = None
write_name = None
correct_angular = None
possible_materials = {}
angular_fit_params = {}
equiv_energy = 50   #keV

#Global variables for when we convert images
centerline_spline = None
angular_spline = None

class Spectrum:
    '''Class to hold the spectrum: energies and spectral power.
    '''
    def __init__(self, energies, spectral_power):
        if len(energies) != len(spectral_power):
            raise ValueError
        self.energies = energies
        self.spectral_power = spectral_power

    def fintegrated_power(self):
        return scipy.integrate.simps(self.spectral_power, self.energies)

    def fmean_energy(self):
        return scipy.integrate.simps(self.spectral_power * self.energies, self.energies) / self.fintegrated_power()
    
    def __len__(self):
        return len(energies)
 
#Copy part of the Material class from Scintillator_Optimization code
class Material:
    '''Class that defines the absorption and attenuation properties of a material.
    Data based off of the xCrossSec database in XOP 2.4.
    '''
    def __init__(self,name,density):
        self.name = name
        self.density = density  #in g/cc
        self.fread_absorption_data()
        self.absorption_interpolation_function = self.interp_function(self.energy_array,self.absorption_array)
        self.attenuation_interpolation_function = self.interp_function(self.energy_array,self.attenuation_array)
    
    def __repr__(self):
        return "Material({0:s}, {1:f})".format(self.name, self.density)
    
    def fread_absorption_data(self):
        raw_data = np.genfromtxt(PurePath.joinpath(data_path, self.name + '_properties_xCrossSec.dat'))
        self.energy_array = raw_data[:,0] / 1000.0      #in keV
        self.absorption_array = raw_data[:,3]   #in cm^2/g, from XCOM in XOP
        self.attenuation_array = raw_data[:,6]  #in cm^2/g, from XCOM in XOP, ignoring coherent scattering
    
    def interp_function(self,energies,absorptions):
        '''Return a function to interpolate logs of energies into logs of absorptions.
        '''
        return scipy.interpolate.interp1d(np.log(energies),np.log(absorptions),bounds_error=False)
    
    def finterpolate_absorption(self,input_energies):
        '''Interpolates absorption on log-log scale and scales back
        '''
        return np.exp(self.absorption_interpolation_function(np.log(input_energies)))
    
    def finterpolate_attenuation(self,input_energies):
        '''Interpolates attenuation on log-log scale and scales back
        '''
        return np.exp(self.attenuation_interpolation_function(np.log(input_energies)))
    
    def fcompute_proj_density(self,thickness):
        '''Computes projected density from thickness and material density.
        Input: thickness in um
        Output: projected density in g/cm^2
        '''
        return thickness /1e4 * self.density
    
    def fcompute_transmitted_spectrum(self,thickness,input_spectrum):
        '''Computes the transmitted spectral power through a filter.
        Inputs:
        thickness: the thickness of the filter in um
        input_spectrum: Spectrum object for incident spectrum
        Output:
        Spectrum object for transmitted intensity
        '''
        output_spectrum = deepcopy(input_spectrum)
        #Compute filter projected density
        filter_proj_density = self.fcompute_proj_density(thickness)
        #Find the spectral transmission using Beer-Lambert law
        output_spectrum.spectral_power *= (
                    np.exp(-self.finterpolate_attenuation(output_spectrum.energies) * filter_proj_density))
        return output_spectrum
    
    def fcompute_absorbed_spectrum(self,thickness,input_spectrum):
        '''Computes the absorbed power of a filter.
        Inputs:
        thickness: the thickness of the filter in um
        input_spectrum: Spectrum object for incident beam
        Output:
        Spectrum objection for absorbed spectrum
        '''
        output_spectrum = deepcopy(input_spectrum)
        #Compute filter projected density
        filter_proj_density = self.fcompute_proj_density(thickness)
        #Find the spectral transmission using Beer-Lambert law
        output_spectrum.spectral_power = (input_spectrum.spectral_power -
                - np.exp(-self.finterpolate_absorption(input_spectrum.energies) * filter_proj_density) 
                * input_spectrum.spectral_power)
        return output_spectrum
    
    def fcompute_absorbed_power(self,thickness,input_spectrum):
        '''Computes the absorbed power of a filter.
        Inputs:
        material: the Material object for the filter
        thickness: the thickness of the filter in um
        input_energies: Spectrum object for incident beam 
        Output:
        absorbed power
        '''
        return self.fcompute_absorbed_spectrum(thickness,input_spectrum).fintegrated_power()

def fread_config_file(config_filename=None):
    '''Read in parameters for beam hardening corrections from file.
    Default file is in same directory as this source code.
    Users can input an alternative config file as needed.
    '''
    global filters
    filters = {}
    if config_filename:
        config_path = Path(config_filename)
        if not config_path.exists():
            raise IOError('Config file does not exist: ' + str(config_path))
    else:
        config_path = Path.joinpath(Path(__file__).parent, 'setup.cfg')
    with open(config_path, 'r') as config_file:
        temp_filters = {}
        while True:
            line = config_file.readline()
            if line == '':
                break
            if line.startswith('#'):
                continue
            if line.startswith('data_path'):
                temp_path = Path(line.split(':')[1].strip())
                global data_path
                if not temp_path:
                    temp_path = Path.joinpath(Path(__file__),'data')
                elif temp_path.is_absolute():
                    data_path = temp_path
                else:
                    data_path = Path(PurePath.joinpath(config_path.parent, temp_path))
                if not data_path.exists():
                    raise IOError('Path to data does not exist: ' + str(data_path))
            elif line.startswith('source_data_file'):
                source_data = line.split(':')[1].strip()
            elif line.startswith('sample_material'):
                sample_name = line.split(':')[1].strip()
            elif line.startswith('scint_material'):
                scintillator_name = line.split(':')[1].strip()
            elif line.startswith('scint_thickness'):
                global scintillator_thickness
                scintillator_thickness = float(line.split(':')[1])
            elif line.startswith('symbol'):
                symbol = line.split(',')[0].split('=')[1].strip()
                density = float(line.split(',')[1].split('=')[1])
                possible_materials[symbol] = Material(symbol, density)
            elif line.startswith('material'):
                symbol = line.split(',')[0].split('=')[1].strip()
                thickness = float(line.split(',')[1].split('=')[1])
                temp_filters[symbol] = thickness
            elif line.startswith('correct_angular'):
                global correct_angular
                if line.split(':')[1].strip() == 'True':
                    correct_angular = True
                else:
                    correct_angular = False
            elif line.startswith('equiv_energy'):
                global equiv_energy
                equiv_energy = float(line.split(':')[1].strip())
            for var in ['ref_trans','center_row','d_source_m','pixel_size_mm']:
                if line.startswith(var):
                    angular_fit_params[var] = float(line.split(':')[1])
    global source_data_file
    source_data_file = PurePath.joinpath(data_path, source_data)
    global write_name
    write_name = PurePath.joinpath(write_path, write_fname)
    for filt, thick in temp_filters.items():
        filters[possible_materials[filt]] = thick
    print(filters)
    global sample_material
    sample_material = possible_materials[sample_name]
    global scintillator_material
    scintillator_material = possible_materials[scintillator_name]


def fread_source_data(file_name = None):
    '''Reads the spectral power data from file.
    Data file comes from the BM spectrum module in XOP.
    '''
    if file_name:
        spectral_data = np.genfromtxt(file_name , comments='!')
    elif source_data_file:
        spectral_data = np.genfromtxt(PurePath.joinpath(data_path, source_data_file), comments='!')
    else:
        raise IOError("File not specified.")
    spectral_energies = spectral_data[:-2,0] / 1000.
    spectral_power = spectral_data[:-2,1]
    return Spectrum(spectral_energies, spectral_power)


def fapply_filters(filters, input_spectrum):
    '''Computes the spectrum after all filters.
        Inputs:
        filters: dictionary giving filter materials as keys and thicknesses in microns as values.
        input_energies: numpy array of the energies for the spectrum in keV
        input_spectral_power: spectral power for input spectrum, in numpy array
        Output:
        spectral power transmitted through the filter set.
        '''
    print("Input power = {0:f}".format(input_spectrum.fintegrated_power()))
    print("Mean input spectrum energy = {0:6.3f} keV".format(input_spectrum.fmean_energy()))
    temp_spectrum = deepcopy(input_spectrum)
    for filt, thickness in filters.items():
        temp_spectrum = filt.fcompute_transmitted_spectrum(thickness, temp_spectrum)
    print("Output power = {0:f}".format(temp_spectrum.fintegrated_power()))
    print("Mean output spectrum energy = {0:6.3f} keV".format(temp_spectrum.fmean_energy()))
    return temp_spectrum


def ffind_calibration_one_angle(input_spectrum):
    '''Makes a scipy interpolation function to be used to correct images.
    '''
    #Make an array of sample thicknesses
    sample_thicknesses = np.sort(np.concatenate((-np.logspace(1,0,11), [0], np.logspace(-1,4.5,56))))
    #For each thickness, compute the absorbed power in the scintillator
    detected_power = np.zeros_like(sample_thicknesses)
    for i in range(sample_thicknesses.size):
        sample_filtered_power = sample_material.fcompute_transmitted_spectrum(sample_thicknesses[i],
                                                                              input_spectrum)
        detected_power[i] = scintillator_material.fcompute_absorbed_power(scintillator_thickness,
                                                                          sample_filtered_power)
    #Compute an effective transmission vs. thickness
    sample_effective_trans = detected_power / scintillator_material.fcompute_absorbed_power(scintillator_thickness,
                                                                                            input_spectrum)
    #Return a spline, but make sure things are sorted in ascending order
    inds = np.argsort(sample_effective_trans)
    #for i in inds:
    #    print(sample_effective_trans[i], sample_thicknesses[i])
    return InterpolatedUnivariateSpline(sample_effective_trans[inds], sample_thicknesses[inds])


def ffind_calibration():
    '''Do the correlation at the reference transmission. 
    Treat the angular dependence as a correction on the thickness vs.
    transmission at angle = 0.
    '''
    ref_trans = float(angular_fit_params['ref_trans'])
    angles_urad = np.array([0,5,10,15,20,30,40])
    cal_curve = []
    for i in angles_urad:
        angle_spectrum = fread_source_data(PurePath.joinpath(data_path,
                                                        'Psi_{0:02d}urad.dat'.format(i)))
        #Filter the beam
        filtered_spectrum = fapply_filters(filters, angle_spectrum)
        #Create an interpolation function based on this
        angle_spline = ffind_calibration_one_angle(filtered_spectrum)
        if i == 0:
            global centerline_spline
            centerline_spline = angle_spline
        cal_curve.append(angle_spline(ref_trans))
    cal_curve /= cal_curve[0]
    global angular_spline
    angular_spline = InterpolatedUnivariateSpline(angles_urad, cal_curve) 


def fconvert_to_transmission(pathlengths, equiv_energy):
    '''Converts pathelength values back to transmission at an equivalent energy.
    Input pathlengths are in microns
    '''
    #Get absorption coefficient in cm^2/g
    equiv_abs_coeff = sample_material.finterpolate_attenuation(equiv_energy)
    print(equiv_abs_coeff)
    #Convert to 1/microns
    equiv_abs_coeff *= sample_material.fcompute_proj_density(1)
    print(equiv_abs_coeff)
    return np.exp(-equiv_abs_coeff * pathlengths)


def fcorrect_as_pathlength_centerline(input_trans):
    """Corrects for the beam hardening, assuming we are in the ring plane.
    Input: transmission
    Output: sample pathlength in microns.
    """
    return centerline_spline(input_trans)


def fcorrect_as_pathlength(input_trans):
    '''Corrects for the angular dependence of the BM spectrum.
    First, use fconvert_data to get in terms of pathlength assuming we are
    in the ring plane.  Then, use this function to correct.
    '''
    angles = np.abs(np.arange(pathlength_image.shape[0]) - angular_fit_params['center_row'])
    angles *= angular_fit_params['pixel_size_mm'] / angular_fit_params['d_source_m'] * 1e3
    correction_factor = angle_spline(angles)
    return centerline_spline(input_trans) * correction_factor[:,None]


def fcorrect_as_transmission(input_trans):
    '''Corrects for beam hardening, putting data in terms of transmission at
    the equiv_energy.'''
    return fconvert_to_transmission(fconvert_to_pathlength(input_trans), equiv_energy)


def fset_up_calculations():
    '''Perform initial steps for analysis.
    Read in config file and source file.
    Apply appropriate filters.
    Find the mean energy of the beam that would be detected after the filters.
    '''
    fread_config_file()
    input_spectrum = fread_source_data()
    filtered_spectrum = fapply_filters(filters, input_spectrum)
    scint_spectrum = scintillator_material.fcompute_absorbed_spectrum(
                        scintillator_thickness, filtered_spectrum)
    equiv_energy = scint_spectrum.fmean_energy()
    ffind_calibration()
