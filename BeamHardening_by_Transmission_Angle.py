'''Code to correct for beam hardening in filtered white beam imaging experiments.

Alan Kastengren, XSD, APS

Started: November 11, 2015

Edits: Apr 27, 2016. Dan & Katie edited fcompute_lookup_table_function to sort the 
       values going into the lookup table. Our version of interp1d was unhappy with
       non monotonic x values and it spat out all NaNs. Also added 'pwd' to allow
       files to be stored somewhere else.
    
        June 16, 2017: several edits to make the code more generally usable.
            * Change to different files, from xCrossSec, for more materials.
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.integrate
import h5py
import os

#Read input spectral data from the data file created in XOP.
pwd = os.getcwd() + '/'
spectral_data = np.genfromtxt(pwd+'SRCOMPW',comments='!')
spectral_energies = spectral_data[:-2,0]/1000
spectral_power = spectral_data[:-2,1]
filtered_power = spectral_power

#Set the file to be used to write the coefficients to file.
hdf_fname = pwd + 'Beam_Hardening_LUT.hdf5'

#Global variables for when we convert images
trans_fit_coeffs = None
angle_fit_coeffs = None

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
    
    def fread_absorption_data(self):
        raw_data = np.genfromtxt(pwd + self.name + '_properties_xCrossSec.dat')
        self.energy_array = raw_data[:,0] / 1000.0      #in keV
        self.absorption_array = raw_data[:,3]   #in cm^2/g, from XCOM in XOP
        self.attenuation_array = raw_data[:,6]  #in cm^2/g, from XCOM in XOP, ignoring coherent scattering
    
    def interp_function(self,energies,absorptions):
        '''Return a function to interpolate logs of energies into logs of absorptions.
        '''
        return scipy.interpolate.interp1d(np.log(energies),np.log(absorptions),bounds_error=False)
    
    def finterpolate_absorption(self,input_energies):
        '''Interpolates on log-log scale and scales back
        '''
        return np.exp(self.absorption_interpolation_function(np.log(input_energies)))
    
    def finterpolate_attenuation(self,input_energies):
        '''Interpolates on log-log scale and scales back
        '''
        return np.exp(self.attenuation_interpolation_function(np.log(input_energies)))
    
    def fcompute_proj_density(self,thickness):
        '''Computes projected density from thickness and material density.
        Input: thickness in um
        Output: projected density in g/cm^2
        '''
        return thickness /1e4 * self.density
    
    def fcompute_transmitted_spectrum(self,thickness,input_energies,input_spectral_power):
        '''Computes the transmitted spectral power through a filter.
        Inputs:
        thickness: the thickness of the filter in um
        input_energies: numpy array of the energies for the spectrum in keV
        input_spectral_power: spectral power for input spectrum, in numpy array
        Output:
        spectral power at same energies as input.
        '''
        #Compute filter projected density
        filter_proj_density = self.fcompute_proj_density(thickness)
        #Find the spectral transmission using Beer-Lambert law
        return input_spectral_power *  np.exp(-self.finterpolate_attenuation(input_energies) * filter_proj_density)
    
    def fcompute_absorbed_spectrum(self,thickness,input_energies,input_spectral_power):
        '''Computes the absorbed power of a filter.
        Inputs:
        thickness: the thickness of the filter in um
        input_energies: numpy array of the energies for the spectrum in keV
        input_spectral_power: spectral power for input spectrum, in numpy array
        Output:
        absorbed power
        '''
        #Compute filter projected density
        filter_proj_density = self.fcompute_proj_density(thickness)
        #Find the spectral transmission using Beer-Lambert law
        return (np.ones_like(input_energies) - np.exp(-self.finterpolate_absorption(input_energies) * filter_proj_density)) * input_spectral_power
    
    def fcompute_absorbed_power(self,thickness,input_energies,input_spectral_power):
        '''Computes the absorbed power of a filter.
        Inputs:
        material: the Material object for the filter
        thickness: the thickness of the filter in um
        input_energies: numpy array of the energies for the spectrum in keV
        input_spectral_power: spectral power for input spectrum, in numpy array
        Output:
        absorbed power
        '''
        return scipy.integrate.simps(self.fcompute_absorbed_spectrum(thickness,input_energies,input_spectral_power),input_energies)
    
Be_filter = Material('Be',1.85)
Cu_filter = Material('Cu',8.96)
Mo_filter = Material('Mo',10.2)
Fe_filter = Material('Fe',7.87)
Si_filter = Material('Si',2.329)
Ge_filter = Material('Ge',5.323)
Ta_filter = Material('Ta',16.69)
Pb_filter = Material('Pb',11.34)
Inconel_filter = Material('Inconel625',8.44)
LuAG_scint = Material('LuAG_Ce',6.73)
YAG_scint = Material('YAG_Ce',4.56)
LYSO_scint = Material('LYSO_Ce',7.20)

#Variables to describe system
#Save filtering as dictionary of material and thicknesses
filters = {Be_filter:750.0,Cu_filter:250.0,Ge_filter:500.0}
sample_material = Inconel_filter
scintillator_material = LuAG_scint
scintillator_thickness = 100.0 
assumed_abs_coeff = 0.001 #In units of 1/microns

def fapply_filters(filters, input_energies=spectral_energies, input_spectral_power=spectral_power):
    '''Computes the spectrum after all filters.
        Inputs:
        filters: dictionary giving filter materials and thicknesses in microns.
        input_energies: numpy array of the energies for the spectrum in keV
        input_spectral_power: spectral power for input spectrum, in numpy array
        Output:
        spectral power transmitted through the filter set.
        '''
    temp_spectral_power = input_spectral_power
    for filter in filters.keys():
        temp_spectral_power = filter.fcompute_transmitted_spectrum(filters[filter],input_energies,temp_spectral_power)
    return temp_spectral_power

def fcompute_lookup_coeffs(input_energies, input_spectrum, order = 5):
    '''Makes a scipy interpolation function to be used to correct images.
    '''
    #Make an array of sample thicknesses
    sample_thicknesses = np.sort(np.concatenate((np.logspace(-1,1,300)[:-2],[0],np.logspace(-1,4.5,550)))) # The [:-2] removes points that overflow.
    #For each thickness, compute the absorbed power in the scintillator
    detected_power = np.zeros_like(sample_thicknesses)
    for i in range(sample_thicknesses.size):
        sample_filtered_power = sample_material.fcompute_transmitted_spectrum(sample_thicknesses[i],
                                                                              input_energies, input_spectrum)
        detected_power[i] = scintillator_material.fcompute_absorbed_power(scintillator_thickness,
                                                                          input_energies, sample_filtered_power)
    #Compute an effective transmission vs. thickness
    sample_effective_trans = detected_power / detected_power[0]
    #This line to give fit to transmission vs. thickness
    return np.polyfit(np.log(sample_effective_trans),sample_thicknesses, order)

def ffind_calibration_angular():
    '''Do the correlation at transmission of 10%.
    Treat the angular dependence as a correction on the thickness vs.
    transmission at angle = 0.
    '''
    angles_urad = np.array([0,5,10,15,20,30,40])
    cal_curve = []
    for i in angles_urad:
        spectral_data = np.genfromtxt(pwd+'Psi_{0:02d}urad.dat'.format(i))
        spectral_energies = spectral_data[:-2,0]/1000
        spectral_power = spectral_data[:-2,1]
        #Filter the beam
        filtered_spectrum = fapply_filters(filters,spectral_energies,spectral_power)
        #Create an interpolation function based on this
        sample_interp_coeffs = fcompute_lookup_coeffs(spectral_energies,filtered_spectrum)
        cal_curve.append(np.polyval(sample_interp_coeffs,np.log(0.1)))
    coeffs = np.polyfit(angles_urad,cal_curve,4)
    with h5py.File(hdf_fname,'r+') as hdf_file:
        hdf_file.create_dataset('Angular_Fits',data=coeffs)

def fsave_coeff_data():
    '''Convert transmission data to material pathlength in microns.
    '''
    #Filter the beam
    filtered_spectrum = fapply_filters(filters)
    #Create an interpolation function based on this
    transmissions = np.logspace(0.02,-2.48,500)
    sample_interp_coeffs = fcompute_lookup_coeffs(spectral_energies,filtered_spectrum)
    sample_thick_interp = np.polyval(sample_interp_coeffs,np.log(transmissions))
    with h5py.File(hdf_fname,'w') as hdf_file:
        hdf_file.create_dataset('Transmission',data=transmissions)
        hdf_file.create_dataset('Microns_Sample',data=sample_thick_interp)
        hdf_file.create_dataset('Fit_Coeff',data=sample_interp_coeffs)
        hdf_file.attrs['Sample Material'] = sample_material.name
        for filt_key in filters.keys():
            hdf_file.attrs['Filter {0:s} Thickness, um'.format(filt_key.name)] = filters[filt_key]
    ffind_calibration_angular()

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
    return np.polyval(trans_fit_coeffs,np.log(input_trans))

def fcorrect_angular(pathlength_image,center_row,d_source_m,pixel_size_mm):
    '''Corrects for the angular dependence of the BM spectrum.
    First, use fconvert_data to get in terms of pathlength assuming we are
    in the ring plane.  Then, use this function to correct.
    '''
    angles = np.abs(np.arange(pathlength_image.shape[0]) - center_row)
    angles *= pixel_size_mm / d_source_m * 1e3
    correction_factor = np.polyval(angle_fit_coeffs,angles)
    return pathlength_image * correction_factor[:,None]

def fmake_plots_angular():
    '''Make plots to see how the fits change for different angular positions.
    '''
    correlation_curves = []
    for i in [0,5,10,15,20,30,40]:
        spectral_data = np.genfromtxt(pwd+'Psi_{0:02d}urad.dat'.format(i))
        spectral_energies = spectral_data[:-2,0]/1000
        spectral_power = spectral_data[:-2,1]
        #Filter the beam
        filtered_spectrum = fapply_filters(filters,spectral_energies,spectral_power)
        #Create an interpolation function based on this
        transmissions = np.logspace(0.02,-2.48,500)
        sample_interp_coeffs = fcompute_lookup_coeffs(spectral_energies,filtered_spectrum)
        sample_thick_interp = np.polyval(sample_interp_coeffs,np.log(transmissions))
        correlation_curves.append(sample_thick_interp)
        plt.figure(1)
        plt.semilogy(sample_thick_interp,transmissions,label=r'{0:d} $\mu$rad'.format(i))
        plt.xlabel(r'Material Thickness, $\mu$m')
        plt.ylabel('Transmission')
        plt.figure(2)
        plt.plot(transmissions,correlation_curves[0]/sample_thick_interp,label=r'{0:d} $\mu$rad'.format(i))
    plt.figure(1)
    plt.legend(loc='lower right', fontsize=6)
    plt.figure(2)
    plt.legend(loc='lower right', fontsize=6)
    plt.show()

def fmake_plots():
    #Filter the beam
    filtered_spectrum = fapply_filters(filters)
    #Create an interpolation function based on this
    transmissions = np.logspace(0.02,-2.48,500)
    sample_interp_coeffs = fcompute_lookup_coeffs(spectral_energies,filtered_spectrum)
    sample_thick_interp = np.polyval(sample_interp_coeffs,np.log(transmissions))
    plt.figure()
    plt.plot(sample_thick_interp,transmissions)
    plt.xlabel(r'Material Thickness, $\mu$m')
    plt.ylabel('Transmission')
    plt.figure()
    plt.semilogy(sample_thick_interp,transmissions,label="Interp")
    plt.xlabel(r'Material Thickness, $\mu$m')
    plt.ylabel('Transmission')
    print(zip(np.log(transmissions),sample_thick_interp))
    for i in range(2,5):
        coeff = np.polyfit(np.log(transmissions),sample_thick_interp,i)
        plt.semilogy(np.polyval(coeff,np.log(transmissions)),transmissions,label=str(i))
    plt.legend(loc='upper right')
    plt.title('Plot of different polynomial fit orders')
    plt.xlabel('Fit Pathlength, microns')
    plt.ylabel('Transmission')
    #Compute the curve between actual and calculated transmission
    plt.figure(3)
    assumed_absorption_coefficient = 0.001
    assumed_transmission = np.exp(-assumed_absorption_coefficient * sample_thick_interp)
    plt.loglog(transmissions, assumed_transmission,'r.',label='Data')
    plt.figure(4)
    plt.plot(transmissions, assumed_transmission,'r.',label='Data')
    for i in range(2,5):
        coeff = np.polyfit(np.log(transmissions),np.log(assumed_transmission),i)
        print(coeff)
        plt.figure(3)
        plt.loglog(transmissions,np.exp(np.polyval(coeff,np.log(transmissions))),label=str(i))
        plt.figure(4)
        plt.plot(transmissions,np.exp(np.polyval(coeff,np.log(transmissions))),label=str(i))
    plt.legend(loc='upper right')
    plt.show()

fsave_coeff_data()
# fmake_plots()
# fmake_plots_angular()