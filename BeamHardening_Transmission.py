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

pwd = os.getcwd() +'/tomoPyScriptUtilities/'
spectral_data = np.genfromtxt(pwd+'SRCOMPW',comments='!')
spectral_energies = spectral_data[:-2,0]/1000
spectral_power = spectral_data[:-2,1]
filtered_power = spectral_power

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
        raw_data = np.genfromtxt(pwd+self.name + '_properties_xCrossSec.dat')
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
        #Compute filter projected density
        filter_proj_density = self.fcompute_proj_density(thickness)
        return scipy.integrate.simps(self.fcompute_absorbed_spectrum(thickness,input_energies,input_spectral_power),input_energies)
    

Be_filter = Material('Be',1.85)
Cu_filter = Material('Cu',8.96)
Mo_filter = Material('Mo',10.2)
Fe_filter = Material('Fe',7.87)
Si_filter = Material('Si',2.329)
Ge_filter = Material('Ge',5.323)
Ta_filter = Material('Ta',16.69)
Pb_filter = Material('Pb',11.34)
LuAG_scint = Material('LuAG_Ce',6.73)
YAG_scint = Material('YAG_Ce',4.56)
LYSO_scint = Material('LYSO_Ce',7.20)

#Variables to describe system
#Save filtering as dictionary of material and thicknesses
filters = {Be_filter:750.0,Cu_filter:500.0,Mo_filter:25.0}
sample_material = Fe_filter
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

def fcompute_lookup_coeffs(input_energies,input_spectrum):
    '''Makes a scipy interpolation function to be used to correct images.
    '''
    #Make an array of sample thicknesses
    sample_thicknesses = np.sort(np.concatenate((np.logspace(-1,1,300)[:-2],[0],np.logspace(-1,4.5,550)))) # The [:-2] removes points that overflow.
    #For each thickness, compute the absorbed power in the scintillator
    detected_power = np.zeros_like(sample_thicknesses)
    for i in range(sample_thicknesses.size):
        sample_filtered_power = sample_material.fcompute_transmitted_spectrum(sample_thicknesses[i],
                                                                              spectral_energies,input_spectrum)
        detected_power[i] = scintillator_material.fcompute_absorbed_power(scintillator_thickness,spectral_energies,sample_filtered_power)
    #Compute an effective transmission vs. thickness
    sample_effective_trans = detected_power / detected_power[0]

    #Compute the transmission if this were mono beam at the assumed abs coeff
    sample_mono_trans = np.exp(-sample_thicknesses * assumed_abs_coeff)
    #This line for giving answer in terms of transmission if mono beam
#     coeff = np.polyfit(np.log(sample_effective_trans),np.log(sample_mono_trans),8)
    #This line to give in terms of pathlength
    return np.polyfit(np.log(sample_effective_trans),sample_thicknesses,5)
    