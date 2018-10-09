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

pwd = os.getcwd() + '/'#+'/tomoPyScriptUtilities/'
spectral_data = np.genfromtxt(pwd+'SRCOMPW',comments='!')
spectral_energies = spectral_data[:-2,0]/1000
spectral_power = spectral_data[:-2,1]

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

Be_filter = Material('Be',1.85)
Cu_filter = Material('Cu',8.96)
Mo_filter = Material('Mo',10.2)
Fe_filter = Material('Fe',7.87)
Si_filter = Material('Si',2.329)
Ge_filter = Material('Ge',5.323)
Ta_filter = Material('Ta',16.69)
Inconel_filter = Material('Inconel625',8.44)
LuAG_scint = Material('LuAG_Ce',6.73)
YAG_scint = Material('YAG_Ce',4.56)
LYSO_scint = Material('LYSO_Ce',7.20)

def fcompute_transmitted_spectrum(material,thickness,input_energies,input_spectral_power):
    '''Computes the transmitted spectral power through a filter.
    Inputs:
    material: the Material object for the filter
    thickness: the thickness of the filter in um
    input_energies: numpy array of the energies for the spectrum in keV
    input_spectral_power: spectral power for input spectrum, in numpy array
    Output:
    spectral power at same energies as input.
    '''
    #Compute filter projected density
    filter_proj_density = material.fcompute_proj_density(thickness)
    #Find the spectral transmission using Beer-Lambert law
    spectral_trans = np.exp(-material.finterpolate_attenuation(input_energies) * filter_proj_density)
    #Return the spectral power
    return input_spectral_power * spectral_trans

def fcompute_absorbed_spectrum(material,thickness,input_energies,input_spectral_power):
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
    filter_proj_density = material.fcompute_proj_density(thickness)
    #Find the spectral transmission using Beer-Lambert law
    return (np.ones_like(input_energies) - np.exp(-material.finterpolate_absorption(input_energies) * filter_proj_density)) * input_spectral_power

def fcompute_absorbed_power(material,thickness,input_energies,input_spectral_power):
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
    filter_proj_density = material.fcompute_proj_density(thickness)
    #Find the spectral transmission using Beer-Lambert law
    spectral_abs = np.ones_like(input_energies) - np.exp(-material.finterpolate_absorption(input_energies) * filter_proj_density)
    return scipy.integrate.simps(spectral_abs * input_spectral_power,input_energies)

#Put in the filtering from the Be, Cu, Mo
plt.plot(spectral_energies,spectral_power,'k',label='Unfiltered')
filtered_power = fcompute_transmitted_spectrum(Be_filter,750,spectral_energies,spectral_power)
plt.plot(spectral_energies,filtered_power,'r',label='Be')
filtered_power = fcompute_transmitted_spectrum(Cu_filter,250,spectral_energies,filtered_power)
plt.plot(spectral_energies,filtered_power,'g',label='Cu')
filtered_power = fcompute_transmitted_spectrum(Ge_filter,500,spectral_energies,filtered_power)
plt.plot(spectral_energies,filtered_power,'b',label='Mo')
plt.legend(loc='upper right')
plt.show()

def fcompute_lookup_table_function():
    '''Makes a scipy interpolation function to be used to correct images.
    '''
    #Make an array of Fe thicknesses
    Fe_thicknesses = np.concatenate((-np.logspace(-1,2,300)[:-2],[0],np.logspace(-1,4.5,550))) # The [:-2] removes points that overflow.
    #print Fe_thicknesses
    detected_power = np.zeros_like(Fe_thicknesses)
    for i in range(Fe_thicknesses.size):
        #print i
        #print Fe_thicknesses[i]
        Fe_filtered_power = fcompute_transmitted_spectrum(Inconel_filter,Fe_thicknesses[i],spectral_energies,filtered_power)
        #print Fe_filtered_power[-1]
        detected_power[i] = fcompute_absorbed_power(LuAG_scint,100,spectral_energies,Fe_filtered_power)
        #print detected_power[i]
    
    Fe_effective_trans = detected_power / detected_power[0]
    
    #Create an interpolation function based on this.
    s=np.argsort(Fe_effective_trans) # Sort values (otherwise interp1d is unhappy)
    return scipy.interpolate.interp1d(Fe_effective_trans[s],Fe_thicknesses[s],bounds_error=False)

def fcompute_polynomial_function():
    '''Makes a scipy interpolation function to be used to correct images.
    '''
   #Create an interpolation function based on this
    transmissions = np.logspace(0.02,-2.48,500)
    Fe_thick_interp = fcompute_lookup_table_function()
    return np.polyfit(np.log(transmissions),Fe_thick_interp(transmissions),4)

def fcompute_polynomial_function_transmission():
    '''Makes a scipy interpolation function to be used to correct images.
    Takes the actual transmission and gives a function to convert it to the
    transmission if the beam were monochromatic.
    '''
   #Create an interpolation function based on this
    transmissions = np.logspace(0.02,-2.48,500)
    Fe_thick_interp = fcompute_lookup_table_function()
    return np.polyfit(np.log(transmissions),Fe_thick_interp(transmissions),4)

if __name__ == '__main__':
    #Create an interpolation function based on this
    transmissions = np.logspace(0.02,-2.48,500)
    Fe_thick_interp = fcompute_lookup_table_function()
    #print Fe_thick_interp(0.5)
    plt.figure()
    plt.plot(Fe_thick_interp(transmissions),transmissions)
    plt.figure()
    plt.semilogy(Fe_thick_interp(transmissions),transmissions,label="Interp")
    #plt.show() 
    print zip(np.log(transmissions),Fe_thick_interp(transmissions))
    for i in range(2,5):
        coeff = np.polyfit(np.log(transmissions),Fe_thick_interp(transmissions),i)
        print coeff
        plt.semilogy(np.polyval(coeff,np.log(transmissions)),transmissions,label=str(i))
    plt.legend(loc='upper right')
    #Compute the curve between actual and calculated transmission
    plt.figure(3)
    assumed_absorption_coefficient = 0.001
    assumed_transmission = np.exp(-assumed_absorption_coefficient * Fe_thick_interp(transmissions))
    plt.loglog(transmissions, assumed_transmission,'r.',label='Data')
    plt.figure(4)
    plt.plot(transmissions, assumed_transmission,'r.',label='Data')
    for i in range(2,11):
        coeff = np.polyfit(np.log(transmissions),np.log(assumed_transmission),i)
        print coeff
        plt.figure(3)
        plt.loglog(transmissions,np.exp(np.polyval(coeff,np.log(transmissions))),label=str(i))
        plt.figure(4)
        plt.plot(transmissions,np.exp(np.polyval(coeff,np.log(transmissions))),label=str(i))
    plt.legend(loc='upper right')
    plt.show()
    with h5py.File('Beam_Hardening_LUT.hdf5','w') as hdf_file:
        hdf_file.create_dataset('Transmission',data=transmissions)
        hdf_file.create_dataset('Microns_Fe',data=Fe_thick_interp(transmissions))
        hdf_file.create_dataset('Mono_Transmission',data=assumed_transmission)
        hdf_file['Mono_Transmission'].attrs['Abs_coeff'] = 0.001
        hdf_file.attrs['Notes'] = '''Created by Alan Kastengren, February 1, 2016.
    LUT between transmission and thickness of Fe, using the BM spectrum with 
    750 um of Be, 25 um Mo, and 250 um Cu.'''
