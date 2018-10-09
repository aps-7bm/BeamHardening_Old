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
                                                                              spectral_energies,input_spectrum)
        detected_power[i] = scintillator_material.fcompute_absorbed_power(scintillator_thickness,spectral_energies,sample_filtered_power)
    #Compute an effective transmission vs. thickness
    sample_effective_trans = detected_power / detected_power[0]
    #This line to give fit to transmission vs. thickness
    return np.polyfit(np.log(sample_effective_trans),sample_thicknesses,5)

def fcompute_2d_lookup_coeffs(trans_order = 5, angle_order = 4):
    '''Performs a 2D least squares optimization to find the sample thickness
    given the transmission and the angle.
    '''
    angles_urad = np.array([0,5,10,15,20,30,40])
    transmissions = np.logspace(0.02,-2.48,500)
    sample_thicknesses = np.sort(np.concatenate((np.logspace(0,2.0,31),np.logspace(2.0,4.5,121)))) 
    A = np.zeros((sample_thicknesses.shape[0] * angles_urad.shape[0],trans_order + angle_order + 1))
    b = np.tile(sample_thicknesses,angles_urad.shape[0])
    for i,urad in enumerate(angles_urad):
        print('Opening new angle = {0:3d} urad.'.format(urad))
        spectral_data = np.genfromtxt(pwd+'Psi_{0:02d}urad.dat'.format(urad))
        spectral_energies = spectral_data[:-2,0]/1000
        spectral_power = spectral_data[:-2,1]
        #Filter the beam
        filtered_spectrum = fapply_filters(filters,spectral_energies,spectral_power)
        zero_thickness_power = 0
        for j in range(sample_thicknesses.size):
            sample_filtered_power = sample_material.fcompute_transmitted_spectrum(sample_thicknesses[j],
                                                                                  spectral_energies,filtered_spectrum)
            detected_power = scintillator_material.fcompute_absorbed_power(scintillator_thickness,spectral_energies,sample_filtered_power)
            if j == 0:
                zero_thickness_power = detected_power
            trans = detected_power / zero_thickness_power
            A_row = np.zeros((trans_order + angle_order + 1))
            A_row[0] = 1
            for k in range(trans_order):
                A_row[1 + k] = np.log(trans)**(k+1)
            for k in range(angle_order-1):
                A_row[1 + trans_order + k] = urad**(k+2)
            A_row[-1] = np.log(trans)**2 * urad**2
            A[j + sample_thicknesses.shape[0] * i,:] = A_row
    #Perform least squares fit
    output = np.linalg.lstsq(A,b)
    print(output)
    with h5py.File('Beam_Hardening_LUT_Raytheon.hdf5','w') as hdf_file:
        hdf_file.create_dataset('Angular_Fits',data=output[0])
    return(output[0])

def fevaluate_2d_lookup(coeffs,angle,trans):
    c = np.zeros((6,5))
    c[0,0] = coeffs[0]
    c[1:,0] = coeffs[1:6]
    c[0,2:] = coeffs[6:9]
    c[2,2] = coeffs[9]
    return np.polynomial.polynomial.polyval2d(np.log(trans),angle,c)

def ftest_2d_lookup(coeffs):
    plt.figure(1)
    transmissions = np.logspace(0.02,-2.48,500)
    plt.semilogy(fevaluate_2d_lookup(coeffs,0,transmissions),transmissions)
    plt.show()
    angles_urad = np.array([0,5,10,15,20,30,40])
    sample_thicknesses = np.sort(np.concatenate((np.logspace(0,2.0,31),np.logspace(2.0,4.5,121)))) 
    for i,urad in enumerate(angles_urad):
        print('Opening new angle = {0:3d} urad.'.format(urad))
        spectral_data = np.genfromtxt(pwd+'Psi_{0:02d}urad.dat'.format(urad))
        spectral_energies = spectral_data[:-2,0]/1000
        spectral_power = spectral_data[:-2,1]
        #Filter the beam
        filtered_spectrum = fapply_filters(filters,spectral_energies,spectral_power)
        effective_trans = np.zeros_like(sample_thicknesses)
        for j in range(sample_thicknesses.size):
            sample_filtered_power = sample_material.fcompute_transmitted_spectrum(sample_thicknesses[j],
                                                                                  spectral_energies,filtered_spectrum)
            effective_trans[j] = scintillator_material.fcompute_absorbed_power(scintillator_thickness,spectral_energies,sample_filtered_power)
        #Normalize by the case with no sample
        effective_trans /= effective_trans[0]
        plt.figure()
        plt.semilogy(sample_thicknesses,effective_trans,'r.')
        fit_thickness = fevaluate_2d_lookup(coeffs,urad,effective_trans)
        plt.semilogy(fit_thickness,effective_trans,'b.')
        plt.xlim(-100,1000)
        plt.ylim(0.1,1)
        plt.title('Angle = {0:d} urad'.format(urad))
        plt.show()

# coeffs = fcompute_2d_lookup_coeffs()
# ftest_2d_lookup(coeffs)
    
