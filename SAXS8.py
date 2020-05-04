#!~/Enthought/Canopy_64bit/User/bin/python
# from matplotlib import cm
import sys
import os
# import matplotlib
# from matplotlib.figure import Figure
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
# from matplotlib import pyplot as plt
from scipy import special
from scipy import integrate
from scipy import polyfit
from scipy import stats
from scipy.interpolate import interpolate
from math import *
import numpy as np
from array import array
import re
# import cPickle
import SASM
# import SASExceptions

path = os.path.dirname(__file__)
if path not in sys.path:
    sys.path.append(path)

"""import SASImage

#This function loads the mask from a cfg file
def load_RAW_mask_from_cfg(fname,xpix,ypix):

    raw_settings = SASImage.RawGuiSettings()
    SASImage.loadSettings(raw_settings,fname)    

    masks=raw_settings.get('Masks')['BeamStopMask'][1]

    mask=SASImage.createMaskMatrix([ypix,xpix],masks)

    return mask, raw_settings

#this function loads the mask from a msk file
def load_RAW_mask_from_msk(fname, xpix, ypix):
    file_obj = open(fname, 'r')
    masks = cPickle.load(file_obj)
    file_obj.close()

    i=0
    for each in masks:
        each.maskID = i
        i = i + 1

    mask=SASImage.createMaskMatrix([ypix,xpix],masks)

    return mask
"""


class SAXS:

    def __init__(self, custom_path="", sample_type='Protein', qRg=1.3, a=150, d=0.2, pp=440.0e21, ps=334.6e21, t=4,
                 total_t=4, P=2.0e11, energy=10.0, mw=10, c=2, hr=2, shape='Sphere', salt_type='Water', salt_c=1,
                 mask="none", detector="100K"):
        """create a SAXS object for X-ray small angle scattering simulation,
           parameter definition see comments below
           
        """
        # custom path to save results (has default)
        self.custom_path = custom_path
        self.results_url = ""
        self.prof_name = "";

        # determine current dictory, as well as directory to store plots, relatively
        # REG: important for WebSAXS implementation

        self.app_path = os.path.abspath(os.path.join(__file__, '..')) + "/"

        if self.app_path not in sys.path:
            sys.path.append(self.app_path)

        self.app_url = re.sub(r'.*/wsgi/', '/', self.app_path)
        self.results_url = self.app_url + "results/"

        #
        # Unit conversions and constants
        #

        self.keV_to_J = 1.602176565e-16

        #
        # diagnostic output level
        #

        self.verbose = 2  # 0 = silent, 1 = concise, 2 = verbose

        #
        # Beamline parameters
        #

        # x-ray energy (keV)
        self.energy = energy
        # wavelength in Angstrom
        self.lam = 12.39842 / self.energy

        print (self.energy, " keV (", self.lam, " A)")

        # distance between sample and detector,cm-given (has default)
        self.a = a

        # thickness of sample, cm-given (has default)
        self.d = d

        # incident flux of photons, # photons/sec on sample 
        self.P = P

        # exposure time, in seconds
        self.t = t

        # total exposure time, before averaging. REG!! Make this an optional parameter to avoid confusion
        self.total_t = total_t

        # Detector parameters
        ##########################################################################################
        # Note: There are  a number of things that could be modeled here in the future.
        # Most important is the deviation of quantum efficiency across the face of the detector
        # Also, larger Pilatus models have dead zones between pixels.
        # It is more complicated to model a CCD detector, but possibly important for some users.
        ##########################################################################################

        # q value limits in the experiment,A**-1; default values, can be changed through create_Mask()
        self.q_min, self.q_max = 0.0098, 0.281
        # the q value at the larger edge of the mask
        self.alt_q_min = -1

        # Pilatus pixels are 172 microns, here pixel_size is in cm

        self.sensor_thickness = 0.0320

        self.detector = detector

        if detector == "100K":
            self.pixel_size = 0.0172
            # number of pixels in horizontal direction
            self.h_pixels = 195
            # number of pixels in vertical direction
            self.v_pixels = 487
        elif detector == "200K":
            self.pixel_size = 0.0172
            # number of pixels in horizontal direction
            self.h_pixels = 487
            # number of pixels in vertical direction
            self.v_pixels = 407
        elif detector == "300K":
            self.pixel_size = 0.0172
            # number of pixels in horizontal direction
            self.h_pixels = 619
            # number of pixels in vertical direction
            self.v_pixels = 487
        elif detector == "300KW":
            self.pixel_size = 0.0172
            # number of pixels in horizontal direction
            self.h_pixels = 195
            # number of pixels in vertical direction
            self.v_pixels = 1475
        elif detector == "1M":
            # Pilatus 1M pixels are 172 microns, here pixel_size is in cm
            self.pixel_size = 0.0172
            # number of pixels in horizontal direction
            self.h_pixels = 1043
            # number of pixels in vertical direction
            self.v_pixels = 981
        elif detector == "Eiger1M":
            # Pilatus 1M pixels are 172 microns, here pixel_size is in cm
            self.pixel_size = 0.0075
            # number of pixels in horizontal direction
            self.h_pixels = 1065
            # number of pixels in vertical direction
            self.v_pixels = 1030
            # sensor thickness
            self.sensor_thickness = 0.0450
        elif detector == "Eiger4M":
            # Pilatus 1M pixels are 172 microns, here pixel_size is in cm
            self.pixel_size = 0.0075
            # number of pixels in horizontal direction
            self.h_pixels = 2070
            # number of pixels in vertical direction
            self.v_pixels = 2167
            # sensor thickness
            self.sensor_thickness = 0.0450
        elif detector == "2M":
            # Pilatus 2M pixels are 172 microns, here pixel_size is in cm
            self.pixel_size = 0.0172
            # number of pixels in horizontal direction
            self.h_pixels = 1679
            # number of pixels in vertical direction
            self.v_pixels = 1475
        elif detector == "6M":
            # Pilatus 6M pixels are 172 microns, here pixel_size is in cm
            self.pixel_size = 0.0172
            # number of pixels in horizontal direction
            self.h_pixels = 2427
            # number of pixels in vertical direction
            self.v_pixels = 2463
        else:
            print ("unrecognized detector")

        print ("Detector: ", detector, " ", self.h_pixels, "(H) ", self.v_pixels, "(V) pixel size = ", self.pixel_size)

        # delta q  (resolution in q-space)
        self.dq = 4 * np.pi * np.sin(self.pixel_size / self.a / 2.0) / self.lam

        # print " calc_dq = ", self.dq

        # avg quantum efficiency (based on Pilatus at 10 keV, Donath et al. SRI 2012, J. Phys: Conference Series 425 (2013) 062001 )
        # note: this really should be a function of energy
        self.det_eff = self.QE(self.lam, self.sensor_thickness)
        # print "default QE (lam = ", self.lam, ") = ",self.det_eff, " d = ", self.sensor_thickness

        # Number of pixels per q bin
        self.NofPixels = []
        # The model of mask ... used mainly for web app where we want some pre-programmed choices
        # self.mask_model = mask_model

        # q arrays   REG: there is a lot of possible confusion here over which q arrays match which I and sigma arrays. Need to address that.

        # q array determined by detector config, by create_Mask(), goes from 0 to q_max 
        self.default_q = []
        # q array that will be determined by create_Mask(), goes from q_min to q_max
        self.mask_q = []
        # q values from experimental profile (read in)
        self.exp_q = []
        # 
        self.short_q = []

        # Intensities and sigmas

        # The I array that can be experimental raw I or interpolated I
        # Intensity: photon per unit time per q_bin
        self.exp_I = []
        # The sigma array that can be experimental raw sigma or interpolated sigma
        self.exp_Sig = []

        # Buffer and background profiles used to construct the model buffer profile

        # The buf_I array is buffer with or without empty cell being subtracted. 
        # When unsubtracted, you cannot model changes in windows and sample cell thickness correctly, but it can 
        # be useful for checking predicted noise levels from an ideal experimental background. The subtracted
        # version represents pure buffer. 

        self.buf_q = []
        self.buf_I = []
        self.buf_Sig = []  # not usually used
        # The starting q position of buffer data compare to specified q
        self.buf_q_start = -1

        # same as buf_ arrays, only for vacuum/instrumental scattering (optional)
        self.vac_q = []
        self.vac_I = []
        self.vac_Sig = []

        # same as buf_ arrays, only for vacuum/instrumental scattering (optional)
        self.win_q = []
        self.win_I = []
        self.win_Sig = []

        # final assembled buffer model used in noise calculation

        self.buf_model_q = []
        self.buf_model_I = []

        # I (calculated) at q points matching sigma below.

        self.Icalc = np.array([])

        # Sigma of buffer-subtracted I (calculated)
        self.sigma = np.array([])

        # I(0) of model in photons/pixel and cm-1

        self.I0_model = 0
        self.I0_model_abs = 0

        # Guinier fit parameters

        self.qRg = qRg  # q*Rg limit for Guinier fit (has default)
        self.Rg_noise = -1
        self.Rg_pure = -1
        self.rms_error_noise = -1
        self.rms_error_pure = -1
        self.I_0 = -1
        self.sigI_0 = -1
        self.I_0_pure = -1

        # Sample and Buffer parameters

        # molecule type of the sample of interest (has default)
        self.sample_type = sample_type

        # default density of buffer (g/cm**3)
        self.dens = 1.0
        # type of "salt" added to the buffer
        self.salt_type = salt_type
        # concentration of "salt" in the buffer;millimol/L for NaCl, volume % for Glycerol
        self.salt_c = salt_c
        # molecular weight, kDa-given
        self.mw = mw
        # concentration of solute,mg/cm***3-given
        self.c = c
        # ratio of Height to Radius for a cylinder-given
        self.hr = hr
        # specific volume of solute,cm**3/g, reference:Mylonas
        if sample_type == "Protein":
            self.v = 0.7425
        elif sample_type == "RNA":
            self.v = 0.569
        # shape of solute, Sphere or Cylinder-given
        self.shape = shape
        # Volume of the solute molecule, cm**3-estimated
        self.V = self.find_V(mw)
        # self.V = 1
        # figure out R,given mw (and hr if necessary)
        self.R = 0
        self.H = 0
        self.D_max = 0
        self.set_R(self.mw, self.hr)

        # default electron density, ro, of protein, number of electrons per centimeter**3
        # reference:Svergun, D. I. & Koch, M. H. J. (2003). Reports on Progress in Physics 66, 1735-1782.
        if sample_type == 'Protein':
            self.pp = pp
        elif sample_type == 'RNA':
            self.pp = 550.0e21


        # electron density, ro, of solvent, number of electrons per centimeter**3
        # REG: these are assuming additive volume ... I think. Should check this.
        # change in density with concentration will make a significant difference in case of glycerol at least.

        if self.salt_type == 'NaCl':
            self.ps = ps + salt_c * 28 * 6.02e17  # LL: What is this 28?
        elif self.salt_type == 'Glycerol':
            self.ps = (1.0 - salt_c / 100.0) * ps + salt_c / 100.0 * 413e21
        elif self.salt_type == 'Water':
            self.ps = ps
        else:
            self.ps = (1.0 - salt_c / 100.0) * ps + salt_c / 100.0 * 466e21

        # =contrast excess scattering density, cm/cm**3-given;
        # for electron radius, plz refer to http://en.wikipedia.org/wiki/Electron_radius
        # This variable is used in I_of_q
        self.p = (self.pp - self.ps) * 2.818e-13

        # =contrast is positive # LL: What's the use of this??????
        if self.p > 0:
            self.contrast_positive = True
        else:
            self.contrast_positive = False

        # self.model_Mask(self.mask_model)
        # self.load_buf(interpolate=True,q_array = self.default_q)       

        # Using power law approximation for water mu

        self.I_I0_ratio = exp(-1.065)
        self.mu = 2.8 * self.lam ** 3

        #
        # based on my comparison with NIST data (see water_mu.py in the Pilatus_QE folder), the power law is valid
        # from lambda = 0.62 (20 keV, about -15% error) up to lambda = 8.0 ( 1.5 keV, about +15% error)
        #
        if (self.lam < 0.6): print ("WARNING: mu power-law approximation breaks down at this energy")
        # print "optimum pathlength = ", 1.0/(2.8*self.lam**3), "attenuation with d = ",self.d," is ", exp(-self.d*2.8*self.lam**3)
        # print "percent error in transmission with +/- 200 um in d", 1.0-exp(-0.02*2.8*self.lam**3)
        # print "percent error in I(0) with +/- 200 um in d", ((self.d + 0.02)/self.d*exp(-(0.02)*2.8*self.lam**3) - 1.0)

        # window material and thickness

        self.d_win = 0.005  # total window thickness, units = cm
        self.mu_win = 55.68  # measured attenuation for our Ruby muscovite (mica) - T = 0.757 11/07/15
        self.mu_factor = 29.5218  # fitted prefactor (see set_window function)

        # x-ray absorption;reference for u:http://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html

        self.dose = 0.0  # x-ray dose on sample (calculated)
        self.MaxQ = -1.0

        self.name = "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%.2g" % (
        self.detector, self.c, self.mw, self.t, self.shape, self.hr, self.qRg, self.salt_type, self.salt_c, self.energy,
        self.a, self.P)
        print ("--------------")

    ######################################################### Methods starts here#########################################################
    def set_energy(self, energy):
        """ Must set window type and sensor thickness first!
        also be aware the q-space range determined from the mask is 
        wavelength dependent and this routine does not recalculate that. """
        self.energy = energy
        # wavelength in Angstrom
        self.lam = 12.39842 / self.energy
        # Using power law approximation for water mu
        self.mu = 2.8 * self.lam ** 3
        # Using power law approximation for window materials
        self.mu_win = self.mu_factor * self.lam ** 2.9
        # detector quantum efficiency
        self.det_eff = self.QE(self.lam, self.sensor_thickness)
        # q increment per pixel (used by create_mask)
        self.dq = 4 * np.pi * np.sin(self.pixel_size / self.a / 2.0) / self.lam
        # 
        self.name = "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%.2g" % (
        self.detector, self.c, self.mw, self.t, self.shape, self.hr, self.qRg, self.salt_type, self.salt_c, self.energy,
        self.a, self.P)

        def set_wavelength(self, lam):
            self.lam = lam

        # wavelength in Angstrom
        self.energy = 12.39842 / self.lam
        # Using power law approximation for water mu
        self.mu = 2.8 * self.lam ** 3
        # Using power law approximation for window materials
        self.mu_win = self.mu_factor * self.lam ** 2.9
        # detector quantum efficiency
        self.det_eff = self.QE(self.lam, self.sensor_thickness)
        #
        self.name = "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%.2g" % (
        self.detector, self.c, self.mw, self.t, self.shape, self.hr, self.qRg, self.salt_type, self.salt_c, self.energy,
        self.a, self.P)
        print ('Energy was reset to %s corresponding to %s Angstroms' % (str(self.energy), str(self.lam)) + '' + 'For the simulated data')

    def set_window(self, type):

        # sample window absorption. Known thickness, mu_win determined by transmission measurement.
        # energy dependence is assumed lam**2.9 (based on fitting NIST data for light atoms)
        # prefactor is chosen to match the measured Transmission at specified wavelength

        if type == 'mica':
            self.d_win = 0.005  # total window thickness, units = cm
            # measured attenuation for our Ruby muscovite (mica) - T = 0.757 11/07/15
            # wavelength = 1.24457
            # 55.68 = -log(0.757)/0.005
            # (55.68/1.24457**2.9) = 29.521823
            self.mu_factor = 29.5218
            self.mu_win = self.mu_factor * self.lam ** 2.9

        elif type == 'glass film':
            # measured attenuation for Nippon Electric glass ribbon - T = 0.97 05/21/16
            # approximate figure close to NIST value for SiO2 dens = 2.5 - 2.2 - T = 0.96
            # 30.46 = -log(0.97)/0.001
            # mu_factor = (30.46/1.24457**2.9) = 16.15
            self.d_win = 0.001  # total window thickness, units = cm
            self.mu_factor = 16.15
            self.mu_win = self.mu_factor * self.lam ** 2.9

        elif type == 'quartz':
            self.d_win = 0.002  # total window thickness, units = cm
            self.mu_factor = 0.0
            self.mu_win = self.mu_factor * self.lam ** 2.9

        elif type == 'Hilgenberg':  # custom quartz capillaries used at P12 (50 um thick, ID = 1.7 mm)
            # measured transmission (see notebook) T = 0.7/0.9 = 0.7777
            #   25.13 = -log(0.7777)/0.01
            # mu_factor = (25.13/1.24457**2.9) = 13.32
            self.d_win = 0.01  # total window thickness, units = cm
            self.mu_factor = 13.32  # borrowed from glass film
            self.mu_win = self.mu_factor * self.lam ** 2.9

        elif type == 'special glass':
            self.d_win = 0.002  # total window thickness, units = cm
            self.mu_factor = 0.0
            self.mu_win = self.mu_factor * self.lam ** 2.9

        elif type == 'polystyrene':
            # measured attenuation for PS film from Goodfellow - T = 0.979 (Transactions paper)
            # NIST value for PS  T = 0.988 = exp(-2.219*1.04*0.005)    1.04 g/cm**3 = density of PS
            #  4.2447 = -log(0.979)/0.005 (2X25 um)
            # mu_factor = (47.1/1.24457**2.9) = 2.25056
            self.d_win = 0.005  # total window thickness, units = cm
            self.mu_factor = 2.25056
            self.mu_win = self.mu_factor * self.lam ** 2.9

        elif type == '7mu kapton':
            self.d_win = 0.0014  # total window thickness, units = cm
            self.mu_factor = 0.0
            self.mu_win = self.mu_factor * self.lam ** 2.9

        else:
            print ("Unrecognized type! Choose: mica, glass film, quartz, Hilgenberg, special glass, polystyrene, or 7mu kapton")
            self.d_win = 0.001  # total window thickness, units = cm
            self.mu_factor = 0.0
            self.mu_win = self.mu_factor * self.lam ** 2.9

        # print "window type: ", type, " thickness = ", self.d_win, " mu = ", self.mu_win, " transmission = ", exp(-self.mu_win*self.d_win)

    def print_settings(self):
        print ("energy: ", self.energy)
        print ("wavelength: %5.3f " % self.lam)
        print ("sample-det (cm): ", self.a)
        print ("sample thickness: ", self.d)
        print ("flux: %4.2e " % self.P)
        print ("exposure time: ", self.t)

    def I_of_q(self, c, mw, q):
        """ Calculate smooth I(q) based on reference:Stuhrmann, H. B. (1980). Small angle x-ray scattering of macromolecules in solution.
        In Synchrotron Radiation Research (Winick, H., Doniach, S., ed.), pp. 513-531. Plenum Press, New York.
        """
        V = np.float(self.V)
        R = np.float(self.R)

        self.name = "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%.2g" % (
        self.detector, c, mw, self.t, self.shape, self.hr, self.qRg, self.salt_type, self.salt_c, self.energy, self.a,
        self.P)

        """
        print "lam", self.lam
        print "energy", self.energy
        print "det_eff", self.det_eff
        print "I_I0_ratio", self.I_I0_ratio
        print "P",self.P
        print "a",self.a
        print "v",self.v
        print "p",self.p
        print "mw",self.mw
        print 'd',self.d
        print 'c',c
        print "t",self.t
        print 'FF(q=0)',self.FF(0.0)
        """
        # convert c from mg/ml to g/cm**3, MW from kDa to g, (note: 1000's cancel), P is photons/s entering the sample. 
        # 
        # print "solid angle correction range: ", np.max(self.SolidAngCor(q)), np.min(self.SolidAngCor(q))

        self.mu_win = np.float(self.mu_factor) * np.float(self.lam) ** 2.9  # energy dependence of window absorption
        # print "mu_win = ", self.mu_win

        self.mu = 2.8 * self.lam ** 3  # energy dependence of water absorption

        I0abs = (np.float(self.v) * np.float(self.p)) ** 2 * np.float(self.mw) * np.float(c) / (6.022e23)
        PreFac = exp(-self.mu_win * self.d_win) * self.det_eff * exp(-self.mu * self.d) * self.P / (
                    self.a ** 2) * self.d
        I0 = PreFac * I0abs

        # print "I(0) = ", I0*self.t*(self.pixel_size)**2, "(photons/pixel)"
        # print "I(0) = ", I0abs, " (cm-1)"
        # Full Prefactor converts cm-1 to photons/pixel
        # print "Full Prefactor = ", PreFac*self.t*(self.pixel_size)**2
        # print "window transmission: ",exp(-self.mu_win*self.d_win)
        # print "water  transmission: ",exp(-self.mu*self.d)

        I = I0 * self.FF(q)

        self.I0_model = I0
        self.I0_model_abs = I0abs

        # Return I: photons per cm**2 per unit time
        # multiply by pixel area (cm**2) and by exposure time to get actual measured intensity (average photons per pixel at each sampled q bin.
        # This formula assumes a delta-function beam profile. Finite beam profile should be implemented someday, but won't make much difference in counts.

        return I

    # RM!

    def I_of_q_input_rho(self, c, mw, q, p):
        """ Calculate smooth I(q) based on reference:Stuhrmann, H. B. (1980). Small angle x-ray scattering of macromolecules in solution.
        In Synchrotron Radiation Research (Winick, H., Doniach, S., ed.), pp. 513-531. Plenum Press, New York.
        """
        V = self.V
        R = self.R

        self.name = "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%.2g" % (
        self.detector, c, mw, self.t, self.shape, self.hr, self.qRg, self.salt_type, self.salt_c, self.energy, self.a,
        self.P)

        """
        print "lam", self.lam
        print "energy", self.energy
        print "det_eff", self.det_eff
        print "I_I0_ratio", self.I_I0_ratio
        print "P",self.P
        print "a",self.a
        print "v",self.v
        print "p",self.p
        print "mw",self.mw
        print 'd',self.d
        print 'c',c
        print "t",self.t
        print 'FF(q=0)',self.FF(0.0)
        """
        # convert c from mg/ml to g/cm**3, MW from kDa to g, (note: 1000's cancel), P is photons/s entering the sample.
        #
        # print "solid angle correction range: ", np.max(self.SolidAngCor(q)), np.min(self.SolidAngCor(q))

        self.mu_win = self.mu_factor * self.lam ** 2.9  # energy dependence of window absorption
        # print "mu_win = ", self.mu_win

        self.mu = 2.8 * self.lam ** 3  # energy dependence of water absorption

        I0abs = (np.float(self.v) * p) ** 2 * np.float(self.mw) * np.float(c) / (6.022e23)
        PreFac = exp(-self.mu_win * self.d_win) * self.det_eff * exp(-self.mu * self.d) * self.P / (
                    self.a ** 2) * self.d
        I0 = PreFac * I0abs

        # print "I(0) = ", I0*self.t*(self.pixel_size)**2, "(photons/pixel)"
        # print "I(0) = ", I0abs, " (cm-1)"
        # Full Prefactor converts cm-1 to photons/pixel
        # print "Full Prefactor = ", PreFac*self.t*(self.pixel_size)**2
        # print "window transmission: ",exp(-self.mu_win*self.d_win)
        # print "water  transmission: ",exp(-self.mu*self.d)

        I = I0 * self.FF(q)

        self.I0_model = I0
        self.I0_model_abs = I0abs

        # Return I: photons per cm**2 per unit time
        # multiply by pixel area (cm**2) and by exposure time to get actual measured intensity (average photons per pixel at each sampled q bin.
        # This formula assumes a delta-function beam profile. Finite beam profile should be implemented someday, but won't make much difference in counts.

        return I

    def I_of_q_variable_contrast(self, c, mw, q, p):
        """ Calculate smooth I(q) based on reference:Stuhrmann, H. B. (1980). Small angle x-ray scattering of macromolecules in solution.
        In Synchrotron Radiation Research (Winick, H., Doniach, S., ed.), pp. 513-531. Plenum Press, New York.

        RM! Currently, this script exports an array of arrays, with length equal to the length of the input contrast list.
        If the user returns I[0], the first array (i.e. the scattering curve at the first contrast value) will be returned.
        Need to clean up the output possibly.
        """
        V = self.V
        R = self.R

        self.name = "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%.2g" % (
        self.detector, c, mw, self.t, self.shape, self.hr, self.qRg, self.salt_type, self.salt_c, self.energy, self.a,
        self.P)
        # self.p = p
        """
        print "lam", self.lam
        print "energy", self.energy
        print "det_eff", self.det_eff
        print "I_I0_ratio", self.I_I0_ratio
        print "P",self.P
        print "a",self.a
        print "v",self.v
        print "p",self.p
        print "mw",self.mw
        print 'd',self.d
        print 'c',c
        print "t",self.t
        print 'FF(q=0)',self.FF(0.0)
        """
        # convert c from mg/ml to g/cm**3, MW from kDa to g, (note: 1000's cancel), P is photons/s entering the sample.
        #
        # print "solid angle correction range: ", np.max(self.SolidAngCor(q)), np.min(self.SolidAngCor(q))

        self.mu_win = self.mu_factor * self.lam ** 2.9  # energy dependence of window absorption
        # print "mu_win = ", self.mu_win

        self.mu = 2.8 * self.lam ** 3  # energy dependence of water absorption

        # RM! Generate series of I0abs values for every contrast value.

        I0abs = []
        for i in range(0, len(p)):
            I0abs.append((np.float(self.v) * p[i]) ** 2 * np.float(self.mw) * np.float(c) / (6.022e23))

        # I0abs = (self.v * self.p) ** 2 * self.mw * c / (6.022e23)
        # PreFac = exp(-self.mu_win * self.d_win) * self.det_eff * exp(-self.mu * self.d) * self.P / (
        #             self.a ** 2) * self.d

        I0 = []
        preFac = exp(-self.mu_win * self.d_win) * self.det_eff * exp(-self.mu * self.d) * self.P / (
                self.a ** 2) * self.d
        for i in range(0, len(I0abs)):
            I0.append(preFac * I0abs[i])

        # print "I(0) = ", I0*self.t*(self.pixel_size)**2, "(photons/pixel)"
        # print "I(0) = ", I0abs, " (cm-1)"
        # Full Prefactor converts cm-1 to photons/pixel
        # print "Full Prefactor = ", PreFac*self.t*(self.pixel_size)**2
        # print "window transmission: ",exp(-self.mu_win*self.d_win)
        # print "water  transmission: ",exp(-self.mu*self.d)

        I = []
        for i in range(0, len(I0)):
            I.append(I0[i] * self.FF(q))

        # I = I0 * self.FF(q)

        self.I0_model = I0
        self.I0_model_abs = I0abs

        # Return I: photons per cm**2 per unit time
        # multiply by pixel area (cm**2) and by exposure time to get actual measured intensity (average photons per pixel at each sampled q bin.
        # This formula assumes a delta-function beam profile. Finite beam profile should be implemented someday, but won't make much difference in counts.

        return I

    def with_noise(self, t, q, I):
        """ Create a scattering profile with simluated noise. I = theoretical scattering profile of molecule in vacuo. 
        Convert I to # of photons, using exposure time, then return a sample for each q from a Poisson distribution
        

        # The goal here is to have three arrays all match properly: computed intensity I,
        # number of pixels per q-bin self.NofPixels, and buffer self.buf_q. All of these array values 
        # need to be interpolated onto self.mask_q which is the q-vector derived from the detector model. 
        # NofPixels is already matched to self.mask_q by virtue of how it was created.
        # 

        # Note: technically I is mutable, but I beleive the line below effectively creates a new local reference
        # so that changes to it do not effect the value of the I outside the scope of this function.

        # trim self.mask_q and self.NoPixels arrays to fit within the given buffer and model profiles
        # self.hr =
        # self.qRg =
        # self.a =
        #
        """
        # RM! Annotation
        ## Fill self.name array with relevant parameters
        self.name = "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%.2g" % (
        self.detector, self.c, self.mw, t, self.shape, self.hr, self.qRg, self.salt_type, self.salt_c, self.energy,
        self.a, self.P)

        ## Define q-range
        #
        start1, stop1 = self.find_q_range(self.mask_q, q)
        temp_mask_q = self.mask_q[start1:stop1 + 1]
        temp_NoPix = self.NofPixels[start1:stop1 + 1]

        start1, stop1 = self.find_q_range(temp_mask_q, self.buf_model_q)
        self.mask_q_short = temp_mask_q[start1:stop1 + 1]
        NofPixels_short = temp_NoPix[start1:stop1 + 1]

        # print "trimmed mask_q: [",self.mask_q[0],self.mask_q_short[0],self.mask_q_short[-1],self.mask_q[-1],"]"

        # interpolate I and buf_I onto mask_q_short

        I_short = self.interpol(self.mask_q_short, q, I)
        buf_I_short = self.interpol(self.mask_q_short, self.buf_model_q, self.buf_model_I)

        self.Icalc = I_short  # store in-register (with mask_q_short) calculated intensity

        # print "interpolation q bounds:", self.short_q[0],temp_mask_short_q[0]
        # print "interpolation q max bounds:", self.short_q[-1],temp_mask_short_q[-1]
        # print "lengths = ", len(self.short_q), len(buf), len(I), len(temp_NofPixels)

        # temp_NofPixels = self.NofPixels[start2:stop2+1]

        ave_ratio = self.total_t / self.t  # accounts for averaging over multiple frames, if present

        ave_ratio = 1.0  # REG!! self.total function has been disabled for safety!! This needs to be fixed if you want average calculations"

        # print "ave_ratio ",ave_ratio, self.total_t, self.t, t

        # flux_ratio = self.P/self.bufP

        # print "ave_ratio = ",ave_ratio, "flux_ratio = ", flux_ratio

        # convert to total number of photons per q-bin
        # compute total number of photons from both protein and buffer (note buffer contributes twice)
        # temp_NofPixels term is number of pixels per bin

        # Important: the factor of 2 in the formula for "Buffer" below is necessary to give the correct Poisson noise level
        # for the difference between (protein in vacuo + buffer) and buffer. We could have defined two curves 
        # and generated independent deviates for each, then subtracted them, but that seems inefficient. 
        # The factor of 2 is rigorously equivalent because sigmas ad in quadrature. We're basically boosting the 
        # intensity up temporarily for the purpose of calculating the Poisson noise.
        # note: pixel size is in cm
        #
        # REG!! also, this formula assumes the protein is dilute. We should probably introduce a 
        # volume fraction term to reduce the solvent contribution slightly.

        # note that I_short already contains flux, but buf_I_short does not (it was divided out)

        # IMPORTANT: energy dependence is already contained in input buffer (self.buf_model_I) 
        # buf_I_short contains window and sample transmission corrections when simulated

        Buffer = 2 * np.array(NofPixels_short) * np.array(buf_I_short) * self.P * self.det_eff / (self.a ** 2)

        I = (I_short * t * np.array(NofPixels_short) * (self.pixel_size) ** 2 + Buffer * t) * ave_ratio

        # Here I becomes photon counts per q_bin with buffer (buf is already photon per q-bin per time)
        # LL: multiplying buf with t makes sense, since N_buf was divided out already in the loading process.

        I_out = np.array([])

        # LL: with_noise() is called multiple times, this makes sure self.sigma is 
        # not appended multiple times.

        self.sigma = np.array([])
        for x in I:
            I_out = np.append(I_out, np.random.poisson(x))
            self.sigma = np.append(self.sigma, np.sqrt(x))

        # need to convert from sum back to average sigma per pixel
        # should this be divided by temp_NofPixels? REG!!

        self.sigma = self.sigma / (NofPixels_short * ave_ratio)

        # print "with_noise(), I_out: ", I_out
        # Now set the amplitude back to the in-vacuo level and 
        # convert it to intensity

        I_out = (I_out / (t * ave_ratio) - Buffer) / NofPixels_short / (self.pixel_size) ** 2

        # I_out = (I_out/(t*ave_ratio)-2*buf_I_short*self.P*NofPixels_short)/NofPixels_short/(self.pixel_size)**2

        # LL: I changed the divided by t to after I_out because, N_buf is already intensity, not photon per bin per area.
        # This function returns photon per time per unit area

        return I_out

    def absorbed_dose(self, t, beam_h, beam_v):
        """ x-ray dose on sample reported in grays (J/kg)
            t = exposure time
            beam_h, beam_v = horizontal and vertical beam diameters (cm)
        """
        # print "t ",t, "flux ", self.P," T ", exp(-self.mu*self.d)," energy ", self.energy, " conv factor ",self.keV_to_J
        self.mu = 2.8 * self.lam ** 3

        # window transmission 
        self.mu_win = self.mu_factor * self.lam ** 2.9  # energy dependence of window absorption
        Trans_win = exp(-self.mu_win * self.d_win / 2.0)  # divide by 2.0 because only one window

        # note: factor of 1000 converts g to kg

        self.dose = t * self.P * Trans_win * self.energy * (1.0 - exp(-self.mu * self.d)) / (
                    beam_h * beam_v * self.d * self.dens) * self.keV_to_J * 1000.0
        return self.dose

    def maximum_exposure(self, Dose, beam_h, beam_v):
        """ x-ray dose on sample reported in grays (J/kg)
            t = exposure time
            beam_h, beam_v = horizontal and vertical beam diameters (cm)
        """
        self.dose = Dose

        self.mu = 2.8 * self.lam ** 3

        # window transmission 
        self.mu_win = self.mu_factor * self.lam ** 2.9  # energy dependence of window absorption
        Trans_win = exp(-self.mu_win * self.d_win / 2.0)  # divide by 2 because only one window!

        # note: factor of 1000 converts g to kg

        self.max_exposure = self.dose * (beam_h * beam_v * self.d * self.dens) / (
                    self.P * self.energy * Trans_win * (1.0 - exp(-self.mu * self.d)) * self.keV_to_J * 1000.0)

        # print "flux ", self.P," T ", exp(-self.mu*self.d)," energy ", self.energy, " Trans_win: ", Trans_win, " max exposure: ", self.max_exposure
        # print self.P*self.energy, Trans_win, 1.0-exp(-self.mu*self.d), self.keV_to_J*1000.0

        return self.max_exposure

    def volume_consumed(self, t, Dose):

        # volume of sampled consumed in cm**3

        self.mu = 2.8 * self.lam ** 3

        # window transmission 
        self.mu_win = self.mu_factor * self.lam ** 2.9  # energy dependence of window absorption
        Trans_win = exp(-self.mu_win * self.d_win / 2.0)  # divide by 2 because only one window!

        # note: factor of 1000 converts g to kg

        Volume = t * self.P * self.energy * Trans_win * (1.0 - exp(-self.mu * self.d)) * self.keV_to_J * 1000.0 / (
                    Dose * self.dens)

        return Volume

    def equivalent_detector_distance(self, current_a, current_E, new_E, q_max):
        """ new calculate sample-to-detector distance that would give the equivalent q-space 
        range at a new energy. For small angles, this expression is independent of q_max.
        """

        lam0 = 12.39842 / current_E
        lam1 = 12.39842 / new_E

        tan0 = np.tan(2 * np.arcsin(q_max * lam0 / (4 * np.pi)))
        tan1 = np.tan(2 * np.arcsin(q_max * lam1 / (4 * np.pi)))

        return current_a * tan0 / tan1

    # Note: this function is commented out in Luna's code ... check it
    def get_count_noise(self, q):
        """ Simulated actual data points, i.e. number of counts per pixel."""
        I = self.I_of_q(self.c, self.mw, q)
        N = self.t * (self.pixel_size) ** 2 * self.with_noise(self.t, q, I)
        # When self.with_noise() is called, I is interpolated onto self.buf_q (part of self.default_q)
        # Hence we need to find the range of self.buf_q within self.default_q
        # Also that's why we can safely divide sqrt(N)
        start, stop = self.find_q_range(self.default_q, self.buf_q)
        temp_NofPixels = self.NofPixels[start:stop + 1]
        N = N * temp_NofPixels
        ## Why divided by self.NofPixels. Then sigma^2 becomes photons per pixel while 
        ## N(intensity) is photon per q bin???
        Sigma = self.sigma

        return N, Sigma

    def find_V(self, mw):
        """given MW(g), returns volume of protein (A**3) approx."""
        volume = np.float(self.v) / 6.02 * np.float(mw) * 9.996 * 1000
        return volume

    def set_R(self, mw, hr):
        """given MW(g/mol), returns radius of protein (A) approx."""
        if self.shape == 'Sphere':
            radius = (self.V * 3 / 4 / np.pi) ** (1.0 / 3)
            self.R = radius
            self.D_max = radius * 2
        else:
            self.R = (self.V / hr / np.pi) ** (1.0 / 3)
            self.H = hr * self.R
            self.D_max = max((self.R * 2), self.H)

    # Luna's code has a WAXS option here. Check this later

    # def model_Mask(self,model ='G1'):
    #    """ A simple function that calls create_Mask() to model
    #    G1 or F2 station"""
    #    if model == 'none':
    #        self.create_Mask(97, 5, 0, 0, wedge=360.0, type="rectangle")
    #    if model == 'G1':
    #        self.create_Mask(97, 5, 40, 10, wedge=360.0, type="rectangle")
    #    if model == 'G1_02_15':
    #        self.create_Mask(129, 10, 40, 12, wedge=360.0, type="rectangle")
    #    elif model == "F2":
    #        self.create_Mask(97, 5, 20, 15, wedge=73.0, type="rectangle")
    #    else:
    #        print "Warning; Provide a valid flag for model, F2 or G1, default is none."
    #    self.mask_model = model
    #    print "model_Mask(): Your model is :", model

    def create_Mask(self, beam_i, beam_j, stop_ri, stop_rj, wedge=23.0, type="rectangle"):

        """ assigns number of pixels to each q bin based on choice of detector mask

        NOTE: NofPixels = array containing counts per q-bin
              mask_q    = array of corresponding q values (covers only unmasked q-range)
              default_q = same array, but going all the way to q = 0

        !!! for wedge angle less than 180, must have "and" instead of "or" REG:!!!

        beam_i = position of beam center horizontal (pixel coordinates)
        beam_j = position of beam center vertical   (pixel coordinates)
        stop_ri = pixels from beam center to edge of beamstop (horizontal)
        stop_rj = pixels from beam center to edge of beamstop (vertical)
        wedge = angle of integration wedge (degrees)
        type = shape of beamstop: rectangle or disk
        By defautl it models G1, but F2 could used used as a flag.

        IMPORTANT: the current version makes the small-angle approximation in q.
        For Pilatus 100K at 1500 mm (lengthwise), that's less than 0.1% error, however. But this code 
        should be modified for WAXS. q = 2.4A-1 (10keV) will give 1% error. 

        Also note that the q-space range calculated here is wavelength-dependent! In other words,
        the q values of the pixels on a detector surface depend on distance and wavelength.

        """

        print ("New mask created: ", beam_i, beam_j, stop_ri, stop_rj, wedge, type)

        # print len(q),beam_i,beam_j,stop_ri,stop_rj,wedge,type

        # express the integration wedge as two intersecting lines in normal form passing through origin,
        # this way it's easy to test if an (i,j) is in between them without concern for special cases (theta = 0 or 180)
        # LL: I added the factor of 2 here. Conversion from degree to radian.
        cos_wedge = np.cos(2 * np.pi * wedge / 360.0)
        sin_wedge = np.sin(2 * np.pi * wedge / 360.0)

        # print "A B",A,B

        # determine maximum possible distance from beam to upper corner of detector (in pixels)
        cUL = np.sqrt((self.v_pixels - beam_j) ** 2 + (beam_i) ** 2)
        cUR = np.sqrt((self.v_pixels - beam_j) ** 2 + (self.h_pixels - beam_i) ** 2)
        MaxQ = max(cUL, cUR)
        self.MaxQ = MaxQ
        MinQ = min(stop_ri, stop_rj)  # LL: I changed from max() to min(), is it correct?
        Alt_MinQ = max(stop_ri, stop_rj)
        # LL: calculate q_max and q_min, first we need to find hypotenuse, calculate cosine of the
        # scattering angle, then using half angle formula to compute sin(theta) 

        hipot_max = np.sqrt((self.a) ** 2 + (MaxQ * self.pixel_size) ** 2)
        self.detector_hypot_max = hipot_max  # save maximum distance on detector face from beam
        cos_2theta_max = self.a / hipot_max
        sin_theta_max = np.sqrt(0.5 * (1 - cos_2theta_max))
        self.q_max = 4 * np.pi * sin_theta_max / self.lam

        hipot_min = np.sqrt((self.a) ** 2 + (MinQ * self.pixel_size) ** 2)
        cos_2theta_min = self.a / hipot_min
        sin_theta_min = np.sqrt(0.5 * (1 - cos_2theta_min))
        self.q_min = 4 * np.pi * sin_theta_min / self.lam

        alt_hipot_min = np.sqrt((self.a) ** 2 + (Alt_MinQ * self.pixel_size) ** 2)
        alt_cos_2theta_min = self.a / alt_hipot_min
        alt_sin_theta_min = np.sqrt(0.5 * (1 - alt_cos_2theta_min))
        self.alt_q_min = 4 * np.pi * alt_sin_theta_min / self.lam

        # print "\ncreate_Mask(): Calculating mask . . ."

        # mask_q is the array that cooresponds with NofPixels

        self.mask_q = np.arange(self.q_min, self.q_max, self.dq)

        print ("Mask q-range: qmin,qmax = ", self.q_min, self.q_max, " (", len(self.mask_q), " pts)\n")

        # default_q is mas_q extended to q = 0.

        self.default_q = np.arange(0.0, self.q_max, self.dq)

        # test each pixel (slow step!)

        # print "Detector MaxQ:",MaxQ
        L = np.zeros(len(self.default_q), dtype='int')
        if type == 'rectangle':
            for i in range(self.h_pixels):
                for j in range(self.v_pixels):
                    q_pix = np.sqrt((i - beam_i) ** 2 + (j - beam_j) ** 2)
                    iq = int((q_pix - MinQ) / (MaxQ - MinQ) * (
                                len(self.default_q) - 1))  # index for which bin this pixel belongs in
                    if not (((i > beam_i - stop_ri) and (i < beam_i + stop_ri)) and (
                            (j > beam_j - stop_rj) and (j < beam_j + stop_rj))):
                        # if outside beamstop rectangle
                        if ((cos_wedge * (i - beam_i) + sin_wedge * (j - beam_j) > 0) or (
                                -cos_wedge * (i - beam_i) + sin_wedge * (j - beam_j) > 0)):
                            # test for between line
                            if (iq >= 0): L[iq] = L[iq] + 1  # accumulate counts in this bin


        elif type == 'disk':  # NOT TESTED!
            # REG!! none of this should be necessary. Calculate for no mask, then simply truncate q_min
            print ("DISK!")
            for i in range(self.h_pixels):
                for j in range(self.v_pixels):
                    if ((A * (i - beam_i) + B * (j - beam_j) > 0) and (
                            -A * (i - beam_i) + B * (j - beam_j) > 0)):  # test for in between lines
                        q_pix = np.sqrt((i - beam_i) ** 2 + (j - beam_j) ** 2)
                        if (q_pix > stop_ri):
                            iq = int((q_pix - MinQ) / (MaxQ - MinQ) * (len(self.default_q)))
                            print (iq)
                            if iq >= 0:
                                L[iq] = L[iq] + 1

        else:
            print ("create_Mask(): unrecognized beamstop type:", type, "\nEnter either rectangle or disk.")
        # Need to erase content of the arrays in case create_Mask() is called
        # after the initiation.
        self.NofPixels = []
        self.NofPixels = np.append(self.NofPixels, L)

    # def create_Mask2(self, beam_i, beam_j, stop_ri, stop_rj, wedge=23.0, type="rectangle"):

    #     """ REG: this is a test version of create_Mask that reads cfg files.

    #     assigns number of pixels to each q bin based on choice of detector mask

    #     NOTE: this code assumes that q[0] = 0

    #     !!! for wedge angle less than 180, must have "and" instead of "or" REG:!!!

    #     beam_i = position of beam center horizontal (pixel coordinates)
    #     beam_j = position of beam center vertical   (pixel coordinates)
    #     stop_ri = pixels from beam center to edge of beamstop (horizontal)
    #     stop_rj = pixels from beam center to edge of beamstop (vertical)
    #     wedge = angle of integration wedge (degrees)
    #     type = shape of beamstop: rectangle or disk
    #     By defautl it models G1, but F2 could used used as a flag.

    #     IMPORTANT: the current version makes the small-angle approximation in q.
    #     For Pilatus 100K at 1500 mm (lengthwise), that's less than 0.1% error, however. But this code
    #     should be modified for WAXS. q = 2.4A-1 (10keV) will give 1% error.

    #     """

    #     print "New mask created: ", beam_i, beam_j, stop_ri, stop_rj, wedge, type

    #     # print len(q),beam_i,beam_j,stop_ri,stop_rj,wedge,type

    #     # express the integration wedge as two intersecting lines in normal form passing through origin,
    #     # this way it's easy to test if an (i,j) is in between them without concern for special cases (theta = 0 or 180)
    #     # LL: I added the factor of 2 here. Conversion from degree to radian.
    #     cos_wedge = np.cos(2 * np.pi * wedge / 360.0)
    #     sin_wedge = np.sin(2 * np.pi * wedge / 360.0)

    #     # print "A B",A,B

    #     # determine maximum possible distance from beam to upper corner of detector (in pixels)
    #     cUL = np.sqrt((self.v_pixels - beam_j) ** 2 + (beam_i) ** 2)
    #     cUR = np.sqrt((self.v_pixels - beam_j) ** 2 + (self.h_pixels - beam_i) ** 2)
    #     MaxQ = max(cUL, cUR)
    #     MinQ = min(stop_ri, stop_rj)  # LL: I changed from max() to min(), is it correct?
    #     Alt_MinQ = max(stop_ri, stop_rj)
    #     # LL: calculate q_max and q_min, first we need to find hypotenuse, calculate cosine of the
    #     # scattering angle, then using half angle formula to compute sin(theta)

    #     hipot_max = np.sqrt((self.a) ** 2 + (MaxQ * self.pixel_size) ** 2)
    #     cos_2theta_max = self.a / hipot_max
    #     sin_theta_max = np.sqrt(0.5 * (1 - cos_2theta_max))
    #     self.q_max = 4 * np.pi * sin_theta_max / self.lam

    #     hipot_min = np.sqrt((self.a) ** 2 + (MinQ * self.pixel_size) ** 2)
    #     cos_2theta_min = self.a / hipot_min
    #     sin_theta_min = np.sqrt(0.5 * (1 - cos_2theta_min))
    #     self.q_min = 4 * np.pi * sin_theta_min / self.lam

    #     alt_hipot_min = np.sqrt((self.a) ** 2 + (Alt_MinQ * self.pixel_size) ** 2)
    #     alt_cos_2theta_min = self.a / alt_hipot_min
    #     alt_sin_theta_min = np.sqrt(0.5 * (1 - alt_cos_2theta_min))
    #     self.alt_q_min = 4 * np.pi * alt_sin_theta_min / self.lam

    #     print "\ncreate_Mask(): Calculating mask . . ."
    #     print "create_Mask(): qmin,qmax = ", self.q_min, self.q_max, "\n"

    #     # LL: create the default q array, that experimental intensity (protein and buffer)
    #     # will be interpolated onto

    #     # Need to erase content of the arrays in case create_Mask() is called
    #     # after the initiation.
    #     temp_q_1 = np.arange(0.0, self.q_max, self.dq)
    #     self.default_q = []
    #     self.default_q = np.append(self.default_q, temp_q_1)
    #     ## CAUTION: using self.mask_q can be very tricky. Though spacing dq is the same,
    #     ## the starting point is different, can potential screw up every thing
    #     ## If it's used, consistency in using this array for all interpolation and calculation is key!
    #     temp_q_2 = np.arange(self.q_min, self.q_max, self.dq)
    #     self.mask_q = []
    #     self.mask_q = np.append(self.mask_q, temp_q_2)

    #     # test each pixel (slow step!)

    #     """
    #     NEW STUFF HERE!!!!!!!!
    #     """
    #     if type[-3:] == 'cfg':
    #         print "using cfg file: ", type

    #         # mask_file = 'path/filename.cfg' #This is the path, relative or absolute, to the RAW .cfg file
    #         detx_pix = self.v_pixels  # This is the number of pixels in the detector x direction
    #         dety_pix = self.h_pixels  # Number of pixels in the detector y direction
    #         self.RAWMask, self.RAWSettings = load_RAW_mask_from_cfg(type, detx_pix, dety_pix)
    #         """Mask is returned as 2D Numpy array in the shape of the detector. Each pixel in this 'detector image'
    #         is either 1, for unmasked, or -1, for masked. 
    #         """

    #         xcen = self.RAWSettings.get('Xcenter')
    #         ycen = self.RAWSettings.get('Ycenter')

    #         print "cfg  center:", xcen, ycen
    #         print "spec center:", beam_i, beam_j
    #         print "h_pixels, v_pixels = ", self.h_pixels, self.v_pixels

    #         xcen = detx_pix - xcen - 1

    #         # plt.imshow(self.RAWMask)
    #         # plt.show()

    #         L = np.zeros(len(self.default_q), dtype='int')
    #         for i in range(self.h_pixels):
    #             for j in range(self.v_pixels):
    #                 if self.RAWMask[i, j] > 0:
    #                     q_pix = np.sqrt((i - ycen) ** 2 + (j - xcen) ** 2)
    #                     iq = (q_pix - MinQ) / (MaxQ - MinQ) * (len(self.default_q) - 1)
    #                     L[iq] = L[iq] + 1
    #     else:
    #         print "need to specify cfg file"

    #     # Need to erase content of the arrays in case create_Mask() is called
    #     # after the initiation.
    #     self.NofPixels = []
    #     self.NofPixels = np.append(self.NofPixels, L)

    def interpol(self, special_q, raw_q, raw_data):
        """                                                                                               
        Interpolate the experimental data or FoXS output at desire q values,                              
        using numpy interpolate function. Default q array is our default_q, starting from 0.              
        However, when loading and interpolating data set with q values that don't reach 0,                
        we need to supply the specific range in the default_q value.
        
        Using default setting, one must have run create_Mask()
        """

        interp_func = interpolate.interp1d(raw_q, raw_data)
        interI = interp_func(special_q)
        return interI

    def find_q_range(self, q_array, expq):

        """ Using searchsorted function from numpy, we find the location in q_array 
            of the first element that is >= the first q value in exp/buf q,
            or the first element that is > the last value in exp q.
            Then we subtract 1 from that value to get the last element
            in q_array that is <= the last exp q value.

            REG: in other words, find the starting and ending indices in the first 
            array of the value-overlap region between the two arrays.

        """
        # print len(q_array), expq[0]
        start_point = np.searchsorted(q_array, expq[0])
        # print "find_q_range(): Start at " + str(start_point)+" q_value."

        stop_point = np.searchsorted(q_array, expq[-1], side='right') - 1
        # print "find_q_range(): Stop at " + str(stop_point)+" q_value."

        return start_point, stop_point

    #   ----------------------------------------------------------------------------------------
    #   Placing imported data points on the same q-space grid:
    # 
    #   The following "load_" functions read scattering profiles and interpolate them 
    #   onto a specified array of q-space points. The q-space range of the result is the 
    #   intersection of the two q-space arrays. Even if load_I, load_buf, load_vac etc. are
    #   based on the same "q_array", the resulting ranges of self.exp_q, self.buf_q, self.vac_q
    #   etc. may not agree because of different q-ranges in the input files. What we need here is 
    #   some way to generate a least common q-space interval given a bunch of different files. For now
    #   I'm using manual truncation on a case-by-case basis. REG - 9/23/17
    #   ----------------------------------------------------------------------------------------

    def load_I(self, fname, interpolate=False, q_array=None, FoxS=False):
        """ 
        Read a scattering profile and create self.exp_q, self.exp_I and self.exp_Sig arrays.

        WARNING: self.exp_q and self.exp_I are used as the theoretical scattering profile. This
        function and the member data really shouldn't be used for anything else. 

        When interpolate = True, this function will interpolate loaded data onto the self.q_array.
        If unspecificed, the default self.q_array is determined by the detector mask. 
        Since the q range of the data may not match the q range of the q_array, the resulting
        self.exp_q may be cropped to fit (along with self.exp_I and self.exp_Sig).            
        
         """

        if FoxS:  # running FoxS in comparison mode (Sig meaningless placeholder)
            q, Sig, I = np.loadtxt(fname, usecols=(0, 1, 2), comments='#', unpack=True)
        else:
            q, I, Sig = np.loadtxt(fname, usecols=(0, 1, 2), comments='#', unpack=True)

        # Fill instance variable arrays with read-in lists 
        # after clearing the arrays first (in case of accidentally loading files multiple times)

        self.exp_q = np.array(q)
        self.exp_I = np.array(I)
        self.exp_Sig = np.array(Sig)

        # self.exp_q = np.append(self.exp_q, q)
        # self.exp_I = np.append(self.exp_I, I)
        # self.exp_Sig = np.append(self.exp_Sig, Sig)

        ###### Code below interpolate exp data onto a q array       #########                            
        ###### To use this function, default_q cannot be null       #########                            
        ###### Run create_Mask() first, and set interpolate to True #########                            
        ###### This specific inter_I function working for FoXS,     #########                            
        ###### pay attention, to starting q value.                  #########                            

        if interpolate == True:
            # If user chooses to interpolate but doesn't provide q_array, 
            # we set q_array to the default_q created in create_Mask()
            # If user provides q_array, then we use that.
            if q_array is None:   q_array = self.default_q
            # print "load_I(): Interpolating I using default q_array."
            # else:
            # print "load_I(): Interpolating I using specified q_array."

            # We first check whether user q_array has overlap with our experimental q
            # If not, print the following error message.
            # If two arrays have overlap area, then we proceed 

            if q_array[-1] < self.exp_q[0]:
                print ("load_I(): Your last q_value must be greater than the first experimental q_value.")
            else:
                start_point, stop_point = self.find_q_range(q_array, q)

                # Construct a new q array that we can interpolate onto safely without crushing
                # [start:stop] notation for arrays is [inclusive:exclusive]

                self.exp_q = q_array[start_point:stop_point + 1]
                self.exp_I = self.interpol(self.exp_q, q, I)
                self.exp_Sig = self.interpol(self.exp_q, q, Sig)
                # print "--------------"
                # print "load_I: self.exp_q is q_array[",start_point,":",stop_point+1,"]"
                # print "q_array ", q_array[0], q_array[start_point]
                # print "exp_q first last length ", self.exp_q[0], self.exp_q[-1],len(self.exp_q)
                # print "--------------"

                # print "\nload_I(): Experimental data loaded and interpolated."

        else:
            print ("\nload_I(): Experimental data loaded, not interpolated.")
        return

        ####################################################################################################

    # The load_buf, load_vac, load_win functions load experimentally-derived scattering profiles
    # for use for calculating noise levels. The input scattering profiles should be in units of photons per
    # pixel. The exposure time, flux, thickness, and transmission of the measurements are divided out in
    # each case so the final units should be inverse cm per pixel.  
    ####################################################################################################                         

    def load_buf(self, buf_file="water.dat", t=None, P=None, interpolate=False, q_array=None, mu=None, d=None,
                 mu_win=None, d_win=None, a=None, norm=None):
        """                                                   
        Works similar to load_I(). Load an experimental buffer profile for use in calculating the noise level. 

        Note: this could be buffer in sample cell or "bare" buffer with empty cell subtracted (both normalized to beamstop diode)
        In all cases, flux and exposure time should be divided out. For the case of bare buffer the photons due to window should
        be gone, but the buffer scattering is still attenuated by window absorption as well as self absorption. Divide out 
        window and water transmission, but only buffer thickness. The case of buffer-in-cell is also attenuated by self and by
        windows, and should be divided by buffer thickness, but the model cannot completely correctly model window scattering contribution. 
        That has to remain fixed (the user shouldn't attempt to vary that, so window and sample thickness should be fixed at the same values
        as experimentally collected). 

        Note: must get detector efficiency (self.det_eff) correct here using the energy at which the buffer was collected!
        I currently don't provide a way to set this explicitly here.

        This function creates self.buf_q, self.buf_I, and self.buf_sig. 
   
        Interpolation works the same as load_I()
        """

        if P is None:
            P = self.P
            print ("WARNING: load_buf used default flux = ", P)

        if t is None:
            t = self.t
            print ("WARNING: load_buf used default exposure time = ", t)

        if norm is None:
            norm = 1.0
            # print "assuming buffer norm = 1.0!"

        if mu is None:
            mu = self.mu

        if d is None:
            d = self.d

        if mu_win is None:
            mu_win = self.mu_win

        if d_win is None:
            d_win = self.d_win

        if a is None:
            a = self.a

        # REG!! we assume detector QE is known for this profile

        print ("load_buf: assuming detector QE = ", self.det_eff)

        q, N, Sig = np.loadtxt(fname=buf_file, usecols=(0, 1, 2), comments="#", unpack=True)

        N = N * norm
        Sig = Sig * norm

        print ("load_buf: P = ", P, " t = ", t, "d = ", d, "mu = ", mu, "d_win = ", d_win, " mu_win = ", mu_win, "a = ", a)

        divide_out = d * np.exp(-mu * d) * np.exp(-mu_win * d_win) * P * t * self.det_eff / a ** 2

        #        q, N, Sig= np.loadtxt(fname="A_water_A1_16_1_0000.dat",usecols = (0,1,2),unpack=True)
        #        q, N, Sig= np.loadtxt(fname="TestSet/LysBuf_A1_9_001_0000.dat",usecols = (0,1,2),unpack=True)
        #        t = 2.0

        # print "loadbuf used: ", buf_file, "npts = ", len(q)

        self.bufP = P  # We store the flux value for external use.

        # Append the experiment buffer arrays to the instance variable, after clearing them
        # to avoid accidentally loading it multiple times
        # REG! check to make sure this isn't a problem with repeat calls to load_buf

        self.buf_q = np.append(self.buf_q, q)
        self.buf_I = np.append(self.buf_I, N / divide_out)
        self.buf_Sig = np.append(self.buf_Sig,
                                 Sig / np.sqrt(divide_out))  # currently NOT used! Check if you ever use this!

        # REG:extrapolate to q = 0?                                                                

        ###### Code below interpolates buffer onto default_q array #########                                                     
        ###### To use this function, default_q cannot be null      #########                                                     
        ###### Run create_Mask() first                             #########                                                     
        ###### This specific inter_I function working for this buffer,                                                           
        ###### pay attention to starting q value.                                                                                

        if interpolate == True:
            # If user chooses to interpolate but doesn't provide q_array, 
            # we set q_array to the default_q created in create_Mask()
            # If user provides q_array, then we use that.

            if q_array is None:   q_array = self.default_q
            #    print "load_buf(): Interpolating buffer using default q_array."
            # else:
            #    print "load_buf(): Interpolating buffer using specified q_array."

            # We first check whether user q_array has overlap with our buffer q
            # If not, print the following error message.
            # If two arrays have overlap area, then we proceed    

            if q_array[-1] < self.buf_q[0]:
                print ("load_buf(): Your last q_value must be greater than the first buffer q_value.\n")
            else:
                # print "finding range"

                start_point, stop_point = self.find_q_range(q_array, q)
                self.buf_q_start = start_point
                # Construct a new q array that we can interpolate onto safely without crushing
                # [start:stop] notation for arrays is [inclusive:exclusive],so add 1 to stop_point
                self.buf_q = q_array[start_point:stop_point + 1]
                # Store inter_q_array to an instance variable self.inter_buf_q
                self.buf_I = self.interpol(self.buf_q, q, N / (divide_out))
                self.buf_Sig = self.interpol(self.buf_q, q, Sig / np.sqrt(divide_out))
                # print "\nload_buf(): Buffer data loaded and interpolated."
                # print "load_buf: self.buf_q is q_array[",start_point,":",stop_point+1,"]"
                # print "q_array ", q_array[0], q_array[start_point]
                # print "buf_q first last length ", self.buf_q[0], self.buf_q[-1],len(self.buf_q)
                # print "--------------"
        else:
            print ("\nload_buf(): Buffer data loaded but not interpolated.")

        return

    """
    def build_buf(self, buffer_type = "buffer_only", q_array = None)
    
    # measured buffer only supplied. 
    if buffer_type == "measured":
        pass  # REG!! user should not vary sample or window thickness in this case

    # in addition to measured buffer, user has also supplied empty cell and vacuum profiles.    
    else if buffer_type == "measured+MT+Vac":

        self.buf_I = self.buffer_I + self.windows_I + self.instrumental_I

    # no experimental data provided. Use model for buffer+windows+instrument
    else if buffer_type == "model":

    """

    # def load_vac(self, vac_file="vac.dat", t=None, P=None, a=None, interpolate=False, q_array=None, norm=None):
    #     """                                                   
    #     This function is the same as load_buf, only for instrumental (vacuum) background. We assume here that 
    #     the vacuum scattering profile has been normalized by transmitted beamstop counts with appropriate 
    #     scale factor so that the buffer profile normalization = 1.0. This is my standard practice. As a result,
    #     the input profile is effectively attenuated by buffer and windows and so must be scaled back up to 
    #     the unattenuated result: flux and time are divided out as well as water and window transmissions.

    #     Note: must get detector efficiency (self.det_eff) correct here using the energy at which the buffer was collected!

    #     """
    #     if P is None:
    #         P = self.P
    #         print "WARNING: load_vac used default flux = ", P

    #     if t is None:
    #         t = self.t
    #         print "WARNING: load_vac used default exposure time = ", t

    #     if a is None:
    #         a = self.a
    #         print "WARNING: load_vac used default exposure a = ", a

    #     if norm is None:
    #         norm = 1.0
    #         # print "assuming buffer norm = 1.0!"

    #     q, N, Sig = np.loadtxt(fname=vac_file, usecols=(0, 1, 2), comments="#", unpack=True)

    #     N = N * norm
    #     Sig = Sig * norm

    #     self.vacP = P

    #     divide_out = t * P * np.exp(-self.mu * self.d) * np.exp(-self.mu_win * self.d_win) * self.det_eff / a ** 2

    #     self.vac_q = np.append(self.vac_q, q)
    #     self.vac_I = np.append(self.vac_I, N / divide_out)
    #     self.vac_Sig = np.append(self.vac_Sig, Sig / np.sqrt(divide_out))

    #     if interpolate == True:
    #         if q_array is None:   q_array = self.default_q
    #         #    print "load_buf(): Interpolating buffer using default q_array."
    #         # else:
    #         #    print "load_buf(): Interpolating buffer using specified q_array."

    #         # We first check whether user q_array has overlap with our buffer q
    #         # If not, print the following error message.
    #         # If two arrays have overlap area, then we proceed    

    #         if q_array[-1] < self.vac_q[0]:
    #             print "load_buf(): Your last q_value must be greater than the first buffer q_value.\n"
    #         else:
    #             # print "finding range"

    #             start_point, stop_point = self.find_q_range(q_array, q)
    #             self.vac_q_start = start_point
    #             # Construct a new q array that we can interpolate onto safely without crushing
    #             # [start:stop] notation for arrays is [inclusive:exclusive],so add 1 to stop_point
    #             self.vac_q = q_array[start_point:stop_point + 1]
    #             # Store inter_q_array to an instance variable self.inter_buf_q
    #             self.vac_I = self.interpol(self.vac_q, q, N / (divide_out))
    #             self.vac_Sig = self.interpol(self.vac_q, q, Sig / np.sqrt(divide_out))
    #             # print "\nload_buf(): Buffer data loaded and interpolated."
    #             # print "load_vac: self.vac_q is q_array[",start_point,":",stop_point+1,"]"
    #             # print "q_array ", q_array[0], q_array[start_point]
    #             # print "vac_q first last length ", self.vac_q[0], self.vac_q[-1],len(self.vac_q)
    #             # print "--------------"
    #     else:
    #         print "\nload_vac(): Vacuum data loaded but not interpolated."

    # def load_win(self, win_file="win.dat", t=None, P=None, q_array=None, interpolate=False, d_win=None, mu_win=None,
    #              a=None, norm=None):
    #     """                                                   
    #     Window profile is empty-subtracted and normalized by beamstop counts and scaled so that buffer profile normalization is 
    #     effectively 1.0. As a result, the window profile as input is attenuated by the beamstop normalization as if by buffer only 
    #     (window transmission is implicitly in the transmitted beamstop counts). To prepare an unattenuated
    #     profile, we must divide out buffer transmission, but also expected window transmission, flux, and time. Since this is 
    #     supposed to be a macroscopic differential crossection of a material, we must also divide out window thickness.

    #     Note: must get detector efficiency (self.det_eff) correct here using the energy at which the buffer was collected!

    #     """
    #     if P is None:
    #         P = self.P
    #         print "WARNING: load_win used default flux = ", P

    #     if t is None:
    #         t = self.t
    #         print "WARNING: load_win used default exposure time = ", t

    #     if norm is None:
    #         norm = 1.0
    #         # print "assuming window profile norm = 1.0!"

    #     if d_win is None:
    #         d_win = self.d_win

    #     if mu_win is None:
    #         mu_win = self.mu_win

    #     if a is None:
    #         a = self.a
    #         print "WARNING: load_win used default a = ", a

    #     # print "load_win: P = ",P," t = ",t, "d_win = ", d_win, " mu_win = ", mu_win

    #     q, N, Sig = np.loadtxt(fname=win_file, usecols=(0, 1, 2), comments="#", unpack=True)

    #     N = N * norm
    #     Sig = Sig * norm

    #     self.winP = P

    #     divide_out = t * P * d_win * np.exp(-mu_win * d_win) * np.exp(-self.mu * self.d) * self.det_eff / a ** 2

    #     self.win_q = np.append(self.win_q, q)
    #     self.win_I = np.append(self.win_I, N / divide_out)
    #     self.win_Sig = np.append(self.win_Sig, Sig / np.sqrt(divide_out))

    #     if interpolate == True:
    #         if q_array is None:   q_array = self.default_q
    #         #    print "load_buf(): Interpolating buffer using default q_array."
    #         # else:
    #         #    print "load_buf(): Interpolating buffer using specified q_array."

    #         # We first check whether user q_array has overlap with our buffer q
    #         # If not, print the following error message.
    #         # If two arrays have overlap area, then we proceed    

    #         if q_array[-1] < self.win_q[0]:
    #             print "load_buf(): Your last q_value must be greater than the first buffer q_value.\n"
    #         else:
    #             # print "finding range"

    #             start_point, stop_point = self.find_q_range(q_array, q)
    #             self.win_q_start = start_point
    #             # Construct a new q array that we can interpolate onto safely without crushing
    #             # [start:stop] notation for arrays is [inclusive:exclusive],so add 1 to stop_point
    #             self.win_q = q_array[start_point:stop_point + 1]
    #             # Store inter_q_array to an instance variable self.inter_buf_q
    #             self.win_I = self.interpol(self.win_q, q, N / divide_out)
    #             self.win_Sig = self.interpol(self.win_q, q, Sig / np.sqrt(divide_out))
    #             # print "\nload_buf(): Buffer data loaded and interpolated."
    #             # print "load_win: self.win_q is q_array[",start_point,":",stop_point+1,"]"
    #             # print "q_array ", q_array[0], q_array[start_point]
    #             # print "vac_q first last length ", self.vac_q[0], self.vac_q[-1],len(self.vac_q)
    #             # print "--------------"
    #     else:
    #         print "\nload_win(): window profile loaded but not interpolated."

    def writeStandardProfiles(self, rootname):
        print ("writeStandardProfiles:", rootname)
        """ Write standardized background profiles for buffer, windows, and vacuum. rootname = path/rootfilename """
        self.writeProf(rootname + "vac_I.dat", self.vac_q, self.vac_I, self.vac_Sig)
        self.writeProf(rootname + "win_I.dat", self.win_q, self.win_I, self.win_Sig)
        self.writeProf(rootname + "buf_I.dat", self.buf_q, self.buf_I, self.buf_Sig)

    def readStandardProfiles(self, rootname):
        print ("readStandardProfiles:", rootname)
        (self.vac_q, self.vac_I, self.vac_Sig) = self.readProf(rootname + "vac_I_interp.dat")
        (self.win_q, self.win_I, self.win_Sig) = self.readProf(rootname + "win_I_interp.dat")
        (self.buf_q, self.buf_I, self.buf_Sig) = self.readProf(rootname + "buf_I_interp.dat")
        print ("--------------")

    def simulate_buf(self, subtracted=False, no_vac=False, q_array=None, interpolate=True):
        """simulate a typical buffer scattering profile including windows, water (+salt/additives?), and beamline parasitic scatter  """
        """if q_array == None:
            q_array = self.default_q
            print "simulate_buf(): Interpolating buffer using default q_array."
        else:
            print "simulate_buf(): Interpolating buffer using specified q_array."
            
        # using cgs units
        chi_T = 4.58e-11      # compressibility of water @ 293K  bayres**-1 (cgs units of pressure 10 barye = 1 Pa)
        chi_T = 4.85e-11      # 4C - check!
        T = 277.15             # temperature (Kelvin)
        k_b = 1.3806488e-16   # Boltzmann constant (ergs)
        rho = self.ps         # average electron density of buffer (electrons/cm**3)
        r0  = 2.818e-13       # classical electron radius = 2.818e-13 (cm)

        print "rho/rho_water = ", self.ps/334.0e21
        print "d = ", self.d

        water_I = (rho*r0)**2*k_b*T*chi_T
        self.mu_win = self.mu_factor*self.lam**2.9 # energy dependence of window absorption
        self.mu = 2.8*self.lam**3   # energy dependence of water absorption
        
        PreFac = self.d*self.det_eff*exp(-self.mu_win*self.d_win)*exp(-self.mu*self.d)*self.P/(self.a**2)
        water_I = water_I*PreFac
        water   = water_I*np.ones(len(q_array))

        window  = 1.0e-06*(q_array)**(-2.8)+ 0.008
        slits   = q_array*0.0
        
        sim_total = self.buf_I
        """
        # Reminder: win_I, buf_I have transmission, sample thickness, flux, exposure time divided out
        #           vac_I has only flux and exposure time divided out (no thickness or transmission)
        # we are assuming buffer = pure water for now.

        # These factors convert unattenuated profiles into measured ones

        # self.mu_win = attentuation coefficient of the window
        # self.mu = attenuation coefficient of WHAT - RM! Buffer?

        total_attenuation = np.exp(-self.mu_win * self.d_win) * np.exp(-self.mu * self.d)

        if no_vac:
            vac_factor = 0.0
        else:
            vac_factor = total_attenuation

        win_factor = self.d_win * total_attenuation
        buf_factor = self.d * total_attenuation

        # print "synth_buffer: ",vac_factor, win_factor, buf_factor

        # In this version of the code, we assume buf, win, and vac are all in register.

        # print "synth_buffer component array lengths: ", len(self.vac_I), len(self.win_I), len(self.buf_I)

        self.buf_model_q = self.buf_q

        if ((len(self.vac_I) != len(self.win_I)) or (len(self.win_I) != len(self.buf_I)) or (
                len(self.win_I) != len(self.buf_I))):
            raise ValueError('standard background profiles not of same length', len(self.vac_I), len(self.win_I),
                             len(self.buf_I))

        if subtracted == True:
            # this line assumes empty cell scatter has been subtracted from buf_I
            # print "treating buffer file as subtracted"
            self.buf_model_I = vac_factor * self.vac_I + win_factor * self.win_I + buf_factor * (self.buf_I)
            synth_MT_cell = vac_factor * self.vac_I + win_factor * self.win_I
            synth_vac = vac_factor * self.vac_I
            synth_win = win_factor * self.win_I
            synth_buf = buf_factor * self.buf_I
        else:
            # this line assumes buf_I contains window and instrumental "vacuum" scatter
            # print "treating buffer file as unsubtracted"
            self.buf_model_I = buf_factor * self.buf_I
            synth_MT_cell = vac_factor * self.vac_I + win_factor * self.win_I
            synth_vac = vac_factor * self.vac_I

            # interpolate back onto current q-space points if the range has changed

        if interpolate == True:
            if q_array is None:   q_array = self.mask_q

            # We first check whether user q_array has overlap with our buffer q
            # If two arrays have overlap area, then we proceed    

            if q_array[-1] < self.buf_model_q[0]:
                print ("simulate_buf(): buffer q-range does not overlap with requested q-range\n")
            else:
                start_point, stop_point = self.find_q_range(q_array, self.buf_model_q)

                # [start:stop] notation for arrays is [inclusive:exclusive],so add 1 to stop_point

                q_trimmed = q_array[start_point:stop_point + 1]

                print ("\nsimulate_buf() called: Buffer data loaded and interpolated.")
                print ("load_win: self.win_q is q_array[", start_point, ":", stop_point + 1, "]")

                self.buf_model_I = self.interpol(q_trimmed, self.buf_model_q, self.buf_model_I)
                synth_MT_cell = self.interpol(q_trimmed, self.buf_model_q, synth_MT_cell)
                synth_vac = self.interpol(q_trimmed, self.buf_model_q, synth_vac)
                synth_win = self.interpol(q_trimmed, self.buf_model_q, synth_win)
                synth_buf = self.interpol(q_trimmed, self.buf_model_q, synth_buf)

                self.buf_model_q = q_trimmed

                print ("q_array start:", q_array[0], q_array[start_point])
                print ("buf_model_q: first, last, length ", self.buf_model_q[0], self.buf_model_q[-1], len(self.buf_model_q))
                print ("--------------")
        else:
            print ("\nload_win(): window profile loaded but not interpolated.")

        return (self.buf_model_I, synth_MT_cell, synth_vac, synth_win, synth_buf)

    # def load_Mask(self):
    #     """ load Mask file for comparison with calculated masks """
    #     self.realMask = np.loadtxt(fname="counts_out.txt")

    # def FF_Cylin(self, q, H, R):
    #     """Form factor of a right circular cylinder (Rod/Disk)"""
    #     FF = np.array([], dtype=float)
    #     for v in q:
    #         if v == 0:
    #             FF = np.append(FF, 1)
    #         else:
    #             Q = lambda q: np.sin(q) / q
    #             F = lambda q, R, x: special.jn(1, (q * R * (1 - x ** 2) ** 0.5)) ** 2
    #             G = lambda q, R, x: (q * R * (1 - x ** 2) ** 0.5) ** 2
    #             J = lambda q, H, x: Q(q * H * x / 2) ** 2
    #             I = integrate.quad(lambda x: F(v, R, x) / G(v, R, x) * J(v, H, x), 0, 1)
    #             FF = np.append(FF, 4 * I[0])
    #     return FF

    # def FF_Sphere(self, q, R):
    #     """Form Factor for a sphere"""
    #     FF = np.array([], dtype=float)
    #     for v in q:
    #         if v == 0:
    #             FF = np.append(FF, 1)

    #         else:
    #             FF = np.append(FF, (3 * (np.sin(v * R) - (v * R) * np.cos(v * R)) / (v * R) ** 3) ** 2)
    #     return FF

    def FF_FoxS(self, q):
        """ use pre-loaded profile as form factor.
        Must call load_I first, also q must match I.
        One can use self.inter_I_q to be safe.
        REG: this code also assumes exp_I[0] corresponds with q = 0
        """

        FF = np.array([], dtype=float)
        for v in q:
            # Basically loading FF with the FoxS profile, with I_0 normalized to 1 

            # print "len q", len(q), "len exp_I: ", len(self.exp_I), " index: ", np.min(np.nonzero(q == v)[0])

            FF = self.interpol(q, self.exp_q, self.exp_I)

            # function np.min(no.nonzero(array == element)[0])
            # returns the index of the element in the first row of the array
            # that equals v

            # REG: this original code was designed to pick the closest point in the
            # FoXS profile that matched the specified q. Here is the code my students 
            # designed to do this: 
            # FF=np.append(FF,self.exp_I[np.min(np.nonzero(q == v)[0])])
            # q == v is a list of booleans true only where v matches q.  Nonzero returns
            # only that element and min converts it to an integer index, equal to v ??

        FF = FF / self.exp_I[0]
        return FF

    def FF(self, q):
        """Form factor, combined, q is an np.array"""
        if type(q) == float:
            q = np.array([q])

        if self.shape == 'Sphere':
            return self.FF_Sphere(q, self.R)
        elif self.shape == 'Cylin':
            return self.FF_Cylin(q, self.H, self.R)
        elif self.shape == 'FoxS':
            return self.FF_FoxS(q)  # LL: FF_FoxS function is now fixed
        else:
            print ("FF(): unknown model type: ", self.shape)

    # def SolidAngCor(self, q):
    #     """ correct for rays non-perpendicular to detector """
    #     return (1.0 - 2 * (q * self.lam / (4 * np.pi)) ** 2) ** 3

    # def q_fit(self, R=1):
    #     """find q_fit such that q*Rg=self.qRg
    #     Now redesigned to show q_fit for FoxS results as well
    #     FoxS doesn't provide R, so set R to have a default value 1

    #     """
    #     if self.shape == 'Sphere':
    #         return self.qRg / (np.sqrt(.6) * R)
    #     elif self.shape == 'Cylin':
    #         return self.qRg / (np.sqrt(self.R ** 2 / 2 + self.H ** 2 / 12))
    #     # for this option of FoxS to run, there must be data in self.inter_I and self.inter_I_q
    #     # which means load_I() must be run first.
    #     elif self.shape == 'FoxS':
    #         q = np.array([], dtype=float)
    #         logI = np.array([], dtype=float)
    #         for v in self.exp_q:
    #             q = np.append(q, v ** 2)
    #         for w in self.exp_I:
    #             logI = np.append(logI, np.log(w))
    #         # print "q_fit function, length of q, logI: " len(q),len(logI)
    #         # Scipy.stats.linregress function gives slope (what we need) 
    #         # to a linear regression
    #         slope, intercept, r_value, p_value, std_err = stats.linregress(q, logI)
    #         Rg = np.sqrt(slope * (-3))
    #         # See numerous reference, Guinier's 1959 book Page 39 for one
    #         return self.qRg / Rg
    #     else:
    #         print "q_fit(): Unknown Rg for FoxS profile. Enter shape!"

    def lsqfity_with_sigmas(self, X, Y, SIG):

        """ This is a special version of linear least squares that uses 
        known sigma values of Y to calculate standard error in slope and 
        intercept. The actual fit is also SIGMA-weighted
        """

        # X, Y, SIG = map(np.asanyarray, (X, Y, SIG))

        #print "SIG: ", SIG
        #print "I: ", Y

        XSIG2 = np.sum(X / (SIG ** 2))
        X2SIG2 = np.sum((X ** 2) / (SIG ** 2))
        INVSIG2 = np.sum(1.0 / (SIG ** 2))

        XYSIG2 = np.sum(X * Y / (SIG ** 2))
        YSIG2 = np.sum(Y / (SIG ** 2))

        delta = INVSIG2 * X2SIG2 - XSIG2 ** 2

        #print ":::: ", XSIG2, X2SIG2, INVSIG2, XYSIG2, YSIG2, "delta: ", delta

        smy = X2SIG2 / delta
        sby = INVSIG2 / delta

        my = (X2SIG2 * YSIG2 - XSIG2 * XYSIG2) / delta
        by = (INVSIG2 * XYSIG2 - XSIG2 * YSIG2) / delta

        return my, by, smy, sby

    def lsqfity(self, X, Y):
        """
        Calculate a "MODEL-1" least squares fit.
    
        The line is fit by MINIMIZING the residuals in Y only.
    
        The equation of the line is:     Y = my * X + by.
    
        Equations are from Bevington & Robinson (1992)
        Data Reduction and Error Analysis for the Physical Sciences, 2nd Ed."
        pp: 104, 108-109, 199.

        REG: these formulas assume all sigmas are equal.
    
        Data are input and output as follows:
    
        my, by, ry, smy, sby = lsqfity(X,Y)
        X     =    x data (vector)
        Y     =    y data (vector)
        my    =    slope
        by    =    y-intercept
        ry    =    correlation coefficient
        smy   =    standard deviation of the slope
        sby   =    standard deviation of the y-intercept
    
        """

        X, Y = map(np.asanyarray, (X, Y))

        # Determine the size of the vector.
        n = len(X)

        # Calculate the sums.

        Sx = np.sum(X)
        Sy = np.sum(Y)
        Sx2 = np.sum(X ** 2)
        Sxy = np.sum(X * Y)
        Sy2 = np.sum(Y ** 2)

        # Calculate re-used expressions.
        num = n * Sxy - Sx * Sy
        den = n * Sx2 - Sx ** 2

        # Calculate my, by, ry, s2, smy and sby.
        my = num / den
        by = (Sx2 * Sy - Sx * Sxy) / den
        ry = num / (np.sqrt(den) * np.sqrt(n * Sy2 - Sy ** 2))

        diff = Y - by - my * X

        s2 = np.sum(diff * diff) / (n - 2)
        smy = np.sqrt(n * s2 / den)
        sby = np.sqrt(Sx2 * s2 / den)

        return my, by, ry, smy, sby

    def get_guinier(self):
        """ Provide guinier q range, and get guinier region data of 
        both the theoretical(smooth) and simulated noise curve"""
        # q0, N0 represent theoretical smooth curve
        # q, N represent curve with simulated noise

        q0 = np.arange(0.0, self.q_fit(self.R), self.dq)
        N0 = (self.pixel_size) ** 2 * self.t * self.I_of_q(self.c, self.mw, q0)

        q = np.arange(0.008, self.q_fit(self.R), self.dq)
        I = self.I_of_q(self.c, self.mw, q)

        N = (self.pixel_size) ** 2 * self.t * self.with_noise(self.t, q, I)
        # Once self.with_noise() is called, N has self.buf_q as it's q array
        start1, stop1 = self.find_q_range(self.buf_q, q)
        q = self.buf_q[start1:stop1 + 1]

        # remove non-positive data points
        L = len(N)
        N_new = np.array([])
        q_new = np.array([])
        N0_new = np.array([])
        q0_new = np.array([])
        for i in range(0, len(N)):
            if N[i] > 0:
                N_new = np.append(N_new, N[i])
                q_new = np.append(q_new, q[i])
                q0_new = np.append(q0_new, q0[i])
                N0_new = np.append(N0_new, N0[i])

        return q0_new, N0_new, q_new, N_new

    def cal_guinier(self):
        """ Calculate guinier parameters"""
        # REG: need to check these error formulas! 
        # q0, I_guinier_pure represent theoretical smooth curve
        # 1, I_guinier_noise represent curve with simulated noise
        q0, I_guinier_pure, q, I_guinier_noise = self.get_guinier()
        # find and plot the simulated Guinier fit, and calculate Rg and I(0)

        # Provide a fit for the curve with simulated noise
        # k_noise,r_noise = np.polyfit(q**2,np.log(I_guinier_noise),1)
        k_noise, r_noise, ry, smy, sby = self.lsqfity(q ** 2, np.log(I_guinier_noise))

        x_fit = np.linspace(self.q_min ** 2, self.q_fit(self.R) ** 2)
        y_fit = x_fit * k_noise + r_noise
        # Calculated the Rg and I_0
        self.Rg_noise = sqrt(-k_noise * 3)
        self.I_0 = np.exp(r_noise)
        self.sigI_0 = np.exp(sby)

        # calculated estimated error (rms error)
        # =actual y-value
        y_i = np.log(I_guinier_noise)
        # =y-value from fitting
        f_i = q ** 2 * k_noise + r_noise
        # =number of data points
        N = len(y_i)
        # =rms error = sqrt(1/N*sigma(f_i-y_i))
        self.rms_error_noise = sqrt(((f_i - y_i) ** 2).sum() / (N - 2) / ((q - np.average(q)) ** 2).sum())

        # find the Guinier fit parameters for the smooth curve
        k_pure, r_pure = np.polyfit(q0 ** 2, np.log(I_guinier_pure), 1)
        self.Rg_pure = sqrt(-k_pure * 3)
        # calculated estimated error (rms error)self.rms_error_pure
        # =actual y-value
        y_i = np.log(I_guinier_pure)
        # =y-value from fitting
        f_i = q0 ** 2 * k_pure + r_pure
        # =number of data points
        N = len(y_i)
        # =I_0 for smooth curve
        self.I_0_pure = np.exp(r_pure)
        # =rms error = sqrt(1/N*sigma(f_i-y_i))
        self.rms_error_pure = sqrt(((f_i - y_i) ** 2).sum() / (N - 2) / ((q0 - np.average(q0)) ** 2).sum())
        # -----------------------------------------------------------------------

        # Generate guinier_fit function, useful in plotting relative errors
        I_guinier_fit = self.I_0 * np.exp(-1.0 / 3.0 * self.Rg_noise ** 2 * q ** 2)

        return x_fit, y_fit, I_guinier_fit

    def sigma_profile_fit(self, I_no_noise, sig):
        """ Calculate I(0) and corresponding sigma(0) by fitting the known smooth profile to 
        the simulated noisy data. The resulting sigma is effectively
        the standard error of the I(0) obtained.
        """
        # alpha = np.sum(I_noise*I_no_noise)/np.sum(I_no_noise*I_no_noise)

        # CAREFUL! arguments should be in photons per pixel units
        # self_I0_model needs to be multiplied by t and pixel area

        w = self.I0_model * (self.t * self.pixel_size ** 2) * I_no_noise / np.sum(I_no_noise * I_no_noise)
        w2 = w * w
        sig2 = sig * sig
        variance = np.dot(w2, sig2)  # variance of a weighted sum
        sigmaI0 = np.sqrt(variance)

        return sigmaI0

    def QE(self, lam=None, d=None):
        """What does this function do? RM!
        """
        # Donath, personal communication
        # Thickness of sensor = d
        # lam = energy
        # density of Si g/cm**3

        if d == None:
            d = self.sensor_thickness

        if lam == None:
            lam = self.lam

        rho_Si = 2.329  # density of silcon
        # empirical correction for non-sensitive layer:
        mu_dark = 0.0048
        q_eff = np.exp(-mu_dark * lam ** 3) * (1.0 - np.exp(-17.19 * d * rho_Si * lam ** 3))

        # based on the reported Si mu values, this looks valid from 50 keV (lam = 0.25) down to about 3 keV (lam = 4.0)

        # print "sensor thickness = ",d," QE = ", q_eff
        return q_eff

    def PDDF(self, r_range):
        """plot pair distance distribution"""
        # reference:"Svergen&Koch,Rep.Phys.Prog 66(2003) 1735-82
        if self.shape == "Sphere" or self.shape == "Cylin":
            q = np.arange(0.01, 20.0, 0.01)
        elif self.shape == "FoxS":
            if self.exp_q[0] == 0:
                q = self.exp_q[1:]
                # Taking the first point in exp_q out if it's 0, avoiding dividing by 0 problem
            else:
                q = self.exp_q

        P_r = np.array([], dtype=float)
        I = self.I_of_q(self.c, self.mw, q)

        for r in r_range:
            p_r = np.sum(q ** 2 * I * np.sin(q * r) / (q * r) * 0.02) * (r ** 2) / (2.0 * np.pi ** 2)
            P_r = np.append(P_r, p_r)

        return P_r

    def readProf(self, name):
        """This function reads common three-column data files treating any line that starts with # as a comment
        Usage: readProf("name")
        """
        q, I, sig = np.loadtxt(name, usecols=(0, 1, 2), comments="#", unpack=True)

        return (q, I, sig)

    def readDat(self, name):
        """ General utility function 
        reads ".dat" files generated by RAW                                                                                    
        Usage: (q,I,sig) = readRad("file name")                                                                                                
        """

        q = []
        I = []
        sig = []

        fline = open(name).readlines()

        npts = int(fline[2])

        i = 0

        while (i < npts):
            tmp = fline[i + 3].split()
            q.append(float(tmp[0]))
            I.append(float(tmp[1]))
            sig.append(float(tmp[2]))
            i = i + 1

        return (np.array(q), np.array(I), np.array(sig))

    def writeProf(self, name, q, I, sig=None):
        """ Usage: writeProf(name,q,I,sig)
        This function writes a profile in a form compatable with GNOM input
        q,I,sig are written in simple columns to filename "name". 
        The file starts with a comment line containing the file name
        The array argument "sig" is optional.
        All arrays must have the same number of points
        """
        f = open(name, 'w')
        print >> f, "# ", name
        if sig is None:
            for i in np.arange(len(q)):
                print >> f, "%f %.15f" % (q[i], I[i])
        else:
            for i in np.arange(len(q)):
                print >> f, "%f %.15f %.15f" % (q[i], I[i], sig[i])

        f.close()

        return

    def Reduced_Chi2(self, I0val, Sig0val, I1val, Sig1val, num_fit_params=0):
        """ calculate reduced Chi-square between two profiles.
        The number of fitted parameters, num_fit_params, if any, reduces the degrees
        of freedom. """

        chi2 = np.sum((I0val - I1val) ** 2 / (Sig0val ** 2 + Sig1val ** 2)) / (len(I0val) - num_fit_params)

        # print "Chi2:", (I0val[2]-I1val[2]), (Sig0val[2]**2 + Sig1val[2]**2), len(I0val)-1

        return chi2


"""
The following code impliments the pairwise probability test for differences in curves,
known as the CORMAP test. It is taken from the freesas project:
https://github.com/kif/freesas
and used under the MIT license

Information from the original module:
__author__ = "Jerome Kieffer"
__license__ = "MIT"
__copyright__ = "2017, ESRF"
"""


class LongestRunOfHeads(object):
    """Implements the "longest run of heads" by Mark F. Schilling
    The College Mathematics Journal, Vol. 21, No. 3, (1990), pp. 196-207

    See: http://www.maa.org/sites/default/files/pdf/upload_library/22/Polya/07468342.di020742.02p0021g.pdf
    """

    def __init__(self):
        "We store already calculated values for (n,c)"
        self.knowledge = {}

    def A(self, n, c):
        """Calculate A(number_of_toss, length_of_longest_run)

        :param n: number of coin toss in the experiment, an integer
        :param c: length of the longest run of
        :return: The A parameter used in the formula

        """
        if n <= c:
            return 2 ** n
        elif (n, c) in self.knowledge:
            return self.knowledge[(n, c)]
        else:
            s = 0
            for j in range(c, -1, -1):
                s += self.A(n - 1 - j, c)
            self.knowledge[(n, c)] = s
            return s

    def B(self, n, c):
        """Calculate B(number_of_toss, length_of_longest_run)
        to have either a run of Heads either a run of Tails

        :param n: number of coin toss in the experiment, an integer
        :param c: length of the longest run of
        :return: The B parameter used in the formula
        """
        return 2 * self.A(n - 1, c - 1)

    def __call__(self, n, c):
        """Calculate the probability of a longest run of head to occur

        :param n: number of coin toss in the experiment, an integer
        :param c: length of the longest run of heads, an integer
        :return: The probablility of having c subsequent heads in a n toss of fair coin
        """
        if c >= n:
            return 0
        delta = 2 ** n - self.A(n, c)
        if delta <= 0:
            return 0
        return 2.0 ** (np.log2(np.array([delta], dtype=np.float64)) - n)

    def probaB(self, n, c):
        """Calculate the probability of a longest run of head or tails to occur

        :param n: number of coin toss in the experiment, an integer
        :param c: length of the longest run of heads or tails, an integer
        :return: The probablility of having c subsequent heads or tails in a n toss of fair coin
        """

        """Adjust C, because probability calc. is done for a run >
        than c. So in this case, we want to know probability of c, means
        we need to calculate probability of a run of length >c-1
        """
        c = c - 1
        if c >= n:
            return 0
        delta = 2 ** n - self.B(n, c)
        if delta <= 0:
            return 0
        return 2.0 ** (np.log2(np.array([delta], dtype=np.float64)) - n)


LROH = LongestRunOfHeads()


def cormap_pval(data1, data2):
    """Calculate the probability for a couple of dataset to be equivalent

    Implementation according to:
    http://www.nature.com/nmeth/journal/v12/n5/full/nmeth.3358.html

    :param data1: numpy array
    :param data2: numpy array
    :return: probablility for the 2 data to be equivalent
    """

    if data1.ndim == 2 and data1.shape[1] > 1:
        data1 = data1[:, 1]
    if data2.ndim == 2 and data2.shape[1] > 1:
        data2 = data2[:, 1]

    if data1.shape != data2.shape:
        raise SASExceptions.CorMapError

    diff_data = data2 - data1
    c = measure_longest(diff_data)
    n = diff_data.size
    if c > 0:
        prob = LROH.probaB(n, c)[0]
    else:
        prob = 1
    return n, c, round(prob, 6)


# This code to find the contiguous regions of the data is based on these
# questions from stack overflow:
# https://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array
# https://stackoverflow.com/questions/12427146/combine-two-arrays-and-sort
def contiguous_regions(data):
    """Finds contiguous regions of the difference data. Returns
    a 1D array where each value represents a change in the condition."""

    if np.all(data == 0):
        idx = np.array([])
    elif np.all(data > 0) or np.all(data < 0):
        idx = np.array([0, data.size])
    else:
        condition = data > 0
        # Find the indicies of changes in "condition"
        d = np.ediff1d(condition.astype(int))
        idx, = d.nonzero()
        idx = idx + 1

        if np.any(data == 0):
            condition2 = data < 0
            # Find the indicies of changes in "condition"
            d2 = np.ediff1d(condition2.astype(int))
            idx2, = d.nonzero()
            idx2 = idx2 + 1
            # Combines the two conditions into a sorted array, no need to remove duplicates
            idx = np.concatenate((idx, idx2))
            idx.sort(kind='mergesort')

        # first and last indices are always in this matrix
        idx = np.r_[0, idx]
        idx = np.r_[idx, condition.size]
    return idx


def measure_longest(data):
    """Find the longest consecutive region of positive or negative values"""
    regions = contiguous_regions(data)
    lengths = np.ediff1d(regions)
    if lengths.size > 0:
        max_len = lengths.max()
    else:
        max_len = 0
    return max_len


def run_cormap(sasm_list, correction='None'):
    pvals = np.ones((len(sasm_list), len(sasm_list)))
    corrected_pvals = np.ones_like(pvals)
    failed_comparisons = []

    if correction == 'Bonferroni':
        m_val = sum(range(len(sasm_list)))

    item_data = []

    for index1 in range(len(sasm_list)):
        sasm1 = sasm_list[index1]
        qmin1, qmax1 = sasm1.getQrange()
        i1 = sasm1.i[qmin1:qmax1]
        for index2 in range(1, len(sasm_list[index1:])):
            sasm2 = sasm_list[index1 + index2]
            qmin2, qmax2 = sasm2.getQrange()
            i2 = sasm2.i[qmin2:qmax2]

            if np.all(np.round(sasm1.q[qmin1:qmax1], 5) == np.round(sasm2.q[qmin2:qmax2], 5)):
                try:
                    n, c, prob = cormap_pval(i1, i2)
                except SASExceptions.CorMapError:
                    n = 0
                    c = -1
                    prob = -1
                    failed_comparisons.append((sasm1.getParameter('filename'), sasm2.getParameter('filename')))

            else:
                n = 0
                c = -1
                prob = -1
                failed_comparisons.append((sasm1.getParameter('filename'), sasm2.getParameter('filename')))

            pvals[index1, index1 + index2] = prob
            pvals[index1 + index2, index1] = prob

            if correction == 'Bonferroni':
                c_prob = prob * m_val
                if c_prob > 1:
                    c_prob = 1
                elif c_prob < -1:
                    c_prob = -1
                corrected_pvals[index1, index1 + index2] = c_prob
                corrected_pvals[index1 + index2, index1] = c_prob

            else:
                c_prob = 1

            item_data.append([str(index1), str(index1 + index2),
                              sasm1.getParameter('filename'), sasm2.getParameter('filename'),
                              c, prob, c_prob]
                             )

    return item_data, pvals, corrected_pvals, failed_comparisons
