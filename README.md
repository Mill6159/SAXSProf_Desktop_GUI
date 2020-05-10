# SAXProf
This is a stripped-down SAXSProf example that was used to create the web version of SAXSProf and has been repurposed for a desktop GUI and more advanced error calculations for SAXS curves at high pressure.


**WARNING:**

If you choose to modify this code, take great care keeping track of q/I(q)-vectors. Due to multiple inputs(Vacuum, buffer, expt) there are multiple sets of
these vectors and interpolating them all properly can be challenging when changing the code, changing the energy via 'set_energy()', etc.
As updates of the code are released, this issue will be addressed.

_For example:_
**Potential issues looping this code while changing parameters.** 
Some parameters, such as energy, require that you re-generate the base q-vector via the mask procedure ... and there is no error-checking to make sure you do this.  
(i.e. if you change the energy, you need to recalculate the mask. Currently, the desktop takes care of all of that on the backend.)
So, the execution model for the code also needs some work to assure that when you change a parameter, all other dependent parameters get properly changed ... in an efficient way. 
I've done that, mostly(REG) ... just be careful when looping over energy.  

The code mask model is simplistic, but works surprisingly well. At one point, the code had the ability to load masks from RAW, but that broke. 
But before it broke, I(REG) did comparisons and found that a simple mask is really pretty good.

These issues, and other small and newly discovered, issues will be addressed in future releases of the code.

## How The Code Works
A "SAXS" object is created and used to generate the simulated profiles. SAXS8.py contains the class and all the implementation details for generating the simulated profiles.  
For more advanced error calculations related to sample concentration, molecular weight, and system pressure the class 'Err_Calcs()' is called. All details for those calculations can
be found in 'SAXSProf_ErrCalcs.py'. Additionally, the derivation for the "Analytical Model" for the relative error in Rg as a function
of various system parameters can be found in the file: Analytical_Derivation_Final.pdf

**Details of the backend of the code:**

1. Define the detector and mask. This determines the q-space points that will be used by everything else (_energy-dependent!_)
2. Load the computed smooth scattering profile from FoXS or some other program. 
3. Load the "standard profiles" describing buffer, windows, and slit scatter "vacuum".
4. Generate the buffer model (simulate_buf)
5. Calculate the smooth model curve on the buffer model q points
6. Calculate sigmas and put noise on the smooth curve
7. Plot
8. Additional calculations (P(r), Rg Error, Etc) if called.

  I've already generated standard profiles for the lysozyme example, but you may want to generate
  "standard profiles" for buffer, window, and "vacuum" for your own systems as in the paper.
  At the beginning of the example code, I have listed the experimental data files
  I used to generate these standard profiles: buffer_model, vac_model, win_model.
  You must specify the flux and exposure (and window types) at which these data were collected.
  By uncommenting these lines, you can generate your own standard profiles:

```python
#saxs1.load_buf(buffer_model, t=buffer_exposure, P=buffer_flux, interpolate=True, q_array = saxs1.default_q)
#saxs1.load_vac(vac_model, t=vac_exposure, P=vac_flux, interpolate=True, q_array = saxs1.default_q)
#saxs1.load_win(win_model, t=win_exposure, P=win_flux, interpolate=True, q_array = saxs1.default_q)
#saxs1.writeStandardProfiles("Nov07_")
```

Note: Currently, standard profiles are read-in but the function 'ReadStandardProfiles()'

Good luck. 

Richard

## Additional notes on error calculations: 

We generated a new class (SAXProf_ErrCalcs.py).  
 
This class calculates the error in Rg as determined by the Guiner fit as is outlined in the Analytical_Derivation_Final.pdf file within this repository.
This allows for comparison of the Rg error from an analytical model (i.e. we know where the error comes from) and the error produced by many fitting routines
such as RAW or primus.
Furthermore, this allowed for us to model the error in Rg as a function of concentration and pressure (contrast), which is also outlined in the Analytical_Derivation_Final.pdf file.

## Running Desktop GUI

Ensure you have Python3 installed locally as well as the following Python modules:  
(1) Matplotlib  
(2) tkinter  
(3) numpy  
(4) scipy  
  
To make this easier, you can simply run the bash_script '''SAXSProf_Setup.sh'''. To run the setup script:  
(1) Open terminal and navigate to the directory containing all of the SAXSProf code/files  
(2) Run the command 'sh SAXSProf_Setup.sh'. Assuming Python3 is already installed locally, the necessary Python packages should be installed.  
(3) If an error message is printed, install the packages manually.