########################################################################
import matplotlib
from SAXS8 import *
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
from tkinter import *
from SAXSProf_ErrCalcs import *
########################################################################
warnings.filterwarnings("ignore", category=RuntimeWarning) # Remove warning caused by dividing through by 0
# GUI framework
root = Tk()
root.title("SAXSProf Desktop GUI")
root.geometry("640x640+0+0")

# Build Header
heading = Label(root, text='Welcome to SAXProf: \nWhere all your SAXS \nsimulation dreams come true',
                font=('helvetica', 40, 'bold'), fg='#F10303').pack()

# Build Entry Labels
label1 = Label(root, text='Enter your concentration (mg/ml): ', font=('helvetica', 20, 'bold'), fg='black').place(x=10,
                                                                                                                  y=200)
label2 = Label(root, text='Enter your MWt(kDa): ', font=('helvetica', 20, 'bold'), fg='black').place(x=10, y=250)
label3 = Label(root, text='Enter your energy (keV): ', font=('helvetica', 20, 'bold'), fg='black').place(x=10, y=300)
label4 = Label(root, text='Enter exposure time (seconds): ', font=('helvetica', 20, 'bold'), fg='black').place(x=10,
                                                                                                               y=350)
label5 = Label(root, text='Enter flux (photons/second): ', font=('helvetica', 20, 'bold'), fg='black').place(x=10,
                                                                                                             y=400)

# Build Entries for Entry Labels
conc = StringVar()
entry_box = Entry(root, textvariable=conc, width=25, bg='#2F8E08').place(x=350, y=200)
MWt = StringVar()
entry_box = Entry(root, textvariable=MWt, width=25, bg='#2F8E08').place(x=350, y=250)
Energy = StringVar()
entry_box = Entry(root, textvariable=Energy, width=25, bg='#2F8E08').place(x=350, y=300)
time = StringVar()
entry_box = Entry(root, textvariable=time, width=25, bg='#2F8E08').place(x=350, y=350)
flux = StringVar()
entry_box = Entry(root, textvariable=flux, width=25, bg='#2F8E08').place(x=350, y=400)
# Set default values #
conc.set(5)
MWt.set(14)
Energy.set(14.14)
time.set(0.50)
flux.set(3.8e12)

'''End General GUI framework
Some buttons for the GUI will be below'''

def sim_params(energy=14.14, P=3.8e12, snaps=1, a=150.47,
               d=0.15, window_type='mica', sensor_thickness=0.032, t=0.4450):
    return energy, P, snaps, a, d, window_type, sensor_thickness, t


# Building outputs
# First, a simple button to return all of the simulations parameters
def sim_param_list():
    energy, P, snaps, a, d, window_type, sensor_thickness, t = sim_params(energy=np.float(Energy.get()),
                                                                          P=np.float(flux.get()),
                                                                          t=np.float(time.get()))
    paramList = [str(energy), str(conc.get()), str(MWt.get()), str(P), str(snaps), str(a), str(d), str(window_type),
                 str(sensor_thickness), str(t)]
    paramNameList = ['Energy (keV): ', 'Conc. (mg/ml) of Sample: ', 'MWt(kDa) of Sample: ', 'Flux (photons/second): ',
                     'Snaps: ', 'Sample-Detector Distance (cm): ',
                     'Sample Cell Length (cm): ', 'Window Type: ', 'Sensor Thickness (cm): ', 'Exposure Time (s): ']
    top_window = Toplevel(root, bg='#898585')
    for ind, paramNameList in enumerate(paramNameList):
        # print names in the tkinter window
        # create a label widget in top_window
        names_label = Label(top_window)
        # values_label = Label(top_window)
        # give it a position using grid
        names_label.grid(row=int(ind) + 1, column=0)
        # values_label.grid(row=int(ind) + 1, column=1)
        # print the fruit name in the label
        names_label.config(text=paramNameList, font=('helvetica', 20, 'bold'), bg='#898585')
    # values_label.config(text=paramList)
    for ind, paramList in enumerate(paramList):
        # print names in the tkinter window
        # create a label widget in top_window
        values_label = Label(top_window)
        # give it a position using grid
        values_label.grid(row=int(ind) + 1, column=1)
        # print the fruit name in the label
        values_label.config(text=paramList, font=('helvetica', 20), bg='#898585')


# Build Buttons:
Simulation_Parameters = Button(root, text="Simulation Parameters", width=30, height=2, bg='lightblue',
                               command=sim_param_list).place(x=320, y=430)

# General plotting function for 1 dependent variable
def plot_S1(X, Y, plotlabel='', savelabel='', xlabel='', ylabel=''):
    if len(X) == len(Y):
        plotlabel = plotlabel
        savelabel = savelabel
        plt.rc("axes", linewidth=2)
        plt.rc("lines", markeredgewidth=2)
        plt.rc('font', **{"sans-serif": ["Helvetica"]})
        fig = plt.Figure(figsize=(8, 8))
        ax1 = fig.add_subplot(1, 1, 1)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
            tick.label1.set_fontname('Helvetica')
        for tick in ax1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
        plt.xlabel(xlabel, size=22)
        plt.ylabel(ylabel, size=22)
        plt.plot(X, Y, '-o', label=plotlabel)
        fig.tight_layout()
        plt.legend(numpoints=1, fontsize=18, loc="best")
        plt.savefig(savelabel + ".png", format='png',
                    bbox_inches='tight')

'''Basic SAXSProf simulation:
Returns and SAXS profile based off simulation parameters
Buffer, window, and vacuum profiles were collected previously
SAXS profile is based off of lysozyme
Where a FoxS input is used'''
def gen_simulation(plot=True):
    #############################################################
    # Initial User Inputs
    #
    #############################################################
    energy = 9.962  # energy (keV)
    P = 8.4e11  # flux (photons/s)
    t = 1.0  # exposure time for individual snapshots (s)
    snaps = 1  # number of snapshots averaged
    a = 150.47  # sample-to-detector distance (cm)
    d = 0.15  # sample cell path length (cm)

    window_type = "mica"  # type of window material on sample cell

    sensor_thickness = 0.032  # Pilatus sensor thickness

    # These files below are only used when generating the "standard profiles."

    # buffer with empty cell subtracted
    buffer_model = "/S_A_nov07_028_lysbufnorm_flat.dat"
    buffer_exposure = t  # exposure used for buffer model
    buffer_flux = P  # flux used for buffer model

    vac_model = "/A_nov07_072_lysbufnorm.dat"
    vac_exposure = t
    vac_flux = P

    win_model = "/S_A_nov07_026_lysbufnorm.dat"
    win_exposure = t
    win_flux = P

    #############################################################################
    #############################################################################

    sample_model_1 = "6lyz.pdb.dat"

    c = 4.0  # concentration (mg.ml)
    t = 10  # Exposure time (seconds)
    mw1 = 14.3  # Molecular Weight (kDa)

    saxs1 = SAXS(mw=mw1,a=a,d=d,energy=energy,P=P,t=t,total_t=t * snaps,c=c,shape='FoxS',detector='100K')
    saxs1.set_window(window_type)
    saxs1.sensor_thickness = sensor_thickness
    saxs1.det_eff = saxs1.QE(saxs1.lam,saxs1.sensor_thickness)

    #  determines the q-space range based on detector and mask. Default_q starts at q = 0, mask_q starts at q_min

    saxs1.create_Mask(98,3,45,14,wedge=360.0,type="rectangle")

    # buffer, vacuum, window, and model profiles read in and interpolated onto default_q
    # saxs1.buf_q are the q points of default_q that fall within buf's q range
    #
    # saxs1.load_buf(buffer_model, t=buffer_exposure, P=buffer_flux, interpolate=True, q_array = saxs1.default_q)
    # saxs1.load_vac(vac_model, t=vac_exposure, P=vac_flux, interpolate=True, q_array = saxs1.default_q)
    # saxs1.load_win(win_model, t=win_exposure, P=win_flux, interpolate=True, q_array = saxs1.default_q)
    saxs1.load_I(sample_model_1,interpolate=True,q_array=saxs1.default_q)

    #############################################################################
    # Here we load in
    #############################################################################

    saxs1.readStandardProfiles("M_Nov07_")

    #############################################################################
    #############################################################################

    # saxs1.writeStandardProfiles("Nov07_")

    #############################################################################
    #############################################################################

    saxs1.v = 0.72  # protein specific volume
    saxs1.pp = 322e21 / saxs1.v  # (from dry mass/specific volume)
    saxs1.ps = 334.6e21  # electrons/cm3 for buffer (water)
    saxs1.p = (saxs1.pp - saxs1.ps) * 2.818e-13  # contrast

    print('Contrast, Protein e Density',saxs1.p,saxs1.pp)

    saxs1.d = 0.15  # microfluidic mixing chip
    # saxs1.P = 1.6e11   # 8 um x 13 um CRL beam
    saxs1.P = 3.8e12  # CHESS-U Flux, Ph/s

    # saxs1.P = 3.8e11   # Oct 2017 30 um x 50 um CRL beam

    # don't forget to set the beam dimensions (cm)

    # h_beam = 0.005
    # v_beam = 0.003

    # h_beam = 0.0008
    # v_beam = 0.0013

    #############################################################################
    #############################################################################

    saxs1.energy, saxs1.P, saxs1.snaps, saxs1.a, saxs1.d, saxs1.window_type, saxs1.sensor_thickness, saxs1.t = sim_params(energy=np.float(Energy.get()),
                                                                          P=np.float(flux.get()),
                                                                          t=np.float(time.get()))
    saxs1.set_energy(14.14)  # this energy is energy for simulated data

    # re-calculate visible q-range and "pixels per bin" array (NofPixels)
    saxs1.create_Mask(98,3,45,14,wedge=360.0,type="rectangle")

    # need to re-load model so we can re-interpolate onto new default_q
    saxs1.load_I(sample_model_1,interpolate=True,q_array=saxs1.default_q)

    # generate buffer profile. simulate_buf uses trimmed mask_q for q-values
    try:
        (synth_buf,MT_cell,Vac_cell,win_cell,buf_cell) = saxs1.simulate_buf(subtracted=True)
    except ValueError as e:
        print(e.args)
        raise e

    # calculate synthetic curve on buf_model profile generated by simulate_buf
    I_no_noise = saxs1.I_of_q(saxs1.c,saxs1.mw,saxs1.buf_model_q)

    # calculate noisy profile on mask_q points that fall within both buf's q range and the specified default_q range (mask_q_short)
    I_w_noise = saxs1.t * saxs1.pixel_size ** 2 * saxs1.with_noise(saxs1.t,saxs1.buf_model_q,I_no_noise)

    # calculated smooth I on default_q (goes all the way to q = 0)
    I_no_noise = saxs1.t * saxs1.pixel_size ** 2 * I_no_noise

    # calculated smooth I and sigma on same q points as I_w_noise
    I_no_noise_calc = saxs1.Icalc * saxs1.t * saxs1.pixel_size ** 2  # in register with buffer
    I_sigma = saxs1.sigma

    if plot == True:
        plotlabel = 'Simulated SAXS Curve'
        savelabel = 'Simulated_SAXS_Curve'
        plt.rc("axes",linewidth=2)
        plt.rc("lines",markeredgewidth=2)
        plt.rc('font',**{"sans-serif":["Helvetica"]})
        top = Toplevel(root)
        fig = plt.Figure(figsize=(5,4),dpi=300)
        ax1 = fig.add_subplot(111)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
            tick.label1.set_fontname('Helvetica')
        for tick in ax1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
        ax1.set_xlabel('$q (\\AA^{-1})$',size=8)
        ax1.set_ylabel('I(q)',size=8)
        ax1.semilogy(saxs1.buf_model_q,I_w_noise,label=plotlabel,markersize=2,color='#009EBD')
        ax1.legend(numpoints=1,fontsize=4,loc="best")
        fig.savefig(savelabel + ".png",format='png',
                    bbox_inches='tight')
        fig.subplots_adjust(left=0.15,bottom=0.20,right=0.95,top=0.92,wspace=0.21,hspace=0.67)
        scatter = FigureCanvasTkAgg(fig,top)
        scatter.get_tk_widget().pack()
    else:
        print('Simulated SAXS profile was NOT generated')
    # plotlabel = 'Simulated SAXS Curve'
    # savelabel = 'Simulated_SAXS_Curve'
    # plt.rc("axes", linewidth=2)
    # plt.rc("lines", markeredgewidth=2)
    # plt.rc('font', **{"sans-serif": ["Helvetica"]})
    # top = Toplevel(root)
    # fig = plt.Figure(figsize=(5, 4), dpi=300)
    # ax1 = fig.add_subplot(111)
    # for tick in ax1.xaxis.get_major_ticks():
    #     tick.label1.set_fontsize(8)
    #     tick.label1.set_fontname('Helvetica')
    # for tick in ax1.yaxis.get_major_ticks():
    #     tick.label1.set_fontsize(8)
    # ax1.set_xlabel('$q (\\AA^{-1})$', size=8)
    # ax1.set_ylabel('I(q)', size=8)
    # ax1.semilogy(saxs1.buf_model_q, I_w_noise, label=plotlabel, markersize=2, color='#009EBD')
    # ax1.legend(numpoints=1, fontsize=4, loc="best")
    # fig.savefig(savelabel + ".png", format='png',
    #             bbox_inches='tight')
    # fig.subplots_adjust(left=0.15, bottom=0.20, right=0.95, top=0.92, wspace=0.21, hspace=0.67)
    # scatter = FigureCanvasTkAgg(fig, top)
    # scatter.get_tk_widget().pack()

    return energy, saxs1




# Associated Button
Generate = Button(root, text="Generate SAXS Curve", width=30, height=2, bg='lightblue', command=gen_simulation).place(
    x=320, y=470)


def err_calcs():

    energy, saxs1 = gen_simulation(plot=False)
    err_data = err_Calcs(saxs1=saxs1)
    concentration, rgError, log_sig = err_data.calc_errRg_conc()

    # err_data.plot_S1(concentration, [x * 100 for x in rgError],
    #                  plotlabel='Simulated Error - Analytical model',
    #                  savelabel='Simulated_Error_Func_Conc',
    #                  xlabel='Conc. ($\\frac{mg}{ml}$)',
    #                  ylabel='($\\frac{\sigma_{R_{g}}}{R_{g}}$) $\cdot 100$')
    # Quick calculate model from initial points (slope) of the simulated data
    inv_err = 1 / np.array(rgError)
    final_slope = (inv_err[3] - inv_err[2]) / (concentration[3] - concentration[2])

    # Technically this final_slope term should be empirically model as it may not be known apriori

    # err_data.plot_S1(concentration, 1.0 / (final_slope * np.array(concentration)),
    #                  plotlabel='($\\frac{1}{concentration}$) Model',
    #                  savelabel='Inv_c_Model',
    #                  xlabel='concentration. ($\\frac{mg}{ml}$)',
    #                  ylabel='Model')

    err_data.plot_S2(concentration, rgError, 1.0 / (final_slope * np.array(concentration)),
                     plotlabel1='Simulated Error - Analytical model',
                     plotlabel2='($\\frac{1}{final \ slope \cdot conc}$)',
                     savelabel='Sim_ErrorRg_vs_Conc_w_Model',
                     xlabel='Conc. ($\\frac{mg}{ml}$)',
                     ylabel='($\\frac{\sigma_{R_{g}}}{R_{g}}$)')

    # RM! 04.28.2020
    # Contrast values taken from RM script.
    density = [0.99707, 1.0015, 1.0184, 1.0379, 1.0719, 1.1010, 1.1142]
    pressure = [0, 10, 50, 100, 200, 300, 350]
    ps = []
    rho = []

    for i in range(len(density)):
        ps.append(density[i] * 3.3428E+23)  # conversion factor from g/ml to electrons per cm^3

    for i in range(len(ps)):
        rho.append((saxs1.pp - ps[i]) * (2.818 * 10 ** -13))

    # Function written by RM: Modified from original
    I = saxs1.I_of_q_variable_contrast(saxs1.c, saxs1.mw, saxs1.buf_model_q, rho)
    err_data.plot_S2(saxs1.buf_model_q, I[0], I[6],
                     plotlabel1='0 MPa',
                     plotlabel2='350 MPa',
                     savelabel='Scattering_Curve_AtMultiplePressures',
                     xlabel='q($\\AA^{-1}$)',
                     ylabel='I(q)')

    rho, Rg_error_contrast, sig2_Rg_out = err_data.calc_errRg_contrast()

    # # A plot of the simulated Rg error WITHOUT a model
    # # commented out unless needed
    # err_data.plot_S1(rho, [x * 100 for x in Rg_error_contrast],
    #                  plotlabel='Simulated Error',
    #                  savelabel='Sim_err_Rg_func_of_Contrast',
    #                  xlabel='$\Delta \\rho (cm^{-2})$',
    #                  ylabel='($\\frac{\sigma_{R_{g}}}{R_{g}}$) $\cdot 100$')

    model, c, Rg_error_contrast = err_data.rgErr_contrast_model()

    err_data.plot_S2(rho, [x * 100 for x in Rg_error_contrast], [x * 100 for x in model],
                     plotlabel1='Simulated Error - Analytical model',
                     plotlabel2='$\\frac{%s}{\\rho}$' % "{:.2e}".format(c[0]),
                     savelabel='Sim_ErrorRg_vs_Contrast_w_Model',
                     xlabel='$\Delta \\rho (cm^{-2})$',
                     ylabel='($\\frac{\sigma_{R_{g}}}{R_{g}}$) $\cdot 100$')

    CytC_data = np.loadtxt("rgErr_Pressure_CytC.txt",
                           dtype={'names': ('Pressure', 'Rg', 'RgErr', 'RgErrRel', 'RgErrRelPercent'),
                                  'formats': (np.float, np.float, np.float, np.float, np.float)}, skiprows=2)
    # This is VERY clunky
    # The user needs to match the simulation pressures of
    # 0,10,50,100,200,300,350
    # If the input data frame has extra pressure data points, they must be removed.
    # In this case, the expt data does NOT include a 50 MPa point, so it was 'popped' off
    # There should be a more automated way to do this added in.
    rho.pop(2)
    popRho = rho
    Rg_error_contrast.pop(2)
    # modelList=list(model)
    # modelList.pop(2)
    popRgErr = Rg_error_contrast
    # popRgErr = modelList
    exptData = np.ndarray.tolist(CytC_data['RgErrRelPercent'])
    exptData.pop(2)

    def plot_S3(X, X2, Y1, Y2, plotlabel1 = '', plotlabel2 = '', savelabel = '', xlabel = '', ylabel = '',xlabel2=''):
        plt.rc("axes",linewidth=2)
        plt.rc("lines",markeredgewidth=2)
        plt.rc('font',**{"sans-serif": ["Helvetica"]})
        top = Toplevel(root)
        fig = plt.Figure(figsize=(8,8),dpi=500)
        DPI = fig.get_dpi()
        fig.set_size_inches(2400.0 / float(DPI),1220.0 / float(DPI))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        for tick in ax1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
            tick.label1.set_fontname('Helvetica')
        for tick in ax1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
        for tick in ax2.xaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
            tick.label1.set_fontname('Helvetica')
        ax1.set_xlabel(xlabel,size=8)
        ax1.set_ylabel(ylabel,size=8)
        ax2.set_xlabel(xlabel2,size=8)
        ax1.plot(X,Y1,'-o',label=plotlabel1,markersize=2,color='#009EBD')
        ax1.plot(X,Y2,'--',
                 linewidth=4,
                 color='k',
                 label=plotlabel2)
        ax2.plot(X2,Y1,linestyle='None')
        ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax2.set_xlim(350,0)
        # ax2.set_xticks([350,300,200,100,10,0])
        ax1.legend(numpoints=1,fontsize=4,loc="best")
        fig.tight_layout()
        fig.savefig(savelabel + ".png",format='png',
                    bbox_inches='tight')
        fig.subplots_adjust(left=0.20,bottom=0.20,right=0.90,top=0.80,wspace=0.21,hspace=0.67)
        scatter = FigureCanvasTkAgg(fig,top)
        scatter.get_tk_widget().pack()

    # generate pressure list for input
    # and label for the analytical model plot
    plist = np.array([350,300,200,100,10,0])
    constantLabel="{:.2e}".format(c[0])
    plot_S3(popRho, plist, [x * 100 for x in popRgErr], exptData,
                     plotlabel1='Analytical model: $\\frac{{%s}}{\\rho}$' % constantLabel,
                     plotlabel2='Experimental Cytochrome C Data',
                     savelabel='SimandExpt_err_Rg_func_of_Contrast',
                     xlabel='$\Delta \\rho (cm^{-2})$',
                     ylabel='($\\frac{\sigma_{R_{g}}}{R_{g}}$) $\cdot 100$',
                     xlabel2='Pressure (MPa)')


# Associated Button
Error_Calcs = Button(root, text="Error Calculations", width=30, height=2, bg='lightblue', command=err_calcs).place(
    x=320, y=510)


def quit_application():
    print("destorying root TK")
    root.destroy()
    print("exiting python")
    exit(0)


# Build a 'quit' button
Button(root, text='Quit', command=quit_application).place(x=565, y=550)
root.mainloop()
