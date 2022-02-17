"""
1D Forward Simulation for a Single Sounding
===========================================

Here we use the module *SimPEG.electromangetics.frequency_domain_1d* to predict
frequency domain data for a single sounding over a 1D layered Earth.
In this tutorial, we focus on the following:

    - Defining receivers, sources and the survey
    - How to predict total field, secondary field or ppm data
    - The units of the model and resulting data
    - Defining and running the 1D simulation for a single sounding

Our survey geometry consists of a vertical magnetic dipole source
located 30 m above the Earth's surface. The receiver is offset
10 m horizontally from the source.


"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
import os
from matplotlib import pyplot as plt
from discretize import TensorMesh

from SimPEG import maps
from SimPEG.electromagnetics import time_domain as tdem
from SimPEG.electromagnetics.utils.em1d_utils import plot_layer


# %% some default setup

#plt.rcParams.update({'font.size': 16})

write_output = True

# sphinx_gallery_thumbnail_number = 2

default_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# %% dictionary with SkyTEM system definition in a 
systems={
    '304':{
        'HM':{'Area': 342,
              'turns':4,
              'Imax': 120,
              'RxArea': 25},
        'LM':{'Area': 342,
              'turns':1,
              'Imax': 9,
              'RxArea':25} },        
    '312HP':{
        'HM':{'Area': 342,
              'turns':12,
              'Imax': 225,
              'RxArea':100},
        'LM':{'Area':    342,
              'turns':2,
              'Imax': 5.5,
              'RxArea':100} },
    '306HP':{
        'HM':{'Area': 342,
              'turns':6,
              'Imax': 240,
              'RxArea':25},
        'LM':{'Area': 342,
              'turns':1,
              'Imax': 9,
              'RxArea':25} }
        }

    
    
# %% define the dipole moment of the TEM system
system_name=['304','HM']
system=systems[system_name[0]][system_name[1]]
turns = system['turns']    # number of loops in the transmitter coil
Imax  = system['Imax']     # max current output
Area = system['Area']   # area of the rtansmitter


frame_hight=30

# %% define the resistivity model:


# model1:
res1=[300 , 1500, 3000]  # resistivities mdoel1
z1=[20 ,45]           # layer depths model1


rel_err=0.03
times = np.logspace(-5.5, -3, 21)               # time channels (s)

# %% ####################################################################
# Create Survey
# -------------
#
#
# Here we demonstrate a general way to define the receivers, sources, waveforms and survey.
# For this tutorial, the source is a horizontal loop whose current waveform
# is a unit step-off. The receiver measures the vertical component of the magnetic flux
# density at the loop's center.
#


# receiver properties
receiver_orientation = "z"                    # "x", "y" or "z"
receiver_location = np.array([0, 0., frame_hight+0.5])


# Define receiver list. In our case, we have only a single receiver for each source.
# When simulating the response for multiple component and/or field orientations,
# the list consists of multiple receiver objects.
receiver_list = []
receiver_list.append(
    tdem.receivers.PointMagneticFluxTimeDerivative(
        receiver_location, times, orientation=receiver_orientation
    )
)



# Source properties
source_current = Imax * turns                            # maximum on-time current
source_radius = np.sqrt(Area/np.pi)                            # source loop radius
#source_current=Imax * turns * Area
#source_radius=1
source_orientation = "z"                      # "x", "y" or "z"
source_location = np.array([0., 0., frame_hight])  
waveform = tdem.sources.StepOffWaveform()


# Define source list. In our case, we have only a single source.
source_list = [
    tdem.sources.CircularLoop(
        receiver_list=receiver_list, location=source_location, waveform=waveform,
        current=source_current, radius=source_radius
    )
]





# Define the survey
survey = tdem.Survey(source_list)

# %% ##############################################
# Defining a 1D Layered Earth Model
# ---------------------------------
#
# Here, we define the layer thicknesses and electrical conductivities for our
# 1D simulation. If we have N layers, we define N electrical conductivity
# values and N-1 layer thicknesses. The lowest layer is assumed to extend to
# infinity. If the Earth is a halfspace, the thicknesses can be defined by
# an empty array, and the physical property values by an array of length 1.
#
# In this case, we have a more conductive layer within a background halfspace.
# This can be defined as a 3 layered Earth model. 
#




# Layer thicknesses
z1.insert(0,0)
thicknesses1 = np.diff(np.array(z1))
n_layer1 = len(thicknesses1) + 1


# physical property models (conductivity models)
model1 = (1/res1[-1]) * np.ones(n_layer1)
for n, r in enumerate(res1):
    model1[n] = 1/res1[n]



# Define a mapping from model parameters to conductivities
model_mapping1 = maps.IdentityMap(nP=n_layer1)

# %% ######################################################################
# Define the Forward Simulation, Predict Data and Plot
# ----------------------------------------------------
# 
# Here we define the simulation and predict the 1D FDEM sounding data.
# The simulation requires the user define the survey, the layer thicknesses
# and a mapping from the model to the conductivities of the layers.
# 
# When using the *SimPEG.electromagnetics.frequency_domain_1d* module,
# predicted data are organized by source, then by receiver, then by frequency.
#

# Define the simulation
simulation1 = tdem.Simulation1DLayered(survey=survey,
                                       thicknesses=thicknesses1,
                                       sigmaMap=model_mapping1)

# Predict sounding data
dpred = simulation1.dpred(model1)





# add some noise to the data
noise = rel_err*dpred*np.random.randn(len(dpred))

ambientnoise_power=-0.5
ambientnoise_amplitude=2e-9 # at 1e-3s (1ms)
ambientnoise=(times*1e3)**ambientnoise_power * ambientnoise_amplitude * np.random.rand(len(dpred))

dpredN = dpred + noise #+ ambientnoise



# %% plotting the stuff
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid(shape=(1, 3), loc=(0, 0), colspan=2)
ax3 = plt.subplot2grid(shape=(1, 3), loc=(0, 2), colspan=1,  rowspan=1)
ax=[ax1, ax3]

#

# Plot sounding data
ax[0].plot(times, -dpred, '.-', lw=2, ms=10, label=r"computed", color=default_colors[0])
ax[0].plot(times, -dpredN, '.-', lw=2, ms=10, label=r"computed+noise", color=default_colors[1])
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel("Times (s)")
ax[0].set_ylabel(r"\partial B/\partial t (T/s)")
ax[0].legend()
ax[0].grid(color="k", alpha=0.5, linestyle="dashed", linewidth=0.5)
#ax[0].set_xlim([2e2, 2e5])
#ax[0].set_ylim([1, 1e3])

#Plot conductivity model
thicknesses_for_plotting1 = np.r_[thicknesses1, 50.]
mesh_for_plotting1 = TensorMesh([thicknesses_for_plotting1])
plot_layer(1/model1, mesh_for_plotting1, ax=ax[1], xscale="log", showlayers=False, color=default_colors[0], label="m_1")
ax[1].set_xlabel("Resistivity (Ohm.m)")
ax[1].invert_yaxis()
ax[1].grid(color="k", alpha=0.5, linestyle="dashed", linewidth=0.5)
ax[1].set_xlim([1e1, 1e4])
plt.tight_layout()
          
          
# %% ######################################################################
# Optional: Export Data
# ---------------------
#
# Write the predicted data. Note that noise has been added.
#

if write_output:

    dir_path = os.path.dirname(__file__).split(os.path.sep)
    dir_path.extend(["outputs"])
    dir_path = os.path.sep.join(dir_path) + os.path.sep

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    
    np.random.seed(222)
    
    
    fname = dir_path + 'em1dtem_data.txt'
    np.savetxt(
        fname,
        np.c_[times, dpredN],
        fmt='%.4e', header='Times dB/dt'
    )
    
    mname=dir_path + "em1dtem_truemodel.txt"
    z1.append(9999.9)
    with open(mname, 'w') as f:
        f.write("Depth\tResitvity\n")
        for n in range(len(res1)):
            f.write(str(z1[n]) + "\t" + str(res1[n]))
            f.write('\n')
            
            
