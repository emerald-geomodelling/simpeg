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


# Anomalous model:
res1=[300, 3000, 3000]  # resistivities mdoel1
z1=[40,50]           # layer depths model1

# model2:
res2=[3000, 3000, 3000]  # resistivities mdoel2
z2=[20, 70]          # layer depths model2


rel_err=0.05
times = np.logspace(-5.5, -2, 31)               # time channels (s)

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

z2.insert(0,0)
thicknesses2 = np.diff(np.array(z2))
n_layer2 = len(thicknesses2) + 1


# physical property models (conductivity models)
model1 = (1/res1[-1]) * np.ones(n_layer1)
for n, r in enumerate(res1):
    model1[n] = 1/res1[n]

model2 = (1/res2[-1]) * np.ones(n_layer2)
for n, r in enumerate(res2):
    model2[n] = 1/res2[n]


# Define a mapping from model parameters to conductivities
model_mapping1 = maps.IdentityMap(nP=n_layer1)
model_mapping2 = maps.IdentityMap(nP=n_layer2)


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

# Define the simulations
simulation1 = tdem.Simulation1DLayered(survey=survey,
                                       thicknesses=thicknesses1,
                                       sigmaMap=model_mapping1)

simulation2 = tdem.Simulation1DLayered(survey=survey,
                                       thicknesses=thicknesses2,
                                       sigmaMap=model_mapping2)


# Predict sounding data
d1 = simulation1.dpred(model1)
d2 = simulation2.dpred(model2)



# %% copute a Sensitivity attribute that ease comparison
#
#  Noise definitions:

# relartive errors
rel_err = 0.05
d1_std = np.abs(d1) * rel_err
d2_std = np.abs(d2) * rel_err


# Ambient nosie


noise_power=-0.5
noise_amplitude=2e-9 # at 1e-3s (1ms)

d_ambientnoise=(times*1e3)**noise_power * noise_amplitude


d2_noise= d2_std + d_ambientnoise
d1_noise= d1_std + d_ambientnoise

# Sensitivity attribute computation:


Sensitivity = np.abs(d1 - d2)/d2_noise



# %% PLotting of the model, data and sensitivity attribute:

fig = plt.figure(figsize=(12, 12))
ax1 = plt.subplot2grid(shape=(2, 3), loc=(0, 0), colspan=2)
ax2 = plt.subplot2grid(shape=(2, 3), loc=(1, 0), colspan=2)
ax3 = plt.subplot2grid(shape=(2, 3), loc=(0, 2), rowspan=2)
ax=[ax1, ax2, ax3]

#ax[0].errorbar(rx.times, np.abs(d1), d1_noise, label="d$_{1}$")
#ax[0].errorbar(rx.times, np.abs(d2), d2_noise, label="d$_{2}$")
ax[0].plot(times, np.abs(d1), '.-', label="d$_{1}$", color=default_colors[0])
ax[0].plot(times, np.abs(d2), '.-', label="d$_{2}$", color=default_colors[1])
ax[0].plot(times, np.abs(d_ambientnoise), 'k:', label="d$_{noise}$")
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlim([times.min()/2, times.max()*2])
ax[0].set_ylim([1e-11, 5e-2])
ax[0].legend()
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("$\partial B_z / \partial t$ (T/s)")
ax[0].grid(color="k", alpha=0.5, linestyle="dashed", linewidth=0.5)
title=' '.join(["TEM data:", system_name[0], system_name[1]])
ax[0].set_title(title)

ax[1].plot(times, Sensitivity, '.-', label="Sensitivity")
ax[1].plot(times, np.ones(np.size(times))*3, 'k--', label="Threshold")
ax[1].set_xscale("log")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Sensitivity (1)")
ax[1].set_title('Sensitivity')
ax[1].grid(color="k", alpha=0.5, linestyle="dashed", linewidth=0.5)
ax[1].legend()
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstring = "\n".join([
    r"SD=$\frac{|d_1 - d_2|}{\delta d}$",
    r"$\delta$d = {} $\cdot$ d$_2$ ".format(rel_err)  + " + d$_{noise}$"
    ])
ax[1].text(0.7, 0.2, textstring, transform=ax[1].transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
ax[1].set_ylim([0, 10])  
ax[1].set_xlim([times.min()/2, times.max()*2])

#Plot conductivity model
thicknesses_for_plotting1 = np.r_[thicknesses1, 100.]
mesh_for_plotting1 = TensorMesh([thicknesses_for_plotting1])
thicknesses_for_plotting2 = np.r_[thicknesses2, 100.]
mesh_for_plotting2 = TensorMesh([thicknesses_for_plotting2])
plot_layer(1/model1, mesh_for_plotting1, ax=ax[2], xscale="log", showlayers=False, color=default_colors[0], label=r"m$_1$")
plot_layer(1/model2, mesh_for_plotting2, ax=ax[2], xscale="log", showlayers=False, color=default_colors[1], label=r"m$_2$")
ax[2].legend()
ax[2].set_xlabel("Resistivity (Ohm.m)")
ax[2].invert_yaxis()
ax[2].grid(color="k", alpha=0.5, linestyle="dashed", linewidth=0.5)
ax[2].set_xlim([1e1, 1e4])
ax[2].set_ylim([150, 0])
plt.tight_layout()

  
plt.tight_layout()
plt.show()



