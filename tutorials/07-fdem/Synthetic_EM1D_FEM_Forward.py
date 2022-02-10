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
from SimPEG.electromagnetics import frequency_domain as fdem
from SimPEG.electromagnetics.utils.em1d_utils import plot_layer


# %% some default setup

#plt.rcParams.update({'font.size': 16})

write_output = True

# sphinx_gallery_thumbnail_number = 2

default_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# %% ####################################################################
# Create Survey
# -------------
#
# Here we demonstrate a general way to define the receivers, sources and survey.
# For this tutorial, the source is a vertical magnetic dipole that will be used
# to simulate data at a number of frequencies. The receivers measure real and
# imaginary ppm data.
# 

# Frequencies being observed in Hz
frequencies = np.array([383, 1820, 3315, 8488, 40835, 133530], dtype=float)

# EM bird geometry
bird_hight=30
bird_length=10


# model1:
res1=[1000, 100, 100]  # resistivities mdoel1
z1=[20, 50]           # layer depths model1

# %% Define a list of receivers. The real and imaginary components are defined
# as separate receivers.
receiver_location = np.array([bird_length, 0., bird_hight])
receiver_orientation = "z"                   # "x", "y" or "z"
data_type = "ppm"                            # "secondary", "total" or "ppm"

receiver_list = []
receiver_list.append(
    fdem.receivers.PointMagneticFieldSecondary(
        receiver_location, orientation=receiver_orientation,
        data_type=data_type, component="real"
    )
)
receiver_list.append(
    fdem.receivers.PointMagneticFieldSecondary(
        receiver_location, orientation=receiver_orientation,
        data_type=data_type, component="imag"
    )
)

# Define the source list. A source must be defined for each frequency.
source_location = np.array([0., 0., bird_hight])
source_orientation = 'z'                      # "x", "y" or "z"
moment = 1.                                   # dipole moment

source_list = []
for freq in frequencies:
    source_list.append(
        fdem.sources.MagDipole(
            receiver_list=receiver_list, frequency=freq,
            location=source_location, orientation=source_orientation, moment=moment
        )
    )

# Define a 1D FDEM survey
survey = fdem.survey.Survey(source_list)


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
simulation1 = fdem.Simulation1DLayered(survey=survey,
                                       thicknesses=thicknesses1,
                                       sigmaMap=model_mapping1)

# Predict sounding data
dpred = simulation1.dpred(model1)

d1_real = np.abs(dpred[0 : len(dpred) : 2])
d1_imag = np.abs(dpred[1 : len(dpred) : 2])



# add some noise to the data
noise = 0.05*np.abs(dpred)*np.random.rand(len(dpred))
dpredN = dpred + noise

d1N_real = np.abs(dpredN[0 : len(dpredN) : 2])
d1N_imag = np.abs(dpredN[1 : len(dpredN) : 2])


# %% plotting the stuff
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid(shape=(1, 3), loc=(0, 0), colspan=2)
ax3 = plt.subplot2grid(shape=(1, 3), loc=(0, 2), colspan=1,  rowspan=1)
ax=[ax1, ax3]

#

# Plot sounding data
ax[0].plot(frequencies, np.abs(d1_real), '.-', lw=2, ms=10, label=r"Real(d)", color=default_colors[0])
ax[0].plot(frequencies, np.abs(d1_imag), '.:', lw=2, ms=10, label=r"Imag(d)", color=default_colors[0])
ax[0].plot(frequencies, np.abs(d1N_real), '.-', lw=2, ms=10, label=r"noisy Real(d)", color=default_colors[1])
ax[0].plot(frequencies, np.abs(d1N_imag), '.:', lw=2, ms=10, label=r"noisy Imag(d)", color=default_colors[1])
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel("Frequency (Hz)")
ax[0].set_ylabel("|Hs/Hp| (ppm)")
ax[0].set_title("Secondary Magnetic Field as ppm")
ax[0].legend()
ax[0].grid(color="k", alpha=0.5, linestyle="dashed", linewidth=0.5)



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
    
    
    fname = dir_path + 'em1dfm_data.txt'
    np.savetxt(
        fname,
        np.c_[frequencies, dpredN[0:len(frequencies)], dpredN[len(frequencies):]],
        fmt='%.4e', header='FREQUENCY HZ_REAL HZ_IMAG'
    )
    
    mname=dir_path + "em1dfm_truemodel.txt"
    z1.append(9999.9)
    with open(mname, 'w') as f:
        f.write("Depth\tResitvity\n")
        for n in range(len(res1)):
            f.write(str(z1[n]) + "\t" + str(res1[n]))
            f.write('\n')
