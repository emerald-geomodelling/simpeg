"""
1D Inversion of for a Single Sounding
=====================================

Here we use the module *SimPEG.electromangetics.frequency_domain_1d* to invert
frequency domain data and recover a 1D electrical conductivity model.
In this tutorial, we focus on the following:

    - How to define sources and receivers from a survey file
    - How to define the survey
    - Sparse 1D inversion of with iteratively re-weighted least-squares

For this tutorial, we will invert 1D frequency domain data for a single sounding.
The end product is layered Earth model which explains the data. The survey
consisted of a vertical magnetic dipole source located 30 m above the
surface. The receiver measured the vertical component of the secondary field
at a 10 m offset from the source in ppm.

"""


#########################################################################
# Import modules
# --------------
#

import os, tarfile
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from discretize import TensorMesh

import SimPEG.electromagnetics.time_domain as tdem
from SimPEG.electromagnetics.utils.em1d_utils import get_vertical_discretization_frequency, plot_layer
from SimPEG.utils import mkvc
from SimPEG import (
    maps, data, data_misfit, inverse_problem, regularization, optimization,
    directives, inversion, utils
    )

default_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


plt.rcParams.update({'font.size': 14, 'lines.linewidth': 2, 'lines.markersize':8})

# sphinx_gallery_thumbnail_number = 2
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

# %% ######Survey data and setup ##########################

# data generated with  to work with Synthetic_EM1D_FEM_Forward.py
data_filename = "./outputs/em1dtem_data.txt"
rel_err =  0.03




# true model:
model_filename = "./outputs/em1dtem_truemodel.txt"
D=np.loadtxt(model_filename, delimiter="\t", skiprows=1)
true_model = 1./D[:,1]
hz = np.diff(D[:,0])
hz = np.r_[hz, 100.]
true_layers = TensorMesh([hz])


# inversion Layer thicknesses
inv_thicknesses = np.logspace(0,1.5,30)
startmodel_res=300
# %% ###########################################
# Load Data and Plot
# ------------------
#
# Here we load and plot the 1D sounding data. In this case, we have the 
# secondary field response in ppm for a set of frequencies.
#

# Load field data
#dobs = np.loadtxt(str(data_filename))
dobs = np.loadtxt(str(data_filename), skiprows=1)

# Define receiver locations and observed data
times = dobs[:, 0]
dobs = dobs[:, 1]



# %% ############################################
# Defining the Survey
# -------------------
# 
# Here we demonstrate a general way to define the receivers, sources and survey.
# The survey consisted of a vertical magnetic dipole source located 30 m above the
# surface. The receiver measured the vertical component of the secondary field
# at a 10 m offset from the source in ppm.
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


# Survey
survey = tdem.Survey(source_list)


# %% ###########################################################
# Assign Uncertainties and Define the Data Object
# -----------------------------------------------
#
# Here is where we define the data that are inverted. The data are defined by
# the survey, the observation values and the uncertainties.
#



noise_power=-0.5
noise_amplitude=2e-9 # at 1e-3s (1ms)
ambient_noise=(times*1e3)**noise_power * noise_amplitude

# 5% of the absolute value
#uncertainties =rel_err *np.abs(dobs)*np.ones(np.shape(dobs))
uncertainties =rel_err *np.abs(dobs)*np.ones(np.shape(dobs)) + ambient_noise


# Define the data object
#data_object = data.Data(survey, dobs=dobs, noise_floor=ambient_noise, standard_deviation=uncertainties)
data_object = data.Data(survey, dobs=dobs, standard_deviation=uncertainties)


# %% ##############################################################
# Defining a 1D Layered Earth (1D Tensor Mesh)
# --------------------------------------------
#
# Here, we define the layer thicknesses for our 1D simulation. To do this, we use
# the TensorMesh class.
#



# Define a mesh for plotting and regularization.
mesh = TensorMesh([(np.r_[inv_thicknesses, inv_thicknesses[-1]])], '0')


# %% ################################################################
# Define a Starting and/or Reference Model and the Mapping
# --------------------------------------------------------
#
# Here, we create starting and/or reference models for the inversion as
# well as the mapping from the model space to the active cells. Starting and
# reference models can be a constant background value or contain a-priori
# structures. Here, the starting model is log(0.1) S/m.
#
# Define log-conductivity values for each layer since our model is the
# log-conductivity. Don't make the values 0!
# Otherwise the gradient for the 1st iteration is zero and the inversion will
# not converge.

# Define model. A resistivity (Ohm meters) or conductivity (S/m) for each layer.
starting_model = np.log(1/startmodel_res*np.ones(mesh.nC))

# Define mapping from model to active cells.
model_mapping = maps.ExpMap()


# %% ######################################################################
# Define the Physics using a Simulation Object
# --------------------------------------------
#

simulation = tdem.Simulation1DLayered(
    survey=survey, thicknesses=inv_thicknesses, sigmaMap=model_mapping
)



# %% ######################################################################
# Define Inverse Problem
# ----------------------
#
# The inverse problem is defined by 3 things:
#
#     1) Data Misfit: a measure of how well our recovered model explains the field data
#     2) Regularization: constraints placed on the recovered model and a priori information
#     3) Optimization: the numerical approach used to solve the inverse problem
#
#

# Define the data misfit. Here the data misfit is the L2 norm of the weighted
# residual between the observed data and the data predicted for a given model.
# The weighting is defined by the reciprocal of the uncertainties.
dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)
dmis.W = 1./uncertainties

# Define the regularization (model objective function)
reg_map = maps.IdentityMap(nP=mesh.nC)

reg = regularization.Sparse(mesh, mapping=reg_map, alpha_s=0.01, alpha_x=1.0)
#reg = regularization.Tikhonov(mesh, mapping=reg_map, alpha_s=0.025, alpha_x=1)


# # reference model
reg.mref = starting_model

# # Define sparse and blocky norms p, q
p = 0
q = 0
reg.norms = np.c_[p, q]

# Define how the optimization problem is solved. Here we will use an inexact
# Gauss-Newton approach that employs the conjugate gradient solver.
opt = optimization.ProjectedGNCG(maxIter=60, maxIterLS=20, maxIterCG=30, tolCG=1e-3)

# Define the inverse problem
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)


#######################################################################
# Define Inversion Directives
# ---------------------------
#
# Here we define any directiveas that are carried out during the inversion. This
# includes the cooling schedule for the trade-off parameter (beta), stopping
# criteria for the inversion and saving inversion results at each iteration.
#

# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e1)

# Update the preconditionner
update_Jacobi = directives.UpdatePreconditioner()

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

# Directive for the IRLS
update_IRLS = directives.Update_IRLS(
    max_irls_iterations=30, minGNiter=1,
    coolEpsFact=1.5, update_beta=True
)

# Updating the preconditionner if it is model dependent.
update_jacobi = directives.UpdatePreconditioner()

# Add sensitivity weights
sensitivity_weights = directives.UpdateSensitivityWeights()

# The directives are defined as a list.
directives_list = [
    sensitivity_weights,
    starting_beta,
    save_iteration,
    update_IRLS,
    update_jacobi,
]

# %% ####################################################################
# Running the Inversion
# ---------------------
#
# To define the inversion object, we need to define the inversion problem and
# the set of directives. We can then run the inversion.
#

# Here we combine the inverse problem and the set of directives
inv = inversion.BaseInversion(inv_prob, directives_list)

# Run the inversion
recovered_model = inv.run(starting_model)


# %% ####################################################################
# Plotting Results
# ---------------------
# 
plotSparse=True
plotL2=True


# Extract Least-Squares model
if plotL2:
    l2_model = inv_prob.l2model

# Plot true model and recovered model

plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(12, 10))
ax1 = plt.subplot2grid(shape=(1, 3), loc=(0, 0), colspan=2)
ax2 = plt.subplot2grid(shape=(1, 3), loc=(0, 2), colspan=1)



#x_min = np.min(np.r_[model_mapping * recovered_model, model_mapping * l2_model, true_model])
#x_max = np.max(np.r_[model_mapping * recovered_model, model_mapping * l2_model, true_model])

plot_layer(true_model, true_layers, ax=ax2, showlayers=False, plotSig=False, color="k",  linestyle="dashed", label="True Model")
if plotL2:
    plot_layer(model_mapping * l2_model, mesh, ax=ax2, showlayers=False, plotSig=False, color="b", label="L2-Model")
if plotSparse:
    plot_layer(model_mapping * recovered_model, mesh, ax=ax2, showlayers=False, plotSig=False, color="r", label="Sparse Model")
ax2.set_ylim(0., 150.0)
ax2.set_title("True and Recovered Models")
ax2.legend(loc="lower right")
plt.gca().invert_yaxis()
ax2.grid(color="k", alpha=0.5, linestyle="dashed", linewidth=0.5)
ax2.set_xlim([1e1, 1e4])


# Plot predicted and observed data
dpred_l2 = simulation.dpred(l2_model)
dpred_final = simulation.dpred(recovered_model)

ax1.loglog(times, -dobs, "k-o", label="Observed")
if plotL2:
    ax1.loglog(times, -dpred_l2, "b-o", label="L2-Model")
if plotSparse:
    ax1.loglog(times, -dpred_final, "r-o", label="Sparse ")
ax1.plot(times, ambient_noise, 'k--')
ax1.set_xlabel("Times (s)")
ax1.set_ylabel(r"\partial B / \partial t (B/s)")
ax1.set_title("Predicted and Observed Data")
ax1.legend(loc="upper left")
ax1.grid(color="k", alpha=0.5, linestyle="dashed", linewidth=0.5)
#ax1.set_xlim([2e2, 2e5])
#ax1.set_ylim([1, 1e3])