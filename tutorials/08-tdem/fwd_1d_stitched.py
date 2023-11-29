"""
Stitched 1D Forward Simulation
==============================

Here we use the module *SimPEG.electromangetics.time_domain_1d* to predict
time electromagnetic domain (TDEM) data for a set of "stitched" 1D soundings. That is, the data
for each source is predicted for a separate, user-defined 1D model.
In this tutorial, we focus on the following:

    - Defining receivers, sources and the survey for the stitched 1D case
    - Constructing a stitched model - a set of 1D vertical conductivity profiels
    - Running a TDEM simulation

For each sounding, our survey geometry consists of a horizontal loop source with a
radius of 10 m located 30 m above the Earth's surface. The receiver is located at the centre
of the loop and measures the vertical component of the response.


"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
from scipy.spatial import Delaunay, cKDTree
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from discretize import TensorMesh
#from pymatsolver import PardisoSolver

from SimPEG import maps, utils
from SimPEG.utils import mkvc
import SimPEG.electromagnetics.time_domain as tdem
from SimPEG.electromagnetics.utils.em1d_utils import plot_layer, get_vertical_discretization_time, set_mesh_1d, Stitched1DModel

plt.rcParams.update({'font.size': 16})
write_output = True
dir_path = "./outputs"

#####################################################################
# Create Survey
# -------------
#
# Here we demonstrate a general way to define receivers, sources and the survey.
# For this tutorial, we define a line of equally spaced 1D soundings along the
# Easting direction. However, there is no restriction on the spacing and position
# of each sounding.
nx = 11
ny = 1
x = np.arange(nx)*50
y = np.arange(ny)*100
z = np.array([30.])

xyz = utils.ndgrid(x, y, z)
np.random.seed(1)
xyz[:,1] += np.random.randn(nx*ny) * 5
n_sounding = xyz.shape[0]
source_locations = xyz  # xyz locations for the centre of the loop
source_current = 1.
source_radius = 10.

receiver_locations = xyz   # xyz locations for the receivers
receiver_orientation = "z"            # "x", "y" or "z"
times = np.logspace(-5, -2, 16)       # time channels

# Define the waveform. In this case all sources use the same waveform.
waveform = tdem.sources.StepOffWaveform()

# For each sounding, we define the source and the associated receivers.
source_list = []
for ii in range(0, n_sounding):

    # Source and receiver locations
    source_location = mkvc(source_locations[ii, :])
    receiver_location = mkvc(receiver_locations[ii, :])

    # Receiver list for source i
    receiver_list = [
        tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_location, times, orientation=receiver_orientation
        )
    ]

    # Source ii
    source_list.append(
        tdem.sources.CircularLoop(
            receiver_list=receiver_list, location=source_location, waveform=waveform,
            radius=source_radius, current=source_current, i_sounding=ii
        )
    )

# Define the survey
survey = tdem.Survey(source_list)
sounding_types = np.ones(n_sounding, dtype=int)
survey._sounding_types = sounding_types

###############################################
# Defining a Global Mesh and Model
# --------------------------------
#
# It is easy to create and visualize 2D and 3D models in SimPEG, as opposed
# to an arbitrary set of local 1D models. Here, we create a 2D model
# which represents the global conductivity structure of the Earth. In the next
# part of the tutorial, we will demonstrate how the set of local 1D models can be
# extracted and organized for the stitched 1D simulation. This process can
# be adapted easily for 3D meshes and models.
#

# line number
line = (np.arange(ny).repeat(nx)).astype(float)
# time stamp
time_stamp = np.arange(n_sounding).astype(float)
# topography
topography = np.c_[xyz[:,:2], np.zeros(n_sounding)]
# vertical cell widths
hz = 10*np.ones(40)

# A function for generating a wedge layer 
def get_y(x):
    y = 30/500 * x + 70.
    return y

# Conductivity values for each unit
background_conductivity = 1./50.
layer_conductivity = 1./10.

# Define a 1D vertical mesh
mesh_1d = set_mesh_1d(hz)
# Generate a stitched 1D model
n_layer = hz.size
conductivity = np.zeros((n_sounding, n_layer), dtype=float)

for i_sounding in range(n_sounding):
    y = get_y(xyz[i_sounding, 0])
    layer_ind = np.logical_and(mesh_1d.vectorCCx>50., mesh_1d.vectorCCx<y)
    conductivity_1d = np.ones(n_layer, dtype=float) * background_conductivity
    conductivity_1d[layer_ind] = layer_conductivity
    conductivity[i_sounding,:]=conductivity_1d

# Note: oder of the conductivity model 
stitched_conductivity_model = conductivity.flatten()

# Generate a Stitched1DModel object for plotting
model_plot = Stitched1DModel(
    hz=hz,
    line=line,
    time_stamp=time_stamp,
    topography=topography,
    physical_property=1./stitched_conductivity_model,
    n_layer=len(hz)
)

_, ax, cb = model_plot.plot_section(cmap='turbo', aspect=0.5, dx=20, i_line=0, clim=(8, 100))
cb.set_label("Resistivity ($\Omega$m)")

plt.savefig(os.path.join(dir_path,'fwd_model.png'))

# the optimum layer thicknesses for a set number of layers. Note that when defining
# the thicknesses, it is the number of layers minus one.
thicknesses = hz[:-1]

#######################################################################
# Define the Mapping, Forward Simulation and Predict Data
# -------------------------------------------------------
#
# Here we define the simulation and predict the TDEM data.
# The simulation requires the user define the survey, the layer thicknesses
# and a mapping from the model to the conductivities.
#
# When using the *SimPEG.electromagnetics.time_domain_1d* module, predicted
# data are organized by source (sounding), then by receiver, then by time channel.
#

# Model and mapping. Here the model is defined by the log-conductivity.
stitched_model = np.log(stitched_conductivity_model)
mapping = maps.ExpMap(nP=len(stitched_model))

# Define the simulation
simulation = tdem.Simulation1DLayeredStitched(
    survey=survey, thicknesses=thicknesses, sigmaMap=mapping,
    parallel=False, n_cpu=2,
    n_layer=n_layer
)

# Predict data
dpred = simulation.dpred(stitched_model)

#######################################################################
# Plotting Results
# ----------------
#

d = np.reshape(dpred, (n_sounding, len(times)))

fig= plt.figure(figsize=(7, 7))
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])

i_line = 0
ind_line = line == i_line
for ii in range(0, len(times)):
    ax.semilogy(receiver_locations[ind_line, 0], np.abs(d[ind_line, ii]), 'k-', lw=3)
ax.set_xlabel("Sounding location (m)")
ax.set_ylabel("|dBdt| (T/s)")
ax.set_title("Line nubmer {:.0f}".format(i_line))

plt.savefig(os.path.join(dir_path,'fwd_data.png'))

#######################################################################
# Write Outputs (Optional)
# ------------------------
#
if write_output:
    import pandas as pd
    import tarfile
    import os.path

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    np.random.seed(1)
    noise = 0.03*np.abs(dpred)*np.random.rand(len(dpred))
    dpred += noise
    fname = os.path.join(dir_path,'em1dtm_stitched_data.csv')
    print(fname)
    fname_times = os.path.join(dir_path,'times.txt')

    DPRED = np.reshape(dpred, (n_sounding, len(times)))
    data_header = ["dbzdt_ch{:d}".format(ii+1)for ii in range(len(times))]
    i_count = 0
    sounding_number = np.arange(n_sounding)
    data = np.c_[sounding_number, line, source_locations, topography[:,2], DPRED]
    header = ['SOUNDINGNUMBER', 'LINENO', 'X', 'Y', 'Z', 'ELEVATION'] + data_header
    df = pd.DataFrame(data=data, columns=header)
    df.to_csv(fname, index=False)
    np.savetxt(fname_times, times)
    output_filename = 'em1dtm_stitched.tar.gz'
    def make_tarfile(output_filename, source_dir):
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
    fname_zip = dir_path + 'em1dtm_stitched_fwd.tar.gz'
    make_tarfile(fname_zip, dir_path)