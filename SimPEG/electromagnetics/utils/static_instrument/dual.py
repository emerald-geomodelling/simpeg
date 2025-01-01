import os
import tarfile
import typing

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import libaarhusxyz
import pandas as pd
import numpy as np
import scipy.stats
from scipy.spatial import cKDTree, Delaunay
from discretize import TensorMesh, SimplexMesh

from SimPEG import (
    maps, data, data_misfit, inverse_problem, regularization, optimization,
    directives, inversion, utils
    )

from SimPEG.utils import mkvc
import SimPEG.electromagnetics.time_domain as tdem
import SimPEG.electromagnetics.utils.em1d_utils
from SimPEG.electromagnetics.utils.em1d_utils import (
    get_2d_mesh, plot_layer, get_vertical_discretization_time
    )
from SimPEG.regularization import LaterallyConstrained, RegularizationMesh

from . import base


class DualMomentTEMXYZSystem(base.XYZSystem):
    """Dual moment system, suitable for describing e.g. the SkyTEM
    instruments. This class can not be directly instantiated, but
    instead, instantiable subclasses can be created using the class
    method

    ```
    MySurveyInstrument = DualMomentTEMXYZSystem.load_gex(
        libaarhusxyz.GEX("instrument.gex"))
    ```

    which reads a gex file containing among other things the waveform
    description of the instrument.

    See the help for `XYZSystem` for more information on basic usage.
    """

    def gt_filt_st(self, inuse_ch_key):
        return np.where((self.xyz.layer_data[inuse_ch_key].sum() == 0).cumsum().diff() == 0)[0][0]

    def gt_filt_end(self, inuse_ch_key):
        return len(self.xyz.layer_data[inuse_ch_key].loc[1, :]) - np.where((self.xyz.layer_data[inuse_ch_key].sum().loc[::-1] == 0).cumsum().diff() == 0)[0][0]

    @property
    def gate_filter__start(self):
        start_list = []
        for channel in range(1, 1 + self.gex.number_channels):
            # str_ch = f"0{channel}"[-2:]
            # inuse_ch_key = f"InUse_Ch{str_ch}"
            inuse_ch_key = f"dbdt_inuse_ch{channel}gt"
            start_list.append(self.gt_filt_st(inuse_ch_key))
        return start_list

    @property
    def gate_filter__end(self):
        end_list = []
        for channel in range(1, 1 + self.gex.number_channels):
            # str_ch = f"0{channel}"[-2:]
            # inuse_ch_key = f"InUse_Ch{str_ch}"
            inuse_ch_key = f"dbdt_inuse_ch{channel}gt"
            end_list.append(self.gt_filt_end(inuse_ch_key))
        return end_list

    # FIXME!!! Tx_orientation should also be broken up by moment/channel,
    #  but without seeing how this would look in a gex I don't know how to redo this
    @property
    def tx_orientation(self):
        return self.gex.tx_orientation

    @property
    def rx_orientation(self):
        return [(self.gex.rx_orientation(channel)).lower() for channel in range(1, 1 + self.gex.number_channels)]
    
    @classmethod
    def load_gex(cls, gex):
        """Accepts a GEX file loaded using libaarhusxyz.GEX() and
        returns a new subclass of XYZSystem that can be used to do
        inversion and forward modelling."""
        
        class GexSystem(cls):
            pass
        GexSystem.gex = gex
        return GexSystem   
    
    @property
    def sounding_filter(self):
        # Exclude soundings with no usable gates
        # return self._xyz.dbdt_inuse_ch1gt.values.sum(axis=1) + self._xyz.dbdt_inuse_ch2gt.sum(axis=1) > 0
        num_gates_per_sounding_per_moment = {}
        for channel in range(self.gex.number_channels):
            iu_key = f"dbdt_inuse_ch{channel + 1}gt"
            num_gates_per_sounding_per_moment[channel] = self._xyz.layer_data[iu_key].values.sum(axis=1)
        num_gates_per_sounding = pd.DataFrame.from_dict(num_gates_per_sounding_per_moment)
        print(f"num_gates_per_sounding = {num_gates_per_sounding}")
        print(f"num_gates_per_sounding.sum(axis=1) > 0 = {num_gates_per_sounding.sum(axis=1) > 0}")
        return num_gates_per_sounding.sum(axis=1) > 0

    @property
    def area(self):
        return self.gex.General['TxLoopArea']

    @property
    def waveform(self):
        return [self.gex.General[f"Waveform{self.gex.transmitter_moment(channel)}Point"] for channel in range(1, 1 + self.gex.number_channels)]


    @property
    def correct_tilt_pitch_for1Dinv(self):
        cos_roll = np.cos(self.xyz.flightlines.tilt_x.values/180*np.pi)
        cos_pitch = np.cos(self.xyz.flightlines.tilt_y.values/180*np.pi)
        return 1 / (cos_roll * cos_pitch)**2
    
    @property
    def lm_data(self):
        dbdt = self.xyz.dbdt_ch1gt.values
        if "dbdt_inuse_ch1gt" in self.xyz.layer_data:
            dbdt = np.where(self.xyz.dbdt_inuse_ch1gt == 0, np.nan, dbdt)
        tiltcorrection = self.correct_tilt_pitch_for1Dinv
        tiltcorrection = np.tile(tiltcorrection, (dbdt.shape[1], 1)).T
        channel = 1
        return - dbdt * self.xyz.model_info.get("scalefactor", 1) * self.gex[f"Channel{channel}"]['GateFactor'] * tiltcorrection
    
    @property
    def hm_data(self):
        dbdt = self.xyz.dbdt_ch2gt.values
        if "dbdt_inuse_ch2gt" in self.xyz.layer_data:
            dbdt = np.where(self.xyz.dbdt_inuse_ch2gt == 0, np.nan, dbdt)
        tiltcorrection = self.correct_tilt_pitch_for1Dinv
        tiltcorrection = np.tile(tiltcorrection, (dbdt.shape[1], 1)).T
        channel = 2
        return - dbdt * self.xyz.model_info.get("scalefactor", 1) * self.gex[f"Channel{channel}"]['GateFactor'] * tiltcorrection

    # NOTE: dbdt_std is a fraction, not an actual standard deviation size!
    @property
    def lm_std(self):
        return self.xyz.dbdt_std_ch1gt.values
    
    @property
    def hm_std(self):
        return self.xyz.dbdt_std_ch2gt.values

    @property
    def data_array_nan(self):
        return np.hstack((self.lm_data, self.hm_data)).flatten()

    @property
    def data_uncert_array(self):
        return np.hstack((self.lm_std, self.hm_std)).flatten()

    @property
    def dipole_moments(self):
        return [self.gex.gex_dict[f'Channel{channel}']['ApproxDipoleMoment'] for channel in range(1, 1 + self.gex.number_channels)]
        # return [self.gex.gex_dict['Channel1']['ApproxDipoleMoment'],
        #         self.gex.gex_dict['Channel2']['ApproxDipoleMoment']]
        
    @property
    def times_full(self):
        return tuple(np.array(self.gex.gate_times(channel)[:, 0]) for channel in range(1, 1 + self.gex.number_channels))
        # return (np.array(self.gex.gate_times(1)[:,0]),
        #         np.array(self.gex.gate_times(2)[:,0]))

    @property
    def times_filter(self):        
        times = self.times_full
        filts = [np.zeros(len(t), dtype=bool) for t in times]
        # FIXME: loop over channel number
        for channel in range(self.gex.number_channels):
            filts[channel][self.gate_filter__start[channel]: self.gate_filter__end[channel]] = True
        return filts

    @property
    def uncertainties__std_data(self):
        return [self.gex.gex_dict[f'Channel{channel}']['UniformDataSTD'] for channel in range(1, 1 + self.gex.number_channels)]

    @property
    def uncertainties__std_data_override(self):
        # "If set to true, use the std_data value instead of data std:s from stacking"
        return [f"dbdt_std_ch{channel}gt" not in self.xyz.layer_data.keys() for channel in range(1, 1 + self.gex.number_channels)]

    def make_waveforms(self):
        time_input_currents = []
        input_currents = []
        for channel in range(self.gex.number_channels):
            time_input_currents.append(self.waveform[channel][:,0])
            input_currents.append(self.waveform[channel][:,1])

        return [tdem.sources.PiecewiseLinearWaveform(time_input_currents[channel], input_currents[channel]) for channel in range(self.gex.number_channels)]

    
    def make_system(self, idx, location, times):
        # FIXME: Martin says set z to altitude, not z (subtract topo), original code from seogi doesn't work!
        # Note: location[2] is already == altitude
        receiver_location = (location[0] + self.gex.General['RxCoilPosition'][0],
                             location[1],
                             location[2] + np.abs(self.gex.General['RxCoilPosition'][2]))
        waveforms = self.make_waveforms()
        return [tdem.sources.MagDipole([tdem.receivers.PointMagneticFluxTimeDerivative(receiver_location,
                                                                                       times[channel],
                                                                                       self.rx_orientation[channel])],
                                       location=location,
                                       waveform=waveforms[channel],
                                       orientation=self.tx_orientation,
                                       i_sounding=idx)
                for channel in range(self.gex.number_channels)]
