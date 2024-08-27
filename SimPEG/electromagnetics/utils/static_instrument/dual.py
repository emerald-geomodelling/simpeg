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
    def gate_filter__start_lm(self):
        inuse_ch_key = "InUse_Ch01"
        return self.gt_filt_st(inuse_ch_key)

    @property
    def gate_filter__end_lm(self):
        inuse_ch_key = "InUse_Ch01"
        return self.gt_filt_end(inuse_ch_key)

    @property
    def gate_filter__start_hm(self):
        inuse_ch_key = "InUse_Ch02"
        return self.gt_filt_st(inuse_ch_key)

    @property
    def gate_filter__end_hm(self):
        inuse_ch_key = "InUse_Ch02"
        return self.gt_filt_end(inuse_ch_key)

    # FIXME!!! Tx_orientation should also be broken up by moment/channel,
    #  but without seeing how this would look in a gex I don't know how to redo this
    @property
    def tx_orientation(self):
        return self.gex.tx_orientation

    @property
    def rx_orientation(self):
        return [self.gex.rx_orientation(channel) for channel in range(self.gex.number_channels)]
        # return self.rx_orientation__lm

    # gate_filter__start_lm = 5
    # "Lowest used gate (zero based)"
    # gate_filter__end_lm = 11
    # "First unused gate above used ones (zero based)"
    # gate_filter__start_hm = 12
    # "Lowest used gate (zero based)"
    # gate_filter__end_hm = 26
    # "First unused gate above used ones (zero based)"
    # rx_orientation: typing.Literal['x', 'y', 'z'] = 'z'
    # "Receiver orientation"
    # tx_orientation: typing.Literal['x', 'y', 'z'] = 'z'
    # "Transmitter orientation"
    
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
        return self._xyz.dbdt_inuse_ch1gt.values.sum(axis=1) + self._xyz.dbdt_inuse_ch2gt.sum(axis=1) > 0

    @property
    def area(self):
        return self.gex.General['TxLoopArea']
    
    @property
    def waveform_hm(self):
        return self.gex.General['WaveformHMPoint']
    
    @property
    def waveform_lm(self):
        return self.gex.General['WaveformLMPoint']

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
        if "dbdt_inuse_ch1gt" in self.xyz.layer_data:
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
        return [self.gex.gex_dict[f'Channel{channel}']['ApproxDipoleMoment'] for channel in range(self.gex.number_channels)]
        # return [self.gex.gex_dict['Channel1']['ApproxDipoleMoment'],
        #         self.gex.gex_dict['Channel2']['ApproxDipoleMoment']]
        
    @property
    def times_full(self):
        return tuple(np.array(self.gex.gate_times(f'Channel{channel}')[:, 0]) for channel in range(self.gex.number_channels))
        # return (np.array(self.gex.gate_times('Channel1')[:,0]),
        #         np.array(self.gex.gate_times('Channel2')[:,0]))

    @property
    def times_filter(self):        
        times = self.times_full
        filts = [np.zeros(len(t), dtype=bool) for t in times]
        filts[0][self.gate_filter__start_lm:self.gate_filter__end_lm] = True
        filts[1][self.gate_filter__start_hm:self.gate_filter__end_hm] = True
        return filts

    @property
    def uncertainties__std_data(self):
        return [self.gex.gex_dict[f'Channel{channel}']['UniformDataSTD'] for channel in range(self.gex.number_channels)]

    @property
    def uncertainties__std_data_override(self):
        # "If set to true, use the std_data value instead of data std:s from stacking"
        return [f"dbdt_std_ch{channel}gt" not in self.xyz.layer_data.keys() for channel in range(self.gex.number_channels)]

    def make_waveforms(self):
        time_input_currents_hm = self.waveform_hm[:,0]
        input_currents_hm = self.waveform_hm[:,1]
        time_input_currents_lm = self.waveform_lm[:,0]
        input_currents_lm = self.waveform_lm[:,1]

        waveform_hm = tdem.sources.PiecewiseLinearWaveform(time_input_currents_hm, input_currents_hm)
        waveform_lm = tdem.sources.PiecewiseLinearWaveform(time_input_currents_lm, input_currents_lm)
        return waveform_lm, waveform_hm
    
    def make_system(self, idx, location, times):
        # FIXME: Martin says set z to altitude, not z (subtract topo), original code from seogi doesn't work!
        # Note: location[2] is already == altitude
        receiver_location = (location[0] + self.gex.General['RxCoilPosition'][0],
                             location[1],
                             location[2] + np.abs(self.gex.General['RxCoilPosition'][2]))
        waveform_lm, waveform_hm = self.make_waveforms()        

        return [
            tdem.sources.MagDipole(
                [tdem.receivers.PointMagneticFluxTimeDerivative(
                    receiver_location, times[0], self.rx_orientation)],
                location=location,
                waveform=waveform_lm,
                orientation=self.tx_orientation,
                i_sounding=idx),
            tdem.sources.MagDipole(
                [tdem.receivers.PointMagneticFluxTimeDerivative(
                    receiver_location, times[1], self.rx_orientation)],
                location=location,
                waveform=waveform_hm,
                orientation=self.tx_orientation,
                i_sounding=idx)]

    
