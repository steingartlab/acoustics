import numpy as np
from obspy.signal.trigger import classic_sta_lta
import pandas as pd
import scipy.interpolate
import scipy.signal
from scipy import stats

import resonance.resonance.limit_memory
import backend
import GT_GM_modulus_exp_params as experimental_params

def _get_sampling_rate(no_samples_per_pass: int, exp_params: dict):
    """Gets oscilloscope sampling rate.
    
    Helper function for get_absolute_tof().
    
    Returns:
        (float): Sampling rate [Hz]
    """
    return no_samples_per_pass / (exp_params['t_1'] - exp_params['t_0'])

def absolute_tof(waves: np.array, exp_params: dict, t_sta: float = 0.05) -> list:
    """Gets absolute Time-of-Flight from waveform amplitude by utilizing the STA/LTA algorithm.
    
    Modified from Wes/Rob's first_break_picker().
    
    Args:
        waves (np.array): The digitized waveform amplitudes
        exp_params (dict): See definition
        t_sta (float): Optional. I don't have a clue why we need it, but everything breaks
            when I remove it so it stays. Defaults to 0.05.

    Returns:
        absolute_ToFs (list): Absolute ToF (as opposed to relative ToF used for ToF shift)
        trigger_indices (list): The waves indices where the wave 'arrives'.
    """

    no_samples_per_pass = waves.shape[1]
    sampling_rate = _get_sampling_rate(no_samples_per_pass, exp_params)
    nsta = t_sta * sampling_rate  # Length of short time average window in samples
    nlta = exp_params['lta2sta'] * nsta  # Length of long time average window in samples
    
    absolute_ToFs = np.zeros(len(waves))
    trigger_indices = np.zeros(len(waves))
    for i, wave in enumerate(waves):
        # This implementation is faster than the custom function b/c it's written in C
        stalta = classic_sta_lta(
            a=wave,
            nsta=int(nsta),
            nlta=int(nlta)
        )

        threshold = 0.75 * np.nanmax(stalta)  # We lover the threshold a little bit
        
        # threshold for first break picker
        trigger_index = (stalta > threshold).argmax() if (stalta > threshold).any() else -1
        absolute_ToFs[i] = exp_params['t_0'] - exp_params['t_ref'] + (trigger_index / no_samples_per_pass) * (exp_params['t_1'] - exp_params['t_0'])
        trigger_indices[i] = trigger_index

    return absolute_ToFs, trigger_indices

def normalize_by_nth(data, n=0):
    """Normalizes data by n-th value"""
    return data / data[n]

def remove_outliers(array: np.array, z_max: float = 3.0) -> np.array:
    """Removes outliers by filtering the z-score (i.e. no of stdevs from mean).
    
    Args:
        array (np.array): Only tested on 1D arrays. If doesn't work on 2D
            then it only needs slight tweaks.
        z_max (float): Optional. Max z-score (st devs) to include when
            when removing outliers. Defaults to 3.0

    Returns:
        (np.array)
    """

    z_score = stats.zscore(array)
    
    return array[(np.abs(z_score) < z_max)]

def total_amplitude(waves: np.array, z_max: float = 3.0) -> np.array:
    """Calculates total waveform amplitude through convolution.
    
    Args:
        waves (np.array): The 2D wave data w shape no_pulsesXno_points
        z_max (float): Optional. Max z-score (st devs) to include when
            when removing outliers. Defaults to 3.0
    
    Returns:
        (np.array)
    """

    total_amplitude_raw = np.zeros(waves.shape[0])
    for i in range(len(waves)):
        total_amplitude_raw[i] = np.dot(waves[i,:], waves[i,:].T)

    # total_amplitude = remove_outliers(array=total_amplitude_raw)
    
    return total_amplitude_raw

def tof_shift(waves: np.array, delay: float = 26.6, duration: float = 6.0, points: int = 10000, kind='cubic', tolerance=3., fill: int = 128) -> np.array:
    """Calculates ToF-shift by utilizing cross-correlation.

    Initially written by GD. Modified by GT.
    
    Args:
        waves (np.array): The waveforms as processed by backend.get_acoustic_amplitudes().
        delay (float): Optional. The time it takes for the wave to pass through the Rexolite spacer [us].
            Defaults to 26.6 (934 temperature chamber)
        duration (float): Optional. The 'open window' we inspect, starting at time _delay_.
            Defaults to 6.0
        points (int): Optional. A resampling factor. Proportional to ToF-shift curve temporal resolution.
            Defaults to 10000.
        tolerance (float): Optional. TBC. Defaults to 3.0
        fill (int): Optional. Value to include in interpolation if NaN is encountered.
            Defaults to 128

    Returns:
        tof_shift (np.array): ToF shift as a 1D time series
    """

    tofs = np.linspace(delay, delay + duration, len(waves[0]))  # Linearly spaced
    t_resampled = np.linspace(delay, delay + duration, int(points))
    wave_reference = scipy.interpolate.interp1d(
        x=tofs,
        y=waves[0],
        kind=kind,
        bounds_error=False,
        fill_value=fill
    )

    tof_shift = np.zeros(len(waves))  # Preallocate
    tof_ = ToFShift(time=t_resampled, tolerance=tolerance)
    for i, wave in enumerate(waves):
        wave_being_inspected = scipy.interpolate.interp1d(
            x=tofs,
            y=wave,
            kind=kind,
            bounds_error=False,
            fill_value=fill
        )
        tof_shift[i] -= tof_.cross_correlate_tolerance(
            crossA=wave_reference(t_resampled),
            crossB=wave_being_inspected(t_resampled),
        )

    return tof_shift


class ToFShift:
    """Lower-level ToF shift stuff. Should generally not be called externally but from tof_shift().
    
    Contains several class attributes for them only to be calculated once which cuts down on
    running time.
    """

    def __init__(self, time: np.array, tolerance: float = 3.0, tof_prev=0.):
        self.time_span = time[-1] - time[0]
        no_steps = len(time) - 1
        self.step_time = self.time_span / no_steps
        self.tolerance = tolerance
        self.tof_prev = tof_prev
        self.old_index = (self.tof_prev + self.time_span) / self.step_time

    def cross_correlate_tolerance(self, crossA, crossB) -> float:
        """Outputs time lag of crossing A&B over t_range time with shifts calculated within a given tolerance.

        Minimizes the error of subtracting shifted wave from reference by incorporating a cross-correlation (convolution integral) to find the wave time-of-flight shift .

        Written by GD and WC. Modified by GT.

        Args:
            crossA (np.array):
            crossB (np.array):
            time (np.array):
            tolerance (float): Optional. TBC. Defaults to 3.0

        Returns:
            delta_t (float): The ToF shift from the initial waveform to the waveform being inspected.
        """
    
        # Determine delta_t
        cross_t = scipy.signal.correlate(crossA, crossB)
        cross_t = cross_t.tolist()
        
        # Find the index of the previous max correlation point
        range_low, range_high = self.get_range(cross_t=cross_t)

        if len(cross_t[range_low:range_high]) > 0:
            max_index = cross_t.index(max(cross_t[range_low:range_high]))
        else:
            max_index = old_index

        delta_t = self.step_time * max_index  # shift from max lag
        delta_t -= self.time_span  # shift from 0 lag
        
        if abs(delta_t-self.tof_prev) > self.tolerance:
            delta_t = self.tof_prev
        
        return delta_t

    def get_range(self, cross_t) -> [int, int]:
        """Helper function for cross_correlate_tolerance()."""

        range_low = self.old_index - 0.5 * self.tolerance / self.time_span * len(cross_t)
        range_high = self.old_index + 0.5 * self.tolerance / self.time_span * len(cross_t)

        return int(range_low), int(range_high)

def test(params):
    dt, waves = backend.get_acoustics(
        exp_id=params['exp_id'],
        machine=params['machine'],
        waves_column='data',
        n=params['nth']
    )
    dt -= pd.Timedelta('1 hour')  # Daylight savings...
    tof_shift_ = tof_shift(waves=waves)
    plt.plot(dt, tof_shift_, 'k', linewidth=1)
    showme()
    
if __name__ == '__main__':
    exp_params = experimental_params.exp_params
    for name, params_ in exp_params.items():
        if name == 'cell_60':
            params = params_
            break

    test(params)