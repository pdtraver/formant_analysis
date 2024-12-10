import pandas as pd
import librosa
import os
import numpy as np
from typing import Dict, Optional, List, Tuple
import constants
from scipy.optimize import shgo
from scipy.linalg import norm
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve1d


class RawAudio:
    signal: np.ndarray
    original_sample_rate: int
    sample_rate: int

    def __init__(
        self,
        signal: np.ndarray,
        original_sample_rate: int,
        sample_rate: int,
    ):
        self.signal = signal
        self.original_sample_rate = original_sample_rate
        self.sample_rate = sample_rate

class BufferParams:
    buffer_size: Optional[int]
    hop_size: Optional[int]
    window_type: str

    def __init__(self, buffer_size: Optional[int], hop_size: Optional[int],
                 window_type: str):
        self.buffer_size = buffer_size
        self.hop_size = hop_size

        # Confirm window type is valid
        assert window_type.lower().split(' ')[0] in constants.WINDOW_TYPES

        self.window_type = window_type
        
class QCPParams:
    '''Parameters for Quasi-closed phase (QCP) analysis windows 
    
    Using t_ramp for now, though more dynamic Ramp_Quotient might serve useful in the future
    '''
    n_ramp: float
    duration_quotient: float
    position_quotient: float
    d: float
    
    def __init__(self, 
                 n_ramp: float, 
                 duration_quotient: float, 
                 position_quotient: float,
                 d: float = 1e-5):
        self.n_ramp = n_ramp
        self.duration_quotient = duration_quotient
        self.position_quotient = position_quotient
        self.d = d
        
class QCPWindowVector:
    """Quasi-closed Phase (QCP) window for Weighted Linear Prediction (WLP)

    Glottal Closure Instants (GCIs) are needed for calculation
    """
    gci_multihot_vector: np.ndarray
    qcp_params: QCPParams
    window_vector: np.ndarray
    
    def __init__(self, gci_multihot: np.ndarray, qcp_params: QCPParams):
        self.gci_multihot_vector = gci_multihot
        self.qcp_params = qcp_params
        self.window_vector = self.get_qcp_window_vector(gci_multihot, qcp_params)
        
    def get_qcp_window_vector(self, gci_multihot: np.ndarray, qcp_params: QCPParams):
        # Get indicies of all the GCI instances
        gci_indices = np.where(gci_multihot == 1)[0]
        
        # Unpack qcp_params:
        position_quotient = qcp_params.position_quotient
        n_samp = qcp_params.n_ramp
        duration_quotient = qcp_params.duration_quotient
        d = qcp_params.d
        
        #Initialize weight array with value d
        weights = np.full(len(gci_multihot), d)
        
        # Iterate across gci_indices & compute windows between all GCIs
        for i in range(len(gci_indices)-1):
            gci_1, gci_2 = gci_indices[i], gci_indices[i+1]
            frame_length = gci_2 - gci_1
            
            position_delay = int(frame_length * position_quotient)
            ramp_length = int(n_samp)
            sustain_duration = int(frame_length * duration_quotient - 2*ramp_length)
            
            # Get indices of non-d values of window
            start_ramp_up = gci_1 + position_delay
            start_sustain = start_ramp_up + ramp_length
            start_ramp_down = start_sustain + sustain_duration
            end_ramp_down = start_ramp_down + ramp_length
            
            # build window for frame
            weights[start_ramp_up:start_sustain] = np.linspace(d, 1, start_sustain - start_ramp_up, endpoint=False)
            weights[start_sustain:start_ramp_down] = 1
            weights[start_ramp_down:end_ramp_down] = np.linspace(1, d, end_ramp_down - start_ramp_down, endpoint=False)
        
        return weights

class BufferedAudio:
    """Prepares buffered and windowed audio given raw_audio
    """
    raw_audio: RawAudio
    buffer_params: BufferParams
    audio_buffers: np.ndarray
    gci_times: np.ndarray
    qcp_params: QCPParams
    qcp_window_vector: QCPWindowVector
    qcp_buffers: np.ndarray

    def __init__(self, raw_audio: RawAudio, buffer_params: BufferParams, gci_times: np.ndarray, qcp_params: QCPParams):
        self.raw_audio = raw_audio
        self.buffer_params = buffer_params
        self.audio_buffers = self.get_audio_buffers(buffer_params, raw_audio=raw_audio)
        self.gci_times = gci_times
        self.qcp_params = qcp_params
        self.qcp_window_vector = self.get_qcp_window_vector(gci_times, qcp_params)
        self.qcp_buffers = self.get_qcp_buffers(self.qcp_window_vector.window_vector, buffer_params)

    def get_audio_buffers(
        self,
        buffer_params: BufferParams,
        raw_audio: Optional[RawAudio] = None,
        gci_multihot: Optional[np.ndarray] = None,
        apply_window: bool = True
    ):
        """Function to generate 2-d array of audio buffers given buffer parameters

        Args:
            raw_audio: Raw audio (signal, resampled to global sr, original sample rate)
            buffer_params: % overlap, buffer size and window type
            
        Returns:
            audio_buffers: 2-d array of audio buffers given params
        """
        if raw_audio:
            full_signal = raw_audio.signal
        else:
            full_signal = gci_multihot
            
        buffer_size = buffer_params.buffer_size
        hop_size = buffer_params.hop_size
        if apply_window:
            window_type = buffer_params.window_type.lower().split(' ')[0]
            window = self.get_window(buffer_size, window_type)

        # Divide the signal length by hop-size, leaving extra room for last buffer
        num_buffers = int(np.floor(
            (len(full_signal) - buffer_size) / hop_size)) + 1

        # Grab each buffer, hopping by hop size each iteration
        # Multiply by window before storing in output array
        audio_buffers = np.zeros((num_buffers, buffer_size))
        for i in range(num_buffers):
            start = i * hop_size
            audio_buffers[i, :] = full_signal[start:(start + buffer_size)]
            if apply_window:
                audio_buffers[i, :] *= window

        return audio_buffers
    
    def get_qcp_window_vector(self, gci_times: np.ndarray, qcp_params: QCPParams):
        gci_multihot_vector = self.convert_to_multihot_sample_vector(gci_times)
        return QCPWindowVector(gci_multihot_vector, qcp_params)
    
    def get_qcp_buffers(self, qcp_multihot: np.ndarray, buffer_params: BufferParams):
        '''Get QCP buffers
        '''
        qcp_buffers = self.get_audio_buffers(buffer_params=buffer_params, gci_multihot=qcp_multihot, apply_window=False)
        return qcp_buffers
    
    def convert_to_multihot_sample_vector(self, gci_times:np.ndarray):
        '''Converts GCI Times into a multihot vector indicating the samples where
        GCI events occur
        
        Args:
            gci_times: Array of timecodes (floats) representing where the GCI events occur
            
        Returns:
            Multihot embedding of timecodes into a signal of equivalent length to the digital audio
        '''
        signal = self.raw_audio.signal
        sr = self.raw_audio.sample_rate
        sample_vector = np.zeros(len(signal))
        sample_indices = (gci_times * sr).astype(int)
        sample_vector[sample_indices] = 1
        return sample_vector
        
    def get_window(self, buffer_size: int, window_type: str):
        """Function to generate array of window values given buffer size & window type

        Args:
            buffer_size: Size of audio buffers
            window_type: Str indicating type of window (hann, hamming, blackman-harris, etc.)
            
        Returns:
            window: 1-d array of size buffer_size of time-domain window values
        """
        window = np.arange(-buffer_size / 2, buffer_size / 2)

        # Coefficient values and equations taken from Wikipedia
        # https://en.wikipedia.org/wiki/Window_function
        match window_type:
            case 'rectangle':
                return np.ones(len(window))
            case 'hann':
                # a0, a1 = .5 for Hann
                return (0.5 *
                        (1 - np.cos(2 * constants.PI * window / buffer_size)))

            case 'hamming':
                # a0 = 25/46 ~ .54 for Hamming -- need elongated form
                factor = 25 / 46
                return (factor - (1 - factor) *
                        np.cos(2 * constants.PI * window / buffer_size))

            case 'blackman':
                # Exact Blackman
                a0 = 7938 / 18608
                a1 = 9240 / 18608
                a2 = 1430 / 18608
                return (a0 -
                        a1 * np.cos(2 * constants.PI * window / buffer_size) +
                        a2 * np.cos(4 * constants.PI * window / buffer_size))

            case 'blackman-harris':
                a0 = 0.35875
                a1 = 0.48829
                a2 = 0.14128
                a3 = 0.01168
                return (a0 -
                        a1 * np.cos(2 * constants.PI * window / buffer_size) +
                        a2 * np.cos(4 * constants.PI * window / buffer_size) -
                        a3 * np.cos(6 * constants.PI * window / buffer_size))


class Person:
    """Class for person with audio files
    
    Ground truth csv not necessary to instantiate person (i.e. for unittests)
    Will be needed for using a Person in a test
    """
    name: str
    initials: str
    gender: str
    ground_truth: Optional[pd.DataFrame]
    audio_dict: Dict[str, RawAudio]
    gci_dict: Optional[Dict[str, np.ndarray]]

    def __init__(self,
                 name: str,
                 initials: str,
                 csv_loc: Optional[str],
                 audio_loc: str,
                 gci_loc: Optional[str],
                 gender: str = None):
        self.name = name
        self.initials = initials
        self.ground_truth, self.gender = self.get_cleaned_ground_truth(csv_loc)
        self.audio_dict = self.get_raw_audio_dict(audio_loc)
        self.gci_dict = self.get_gci_data(gci_loc)
        self.gender = gender if gender is not None else self.gender

    def get_raw_audio_dict(self, audio_loc):
        audio_dict = {}
        for file in os.listdir(audio_loc):
            file = os.path.join(audio_loc, file)
            # Force native sample rate for storage
            audio, original_sr = librosa.load(file, sr=None)
            # TODO -- Check resample doesn't do any weird processing
            audio = librosa.resample(audio,
                                     orig_sr=original_sr,
                                     target_sr=constants.SAMPLE_RATE)
            vowel = file[-6:-4]
            audio_dict[vowel] = RawAudio(audio, original_sr,
                                         constants.SAMPLE_RATE)
        return audio_dict
    
    def get_cleaned_ground_truth(self, csv_loc):
        if csv_loc is None:
            return None
        ground_truth = pd.read_csv(csv_loc)
        gender = ground_truth[[self.initials]].iloc[0].item()
        ground_truth = ground_truth[['Orthography', 'F1', 'F2', 'F3']].set_index('Orthography')
        return ground_truth, gender

    def set_gender_from_df(self):
        gender_row = self.ground_truth.iloc[0]
        initials = self.initials
        return gender_row[initials]
    
    def get_gci_data(self, gci_loc):
        gci_dict = {}
        for file in os.listdir(gci_loc):
            if file[-3:] != 'csv':
                continue
            file = os.path.join(gci_loc, file)
            gci_data = pd.read_csv(file).map(lambda x: x[:-4])
            gci_data = gci_data.rename(columns={'times labels': 'GCI_times'}).to_numpy(dtype=float).T[0]
            vowel = file[-10:-8]
            gci_dict[vowel] = gci_data
        return gci_dict


class LPC:
    """Short Time LPC Calculation given Audio Buffers
    
    Generates lpc coefficients for all frames and stores in 2-d array
    """
    audio_buffers: np.ndarray
    coeffs: np.ndarray
    filter_order: int = 40
    sample_rate: int = 44100

    def __init__(
        self,
        audio_buffers: np.ndarray,
        filter_order: Optional[int] = None,
        sample_rate: Optional[int] = None,
    ):
        self.audio_buffers = audio_buffers
        self.filter_order = filter_order if filter_order is not None else self.filter_order
        self.sample_rate = sample_rate if sample_rate is not None else self.sample_rate
        self.coeffs = self.get_st_lpc(self.audio_buffers, self.filter_order)

    def hann_window(self, buffer_size):
        win = []
        W = 0.5
        for i in range(buffer_size):
            win.append(W * (1 - np.cos(2 * constants.PI * i / buffer_size)))
        return np.array(win)

    def meanv(self, audio_buffer, buffer_size):
        sum = 0
        for i in range(buffer_size):
            sum += audio_buffer[i]
        return sum / (buffer_size * 1)

    def vsadd(self, audio_buffer, mean, buffer_size):
        mean = -1 * mean
        output = []
        for i in range(buffer_size):
            output.append(audio_buffer[i] + mean)
        return np.array(output)

    def vmul(self, buffer_a, buffer_b, buffer_size):
        result = []
        for i in range(buffer_size):
            result.append(buffer_a[i] * buffer_b[i])

        return np.array(result)

    def high_pass_filter(self, audio_buffer, buffer_size):
        result = []
        magic_factor = 0.97
        delsmp = 0
        for i in range(buffer_size):
            result.append(audio_buffer[i] - magic_factor * delsmp)
            delsmp = audio_buffer[i]
        return np.array(result)

    def autocorr_simple(self, order, data):
        result = []
        for i in range(order + 1):
            sum = 0
            for j in range(len(data) - i):
                sum += data[j] * data[j + i]
            result.append(sum)
        return np.array(result)

    def levinsonDurbin(self, autocorr, order):
        alpha = [autocorr[1] / autocorr[0]]

        for i in range(1, order):
            beta = alpha[::-1]
            r = autocorr[1:i + 1]
            p = r[::-1]

            k = (autocorr[i + 1] - np.dot(p, alpha)) / (autocorr[0] -
                                                        np.dot(r, alpha))
            epsilon = [-1 * k * x for x in beta]

            alpha.append(0)
            epsilon.append(k)

            alpha = [a + e for a, e in zip(alpha, epsilon)]

        return alpha

    def lpc_from_data(self, audio_buffer, filter_order):
        corr = self.autocorr_simple(filter_order, audio_buffer)
        coeffs = self.levinsonDurbin(corr, filter_order)

        return np.array(coeffs)

    def compute_lpc(self, audio_buffer, filter_order):
        lpc_coeffs = []
        for _ in range(filter_order):
            lpc_coeffs.append(0)

        buffer_size = len(audio_buffer)
        win = self.hann_window(buffer_size)

        mean = self.meanv(audio_buffer, buffer_size)
        # Negating mean takes place within function
        audio_buffer = self.vsadd(audio_buffer, mean, buffer_size)

        #audio_buffer = self.vmul(audio_buffer, win, buffer_size)

        coeffs = self.lpc_from_data(audio_buffer, filter_order)

        return coeffs

    def get_st_lpc(self, audio_buffers, filter_order):
        '''
        Short-time lpc
        '''
        st_lpc = np.zeros((np.shape(audio_buffers)[0], filter_order))
        for idx, buffer in enumerate(audio_buffers):
            st_lpc[idx, :] = self.compute_lpc(buffer, filter_order)

        return st_lpc
    
class WLP(LPC):
    qcp_buffers: np.ndarray
    
    def __init__(
        self,
        audio_buffers: np.ndarray,
        qcp_buffers: np.ndarray,
        filter_order: Optional[int] = 40,
        sample_rate: Optional[int] = 44100,
    ):
        super().__init__(audio_buffers, filter_order, sample_rate)
        self.qcp_windows = qcp_buffers
        self.coeffs = self.get_st_wlp(audio_buffers, qcp_buffers, filter_order)
    
    def weighted_autocorr_simple(self, order, data, weights):
        result = []
        for i in range(order + 1):
            sum = 0
            for j in range(len(data) - i):
                sum += weights[j] * data[j] * data[j + i]
            result.append(sum)
        return np.array(result)
    
    def weighted_autocorr_forward_backward(self, order, data, weights):
        result = []
        n = len(data)
        for lag in range(order + 1):
            sum = 0
            # Backward component: x(t) * x(t - lag)
            for t in range(lag, n):
                sum += weights[t] * data[t] * data[t - lag]
            # Forward component: x(t) * x(t + lag)
            for t in range(n - lag):
                sum += weights[t] * data[t] * data[t + lag]
            result.append(sum)
        return np.array(result)

    def build_autocorrelation_matrix(self, autocorrelation_values):
        n = len(autocorrelation_values)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                lag = abs(i - j)
                matrix[i, j] = autocorrelation_values[lag]
        return matrix

    def compute_wlp(self, audio_buffer, weights, filter_order):
        """
        Compute Weighted LPC coefficients for a single audio buffer.
        """
        buffer_size = len(audio_buffer)
        win = self.hann_window(buffer_size)
        
        # Preprocess audio buffer (high-pass filtering, windowing, etc.)
        mean = self.meanv(audio_buffer, buffer_size)
        audio_buffer = self.vsadd(audio_buffer, mean, buffer_size)
        audio_buffer = self.vmul(audio_buffer, win, buffer_size)

        # Compute weighted autocorrelation
        weighted_corr = self.weighted_autocorr_forward_backward(filter_order, audio_buffer, weights)
        
        # Solve the weighted Yule-Walker equations (not using Levinson-Durbin)
        r = weighted_corr[1:filter_order + 1]
        # Construct the Toeplitz autocorrelation matrix R
        R = self.build_autocorrelation_matrix(weighted_corr[:-1])
        R += np.eye(len(R)) * 1e-6
        coeffs = np.linalg.solve(R, r)  # Solve linear system
        return coeffs

    def get_st_wlp(self, audio_buffers, qcp_windows, filter_order):
        """
        Compute short-time Weighted LPC coefficients for all frames.
        """
        st_wlp = np.zeros((audio_buffers.shape[0], filter_order))
        for idx, (buffer, weights) in enumerate(zip(audio_buffers, qcp_windows)):
            st_wlp[idx, :] = self.compute_wlp(buffer, weights, filter_order)
        return st_wlp

class FormantCandidate:
    frequency: float
    bandwidth: float

    def __init__(self, frequency: float, bandwidth: float):
        self.frequency = frequency
        self.bandwidth = bandwidth


class CoeffWeightParams:
    method: str
    alpha: float
    beta: float
    
    def __init__(self, method: str, alpha: float, beta: float):
        self.method = method
        self.alpha = alpha
        self.beta = beta


class FormantMethod:
    method_type: str
    coeffs: np.ndarray
    num_formants: int
    coeff_weight_params: Optional[CoeffWeightParams]
    formant_values: Optional[Dict[str, np.ndarray]]
    formant_candidates: Optional[Dict[str, np.ndarray]]

    def __init__(self,
                 method_type: str,
                 coeffs: np.ndarray,
                 num_formants: int,
                 coeff_weight_params: Optional[CoeffWeightParams] = None,
                 formant_values: Optional[Dict[str, np.ndarray]] = None,
                 formant_candidates: Optional[Dict[str, np.ndarray]] = None):

        assert method_type.lower().split(' ')[0] in constants.FORMANT_TYPES

        self.method_type = method_type
        self.coeffs = coeffs
        self.num_formants = num_formants
        self.coeff_weights = coeff_weight_params
        self.formant_values = formant_values
        self.formant_candidates = formant_candidates


class Peak:
    index: int
    frequency: int
    amplitude: float

    def __init__(self, index: int, frequency: int, amplitude: float):
        self.index = index
        self.frequency = frequency
        self.amplitude = amplitude


class MagsAndPeaks(FormantMethod):
    """Peak Picker to get Formants
    
    Note (8/23/24) -- formant_values are set as avg across all frames
    Likely will update to percentiles or another metric, but this reflects
    current use in staRt

    Inherits from generic FormantMethod
    Initialized with inherited attributes:
    
        1. method_type (str): type of formant calculation (see constants.py)
        2. num_formants (int): max number of formants to compute
        3. coeffs (np.ndarray): lpc coefficients for all frames of audio
    """
    resolution: int = 512
    max_frequency: int = 4096
    sample_rate: int = 44100
    min_peak_energy: float = .315
    theta: np.ndarray
    magnitudes: np.ndarray
    peaks: List[List[Peak]]
    formants_by_frame: np.ndarray
    formant_candidates_by_frame: np.ndarray
    gaussian_formants_by_frame: np.ndarray
    gaussian_formants: np.ndarray

    def __init__(self,
                 coeffs: Optional[np.ndarray] = None,
                 num_formants: Optional[int] = None,
                 resolution: Optional[int] = None,
                 max_frequency: Optional[int] = None,
                 sample_rate: Optional[int] = None,
                 minPeakEnergy: Optional[float] = None):
        self.method_type = 'MagsAndPeaks'
        self.coeffs = coeffs
        self.num_formants = num_formants

        # Force test_variables to be filled immediately or overridden later
        self.resolution = resolution if resolution is not None else self.resolution
        self.max_frequency = max_frequency if max_frequency is not None else self.max_frequency
        self.sample_rate = sample_rate if sample_rate is not None else self.sample_rate
        self.min_peak_energy = minPeakEnergy if minPeakEnergy is not None else self.min_peak_energy

        # If not cold init, fill parameters here
        if self.coeffs is not None and self.num_formants is not None:
            self.fill_params_from_cold_init(self.coeffs, self.num_formants)

    def fill_params_from_cold_init(self, coeffs: np.ndarray,
                                   num_formants: int):
        if self.coeffs is None and self.num_formants is None:
            self.coeffs = coeffs
            self.num_formants = num_formants
        self.theta = self.get_theta()
        self.magnitudes = self.get_frequencies_from_lpc()
        self.peaks = self.new_get_peaks()
        self.formants_by_frame = self.get_formants_from_peaks()
        self.formant_values = np.mean(self.formants_by_frame, axis=0)
        self.formant_candidates = None
        self.gaussian_formants_by_frame = self.gaussian_window_derivative_formants()
        self.gaussian_formants = np.mean(self.gaussian_formants_by_frame, axis=0)
        #CHANGE THIS
        self.formant_candidates = self.gaussian_formants
        
    def get_theta(self):
        # Get equally spaced normalized frequencies
        inc = (self.max_frequency /
               self.resolution) * 2 * constants.PI / self.sample_rate
        theta = np.zeros(self.resolution)
        for i in range(self.resolution):
            theta[i] = inc * i

        return theta

    def get_frequencies_from_lpc(self):
        coeffs = self.coeffs
        resolution = self.resolution

        full_mags = np.zeros((len(coeffs), len(self.theta)))
        for idx, lpc_coeffs in enumerate(coeffs):
            # I'm not convinced this is functioning properly
            # We multiply theta by the order of each coeff
            # but we should be raising the entirety of z to -(j+1)
            # Will leave for now but TODO -- check this function
            complex_components = np.zeros(resolution, dtype=np.complex_)
            mags = np.zeros(resolution)
            for i in range(len(self.theta)):
                temp = complex(0, 0)
                for j in range(len(lpc_coeffs)):
                    z = complex(np.cos(self.theta[i] * (j + 1)),
                                np.sin(self.theta[i] * (j + 1)))
                    # does this need to be z^-i?
                    az = complex(lpc_coeffs[j], 0) * z
                    temp += az

                denom = complex(1, 0) - temp
                complex_components[i] = complex(1, 0) / denom

            for i in range(len(mags)):
                mags[i] = 1 * np.log10(abs(complex_components[i]))

            full_mags[idx, :] = mags

        return full_mags

    def bigger_than_my_neighbors(mags, myLoc, numNeighbors):
        result = True
        for i in range(myLoc - numNeighbors, myLoc + numNeighbors):
            result = result and (mags[myLoc] > mags[i])
        return result

    def get_peaks(self, mags):
        how_local = np.floor(len(mags) / 10)
        slope = 0
        i = 0
        peaks = []
        while (slope == 0) and (i < len(mags)):
            if (mags[i] < mags[i + 1]):
                slope = 1
                break
            elif (mags[i] > mags[i + 1]):
                slope = -1
            i += 1

        for i in range(len(mags)):
            pass
        ## This function looks really weird... gonna rewrite it

    def new_get_peaks(self):
        # maximum five formants
        full_peaks = []
        for mags in self.magnitudes:
            slopes = np.diff(mags)
            peaks = []
            curr_zero_string = []
            curr_zero = False
            for i in range(len(slopes) - 1):
                curr = slopes[i]
                next_slope = slopes[i + 1]
                if curr == 0:
                    curr_zero_string.append(i)
                    curr_zero = True
                if next_slope < 0:
                    # Mag value is offset by one from slopes
                    if mags[i + 1] > self.min_peak_energy:
                        if curr_zero:
                            # Takes the middle value if a bunch of zeros are in a row
                            peaks.append(
                                curr_zero_string(
                                    np.ceil(len(curr_zero_string) / 2) - 1))
                            curr_zero_string = []
                            curr_zero = False
                        # Otherwise check if change of slope & add to peaks
                        elif curr > 0:
                            factor = self.sample_rate / (2 * constants.PI)
                            peaks.append(
                                Peak(i, self.theta[i] * factor, mags[i]))
            full_peaks.append(peaks)

        return full_peaks

    def get_formants_from_peaks(self):
        # Right now, formants are the peaks with highest amplitude
        full_formants = np.ndarray((len(self.peaks), self.num_formants))
        for idx, peak_list in enumerate(self.peaks):
            # Sort by amplitude in descending order
            sort_by_amplitude = sorted(peak_list,
                                       key=lambda x: x.amplitude,
                                       reverse=True)
            # Take highest peaks by amplitude
            unsorted_formants = sort_by_amplitude[:self.num_formants]
            # Resort by frequency in ascending order
            sorted_formants = sorted(unsorted_formants,
                                     key=lambda x: x.frequency)

            full_formants[idx, :len(sorted_formants)] = [
                peak.frequency for peak in sorted_formants
            ]

        return full_formants
    
    def gaussian_window_derivative_formants(self):
        resolution = self.resolution
        magnitudes = self.magnitudes
        theta = self.theta
        width_hz = 100
        
        freq_resolution = self.sample_rate / (2 * resolution)
        std_samples = width_hz /(2 * np.sqrt(2 * np.log(2))) / freq_resolution
        
        gaussian_window = gaussian(resolution, std_samples)
        gaussian_derivative = np.gradient(gaussian_window)
        
        master_formant_frequencies = np.ndarray((len(self.peaks), self.num_formants))
        for idx, frame in enumerate(magnitudes):
        
            convolved = convolve1d(frame, gaussian_derivative, mode='reflect')
            
            zero_crossings = np.where((convolved[:-1] > 0) & (convolved[1:] <= 0))[0]
            
            formant_frequencies = theta[zero_crossings] * self.sample_rate / (2 * constants.PI)
            
            if len(formant_frequencies) != self.num_formants:
                formant_frequencies = np.concatenate((formant_frequencies, np.zeros(self.num_formants-len(formant_frequencies))))
            
            master_formant_frequencies[idx, :] = formant_frequencies
        
        return master_formant_frequencies
        

class FormantParams:
    bandwidth_range: Tuple[int, int] = constants.FORMANT_BANDWIDTH_RANGE
    frequency_range: Tuple[int, int] = constants.FORMANT_FREQUENCY_RANGE

    def __init__(
        self,
        bandwidth_range: Optional[Tuple[int, int]] = None,
        frequency_range: Optional[Tuple[int, int]] = None,
    ):
        self.bandwidth_range = bandwidth_range if bandwidth_range is not None else self.bandwidth_range
        self.frequency_range = frequency_range if frequency_range is not None else self.frequency_range


class DurandKerner(FormantMethod):
    """Durand Kerner implementation & calculations
    
    Note (8/23/24) -- current version reflects implementation in staRt
    i.e. formant_values are calculated from the average filter across 
    all frames of audio

    Inherits from generic FormantMethod
    Initialized with inherited attributes:
    
        1. method_type (str): type of formant calculation (see constants.py)
        2. num_formants (int): max number of formants to compute
        3. coeffs (np.ndarray): lpc coefficients for all frames of audio
    """
    tolerance: float = 1e-6
    max_iterations: int = 100
    sample_rate: int = 44100
    formant_params: FormantParams
    order: int
    initial_guess: np.ndarray
    roots_by_frame: np.ndarray
    dk_iters_by_frame: int
    formants_by_frame: np.ndarray
    averaging_weights: np.ndarray
    average_coeffs: np.ndarray
    roots_from_avg_coeffs: np.ndarray
    dk_iters_avg: int
    formants_avg_from_frames: np.ndarray

    def __init__(
        self,
        coeff_weight_params: CoeffWeightParams,
        coeffs: Optional[np.ndarray] = None,
        num_formants: Optional[int] = None,
        tolerance: Optional[float] = None,
        max_iterations: Optional[int] = None,
        sample_rate: Optional[int] = None,
        formant_params: Optional[FormantParams] = None,
    ):
        self.coeff_weight_params = coeff_weight_params
        self.coeffs = coeffs
        self.num_formants = num_formants

        # Force test_variables to be set here or overridden later
        self.tolerance = tolerance if tolerance is not None else self.tolerance
        self.max_iterations = max_iterations if max_iterations is not None else self.max_iterations
        self.sample_rate = sample_rate if sample_rate is not None else self.sample_rate
        self.formant_params = formant_params if formant_params is not None else FormantParams(
        )

        # If not cold init, initialize other params here
        if self.coeffs is not None and self.num_formants is not None:
            self.fill_params_from_cold_init(self.coeffs, self.num_formants)

    def fill_params_from_cold_init(self, coeffs: np.ndarray,
                                   num_formants: int):
        if self.coeffs is None and self.num_formants is None:
            self.coeffs = coeffs
            self.num_formants = num_formants
        self.order = np.shape(coeffs)[1]
        self.initial_guess = constants.DK_INITIAL_Z**np.arange(
            self.order, dtype=np.complex_)
        self.roots_by_frame, self.dk_iters_by_frame = self.get_roots()
        self.formants_by_frame = self.get_formants()
        print(self.formants_by_frame)
        self.averaging_weights = self.get_average_weights(self.formants_by_frame)
        self.average_coeffs = np.average(self.coeffs, axis=0, weights=self.averaging_weights)
        self.roots_from_avg_coeffs, self.dk_iters_avg = self.get_single_frame_roots(
            self.average_coeffs)
        self.formants_avg_from_frames = np.mean(self.formants_by_frame, axis=0)
        self.formant_values, self.formant_candidates = self.get_single_frame_formants(
            self.roots_from_avg_coeffs)

    def get_roots(self):
        """Get roots for all frames of coefficients

        Returns:
            master_roots: Roots for each frame of lpc coefficients
            master_iters: Number of iterations to converge for each frame
        """
        master_roots = np.zeros(np.shape(self.coeffs), dtype=np.complex_)
        master_iters = np.zeros(np.shape(self.coeffs)[0])
        for idx, frame in enumerate(self.coeffs):
            roots, iter = self.get_single_frame_roots(frame)
            master_roots[idx, :] = roots
            master_iters[idx] = iter

        return master_roots, master_iters

    def get_formants(self):
        """Gets formants for all frames of coefficients
        
        Returns:
            Formants for each frame of lpc coefficients
        """
        master_formants = np.zeros(
            (np.shape(self.coeffs)[0], self.num_formants))
        for idx, frame in enumerate(self.roots_by_frame):
            roots, _ = self.get_single_frame_formants(frame)
            master_formants[idx, :] = roots

        return master_formants

    def get_single_frame_roots(self, coeffs: np.ndarray):
        """Calculate roots of LPC polynomial for a single frame
        
        Will be used across all frames in separate function
        """
        # We need to negate the coefficients to get it in proper form
        coeffs = -1 * coeffs
        roots = self.initial_guess

        for iter in range(self.max_iterations):
            is_converged = True

            for i in range(self.order):
                # Get denominator for root i
                denom = complex(1, 0)
                for j in range(self.order):
                    if i != j:
                        denom = denom * (roots[i] - roots[j])

                num = self.evaluate_polynomial(coeffs, roots[i])
                new_root = roots[i] - (num / denom)

                # Set is_converged to false if new_root doesn't meet tolerance
                # If all roots meet tolerance, is_converged will remain true
                if abs((new_root - roots[i])) > self.tolerance:
                    is_converged = False

                roots[i] = new_root

            if is_converged:
                break

        return roots, iter

    def evaluate_polynomial(self, coeffs: np.ndarray, z: complex) -> complex:
        """Evaluate polynomial with coefficients given guess z
        
        Polynomial in the form below
        z^k + a1*z^(k-1) + ... + ak = 0
        
        Negation of coefficients in get_single_frame_roots allows addition

        Args:
            coeffs: Coefficients of polynominal
            z: Guess at pole

        Returns:
            Complex output of polynomial evaluated at z
        """
        assert len(coeffs) == self.order

        result = z**len(coeffs)
        for idx, coeff in enumerate(coeffs):
            result += complex(coeff, 0) * (z**(self.order - idx - 1))

        return result

    def get_single_frame_formants(self, roots: np.ndarray):
        sr = self.sample_rate
        freq_range = self.formant_params.frequency_range
        band_range = self.formant_params.bandwidth_range
        candidates = []
        for root in roots:
            radius = abs(root)
            angle = np.arctan2(root.imag, root.real)

            freq = (angle / (2 * constants.PI)) * sr
            band = (-np.log(radius) / (2 * constants.PI)) * sr

            if ((freq_range[0] <= freq <= freq_range[1]) &
                (band_range[0] <= band <= band_range[1])):
                candidates.append(FormantCandidate(freq, band))

        formant_candidates = np.array([candidate.frequency for candidate in candidates])

        formants = np.zeros(self.num_formants)

        ## TODO -- move similar code to parent class
        # Sort by amplitude in descending order
        sort_by_bandwidth = sorted(candidates, key=lambda x: x.bandwidth)
        # Take highest five peaks by amplitude
        unsorted_formants = sort_by_bandwidth[:self.num_formants]
        # Resort by frequency in ascending order
        sorted_formants = sorted(unsorted_formants, key=lambda x: x.frequency)

        formants[:len(sorted_formants)] = [
            formant.frequency for formant in sorted_formants
        ]

        return formants, formant_candidates
    
    def get_average_weights(self, coeffs):
        norms = np.linalg.norm(coeffs, axis=1)  # Compute vector norms
        norm_min, norm_max = np.min(norms), np.max(norms)
        
        method = self.coeff_weight_params.method
        alpha = self.coeff_weight_params.alpha
        beta = self.coeff_weight_params.beta

        if method == "linear":
            weights = (norms - norm_min) / (norm_max - norm_min)
        elif method == "sigmoid":
            c = (norm_max + norm_min) / 2
            weights = 1 / (1 + np.exp(-alpha * (norms - c)))
        elif method == "exponential":
            weights = np.exp(beta * (norms - norm_min) / (norm_max - norm_min))
        elif method == "percentile":
            ranks = np.argsort(np.argsort(norms))  # Rank norms
            weights = ranks / len(norms)
        elif method == "no_weights":
            weights = np.ones(np.shape(norms))
        else:
            raise ValueError("Invalid weighting method.")
        
        if method != 'no_weights':
            normalized_weights = weights / np.sum(weights)
        else:
            normalized_weights = weights

        return normalized_weights


class TestVariables:
    buffer_params: BufferParams
    qcp_params: QCPParams
    coeff_weight_params: CoeffWeightParams
    filter_order: Optional[int]
    formant_method: FormantMethod
    num_formants: int

    def __init__(self, buffer_params: BufferParams,
                 qcp_params: QCPParams,
                 coeff_weight_params: CoeffWeightParams,
                 filter_order: Optional[int], 
                 formant_method: FormantMethod,
                 num_formants: int):
        self.buffer_params = buffer_params
        self.qcp_params = qcp_params
        self.coeff_weight_params = coeff_weight_params
        self.filter_order = filter_order
        self.formant_method = formant_method
        self.num_formants = num_formants


class Test:
    person: Person
    ground_truth: pd.DataFrame
    buffered_audio: Dict[str, BufferedAudio]
    lpc: Dict[str, LPC]
    wlp: Dict[str, WLP]
    test_variables: TestVariables
    formant_values: Dict[str, pd.DataFrame]

    def __init__(
        self,
        person: Person,
        test_variables: TestVariables,
    ):
        self.person = person
        self.ground_truth = person.ground_truth
        self.test_variables = test_variables
        self.buffered_audio = self.get_buffered_audio(person, test_variables)
        self.lpc = self.get_lpc_frames(self.buffered_audio, test_variables)
        self.wlp = self.get_wlp_frames(self.buffered_audio, test_variables)
        self.formant_values = {
            'lpc': self.get_formant_values(self.lpc, test_variables),
            'wlp': self.get_formant_values(self.wlp, test_variables)
        }

    def get_buffered_audio(
        self,
        person: Person,
        test_variables: TestVariables,
    ) -> Dict[str, BufferedAudio]:
        """Given a person and test_variables, produce buffered_audio for each vowel

        Args:
            person: Person object with raw_audio
            test_variables: Test_Variables object with buffer parameters

        Returns:
            Dict[str, BufferedAudio]: Mapping vowel to BufferedAudio objects
        """

        buffer_params = test_variables.buffer_params
        qcp_params = test_variables.qcp_params
        audio_dict = person.audio_dict
        gci_dict = person.gci_dict

        buffered_audio = {}
        for vowel in audio_dict:
            gci_times = gci_dict[vowel]
            buffered_audio[vowel] = BufferedAudio(audio_dict[vowel],
                                                  buffer_params,
                                                  gci_times,
                                                  qcp_params)

        return buffered_audio

    def get_lpc_frames(
        self,
        buffered_audio: Dict[str, BufferedAudio],
        test_variables: TestVariables,
    ) -> Dict[str, LPC]:
        """Given dictionary of buffered audio and test_variables
        generate LPC frames for each audio buffer

        Args:
            buffered_audio: Mapping of vowels to Buffered_Audio objects
            test_variables: Test_Variables object with filter_order param

        Returns:
            Dict[str, Dict[int, LPC]]: Mapping of vowel to ordered LPC frames
        """
        filter_order = test_variables.filter_order
        vowel_to_lpc = {}
        # For each vowel, get the buffers
        for vowel in buffered_audio:
            audio_buffers = buffered_audio[vowel].audio_buffers

            vowel_to_lpc[vowel] = LPC(audio_buffers, filter_order)

        return vowel_to_lpc
    
    def get_wlp_frames(
        self,
        buffered_audio: Dict[str, BufferedAudio],
        test_variables: TestVariables,
    ) -> Dict[str, LPC]:
        """Given dictionary of buffered audio and test_variables
        generate LPC frames for each audio buffer

        Args:
            buffered_audio: Mapping of vowels to Buffered_Audio objects
            test_variables: Test_Variables object with filter_order param

        Returns:
            Dict[str, Dict[int, LPC]]: Mapping of vowel to ordered LPC frames
        """
        filter_order = test_variables.filter_order
        vowel_to_lpc = {}
        # For each vowel, get the buffers
        for vowel in buffered_audio:
            audio_buffers = buffered_audio[vowel].audio_buffers
            qcp_buffers = buffered_audio[vowel].qcp_buffers

            vowel_to_lpc[vowel] = WLP(audio_buffers, qcp_buffers, filter_order)

        return vowel_to_lpc

    def get_formant_values(
            self, lp_class: Dict[str, LPC] | Dict[str, WLP],
            test_variables: TestVariables) -> Dict[str, np.ndarray]:
        """Given mapping of vowel to ordered lpc frames, produce
        mapping of vowel to average formant across audio file

        Args:
            lpc: LPC frames for each vowel

        Returns:
            Dict[str, np.ndarray]: Mapping of vowel to average formants (F1, F2, F3)
        """
        weight_params = test_variables.coeff_weight_params
        
        match test_variables.formant_method:
            case 'magsandpeaks':
                formant_method = MagsAndPeaks()
            case 'durandkerner':
                formant_method = DurandKerner(weight_params)

        num_formants = test_variables.num_formants

        formants_from_test = {}
        formant_candidates_from_test = {}

        for vowel in lp_class:
            coeffs = lp_class[vowel].coeffs
            formant_method.fill_params_from_cold_init(
                coeffs=coeffs,
                num_formants=num_formants,
            )
            formants_from_test[vowel] = formant_method.formant_values
            formant_candidates_from_test[vowel] = formant_method.formant_candidates

        print(formants_from_test[vowel])
        print(formant_candidates_from_test[vowel])
        columns = []
        for i in range(test_variables.num_formants):
            columns.append(f'F{i+1}')
        return pd.DataFrame.from_dict(formants_from_test,
                                      orient='index',
                                      columns=columns)


class OptimParams:
    params: List[str] = constants.TOTAL_PARAMS
    values_or_bounds: List[Tuple[float, float] | float | str]
    dictionary: List[Tuple[str, Tuple[float, float] | float | str]]

    def __init__(
        self,
        values_or_bounds: List[Tuple[float, float] | float | str],
    ):
        assert len(values_or_bounds) == len(self.params)
        self.values_or_bounds = values_or_bounds
        self.dictionary = zip(self.params, values_or_bounds)


class FormantComparison:
    loss_func: str
    optim_func: str
    people: List[Person]
    optim_params: OptimParams
    optimized_params: Dict[str, float]

    def __init__(
        self,
        loss_func: str,
        optim_func: str,
        people: List[Person],
        optim_params: OptimParams,
    ):
        assert loss_func.lower().split(' ')[0] in constants.LOSS_FUNCS
        self.loss_func = loss_func
        assert optim_func.lower().split(' ')[0] in constants.OPTIM_FUNCS
        self.optim_func = optim_func

        self.people = people
        self.optim_params = optim_params
        self.optimized_params = self.run_optimization()

    def get_args(self):
        test_params = []
        bounds = []
        args = []
        arg_values = []
        loop_params = []

        for param, value in self.optim_params.dictionary:
            if type(value) is float or type(value) is int:
                args.append(param)
                arg_values.append(value)
            elif type(value) is str:
                if value == 'SEARCH':
                    loop_params.append(param)
                else:
                    args.append(param)
                    arg_values.append(value)
            elif type(value) is tuple:
                test_params.append(param)
                bounds.append(value)

        return test_params, bounds, args, arg_values, loop_params

    def loss_only_order(
        self,
        x,
        buffer_size,
        hop_size,
        formant_method,
        num_formants,
        window_type,
        n_ramp,
        duration_quotient,
        position_quotient,
        d,
        method,
        alpha,
        beta,
        lp_type
    ):
        """Computer formants using shgo variables to optimize

        Args:
            x: shgo optimization variables (in this case, just filter order)
            args: other necessary parameters to calculate formants
        """
        # Args sorted alphabetically
        buffer_params = BufferParams(buffer_size=buffer_size,
                                     hop_size=hop_size,
                                     window_type=window_type)
        qcp_params = QCPParams(n_ramp=n_ramp,
                               duration_quotient=duration_quotient,
                               position_quotient=position_quotient,
                               d=d)
        coeff_weight_params = CoeffWeightParams(method=method,
                                                alpha=alpha,
                                                beta=beta)
        test_variables = TestVariables(buffer_params=buffer_params,
                                       qcp_params=qcp_params,
                                       coeff_weight_params=coeff_weight_params,
                                       filter_order=int(x[0]),
                                       formant_method=formant_method,
                                       num_formants=num_formants)

        differences = pd.DataFrame(columns=['F1', 'F2', 'F3'])
        
        for person in self.people:
            test = Test(person=person, test_variables=test_variables)
            formants = test.formant_values[lp_type]
            ground_truth = person.ground_truth
            diff = formants - ground_truth
            differences = pd.concat([differences, diff])

        return norm(differences)

    def loss_only_buffering(self, x, args):
        """Computer formants using shgo variables to optimize

        Args:
            x: shgo optimization variables (in this case, just filter order)
            args: other necessary parameters to calculate formants
        """
        buffer_params = BufferParams(buffer_size=x[0],
                                     hop_size=x[1],
                                     window_type=args[3])
        test_variables = TestVariables(buffer_params=buffer_params,
                                       filter_order=args[0],
                                       formant_method=args[1],
                                       num_formants=args[2])

        differences = pd.DataFrame(columns=['F1', 'F2', 'F3'])
        for person in self.people:
            test = Test(person=person, test_variables=test_variables)
            formants = test.formant_values
            ground_truth = person.ground_truth
            diff = formants - ground_truth
            differences = pd.concat([differences, diff])

        return norm(differences)

    def loss_order_and_buffering(
        self,
        x,
        formant_method,
        num_formants,
        window_type,
    ):
        buffer_params = BufferParams(buffer_size=int(x[0]),
                                     hop_size=int(x[1]),
                                     window_type=window_type)
        test_variables = TestVariables(buffer_params=buffer_params,
                                       filter_order=int(x[2]),
                                       formant_method=formant_method,
                                       num_formants=num_formants)

        differences = pd.DataFrame(columns=['F1', 'F2', 'F3'])
        for person in self.people:
            test = Test(person=person, test_variables=test_variables)
            formants = test.formant_values
            ground_truth = person.ground_truth
            diff = formants - ground_truth
            differences = pd.concat([differences, diff])

        return norm(differences)

    def sort_args(self, args, arg_values):
        sorted_args = sorted(zip(args, arg_values))
        final_args = [x[0] for x in sorted_args]
        final_values = tuple([x[1] for x in sorted_args])
        return final_args, final_values

    def run_optimization(self):
        test_params, bounds, args, arg_values, loop_params = self.get_args()
        print(args)
        print(arg_values)
        
        if 'filterorder' in test_params and 'buffersize' in test_params:
            func = self.loss_order_and_buffering
        elif 'filterorder' in test_params:
            func = self.loss_only_order
        elif 'buffersize' in test_params:
            func = self.loss_only_buffering
        else:
            AssertionError

        best_solution = 100000
        best_res = None
        if 'formantmethod' in loop_params:
            # Loop formant methods
            for formant_method in constants.FORMANT_TYPES:
                if 'windowtype' in loop_params:
                    # Loop window_type
                    for window_type in constants.WINDOW_TYPES:
                        args.append(['formantmethod', 'windowtype'])
                        arg_values.append([formant_method, window_type])

                        args, arg_values = self.sort_args(args, arg_values)

                        res = shgo(func, bounds=bounds, args=arg_values)
                        if res.fun < best_solution:
                            best_solution = res.fun
                            best_res = res
                # just formant method
                else:
                    args.append('formantmethod')
                    arg_values.append(formant_method)

                    args, arg_values = self.sort_args(args, arg_values)

                    res = shgo(func, bounds=bounds, args=arg_values)
                    if res.fun < best_solution:
                        best_solution = res.fun
                        best_res = res

        elif 'windowtype' in loop_params:
            # Just loop window type
            for window_type in constants.WINDOW_TYPES:
                args.append('windowtype')
                arg_values.append(window_type)

                args, arg_values = self.sort_args(args, arg_values)

                res = shgo(func, bounds=bounds, args=arg_values)
                if res.fun < best_solution:
                    best_solution = res.fun
                    best_res = res

        else:
            #args, arg_values = self.sort_args(args, arg_values)
            best_res = shgo(func, bounds=bounds, args=arg_values)

        optimized_params = {}
        for idx, key in enumerate(test_params):
            optimized_params[key] = best_res.x[idx]

        return optimized_params
