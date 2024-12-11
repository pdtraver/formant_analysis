import os
import sys
import numpy as np
import pandas as pd
import time
from typing import List

import constants
from classes import (Person, BufferParams, QCPParams, QCPWindowVector, BufferedAudio, 
                     LPC, WLP, MagsAndPeaks, DurandKerner, TestVariables, Test, CoeffWeightParams)
import warnings

warnings.filterwarnings("ignore")


def buffered_audio_test(person: Person, buffer_params: BufferParams, qcp_params: QCPParams):
    start_time = time.time()

    # Test script
    buffered_audio_dict = {}
    for vowel in person.audio_dict:
        raw_audio = person.audio_dict[vowel]
        gci_times = person.gci_dict[vowel]
        buffered_audio = BufferedAudio(raw_audio, buffer_params, gci_times, qcp_params)

        overlap = buffer_params.hop_size / buffer_params.buffer_size
        flatten_buffer_shape = np.shape(
            buffered_audio.audio_buffers.reshape(-1))[0] * overlap
        flatten_qcp_shape = np.shape(
            buffered_audio.qcp_buffers.reshape(-1))[0] * overlap

        # Add extra hop_size if overlap exists
        # (Off by one hop_size due to excluding one extra overlap region)
        if overlap < 1:
            flatten_buffer_shape += buffer_params.hop_size
        orig_signal_shape = np.shape(raw_audio.signal)[0]

        assert orig_signal_shape == flatten_buffer_shape, (
            flatten_buffer_shape, orig_signal_shape)
        assert orig_signal_shape == flatten_qcp_shape, (
            flatten_qcp_shape, orig_signal_shape)

        buffered_audio_dict[vowel] = buffered_audio

    end_time = time.time()
    execution_time = round(end_time - start_time, 5)

    print(f'~~~ Buffered Audio Tests Passed {execution_time} s ~~~')

    return buffered_audio_dict


def lpc_test(buffered_audio: BufferedAudio):
    start_time = time.time()

    # Test script
    lpc = LPC(buffered_audio.audio_buffers)

    end_time = time.time()
    execution_time = round(end_time - start_time, 5)

    assert np.shape(lpc.coeffs) == (np.shape(buffered_audio.audio_buffers)[0],
                                    lpc.filter_order)

    print(f'~~~ LPC Tests Passed in {execution_time} s ~~~')

    return lpc

def wlp_test(buffered_audio: BufferedAudio):
    start_time = time.time()

    # Test script
    wlp = WLP(buffered_audio.audio_buffers, buffered_audio.qcp_buffers)

    end_time = time.time()
    execution_time = round(end_time - start_time, 5)

    assert np.shape(wlp.coeffs) == (np.shape(buffered_audio.qcp_buffers)[0],
                                    wlp.filter_order)

    print(f'~~~ WLP Tests Passed in {execution_time} s ~~~')

    return wlp


def mags_and_peaks_test(coeffs: np.ndarray, num_formants: int):
    start_time = time.time()

    # Test script
    formant_method = MagsAndPeaks(coeffs=coeffs, num_formants=num_formants)

    end_time = time.time()
    execution_time = round(end_time - start_time, 5)

    assert np.shape(formant_method.formants_by_frame) == (np.shape(coeffs)[0],
                                                          num_formants)

    print(f'~~~ Mags and Peaks Tests Passed in {execution_time} s ~~~')

    return formant_method


def durand_kerner_test(coeff_weight_params: CoeffWeightParams, coeffs: np.ndarray, num_formants: int):
    start_time = time.time()

    # Test script
    formant_method = DurandKerner(coeff_weight_params, coeffs=coeffs, num_formants=num_formants)

    end_time = time.time()
    execution_time = round(end_time - start_time, 5)

    assert np.shape(formant_method.formants_by_frame) == (np.shape(coeffs)[0],
                                                          num_formants)

    print(f'~~~ Durand Kerner Tests Passed in {execution_time} s ~~~')

    return formant_method


def test_test_object(person: Person, test_variables: TestVariables):
    start_time = time.time()

    # Test script
    test = Test(person=person, test_variables=test_variables)

    end_time = time.time()
    execution_time = round(end_time - start_time, 5)

    assert np.shape(
        test.formant_values['lpc'].loc['ee']) == (constants.NUM_FORMANTS, )
    assert np.shape(
        test.formant_values['wlp'].loc['ee']) == (constants.NUM_FORMANTS, )

    print(
        f'~~~ Test Object Tests Passed with {test_variables.formant_method} in {execution_time} s ~~~'
    )

    return test


def main():
    dataset_dir = os.path.join(os.path.dirname(os.getcwd()), 'dataset')
    
    # Free variables (Person, BufferParams, TestVariables) for testing
    test_person = Person(
        name='test',
        initials='test',
        csv_loc=os.path.join(dataset_dir, 'test_csv', 'ee.csv'),
        audio_loc=os.path.join(dataset_dir, 'test_audio'),
        gci_loc=os.path.join(dataset_dir, 'test_GCI_estimate'), 
        gender='Cismale',
    )

    test_buffer_params = BufferParams(
        buffer_size=512,
        hop_size=512,
        window_type='hann',
    )
    
    test_qcp_params = QCPParams(
        n_ramp=7,
        duration_quotient=0.8,
        position_quotient=0.05,
        d=1e-5
    )
    
    test_coeff_weight_params = CoeffWeightParams(
        method='sigmoid',
        alpha=0.65,
        beta=1.0
    )

    test_test_variables_1 = TestVariables(
        buffer_params=test_buffer_params,
        qcp_params=test_qcp_params,
        coeff_weight_params=test_coeff_weight_params,
        filter_order=42,
        formant_method="magsandpeaks",
        num_formants=constants.NUM_FORMANTS,
    )

    test_test_variables_2 = TestVariables(
        buffer_params=test_buffer_params,
        qcp_params=test_qcp_params,
        coeff_weight_params=test_coeff_weight_params,
        filter_order=42,
        formant_method="durandkerner",
        num_formants=constants.NUM_FORMANTS,
    )

    #Test buffered_audio object
    buffered_audio_dict = buffered_audio_test(
        person=test_person, 
        buffer_params=test_test_variables_1.buffer_params,
        qcp_params=test_qcp_params)

    test_buffered_audio = buffered_audio_dict['ee']

    # Test LPC calculations
    lpc = lpc_test(test_buffered_audio)
    lpc_coeffs = lpc.coeffs
    num_formants = constants.NUM_FORMANTS

    # # Mags and Peaks testing
    # mags_and_peaks_lpc = mags_and_peaks_test(coeffs=coeffs,
    #                                      num_formants=num_formants)

    # # Durand Kerner testing
    # durand_kerner_lpc = durand_kerner_test(coeffs=coeffs,
    #                                    num_formants=num_formants)
    
    # Test WLP calculations
    wlp = wlp_test(test_buffered_audio)
    wlp_coeffs = wlp.coeffs
    num_formants = constants.NUM_FORMANTS

    # Mags and Peaks testing
    mags_and_peaks_wlp = mags_and_peaks_test(coeffs=wlp_coeffs,
                                         num_formants=num_formants)

    # Durand Kerner testing
    durand_kerner_wlp = durand_kerner_test(test_coeff_weight_params,
                                           coeffs=wlp_coeffs,
                                           num_formants=num_formants)

    # Test & TestVariables object testing
    test1 = test_test_object(person=test_person,
                             test_variables=test_test_variables_1)

    test2 = test_test_object(person=test_person,
                             test_variables=test_test_variables_2)

    print(test1.formant_values['lpc'])
    print(test1.formant_values['wlp'])
    print(test2.formant_values['lpc'])
    print(test2.formant_values['wlp'])


if __name__ == '__main__':
    main()
