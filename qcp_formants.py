import os
import numpy as np
import constants
from classes import (Person, BufferParams, QCPParams, QCPWindowVector, BufferedAudio, 
                     LPC, WLP, MagsAndPeaks, DurandKerner, TestVariables, Test, CoeffWeightParams)
from tqdm import tqdm

# Get updated people dictionary
data_file = os.path.join(os.getcwd(), 'people_dict.npy')
people_dict = np.load(data_file, allow_pickle=True).item()

# Set output file for data
formant_method = 'magsandpeaks'
output_file = os.path.join(os.getcwd(), f'qcp_formants_{formant_method}.npy')

# Set up calculation parameters -- universal for all people
test_buffer_params = BufferParams(
    buffer_size=1102,
    hop_size=441,
    window_type='rectangle',
)

test_qcp_params = QCPParams(
    n_ramp=7,
    duration_quotient=0.7,
    position_quotient=0.05,
    d=1e-5
)

test_coeff_weight_params = CoeffWeightParams(
    method='sigmoid',
    alpha=0.65,
    beta=1.0
)

filter_order = 42

predicted_formants = {}
for initials in tqdm(people_dict):
    person = people_dict[initials]
    test_variables = TestVariables(
        buffer_params=test_buffer_params,
        qcp_params=test_qcp_params,
        coeff_weight_params=test_coeff_weight_params,
        filter_order=filter_order,
        formant_method=formant_method,
        num_formants=6,
    )
    test = Test(person=person, test_variables=test_variables)
    predicted_formants[initials] = test.formant_values
    break
  
with open(output_file, 'wb') as f:
    np.save(f, predicted_formants)
print(predicted_formants)
    