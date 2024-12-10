"""Constants used for testing framework
"""

SAMPLE_RATE = 8000
PI = 3.14159265
WINDOW_TYPES = set(['rectangle', 'hann', 'hamming', 'blackman', 'blackman-harris'])
LP_TYPE = set(['lpc', 'wlp'])
FORMANT_TYPES = set(['magsandpeaks', 'durandkerner'])
LOSS_FUNCS = set(['mse'])
OPTIM_FUNCS = set(['shgo'])
GRID_PARAMS = ['buffersize', 'filterorder', 'hopsize']
STATIC_PARAMS = ['formantmethod', 'numformants', 'windowtype']
QCP_PARAMS = ['n_ramp', 'duration_quotient', 'position_quotient', 'd']
COEFF_WEIGHT_PARAMS = ['method', 'alpha', 'beta']
TOTAL_PARAMS = GRID_PARAMS + STATIC_PARAMS + QCP_PARAMS + COEFF_WEIGHT_PARAMS + ['lp_type']

# Soft Variables -- nice to have constant version but could be tested
NUM_FORMANTS = 3
DK_INITIAL_Z = complex(0.4, 0.8)
FORMANT_BANDWIDTH_RANGE = (0, 400)
FORMANT_FREQUENCY_RANGE = (20, 4096)
N_RAMP = 7
DURATION_QUOTIENT = 0.8
POSITION_QUOTIENT = 0.05
D = 1e-5