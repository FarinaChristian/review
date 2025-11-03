# --- mmWave Board settings ---
from constants.board import f0
from constants.numeric import SoL
import numpy as np

# --- Sensing configurations
Tx_antenna = 1
Rx_antenna = 1
S = 30.13e12
ADC_sample_rate = 10_000_000  # Sample per second
Sample_per_chirp = 374
Num_of_chirp_loops = 99
idle_time = 100e-6  # ti (second)
ADC_valid_start_time = 6.4e-6  # ts (second)
Tx_start_time = 0
Ramp_end_time = 132.07e-6
ADC_sampling_time = Sample_per_chirp / ADC_sample_rate  # ta (second)
# ---

# --- Information only parameters ---
Total_BW = Ramp_end_time * S
Valid_BW = ADC_sampling_time * S
Chirp_cycle_time = idle_time + Ramp_end_time
Chirp_repetition_period = Chirp_cycle_time * Tx_antenna
Active_frame_time = Num_of_chirp_loops * Chirp_repetition_period  # second
# ---

# --- Chirp configuration parameters ---
Measurement_rate = min(70, int(1 / Active_frame_time))  # (MANUAL INPUT) Measurement rate (frame per second)
Frame_periodicity = 1 / Measurement_rate
# ---

# --- Scene parameters ---
d_max = min(10, int(ADC_sample_rate * SoL / (2 * S)))  # maximum detectable range (meter)
d_min = 0.0  # (MANUAL INPUT) minimum detectable range (meter). We avoid detecting objects very close to the antenna
d_res = SoL / (2 * Valid_BW)
v_max = SoL / (f0 * Chirp_repetition_period * 4)  # meter/second
v_res = SoL / (f0 * Active_frame_time * 2)  # meter/second
# ---

# --- Processing ---
Num_range_fft_bins = int(2 ** np.floor(np.log2(Sample_per_chirp) + 1))
Num_doppler_fft_bins = int(2 ** np.floor(np.log2(Num_of_chirp_loops) + 1))
Range_inter_bin_resolution = SoL * (ADC_sample_rate / Num_range_fft_bins) / (2 * S)  # meters
Velocity_inter_bin_resolution = SoL / f0 * 1 / 2 * (
            Num_of_chirp_loops / (Num_doppler_fft_bins * Active_frame_time)
)  # meter/second

slow_sampling_rate = Num_of_chirp_loops / Frame_periodicity
fd_max = slow_sampling_rate / 2  # This is the maximum doppler frequency that can be produced
max_IF = S * 2 * d_max / SoL  # Hertz (beat frequency)
min_IF = S * 2 * d_min / SoL  # Hertz
# ---