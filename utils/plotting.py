import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from constants.numeric import SoL
from constants.settings import v_max, Num_doppler_fft_bins, Num_range_fft_bins, S, ADC_sample_rate



def plot_doppler_frame(data: np.ndarray, antenna: int, frame_index: int):
    velocity_axis = np.linspace(-v_max, v_max, Num_doppler_fft_bins)
    distance_axis = np.linspace(0, ADC_sample_rate * SoL / (2 * S), Num_range_fft_bins)

    data = np.transpose(data, (0, 2, 1, 3))
    data = data[:, ::-1, :, :]
    plt.imshow(
        np.abs(data[frame_index, :, :, antenna]),
        extent=[velocity_axis[0], velocity_axis[-1], distance_axis[0], distance_axis[-1]],
        aspect='auto',
        cmap='jet'
    )
    plt.xlabel('Velocity - meters/sec')
    plt.ylabel('Distance - meters')

    plt.title(f'Frame {frame_index}')
    plt.show()

def animated_plot(data: np.ndarray, antenna: int):
    def animate(frame_index):
        ax.clear()
        ax.imshow(
            np.abs(data[frame_index, :, :, antenna]),
            aspect='auto',
            cmap='jet',
            extent=[velocity_axis[0], velocity_axis[-1], distance_axis[0], distance_axis[-1]]
        )
        ax.set_title(f'Raw Data  2 FFT - Frame {frame_index}')
        ax.set_xlabel('Velocity - meters/sec')
        ax.set_ylabel('Distance - meters')

    velocity_axis = np.linspace(-v_max, v_max, Num_doppler_fft_bins)
    distance_axis = np.linspace(0, ADC_sample_rate * SoL / (2 * S), Num_range_fft_bins)

    data = np.transpose(data, (0, 2, 1, 3))
    data = data[:, ::-1, :, :]

    fig, ax = plt.subplots(1, 1)
    animation = FuncAnimation(fig, animate, interval=30, repeat=False, frames=data.shape[0])
    plt.show()
