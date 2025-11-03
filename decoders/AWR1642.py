import numpy as np
import torch

class AWR1642(object):

    def __init__(self, adc_samples: int, chirp_loop: int):
        self.adc_samples = adc_samples
        self.chirp_loop = chirp_loop
        self.n_lvds_lanes = 2

    def decode(self, file: str):
        lines_per_chirp = 2 * 4 * self.adc_samples

        with open(file, 'rb') as fd:
            data = np.fromfile(fd, dtype=np.int16)

            i_matrix = np.array([data[i: i + 2] for i in range(0, len(data), 4)]).reshape(-1)
            q_matrix = np.array([data[i: i + 2] for i in range(2, len(data), 4)]).reshape(-1)

            iq_matrix = i_matrix + 1j * q_matrix

            # [frame, chirp, adc_samples, lanes]
            iq_matrix = torch.tensor(iq_matrix.reshape(-1, self.adc_samples, 4, self.chirp_loop))
            iq_matrix = torch.permute(iq_matrix, (0, 3, 1, 2)).numpy()
            return iq_matrix




if __name__ == '__main__':
    directory = "/mnt/hdd16T/mmwave/RVS_Dataset/Radar data/Participant 1/1. Distance Scenario/160 cm/1"
    decoder = AWR1642(adc_samples=250, chirp_loop=128)
    decoder.decode(f"{directory}/data_Raw_0.bin")