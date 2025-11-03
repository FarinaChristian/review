import os
from base64 import decode

from constants.settings import Sample_per_chirp, Num_of_chirp_loops
import numpy as np
from pathlib import Path


class AWR1243(object):

    def __init__(self, adc_samples: int = Sample_per_chirp, chirp_loop: int = Num_of_chirp_loops):
        self.adc_samples = adc_samples
        self.chirp_loop = chirp_loop
        self.n_lanes = 4


    def decode(self, file: str):
        iq_matrix = None

        if file.endswith("*"):
            # Long record decoding, composed by multiple files
            recording_dir = str(Path(file).parent.absolute())#"/".join(Path(file).absolute().parts[0:-1])[1:]
            recording_files = sorted([f for f in os.listdir(recording_dir) if f.endswith(".bin")])

            data = []
            for file_i in recording_files:
                print(f"\rDecoding {file_i}", end="", flush=True)
                with open(os.path.join(recording_dir, file_i), 'rb') as fd:
                    data.append(np.fromfile(fd, dtype=np.int16))
            data = np.concat(data)

        else:
            # Single file decoding
            with open(file, 'rb') as fd:
                data = np.fromfile(fd, dtype=np.int16)

        try:
            raw_data_matrix = data.reshape((-1, 8))  # first 4 columns are I, second 4 columns are Q
            iq_matrix = raw_data_matrix[:, :4] + 1j * raw_data_matrix[:, 4:]
            # [frame, chirp, adc_samples, lanes]
            iq_matrix = iq_matrix.reshape(-1, self.chirp_loop, self.adc_samples, self.n_lanes)

        except Exception as e:
            print(f"Error: {e} on file {file}")

        print("\rDecoding completed")
        return iq_matrix

    def to_npy(self, input_file: str, output_file: str):
        output_file = Path(output_file)
        Path("/".join(output_file.parts[:-1])).mkdir(parents=True, exist_ok=True)

        iq_matrix = self.decode(input_file)
        if iq_matrix:
            Path(output_file.suffix)
            np.save(output_file, iq_matrix)
        else:
            print(f"Error in saving {output_file}...")
