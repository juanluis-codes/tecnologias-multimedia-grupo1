import numpy as np
import minimal
from basic_ToH import *
from temporal_overlapped_DWT_coding import *


class AdvancedTreshhold(Treshold):

    def __init__(self):
        super().__init__()

    def analyze(self, chunk, size):
        chunk_DWT = Temporal_Overlapped_DWT.analyze(chunk)

        # Quantize the subbands
        chunk_DWT[self.slices[0][0]] = (chunk_DWT[self.slices[0][0]] / self.quantization_steps[0]).astype(np.int32)
        for i in range (self.dwt_levels):
            chunk_DWT[self.slices[i+1]['d'][0]] = (chunk_DWT[self.slices[i+1]['d'][0]] / self.quantization_steps[i+1]).astype(np.int32)

        blackman_window = np.blackman(size)

        chunk_DWT[:, 0] = (chunk_DWT[:, 0] / blackman_window).astype(np.int32)
        chunk_DWT[:, 1] = (chunk_DWT[:, 1] / blackman_window).astype(np.int32)

        chunk_DWT = np.fft(chunk_DWT)

        return chunk_DWT


    def synthesize(self, chunk_DWT, size):

        # Dequantize the subbands
        chunk_DWT[self.slices[0][0]] = chunk_DWT[self.slices[0][0]] * self.quantization_steps[0]
        for i in range (self.dwt_levels):
            chunk_DWT[self.slices[i+1]['d'][0]] = chunk_DWT[self.slices[i+1]['d'][0]] * self.quantization_steps[i+1]

        blackman_window = np.blackman(size)

        chunk_DWT[:, 0] = (chunk_DWT[:, 0] / blackman_window).astype(np.int32)
        chunk_DWT[:, 1] = (chunk_DWT[:, 1] / blackman_window).astype(np.int32)

        chunk_DWT = np.ifft(chunk_DWT)

        return Temporal_Overlapped_DWT.synthesize(chunk_DWT)

class Advanced_Treshold__verbose(Treshold, Temporal_Overlapped_DWT__verbose):
    pass

if __name__ == "__main__":
    minimal.parser.description = __doc__

    minimal.args = minimal.parser.parse_known_args()[0]

    if minimal.args.show_stats or minimal.args.show_samples:
        intercom = Advanced_Treshold__verbose()
    else:
        intercom = Treshold()
    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nSIGINT received")
    finally:
        intercom.print_final_averages()
