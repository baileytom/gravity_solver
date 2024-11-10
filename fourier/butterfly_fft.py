import numpy as np
from interface import FFTWorkGroupConfig, FFTImplementation


class ButterflyFFT(FFTImplementation):
    """Butterfly algorithm FFT implementation"""

    def __init__(self, config: FFTWorkGroupConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "ButterflyFFT"

    def _bit_reverse(self, n: int, bits: int) -> int:
        """Reverse bits of a number"""
        return int(format(n, f"0{bits}b")[::-1], 2)

    def _direct_dft(self, x: np.ndarray) -> np.ndarray:
        """Direct DFT computation for small segments"""
        n = len(x)
        k = np.arange(n)
        M = np.exp(-2j * np.pi * k.reshape(-1, 1) * k / n)
        return np.dot(M, x)

    def fft1d(self, data: np.ndarray) -> np.ndarray:
        """Compute 1D FFT using butterfly algorithm"""
        n = len(data)

        if n <= self.config.min_segment_size:
            return self._direct_dft(data)

        bits = int(np.log2(n))
        indices = [self._bit_reverse(i, bits) for i in range(n)]
        x = data[indices].copy()

        for stage in range(bits):
            stage_size = 1 << (stage + 1)
            half_size = stage_size >> 1
            w = np.exp(-2j * np.pi / stage_size)

            for group in range(0, n, stage_size):
                for k in range(half_size):
                    even_idx = group + k
                    odd_idx = even_idx + half_size
                    twiddle = w**k
                    temp = x[odd_idx] * twiddle
                    x[odd_idx] = x[even_idx] - temp
                    x[even_idx] = x[even_idx] + temp

        return x

    def ifft1d(self, data: np.ndarray) -> np.ndarray:
        """Compute 1D inverse FFT"""
        conj_input = np.conjugate(data)
        fft_result = self.fft1d(conj_input)
        return np.conjugate(fft_result) / len(data)
