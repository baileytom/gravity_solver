import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from multidim_fft import FFTDimensionalSolver


class FFTEvaluator:
    """Evaluates FFT implementation by comparing original signal with reconstructed signal"""

    def __init__(self, solver: FFTDimensionalSolver):
        self.solver = solver

    def compute_error(self, data: np.ndarray) -> Dict[str, float]:
        """Compute error metrics between original and reconstructed signal"""
        # Forward FFT followed by inverse FFT
        fft_result = self.solver.fft(data)
        reconstructed = np.real(self.solver.ifft(fft_result))

        # Compute various error metrics between original and reconstructed
        abs_error = np.abs(data - reconstructed)
        return {
            "max_error": np.max(abs_error),
            "mean_error": np.mean(abs_error),
            "rmse": np.sqrt(np.mean(abs_error**2)),
            "relative_error": np.max(abs_error / (np.abs(data) + 1e-10)),
        }

    def visualize_comparison(self, data: np.ndarray) -> None:
        """Visualize comparison between original and reconstructed signal"""
        # Forward FFT followed by inverse FFT
        fft_result = self.solver.fft(data)
        reconstructed = np.real(self.solver.ifft(fft_result))

        if data.ndim <= 2:
            # Handle 1D and 2D cases as before
            plt.figure(figsize=(15, 5))

            plt.subplot(131)
            plt.title("Original Signal")
            if data.ndim == 1:
                plt.plot(data)
            else:
                plt.imshow(data)
                plt.colorbar()

            plt.subplot(132)
            plt.title("Reconstructed Signal")
            if data.ndim == 1:
                plt.plot(reconstructed)
            else:
                plt.imshow(reconstructed)
                plt.colorbar()

            plt.subplot(133)
            plt.title("Absolute Error")
            error = np.abs(data - reconstructed)
            if data.ndim == 1:
                plt.plot(error)
            else:
                plt.imshow(error)
                plt.colorbar()

        else:  # 3D case
            # Show middle slices of 3D volume
            mid_z = data.shape[2] // 2
            plt.figure(figsize=(15, 5))

            plt.subplot(131)
            plt.title("Original Signal (Middle Z-slice)")
            plt.imshow(data[:, :, mid_z])
            plt.colorbar()

            plt.subplot(132)
            plt.title("Reconstructed Signal (Middle Z-slice)")
            plt.imshow(reconstructed[:, :, mid_z])
            plt.colorbar()

            plt.subplot(133)
            plt.title("Absolute Error (Middle Z-slice)")
            error = np.abs(data - reconstructed)
            plt.imshow(error[:, :, mid_z])
            plt.colorbar()

        plt.tight_layout()
        plt.show()
