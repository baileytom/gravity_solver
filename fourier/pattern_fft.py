import numpy as np
from typing import List, Tuple


class TestPatternGenerator:
    """Generates various test patterns for FFT validation"""

    @staticmethod
    def generate_mass_distribution(
        shape: Tuple[int, ...], num_masses: int = 5
    ) -> np.ndarray:
        """Generate random point masses in n-dimensional space"""
        result = np.zeros(shape)
        for _ in range(num_masses):
            position = tuple(np.random.randint(0, s) for s in shape)
            result[position] = np.random.normal()
        return result

    @staticmethod
    def generate_sinusoidal(
        shape: Tuple[int, ...], frequencies: List[float]
    ) -> np.ndarray:
        """Generate n-dimensional sinusoidal pattern"""
        coords = [np.linspace(-np.pi, np.pi, s) for s in shape]
        grid = np.meshgrid(*coords, indexing="ij")

        result = np.zeros(shape)
        for freq in frequencies:
            phase = np.random.uniform(0, 2 * np.pi)
            result += np.sin(freq * sum(grid) + phase)

        return result

    @staticmethod
    def generate_gaussian_blob(
        shape: Tuple[int, ...], sigma: float = 1.0
    ) -> np.ndarray:
        """Generate n-dimensional Gaussian blob"""
        coords = [np.linspace(-2, 2, s) for s in shape]
        grid = np.meshgrid(*coords, indexing="ij")

        r_squared = sum(x**2 for x in grid)
        return np.exp(-r_squared / (2 * sigma**2))
