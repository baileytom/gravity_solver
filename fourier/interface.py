import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple


@dataclass
class FFTWorkGroupConfig:
    """Configuration for FFT work groups"""

    dimensions: Tuple[int, ...]  # Size of each dimension
    work_group_sizes: Tuple[int, ...]  # Work group size for each dimension
    min_segment_size: int = 4

    def __post_init__(self):
        if len(self.dimensions) != len(self.work_group_sizes):
            raise ValueError("Dimensions and work group sizes must match")
        for dim, wg in zip(self.dimensions, self.work_group_sizes):
            if not (isinstance(dim, int) and isinstance(wg, int)):
                raise TypeError("All dimensions must be integers")
            if dim <= 0 or wg <= 0:
                raise ValueError("All dimensions must be positive")
            if not (dim & (dim - 1) == 0):
                raise ValueError("All dimensions must be powers of 2")
            if dim % wg != 0:
                raise ValueError(
                    f"Dimension {dim} must be divisible by work group size {wg}"
                )


class FFTImplementation(ABC):
    """Abstract base class for FFT implementations"""

    @abstractmethod
    def fft1d(self, data: np.ndarray) -> np.ndarray:
        """Compute 1D FFT"""
        pass

    @abstractmethod
    def ifft1d(self, data: np.ndarray) -> np.ndarray:
        """Compute 1D inverse FFT"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Implementation name"""
        pass
