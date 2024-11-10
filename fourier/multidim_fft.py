import numpy as np
from interface import FFTImplementation


class FFTDimensionalSolver:
    """Handles multi-dimensional FFT computations using work groups"""

    def __init__(self, implementation: FFTImplementation):
        self.implementation = implementation

    def _process_dimension(self, data: np.ndarray, axis: int) -> np.ndarray:
        """Process a single dimension of the data"""
        result = data.copy()
        shape = data.shape

        # Handle the case when axis is beyond the current dimensions
        if axis >= len(shape):
            return result

        # Calculate work group divisions for this axis
        dim_size = shape[axis]
        wg_size = self.implementation.config.work_group_sizes[axis]
        num_groups = dim_size // wg_size

        # Reshape to handle the current axis
        temp_shape = list(shape)
        temp_shape[axis] = num_groups
        temp_shape.insert(axis + 1, wg_size)
        result = result.reshape(temp_shape)

        # Process each line in work groups
        # Create the correct number of indices based on data dimensionality
        idx_shape = result.shape[:axis] + result.shape[axis + 2 :]
        for idx in np.ndindex(idx_shape):
            for group in range(num_groups):
                # Create index for the current line in the work group
                full_idx = idx[:axis] + (group,) + (slice(None),) + idx[axis:]
                result[full_idx] = self.implementation.fft1d(result[full_idx])

        # Restore original shape
        result = result.reshape(shape)
        return result

    def fft(self, data: np.ndarray) -> np.ndarray:
        """Compute n-dimensional FFT"""
        result = data.astype(np.complex128)

        # Process each dimension sequentially
        for axis in range(data.ndim):
            result = self._process_dimension(result, axis)

        return result

    def ifft(self, data: np.ndarray) -> np.ndarray:
        """Compute n-dimensional inverse FFT"""
        result = data.astype(np.complex128)

        # Process each dimension sequentially
        for axis in range(data.ndim):
            shape = result.shape

            # Handle the case when axis is beyond the current dimensions
            if axis >= len(shape):
                continue

            dim_size = shape[axis]
            wg_size = self.implementation.config.work_group_sizes[axis]
            num_groups = dim_size // wg_size

            temp_shape = list(shape)
            temp_shape[axis] = num_groups
            temp_shape.insert(axis + 1, wg_size)
            result = result.reshape(temp_shape)

            # Create the correct number of indices based on data dimensionality
            idx_shape = result.shape[:axis] + result.shape[axis + 2 :]
            for idx in np.ndindex(idx_shape):
                for group in range(num_groups):
                    full_idx = idx[:axis] + (group,) + (slice(None),) + idx[axis:]
                    result[full_idx] = self.implementation.ifft1d(result[full_idx])

            result = result.reshape(shape)

        return result
