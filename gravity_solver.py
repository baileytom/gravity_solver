import numpy as np
from scipy.fft import fftn, ifftn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
import sys
from typing import Tuple


class GravitySolver(ABC):
    def __init__(self, G):
        self.show_progress = True
        self.G = G

    @abstractmethod
    def compute_gravity(self, particles, masses, world_size, tensor_size):
        """
        Compute gravitational forces for the particle system

        Returns:
            particle_forces: (N,3) array of force vectors for each particle
            gradient_field: (tensor_size, tensor_size, tensor_size, 3) array of force vectors
            mass_field: (tensor_size, tensor_size, tensor_size) array of mass distribution
        """
        pass

    def get_latest_metrics(self):
        """Get the most recent comparison metrics"""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(G={self.G})"


@dataclass
class ComparisonMetrics:
    """
    Comprehensive metrics for comparing force calculations between solvers.
    All error values are non-negative. Angles are in degrees.
    """

    # Magnitude error metrics
    relative_magnitude_error: float  # Mean relative error in force magnitudes
    max_relative_magnitude_error: float  # Maximum relative error in force magnitudes
    magnitude_log_error: float  # Mean absolute log10 ratio error in magnitudes

    # Direction error metrics (in degrees)
    mean_direction_error: float  # Mean angular difference between force vectors
    max_direction_error: float  # Maximum angular difference between force vectors
    median_direction_error: float  # Median angular difference between force vectors

    # Vector error metrics (combining magnitude and direction)
    rmse: float  # Root mean square error of force vectors
    nrmse: float  # Normalized RMSE (divided by mean reference force magnitude)

    # Gradient error metrics (optional)
    gradient_rmse: Optional[float] = None  # RMSE of gradient tensors
    relative_gradient_error: Optional[float] = (
        None  # Mean relative error in gradient magnitudes
    )

    def __str__(self) -> str:
        """Format metrics nicely for display"""
        lines = [
            "Force Magnitude Errors:",
            f"  Relative: {self.relative_magnitude_error:.2e}",
            f"  Max Relative: {self.max_relative_magnitude_error:.2e}",
            f"  Log Error: {self.magnitude_log_error:.2e}",
            "",
            "Direction Errors (degrees):",
            f"  Mean: {self.mean_direction_error:.2f}°",
            f"  Median: {self.median_direction_error:.2f}°",
            f"  Max: {self.max_direction_error:.2f}°",
            "",
            "Vector Errors:",
            f"  RMSE: {self.rmse:.2e}",
            f"  NRMSE: {self.nrmse:.2e}",
        ]

        # Add gradient metrics if available
        if self.gradient_rmse is not None:
            lines.extend(
                [
                    "",
                    "Gradient Errors:",
                    f"  RMSE: {self.gradient_rmse:.2e}",
                    f"  Relative: {self.relative_gradient_error:.2e}",
                ]
            )

        return "\n".join(lines)

    def as_dict(self) -> dict:
        """Convert metrics to a dictionary, useful for logging or plotting"""
        return {
            # Magnitude metrics
            "relative_magnitude_error": self.relative_magnitude_error,
            "max_relative_magnitude_error": self.max_relative_magnitude_error,
            "magnitude_log_error": self.magnitude_log_error,
            # Direction metrics
            "mean_direction_error": self.mean_direction_error,
            "max_direction_error": self.max_direction_error,
            "median_direction_error": self.median_direction_error,
            # Vector metrics
            "rmse": self.rmse,
            "nrmse": self.nrmse,
            # Gradient metrics (if available)
            **(
                {
                    "gradient_rmse": self.gradient_rmse,
                    "relative_gradient_error": self.relative_gradient_error,
                }
                if self.gradient_rmse is not None
                else {}
            ),
        }


class ComparisonSolver(GravitySolver):
    """
    A GravitySolver that compares a test solver against a reference solver.
    Returns the test solver's results while computing comprehensive error metrics.
    """

    def __init__(self, test_solver: GravitySolver, reference_solver: GravitySolver):
        super().__init__(G=test_solver.G)
        self.test_solver = test_solver
        self.reference_solver = reference_solver
        self.latest_metrics = None

    def compute_metrics(
        self,
        ref_forces: np.ndarray,
        test_forces: np.ndarray,
        ref_gradient: np.ndarray = None,
        test_gradient: np.ndarray = None,
    ) -> ComparisonMetrics:
        """
        Compute comparison metrics between reference and test solutions.

        Args:
            ref_forces: Reference force vectors (N x 3)
            test_forces: Test force vectors (N x 3)
            ref_gradient: Reference force gradients (N x 3 x 3) if available
            test_gradient: Test force gradients (N x 3 x 3) if available
        """
        # Compute force magnitudes
        ref_magnitudes = np.linalg.norm(ref_forces, axis=1)
        test_magnitudes = np.linalg.norm(test_forces, axis=1)

        # Mask for significant forces (both vectors non-zero)
        significant_mask = (ref_magnitudes > 1e-10) & (test_magnitudes > 1e-10)

        # Initialize metrics
        metrics = {}

        # 1. Magnitude error metrics
        if np.any(significant_mask):
            # Relative magnitude error
            magnitude_diff = np.abs(test_magnitudes - ref_magnitudes)
            relative_magnitude_error = np.mean(
                magnitude_diff[significant_mask] / ref_magnitudes[significant_mask]
            )
            max_relative_magnitude_error = np.max(
                magnitude_diff[significant_mask] / ref_magnitudes[significant_mask]
            )

            # Log-ratio error (better for comparing magnitudes across scales)
            log_ratio = np.log10(
                test_magnitudes[significant_mask] / ref_magnitudes[significant_mask]
            )
            magnitude_log_error = np.mean(np.abs(log_ratio))

            metrics.update(
                {
                    "relative_magnitude_error": relative_magnitude_error,
                    "max_relative_magnitude_error": max_relative_magnitude_error,
                    "magnitude_log_error": magnitude_log_error,
                }
            )
        else:
            metrics.update(
                {
                    "relative_magnitude_error": 0.0,
                    "max_relative_magnitude_error": 0.0,
                    "magnitude_log_error": 0.0,
                }
            )

        # 2. Direction error metrics
        if np.any(significant_mask):
            # Normalize vectors for direction comparison
            ref_normalized = (
                ref_forces[significant_mask]
                / ref_magnitudes[significant_mask, np.newaxis]
            )
            test_normalized = (
                test_forces[significant_mask]
                / test_magnitudes[significant_mask, np.newaxis]
            )

            # Compute angles (in degrees)
            cos_angles = np.sum(ref_normalized * test_normalized, axis=1)
            cos_angles = np.clip(cos_angles, -1.0, 1.0)
            angles = np.degrees(np.arccos(cos_angles))

            metrics.update(
                {
                    "mean_direction_error": np.mean(angles),
                    "max_direction_error": np.max(angles),
                    "median_direction_error": np.median(angles),
                }
            )
        else:
            metrics.update(
                {
                    "mean_direction_error": 0.0,
                    "max_direction_error": 0.0,
                    "median_direction_error": 0.0,
                }
            )

        # 3. Vector error metrics
        # RMSE considering both magnitude and direction
        rmse = np.sqrt(np.mean(np.sum((test_forces - ref_forces) ** 2, axis=1)))

        # Normalized RMSE
        if np.any(significant_mask):
            nrmse = rmse / np.mean(ref_magnitudes[significant_mask])
        else:
            nrmse = 0.0

        metrics.update(
            {
                "rmse": rmse,
                "nrmse": nrmse,
            }
        )

        # 4. Gradient error metrics (if available)
        if ref_gradient is not None and test_gradient is not None:
            # Get the original shape
            orig_shape = ref_gradient.shape

            # Reshape to (N, 9) while preserving the particle dimension
            ref_grad_flat = ref_gradient.reshape(orig_shape[0], -1)
            test_grad_flat = test_gradient.reshape(orig_shape[0], -1)

            # Compute RMSE
            grad_rmse = np.sqrt(np.mean((test_grad_flat - ref_grad_flat) ** 2))

            # Compute gradient magnitudes per particle
            ref_grad_magnitudes = np.linalg.norm(ref_grad_flat, axis=1)
            significant_grad = ref_grad_magnitudes > 1e-10

            if np.any(significant_grad):
                grad_diff_norms = np.linalg.norm(test_grad_flat - ref_grad_flat, axis=1)
                relative_grad_error = np.mean(
                    grad_diff_norms[significant_grad]
                    / ref_grad_magnitudes[significant_grad]
                )
            else:
                relative_grad_error = 0.0

            metrics.update(
                {
                    "gradient_rmse": grad_rmse,
                    "relative_gradient_error": relative_grad_error,
                }
            )

        return ComparisonMetrics(**metrics)

    def compute_gravity(
        self, particles, masses, world_size, tensor_size, compute_gradient=True
    ):
        """
        Compute gravity using both solvers and compare results.
        Returns test solver results and stores comparison metrics.
        """
        # Compute solutions from both solvers
        ref_forces, ref_gradient, ref_mass = self.reference_solver.compute_gravity(
            particles, masses, world_size, tensor_size, compute_gradient
        )
        test_forces, test_gradient, test_mass = self.test_solver.compute_gravity(
            particles, masses, world_size, tensor_size, compute_gradient
        )

        # Compute and store metrics with gradients if available
        self.latest_metrics = self.compute_metrics(
            ref_forces,
            test_forces,
            ref_gradient if compute_gradient else None,
            test_gradient if compute_gradient else None,
        )

        # Return test solver results
        return test_forces, test_gradient, test_mass

    def get_latest_metrics(self) -> ComparisonMetrics:
        """Get the most recent comparison metrics"""
        return self.latest_metrics

    def __repr__(self):
        return f"ComparisonSolver(\ntest_solver={self.test_solver},\nreference_solver={self.reference_solver})"


class FrequencyGradientSolver(GravitySolver):
    """
    A gravity solver that computes gradients in frequency space using ik multiplication.
    Properly normalized to match the N^2 solver's force magnitudes.
    """

    def __init__(self, G):
        super().__init__(G)
        self._initialized = False

    def _initialize(self, world_size: float, tensor_size: int):
        """Initialize solver parameters and precompute frequency-space values."""
        if self._initialized:
            return

        self.tensor_size = tensor_size
        self.world_size = world_size
        self.voxel_size = world_size / tensor_size

        # Pre-compute frequency grid
        freqs = np.fft.fftfreq(tensor_size, d=self.voxel_size)
        self.kx, self.ky, self.kz = np.meshgrid(freqs, freqs, freqs, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2

        # Avoid division by zero while maintaining DC component
        self.k2[0, 0, 0] = 1.0

        # Green's function with proper FFT normalization
        scale_factor = 1.0 / (tensor_size**3)
        self.green_function = -4 * np.pi * self.G / self.k2 * scale_factor

        # scale_factor = 1.0
        # self.green_function = -4 * np.pi * self.G / self.k2 * scale_factor

        # Pre-compute i*k factors for gradient computation
        self.ikx = 1j * self.kx
        self.iky = 1j * self.ky
        self.ikz = 1j * self.kz

        self._initialized = True

    def world_to_voxel(self, world_pos: np.ndarray) -> np.ndarray:
        """Convert world coordinates to voxel coordinates."""
        return np.clip(
            (world_pos / self.voxel_size).astype(int), 0, self.tensor_size - 1
        )

    def compute_mass_field(self, particles: np.ndarray, masses: float) -> np.ndarray:
        """Convert particle positions to mass field."""
        mass_field = np.zeros((self.tensor_size, self.tensor_size, self.tensor_size))
        positions = self.world_to_voxel(particles)

        if np.isscalar(masses):
            masses = np.full(len(particles), masses)

        np.add.at(
            mass_field, (positions[:, 0], positions[:, 1], positions[:, 2]), masses
        )

        return mass_field

    def compute_gravity(
        self,
        particles: np.ndarray,
        masses: float,
        world_size: float,
        tensor_size: int,
        compute_gradient: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gravitational forces using frequency space operations.
        Properly normalized to match N^2 solver results.
        """
        self._initialize(world_size, tensor_size)

        # Convert particles to mass field
        mass_field = self.compute_mass_field(particles, masses)

        # Forward FFT of mass field
        mass_spectrum = fftn(mass_field)

        # Compute potential spectrum with proper normalization
        potential_spectrum = mass_spectrum * self.green_function

        # Compute force components in frequency space
        # Note: -ik * Φ gives force components directly
        fx_spectrum = -self.ikx * potential_spectrum
        fy_spectrum = -self.iky * potential_spectrum
        fz_spectrum = -self.ikz * potential_spectrum

        # Compute force field in real space
        if compute_gradient:
            fx = np.real(ifftn(fx_spectrum))
            fy = np.real(ifftn(fy_spectrum))
            fz = np.real(ifftn(fz_spectrum))
            gradient_field = np.stack([fx, fy, fz], axis=-1)
        else:
            gradient_field = np.zeros((tensor_size, tensor_size, tensor_size, 3))

        # Sample forces at particle positions
        positions = self.world_to_voxel(particles)
        particle_forces = gradient_field[
            positions[:, 0], positions[:, 1], positions[:, 2]
        ]

        return particle_forces, gradient_field, mass_field


class FFTGreensSolver(GravitySolver):
    """
    A physically correct FFT-based gravity solver.
    Uses the fact that Φ(k) = -4πGρ(k)/k² in Fourier space.
    """

    def __init__(self, G):
        super().__init__(G)
        self._initialized = False

    def _initialize(self, world_size: float, tensor_size: int):
        """Initialize solver parameters and precompute frequency-space values."""
        if self._initialized:
            return

        self.tensor_size = tensor_size
        self.world_size = world_size
        self.voxel_size = world_size / tensor_size

        # Pre-compute frequency grid
        freqs = np.fft.fftfreq(tensor_size, d=self.voxel_size)
        self.kx, self.ky, self.kz = np.meshgrid(freqs, freqs, freqs, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2[0, 0, 0] = 1.0  # Avoid division by zero while maintaining DC component

        # Create Green's function with correct physics:
        # Φ(k) = -4πGρ(k)/k²
        # scale_factor = 1.0 / (tensor_size**3)  # FFT normalization
        scale_factor = 1 / self.voxel_size**3  # FFT normalization
        self.green_function = (-4 * np.pi * self.G / self.k2) * scale_factor

        self._initialized = True

    def world_to_voxel(self, world_pos: np.ndarray) -> np.ndarray:
        """Convert world coordinates to voxel coordinates."""
        return np.clip(
            (world_pos / self.voxel_size).astype(int), 0, self.tensor_size - 1
        )

    def compute_mass_field(self, particles: np.ndarray, masses: float) -> np.ndarray:
        """Convert particle positions to mass field."""
        mass_field = np.zeros((self.tensor_size, self.tensor_size, self.tensor_size))
        positions = self.world_to_voxel(particles)

        # Handle both scalar and array masses
        if np.isscalar(masses):
            masses = np.full(len(particles), masses)

        # Use np.add.at to handle multiple particles in same voxel
        np.add.at(
            mass_field, (positions[:, 0], positions[:, 1], positions[:, 2]), masses
        )

        return mass_field

    def compute_potential_spectrum(self, mass_spectrum: np.ndarray) -> np.ndarray:
        """Compute potential in frequency space."""
        return mass_spectrum * self.green_function

    def compute_gradient(self, potential: np.ndarray) -> np.ndarray:
        """
        Compute gradient field using 4th order accurate central differences.
        The negative gradient of potential gives the gravitational force.
        """
        # Pad the potential field for accurate edge gradients
        padded = np.pad(potential, 2, mode="edge")

        # Compute gradients using 4th order central differences
        # Note: The negative is applied later since F = -∇Φ
        dx = (
            -padded[4:, 2:-2, 2:-2]
            + 8 * padded[3:-1, 2:-2, 2:-2]
            - 8 * padded[1:-3, 2:-2, 2:-2]
            + padded[:-4, 2:-2, 2:-2]
        ) / (12 * self.voxel_size)

        dy = (
            -padded[2:-2, 4:, 2:-2]
            + 8 * padded[2:-2, 3:-1, 2:-2]
            - 8 * padded[2:-2, 1:-3, 2:-2]
            + padded[2:-2, :-4, 2:-2]
        ) / (12 * self.voxel_size)

        dz = (
            -padded[2:-2, 2:-2, 4:]
            + 8 * padded[2:-2, 2:-2, 3:-1]
            - 8 * padded[2:-2, 2:-2, 1:-3]
            + padded[2:-2, 2:-2, :-4]
        ) / (12 * self.voxel_size)

        # Stack components and apply negative since F = -∇Φ
        gradient = -np.stack([dx, dy, dz], axis=-1)

        # Apply FFT normalization
        return gradient / (self.tensor_size**3)

    def compute_gravity(
        self,
        particles: np.ndarray,
        masses: float,
        world_size: float,
        tensor_size: int,
        compute_gradient: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gravitational forces for particle system.

        Args:
            particles: (N, 3) array of particle positions
            masses: Scalar mass value or (N,) array of masses
            world_size: Physical size of simulation space
            tensor_size: Size of 3D grid (assumed cubic)
            compute_gradient: Whether to compute full gradient field

        Returns:
            particle_forces: (N, 3) array of force vectors for each particle
            gradient_field: (tensor_size, tensor_size, tensor_size, 3) array of force vectors
            mass_field: (tensor_size, tensor_size, tensor_size) array of mass distribution
        """
        # Initialize or update solver parameters if needed
        self._initialize(world_size, tensor_size)

        # Convert particles to mass field
        mass_field = self.compute_mass_field(particles, masses)

        # Forward FFT of mass field
        mass_spectrum = fftn(mass_field)

        # Compute potential in frequency space
        potential_spectrum = self.compute_potential_spectrum(mass_spectrum)

        # Inverse FFT to get potential
        potential = np.real(ifftn(potential_spectrum))

        # Compute gradient field
        gradient_field = (
            self.compute_gradient(potential)
            if compute_gradient
            else np.zeros((tensor_size, tensor_size, tensor_size, 3))
        )

        # Sample gradient at particle positions
        positions = self.world_to_voxel(particles)
        particle_forces = gradient_field[
            positions[:, 0], positions[:, 1], positions[:, 2]
        ]

        return particle_forces, gradient_field, mass_field


class NSquaredSolver(GravitySolver):
    def compute_gravity(
        self, particles, masses, world_size, tensor_size, compute_gradient=False
    ):
        voxel_size = world_size / tensor_size
        particle_forces = np.zeros_like(particles)

        # Ensure masses is always an array for vectorized operations
        if np.isscalar(masses):
            masses = np.full(len(particles), masses)

        # Vectorized particle force computation for all particles at once
        # Compute all pairwise differences - changing direction for attraction
        r = -(
            particles[:, np.newaxis, :] - particles[np.newaxis, :, :]
        )  # Added negative sign

        # Compute all pairwise distances
        r_mag = np.sqrt(np.sum(r * r, axis=2))

        # Set minimum distance to voxel_size to prevent infinite forces
        r_mag = np.maximum(r_mag, voxel_size)

        # Set diagonal to infinity to eliminate self-interactions
        np.fill_diagonal(r_mag, float("inf"))

        # Compute force magnitudes
        force_mags = (self.G * masses[:, np.newaxis] * masses[np.newaxis, :]) / (
            r_mag**3
        )

        # Compute force vectors
        forces = force_mags[:, :, np.newaxis] * r

        # Sum forces for each particle
        particle_forces = np.sum(forces, axis=1)

        # Initialize fields
        gradient_field = np.zeros((tensor_size, tensor_size, tensor_size, 3))
        mass_field = np.zeros((tensor_size, tensor_size, tensor_size))

        # Compute mass field
        positions = np.clip((particles / voxel_size).astype(int), 0, tensor_size - 1)
        np.add.at(
            mass_field, (positions[:, 0], positions[:, 1], positions[:, 2]), masses
        )

        # Compute gradient field if requested
        if compute_gradient:
            x = np.linspace(0, world_size, tensor_size)
            y = np.linspace(0, world_size, tensor_size)
            z = np.linspace(0, world_size, tensor_size)
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            ghost_positions = np.stack([X, Y, Z], axis=-1)
            ghost_flat = ghost_positions.reshape(-1, 3)
            ghost_forces = np.zeros_like(ghost_flat)

            # Process ghosts in chunks to manage memory
            chunk_size = tensor_size * tensor_size
            for chunk_start in range(0, len(ghost_flat), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(ghost_flat))
                chunk_positions = ghost_flat[chunk_start:chunk_end]

                # Compute r vectors between particles and chunk positions - changed direction for attraction
                r = -(
                    chunk_positions[:, np.newaxis, :] - particles[np.newaxis, :, :]
                )  # Added negative sign
                r_mag = np.sqrt(np.sum(r * r, axis=2))
                r_mag = np.maximum(r_mag, voxel_size)

                # Compute forces for chunk
                force_mags = (self.G * masses[np.newaxis, :]) / (r_mag**3)
                chunk_forces = np.sum(r * force_mags[:, :, np.newaxis], axis=1)

                ghost_forces[chunk_start:chunk_end] = chunk_forces

            gradient_field = ghost_forces.reshape(
                tensor_size, tensor_size, tensor_size, 3
            )

        return particle_forces, gradient_field, mass_field
