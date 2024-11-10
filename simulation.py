import numpy as np
from threading import Thread, Lock, Event
from queue import Queue
import time
import matplotlib.pyplot as plt


class ParticleSim:
    def __init__(
        self, gravity_solver, num_particles, particle_mass, tensor_size, world_size
    ):
        self.gravity_solver = gravity_solver

        # Particle settings
        self.NUM_PARTICLES = num_particles
        self.PARTICLE_MASS = particle_mass
        self.MASS_MODIFIER = 1.1

        # World settings
        self.TENSOR_SIZE = tensor_size
        self.WORLD_SIZE = world_size
        self.VOXEL_SIZE = world_size / tensor_size

        # Time settings
        self.TIME_CURRENT_SPEED = 0.05
        self.TIME_MAX_SPEED = 1000.0
        self.TIME_STEP = 0.005

        # Initialization settings
        self.EDGE_PADDING = 0.1

        # Threading and synchronization
        self.computation_lock = Lock()
        self.computation_queue = Queue(maxsize=1)
        self.is_computing = False
        self.paused = Event()  # New Event for pause control
        self.computation_thread = None

        # Initialize fields
        self.mass_field = np.zeros((tensor_size, tensor_size, tensor_size))
        self.gradient_field = np.zeros((tensor_size, tensor_size, tensor_size, 3))

        # Visualization settings
        self.show_particles = True
        self.show_gradient = False

        # State variables
        self.running = True
        self.needs_update = True  # Start with True to compute initial state

        # Initialize particles and velocities
        self.initialize_particles()
        self.velocities = np.zeros((num_particles, 3))

        # Add double buffers for gradient and mass fields

        self.solution_applied = False  # Track if current solution has been applied
        self.gradient_buffer = {
            "current": np.zeros((tensor_size, tensor_size, tensor_size, 3)),
            "next": np.zeros((tensor_size, tensor_size, tensor_size, 3)),
        }
        self.mass_buffer = {
            "current": np.zeros((tensor_size, tensor_size, tensor_size)),
            "next": np.zeros((tensor_size, tensor_size, tensor_size)),
        }
        self.particle_forces_buffer = {
            "current": np.zeros((num_particles, 3)),
            "next": np.zeros((num_particles, 3)),
        }
        self.buffer_lock = Lock()

    def initialize_particles(self):
        min_bound = self.WORLD_SIZE * self.EDGE_PADDING
        max_bound = self.WORLD_SIZE * (1 - self.EDGE_PADDING)
        effective_size = max_bound - min_bound
        center = np.array([self.WORLD_SIZE / 2] * 3)
        radius = effective_size / 3

        self.particles = np.zeros((self.NUM_PARTICLES, 3))
        accepted_count = 0

        while accepted_count < self.NUM_PARTICLES:
            batch_size = (self.NUM_PARTICLES - accepted_count) * 2
            candidates = np.random.uniform(
                center - radius, center + radius, (batch_size, 3)
            )
            distances = np.linalg.norm(candidates - center, axis=1)
            valid_mask = distances <= radius
            valid_points = candidates[valid_mask]
            points_to_add = min(len(valid_points), self.NUM_PARTICLES - accepted_count)
            self.particles[accepted_count : accepted_count + points_to_add] = (
                valid_points[:points_to_add]
            )
            accepted_count += points_to_add

    def swap_buffers(self):
        """Swap the current and next buffers"""
        with self.buffer_lock:
            self.gradient_buffer["current"], self.gradient_buffer["next"] = (
                self.gradient_buffer["next"],
                self.gradient_buffer["current"],
            )
            self.mass_buffer["current"], self.mass_buffer["next"] = (
                self.mass_buffer["next"],
                self.mass_buffer["current"],
            )
            (
                self.particle_forces_buffer["current"],
                self.particle_forces_buffer["next"],
            ) = (
                self.particle_forces_buffer["next"],
                self.particle_forces_buffer["current"],
            )
            self.solution_applied = False

    def compute_gravity_async(self):
        while self.running:
            if self.needs_update and not self.is_computing and not self.paused.is_set():
                self.is_computing = True

                # Pass show_gradient flag to solver
                particle_forces, gradient_field, mass_field = (
                    self.gravity_solver.compute_gravity(
                        self.particles.copy(),
                        self.PARTICLE_MASS,
                        self.WORLD_SIZE,
                        self.TENSOR_SIZE,
                        compute_gradient=self.show_gradient,  # Only compute if visualization is enabled
                    )
                )

                self.particle_forces_buffer["next"] = particle_forces
                self.gradient_buffer["next"] = gradient_field
                self.mass_buffer["next"] = mass_field

                self.swap_buffers()

                self.is_computing = False
                self.needs_update = False
            time.sleep(0.001)

    def start_computation_thread(self):
        self.computation_thread = Thread(target=self.compute_gravity_async, daemon=True)
        self.computation_thread.start()

    def stop_computation_thread(self):
        self.running = False
        if self.computation_thread:
            self.computation_thread.join()

    def update(self, dt):
        """Update particle positions once per gravity solution"""
        if self.paused.is_set() or self.solution_applied:
            return

        # Use particle forces directly instead of sampling from grid
        forces = self.particle_forces_buffer["current"]

        # Apply time direction to velocity updates
        self.velocities += forces * dt * self.TIME_CURRENT_SPEED
        self.particles += self.velocities * dt * self.TIME_CURRENT_SPEED

        # Bounce off boundaries
        for i in range(3):
            out_of_bounds = (self.particles[:, i] < 0) | (
                self.particles[:, i] >= self.WORLD_SIZE
            )
            self.particles[out_of_bounds, i] = np.clip(
                self.particles[out_of_bounds, i], 0, self.WORLD_SIZE
            )
            self.velocities[out_of_bounds, i] *= -0.5

        metrics = self.gravity_solver.get_latest_metrics()
        if metrics:
            print(metrics)  # Will print nicely formatted metrics

        self.solution_applied = True
        self.needs_update = True

    def visualize(self, ax):
        """Visualize the current state with gradient vectors scaled by magnitude"""
        ax.clear()
        ax.set_axis_off()

        visualizers = []

        if self.show_particles:
            velocities_mag = np.linalg.norm(self.velocities, axis=1)
            scatter = ax.scatter(
                self.particles[:, 0],
                self.particles[:, 1],
                self.particles[:, 2],
                c=velocities_mag,
                cmap="viridis",
                alpha=0.9,
                s=1,
            )
            visualizers.append(scatter)

        if self.show_gradient:
            x = np.linspace(0, self.WORLD_SIZE, self.TENSOR_SIZE)
            y = np.linspace(0, self.WORLD_SIZE, self.TENSOR_SIZE)
            z = np.linspace(0, self.WORLD_SIZE, self.TENSOR_SIZE)
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

            # Use current buffer without locking
            current_gradient = self.gradient_buffer["current"]

            # Compute gradient magnitudes
            gradient_magnitudes = np.linalg.norm(current_gradient, axis=3)

            # Scale factor to make the vectors visible but not overwhelming
            # You might need to adjust this based on your data
            scale_factor = self.WORLD_SIZE / gradient_magnitudes.max() * 0.1

            # Apply stride
            stride = 4
            strided_X = X[::stride, ::stride, ::stride]
            strided_Y = Y[::stride, ::stride, ::stride]
            strided_Z = Z[::stride, ::stride, ::stride]
            strided_gradient = current_gradient[::stride, ::stride, ::stride]
            strided_magnitudes = gradient_magnitudes[::stride, ::stride, ::stride]

            # Create a colormap based on magnitude
            colors = plt.cm.viridis(strided_magnitudes / strided_magnitudes.max())

            quiver = ax.quiver(
                strided_X,
                strided_Y,
                strided_Z,
                strided_gradient[..., 0],
                strided_gradient[..., 1],
                strided_gradient[..., 2],
                length=scale_factor,
                normalize=False,  # Don't normalize so we keep magnitude information
                alpha=0.3,
                # colors=colors,
            )
            visualizers.append(quiver)

            # Optionally add a colorbar for the gradient magnitudes
            if len(visualizers) == 1:  # Only if we're not showing particles
                sm = plt.cm.ScalarMappable(
                    cmap="viridis",
                    norm=plt.Normalize(
                        vmin=gradient_magnitudes.min(), vmax=gradient_magnitudes.max()
                    ),
                )
                plt.colorbar(sm, ax=ax, label="Gradient Magnitude")

        return tuple(visualizers)
