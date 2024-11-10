import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gravity_solver import (
    NSquaredSolver,
    FFTGreensSolver,
    ComparisonSolver,
    FrequencyGradientSolver,
)
from simulation import ParticleSim

# Move global variables to top level
NUM_PARTICLES = 128
PARTICLE_MASS = 1e2
GRAVITY_CONSTANT = 6.67430e-11
TENSOR_SIZE = 16
WORLD_SIZE = 16.0

# Initialize solver configuration
SOLVER_A_FIRST = True
SOLVER_A_TYPE = NSquaredSolver
SOLVER_B_TYPE = FrequencyGradientSolver


def create_new_simulation():
    """Create a new simulation with current settings"""
    solver_a = SOLVER_A_TYPE(G=GRAVITY_CONSTANT)
    solver_b = SOLVER_B_TYPE(G=GRAVITY_CONSTANT)
    solver = ComparisonSolver(test_solver=solver_b, reference_solver=solver_a)

    return ParticleSim(
        gravity_solver=solver,
        num_particles=NUM_PARTICLES,
        particle_mass=PARTICLE_MASS,
        tensor_size=TENSOR_SIZE,
        world_size=WORLD_SIZE,
    )


def handle_keyboard(event):
    global sim, SOLVER_A_FIRST

    if event.key == " ":
        if sim.paused.is_set():
            sim.paused.clear()
            print("Simulation resumed")
        else:
            sim.paused.set()
            print("Simulation paused")
    elif event.key == "p":
        sim.show_particles = not sim.show_particles
        print("Showing particles" if sim.show_particles else "Hiding particles")
    elif event.key == "g":
        sim.show_gradient = not sim.show_gradient
        print(
            "Showing gradient field" if sim.show_gradient else "Hiding gradient field"
        )
    elif event.key == "r":
        print("Resetting sim")
        # Stop the old simulation
        sim.stop_computation_thread()
        # Create and start a new simulation
        sim = create_new_simulation()
        sim.start_computation_thread()
    elif event.key == "up":
        sim.TIME_CURRENT_SPEED = min(
            sim.TIME_MAX_SPEED, sim.TIME_CURRENT_SPEED + sim.TIME_STEP
        )
        print(f"Speed: {sim.TIME_CURRENT_SPEED:.2f}")
    elif event.key == "down":
        sim.TIME_CURRENT_SPEED = max(
            -sim.TIME_MAX_SPEED, sim.TIME_CURRENT_SPEED - sim.TIME_STEP
        )
        print(f"Speed: {sim.TIME_CURRENT_SPEED:.2f}")
    elif event.key == "left":
        sim.PARTICLE_MASS /= sim.MASS_MODIFIER
        print(f"Particle mass: {sim.PARTICLE_MASS:.4e}")
    elif event.key == "right":
        sim.PARTICLE_MASS *= sim.MASS_MODIFIER
        print(f"Particle mass: {sim.PARTICLE_MASS:.4e}")
    elif event.key == "t":
        sim.TIME_CURRENT_SPEED *= -1
        print(f"Time reversed. Speed: {sim.TIME_CURRENT_SPEED:.2f}")
    elif event.key == "m":
        with sim.computation_lock:
            if SOLVER_A_FIRST:
                sim.gravity_solver = ComparisonSolver(
                    test_solver=SOLVER_A_TYPE(G=GRAVITY_CONSTANT),
                    reference_solver=SOLVER_B_TYPE(G=GRAVITY_CONSTANT),
                )
            else:
                sim.gravity_solver = ComparisonSolver(
                    test_solver=SOLVER_B_TYPE(G=GRAVITY_CONSTANT),
                    reference_solver=SOLVER_A_TYPE(G=GRAVITY_CONSTANT),
                )
            SOLVER_A_FIRST = not SOLVER_A_FIRST
            sim.needs_update = True


if __name__ == "__main__":
    # Create initial simulation
    sim = create_new_simulation()

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()

    frame_count = 0
    azim, elev = 45, 30
    zoom_level = 1.0
    last_x = last_y = 0
    is_rotating = False

    # Start the computation thread
    sim.start_computation_thread()

    def handle_mouse_button(event):
        global is_rotating, last_x, last_y
        if event.button == 1:
            is_rotating = event.name == "button_press_event"
            if event.xdata is not None and event.ydata is not None:
                last_x = event.x
                last_y = event.y

    def handle_mouse_move(event):
        global is_rotating, azim, elev, last_x, last_y
        if is_rotating and hasattr(event, "x") and hasattr(event, "y"):
            dx = event.x - last_x
            dy = event.y - last_y
            azim += dx
            elev = np.clip(elev + dy, -90, 90)
            last_x = event.x
            last_y = event.y
            ax.view_init(elev=elev, azim=azim)
            fig.canvas.draw_idle()

    def handle_mouse_wheel(event):
        global zoom_level
        if event.button == "up":
            zoom_level = max(0.1, zoom_level * 0.9)
        elif event.button == "down":
            zoom_level = min(10.0, zoom_level * 1.1)

        padding = sim.WORLD_SIZE * 0.2 * zoom_level
        ax.set_xlim(-padding, sim.WORLD_SIZE + padding)
        ax.set_ylim(-padding, sim.WORLD_SIZE + padding)
        ax.set_zlim(-padding, sim.WORLD_SIZE + padding)
        fig.canvas.draw_idle()

    def update(frame):
        global frame_count
        if not sim.paused.is_set():
            sim.update(0.016)

        scatter = sim.visualize(ax)
        frame_count += 1

        title = f"Speed: {sim.TIME_CURRENT_SPEED:.4f}x \nSolver: {sim.gravity_solver}"

        ax.set_title(title)

        padding = sim.WORLD_SIZE * 0.2 * zoom_level
        ax.set_xlim(-padding, sim.WORLD_SIZE + padding)
        ax.set_ylim(-padding, sim.WORLD_SIZE + padding)
        ax.set_zlim(-padding, sim.WORLD_SIZE + padding)
        ax.view_init(elev=elev, azim=azim)

        return scatter

    # Connect event handlers
    fig.canvas.mpl_connect("button_press_event", handle_mouse_button)
    fig.canvas.mpl_connect("button_release_event", handle_mouse_button)
    fig.canvas.mpl_connect("motion_notify_event", handle_mouse_move)
    fig.canvas.mpl_connect("scroll_event", handle_mouse_wheel)
    fig.canvas.mpl_connect("key_press_event", handle_keyboard)

    anim = FuncAnimation(fig, update, interval=16, cache_frame_data=False, blit=False)

    help_text = (
        "Controls:\n"
        "Left Mouse: Rotate camera\n"
        "Scroll: Zoom in/out\n"
        "Space: Pause/Resume\n"
        "P: Toggle particles\n"
        "G: Toggle gradient field\n"
        "R: Reset view\n"
        "T: Reverse time\n"
        "M: Switch solver method\n"
        "Up/Down: Adjust speed"
    )
    plt.figtext(0.02, 0.02, help_text, fontsize=8, color="white")

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nSimulation interrupted")
    finally:
        sim.stop_computation_thread()
        plt.close("all")
