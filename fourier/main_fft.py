from butterfly_fft import ButterflyFFT
from evaluate_fft import FFTEvaluator
from pattern_fft import TestPatternGenerator
from multidim_fft import FFTDimensionalSolver
from interface import FFTWorkGroupConfig


def main():
    # Create configuration for 3D FFT with proper work group sizes
    config = FFTWorkGroupConfig(
        dimensions=(32, 32, 32),  # 3D dimensions
        work_group_sizes=(8, 8, 8),  # Work group sizes for each dimension
        min_segment_size=4,
    )

    # Create implementation and solver
    fft_impl = ButterflyFFT(config)
    solver = FFTDimensionalSolver(fft_impl)

    # Create test pattern generator
    gen = TestPatternGenerator()

    # Generate various 3D test patterns
    patterns = {
        "mass_distribution": gen.generate_mass_distribution((32, 32, 32)),
        "sinusoidal": gen.generate_sinusoidal((32, 32, 32), [1, 2, 4]),
        "gaussian": gen.generate_gaussian_blob((32, 32, 32)),
    }

    # Evaluate implementation
    evaluator = FFTEvaluator(solver)

    for name, pattern in patterns.items():
        print(f"\nTesting {name} pattern:")
        errors = evaluator.compute_error(pattern)
        for metric, value in errors.items():
            print(f"{metric}: {value}")
        evaluator.visualize_comparison(pattern)


if __name__ == "__main__":
    main()
