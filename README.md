# hdf: Holographic Diffraction Fingerprints Generator

`hdf` is a command-line application written in C that processes SMILES (Simplified Molecular Input Line Entry System) strings from a CSV file to generate holographic diffraction fingerprints. It can optionally render and save images of the simulated diffraction patterns.

The core process involves:
1.  Parsing SMILES strings to understand molecular structure.
2.  Optimizing the 2D/3D layout of the molecule.
3.  Simulating a diffraction pattern by drawing atoms and bonds onto a grid.
4.  Performing a 2D Fast Fourier Transform (FFT) on this grid.
5.  Generating a fingerprint from the resulting intensity pattern.

## Features

*   **SMILES Processing:** Parses a wide range of SMILES features including aromaticity, ring closures, branches, isotopes, charges, and basic stereochemistry.
*   **Layout Optimization:** Uses a force-directed algorithm to arrange atoms.
*   **Diffraction Simulation:**
    *   Generates a complex aperture grid representing the molecule.
    *   Considers atomic properties (Z, valence, electronegativity, etc.).
    *   Optional quantum mechanical effects for atom/bond phase and amplitude.
    *   Optional simplified molecular orbital effects.
    *   3D awareness in layout and diffraction based on SMILES stereochemistry.
*   **FFT Processing:** Utilizes the FFTW3 library for efficient 2D FFT.
*   **Fingerprint Generation:**
    *   Outputs log-scaled intensity values from the diffraction pattern.
    *   Option to condense fingerprints by averaging blocks.
    *   Flexible output format (space-separated string or multiple CSV columns).
*   **Image Output:**
    *   Optionally saves diffraction pattern images (PPM for color, PGM for grayscale).
    *   Multiple colormaps available (grayscale, jet, viridis, plasma, heat).
*   **Command-Line Interface:**
    *   Comprehensive CLI options for controlling all aspects of the simulation and output.
    *   Detailed `--help` message.
*   **Parallel Processing:** Supports multi-threaded processing using pthreads and OpenMP for improved performance on multi-core CPUs.
*   **Streaming Mode:** Efficiently processes large CSV files with low memory usage.

## Prerequisites (Ubuntu)

To build and run `hdf` on Ubuntu, you will need the following:

*   **C Compiler:** `gcc` (GNU Compiler Collection) or `clang`.
*   **Make:** The `make` utility.
*   **FFTW3 Library:** Development files for the FFTW3 library.
*   **(Optional) OpenMP:** If your compiler supports it (standard with modern GCC/Clang).
*   **(Optional) libatomic:** If your compiler requires explicit linking for C11 atomics (often not needed with modern GCC/Clang).

You can install these on a Debian-based system (like Ubuntu) using:

```bash
sudo apt update
sudo apt install build-essential libfftw3-dev
# build-essential usually includes gcc, g++, make, etc.
# libatomic1 might be needed if you encounter linker errors related to atomic operations,
# but typically GCC links it automatically when <stdatomic.h> is used.
```

## Building the Project

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <your-repo-url>
    # cd hdf
    ```

2.  **Compile the project using the Makefile:**
    ```bash
    make
    ```
    This will create an executable named `hdf` in the project's root directory.

3.  **Clean build files:**
    ```bash
    make clean
    ```

## Usage

The basic command to run the program is:

```bash
./hdf -i <input_file.csv> -o <output_file.csv> [options]
```

**Example:**

```bash
./hdf -i data/input_molecules.csv -o results/fingerprints.csv --output-dir results/images -r 1024 -j 4 --verbose
```

This command will:
*   Read SMILES from `data/input_molecules.csv`.
*   Save fingerprints to `results/fingerprints.csv`.
*   Save diffraction images to the `results/images/` directory.
*   Use a grid resolution of 1024x1024.
*   Use 4 parallel jobs for processing.
*   Enable verbose output.

### Command-Line Options

For a full list of available command-line options and their descriptions, run:

```bash
./hdf --help
```

Key options include:
*   `-i, --input-csv`: Path to input CSV.
*   `-o, --output-csv`: Path to output CSV.
*   `--output-dir`: Directory for saving images.
*   `-n, --no-images`: Suppress image generation.
*   `-r, --resolution`: Diffraction grid resolution.
*   `-c, --colormap`: Image colormap.
*   `-j, --jobs`: Number of parallel threads.
*   `-q, --quantum-model`: Enable quantum mechanical effects.
*   `-m, --mo-effects`: Enable molecular orbital effects.
*   `--streaming`: Force streaming mode.
*   `--column-format`: Output fingerprint as separate CSV columns.
*   `--condense-block-size`: Condense fingerprint.

## Development Notes

*   The project uses pthreads for parallel task management and can leverage OpenMP for loop-level parallelism within processing steps.
*   CPU affinity is set for worker threads on Linux to potentially improve cache performance.
*   Signal handling (SIGINT, SIGTERM) is implemented for graceful shutdown.

## Future Work

*   **CUDA Acceleration:** Porting computationally intensive sections (layout, FFT, grid drawing) to CUDA for significant performance gains on NVIDIA GPUs.