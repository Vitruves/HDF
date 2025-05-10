#ifndef DIFFRACTION_ENGINE_H
#define DIFFRACTION_ENGINE_H

#include "molecule.h"
#include <stdbool.h>
#include <complex.h>
#include <fftw3.h>

// Define a type to use for proper FFTW access
typedef double fftw_complex_t[2];
typedef fftw_complex_t *fftw_complex_array;

void optimize_molecule_layout(int iterations, double k_spring, double k_repulsive, double damping_factor, double time_step_factor);
void draw_atom_on_grid(complex double *aperture_grid, int grid_width, AtomPos atom, int atom_idx, bool use_quantum_model);
void draw_bond_on_grid(complex double *aperture_grid, int grid_width, BondSeg bond, bool use_quantum_model);
void add_molecular_orbital_effects(complex double *aperture_grid, int grid_width);
void fft_2d(complex double *data, int width, int height, int direction);
void fft_shift_2d(complex double *data, int width, int height);

// Fingerprint condensation
void condense_fingerprint_average(const double *input_fp, int input_w, int input_h, int block_size, double *output_fp, int *output_w, int *output_h);


#endif // DIFFRACTION_ENGINE_H