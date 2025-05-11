#ifndef DIFFRACTION_ENGINE_H
#define DIFFRACTION_ENGINE_H

#include <complex.h>
#include "molecule.h"
#include <stdbool.h>

// Function declarations
void optimize_molecule_layout(int iterations, double k_spring, double k_repulsive, 
                             double damping_factor, double time_step_factor);
void draw_molecule_on_grid(complex double *aperture_grid, int grid_width, bool use_quantum_model);
void fft_2d(complex double *data, int width, int height, int direction);
void fft_shift_2d(complex double *data, int width, int height);
void apply_log_scale_intensity_cuda(double *intensity, double *scaled_intensity, 
                                  int width, int height, double max_intensity, double epsilon);
void add_molecular_orbital_effects(complex double *aperture_grid, int grid_width);
void condense_fingerprint_average(const double *input_fp, int input_w, int input_h, 
                                 int block_size, double *output_fp, int *output_w, int *output_h);
void optimize_molecule_layout_batch(
    AtomPos **atoms_batch_ptr_array,
    int *atom_counts_batch,
    BondSeg **bonds_batch_ptr_array,
    int *bond_counts_batch,
    int num_molecules_in_batch,
    int iterations, double k_spring, double k_repulsive,
    double damping_factor, double time_step_factor
);
void enhance_diffraction_with_electronic_effects(complex double *aperture_grid, int grid_width, 
                                              AtomPos *atoms, int atom_count, 
                                              BondSeg *bonds, int bond_count);

// QM model calculation functions (will be defined in quantum_engine.c)
double calculate_atom_phase_qm(AtomPos atom, int idx);
double calculate_bond_phase_qm(BondSeg bond);
double electron_density(AtomPos atom, double rel_x, double rel_y, double rel_z);
complex double calculate_molecular_form_factor(AtomPos atom, double q_magnitude);

#endif /* DIFFRACTION_ENGINE_H */