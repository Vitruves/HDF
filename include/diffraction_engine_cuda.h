#ifndef DIFFRACTION_ENGINE_CUDA_H
#define DIFFRACTION_ENGINE_CUDA_H

#include <complex.h>
#include "molecule.h"

// Only include CUDA headers in CUDA compilation or when HAS_CUDA is defined
#if defined(HAS_CUDA) || defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cuComplex.h>
#else
// Forward declaration of CUDA types
#ifndef CUDA_COMPLEX_DEFINED
typedef struct { double x, y; } cuDoubleComplex;
#define CUDA_COMPLEX_DEFINED
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

void optimize_molecule_layout_cuda(int iterations, double k_spring, double k_repulsive, 
                                  double damping_factor, double time_step_factor);

int cuda_check_available(void);

/* CUDA-accelerated function declarations */
void draw_molecule_on_grid_cuda(cuDoubleComplex *h_aperture_grid, int grid_width, 
                               AtomPos *h_atoms, int atom_count, 
                               BondSeg *h_bonds, int bond_count);

void compute_diffraction_pattern_cuda(cuDoubleComplex *h_aperture_grid, double *h_intensity, 
                                     int grid_width);

void apply_log_scale_cuda(double *h_intensity, double *h_scaled_intensity, 
                          int grid_width, double epsilon);

void optimize_molecule_layout_cuda_batched(
    AtomPos **h_atoms_batch_ptr_array,       // Array of pointers to AtomPos arrays
    int *h_atom_counts_batch,                // Array of atom counts for each molecule
    BondSeg **h_bonds_batch_ptr_array,       // Array of pointers to BondSeg arrays
    int *h_bond_counts_batch,                // Array of bond counts for each molecule
    int num_molecules_in_batch,              // Number of molecules in this batch
    int iterations, double k_spring, double k_repulsive,
    double damping_factor, double time_step_factor
);

#ifdef __cplusplus
}
#endif

#endif /* DIFFRACTION_ENGINE_CUDA_H */ 