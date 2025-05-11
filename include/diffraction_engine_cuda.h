#ifndef DIFFRACTION_ENGINE_CUDA_H
#define DIFFRACTION_ENGINE_CUDA_H

#include "molecule.h"
#include <stdbool.h>
#include <stddef.h>

// Make this header compatible with both C and CUDA compilation
#if defined(HAS_CUDA) || defined(__CUDACC__)
// When compiled with CUDA, use the real CUDA definitions
#include <cuda_runtime.h>
#include <cuComplex.h>
#else
// Define CUDA types for regular C compilation ONLY when CUDA is not available
typedef enum {
    cudaSuccess = 0
} cudaError_t;

typedef struct { double x, y; } cuDoubleComplex;
static inline cuDoubleComplex make_cuDoubleComplex(double r, double i) {
    cuDoubleComplex c;
    c.x = r;
    c.y = i;
    return c;
}
#endif

// Include complex.h after handling CUDA to avoid conflicts
#ifdef __cplusplus
#include <complex>
#else
#include <complex.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// CUDA functions with C linkage
bool cuda_generate_diffraction_pattern(float *d_diffraction_pattern, int width, int height,
                                    AtomPos *atoms, int atom_count,
                                    double min_q, double max_q, double incident_energy);

// CUDA support functions
int check_cuda_available(void);
int cuda_check_available(void);
void* cuda_malloc(size_t size);
void cuda_free(void *ptr);
bool cuda_copy_to_device(void *dst, const void *src, size_t size);
bool cuda_copy_to_host(void *dst, const void *src, size_t size);

// Molecule layout and processing functions
void optimize_molecule_layout_cuda(int iterations, double k_spring, double k_repulsive, 
                                double damping_factor, double time_step_factor);
void draw_molecule_on_grid_cuda(cuDoubleComplex* cu_grid, int grid_width, AtomPos* atoms, int atom_count, 
                             BondSeg* bonds, int bond_count);
void compute_diffraction_pattern_cuda(cuDoubleComplex* cu_data, double* intensity, int width);
void apply_log_scale_cuda(double* intensity, double* scaled_intensity, int width, double epsilon);
void optimize_molecule_layout_cuda_batched(AtomPos** atoms_batch_ptr_array, int* atom_counts_batch,
                                        BondSeg** bonds_batch_ptr_array, int* bond_counts_batch,
                                        int num_molecules_in_batch, int iterations, 
                                        double k_spring, double k_repulsive,
                                        double damping_factor, double time_step_factor);

#ifdef __cplusplus
}
#endif

#endif // DIFFRACTION_ENGINE_CUDA_H 