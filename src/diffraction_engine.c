// Simplify CUDA handling
#if defined(HAS_CUDA) || defined(__CUDACC__)
// Use real CUDA headers
#include <cuda_runtime.h>
#include <cuComplex.h>
#else
// Define minimal CUDA types when not using real CUDA
typedef enum {
    cudaSuccess = 0
} cudaError_t;
#define cudaFree(ptr) (cudaSuccess)

#ifndef CUDA_COMPLEX_DEFINED
typedef struct { double x, y; } cuDoubleComplex;
static cuDoubleComplex make_cuDoubleComplex(double x, double y) {
    cuDoubleComplex z;
    z.x = x;
    z.y = y;
    return z;
}
#define CUDA_COMPLEX_DEFINED
#endif
#endif

#include "diffraction_engine.h"
#include "globals.h" // Access to atoms, bonds, atom_count, bond_count
#include "quantum_engine.h" // For QM calculations if use_quantum_model is true
#include "utils.h"   // For PI, hash_string, clamp_double, min_double
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>    // For sqrt, cos, sin, exp, ceil, fabs, atan2, fmod, cbrt
#include <fftw3.h>   // FFTW3 library for efficient FFT

// Include this after we've handled the CUDA types
#include "diffraction_engine_cuda.h"

// Add this variable to track if CUDA is available
static int cuda_available = -1; // -1: not checked, 0: not available, 1: available

// Function to check CUDA availability (declared in diffraction_engine_cuda.h)
extern int cuda_check_available(void);

// Wrapper for checking CUDA availability
static int check_cuda_available() {
    // If CPU is forced, return 0
    if (use_cuda == 0) {
        printf("CPU mode selected. Using CPU implementation.\n");
        return 0;
    }

    if (cuda_available == -1) {
        // Try to initialize CUDA
        cuda_available = cuda_check_available();
        
        if (cuda_available) {
            printf("CUDA accelerated layout optimization enabled (batch size: %d)\n", cuda_batch_size);
        } else {
            if (use_cuda == 2) {  // CUDA was forced
                fprintf(stderr, "Error: CUDA was forced but is not available.\n");
                exit(1);
            }
            printf("CUDA not available, using CPU implementation\n");
        }
    }
    return cuda_available;
}

// Classical phase calculation (simpler model if QM is off)
static double calculate_atom_phase_classical(AtomPos atom, int idx) {
    double base_phase = (atom.atomic_number * PI) / 36.0;
    base_phase += (atom.isotope % 10) * PI / 20.0; // Isotope effect
    base_phase += atom.explicit_h_count * PI / 30.0; // H count effect

    double en_factor = atom.electronegativity / 4.0;
    double valence_effect = (atom.valence / 8.0) * PI;
    unsigned int hash = hash_string(atom.atom, idx + atom.isotope); // Salt with isotope
    double position_salt = (hash % 1000) / 1000.0;
    double charge_effect = (atom.charge != 0) ? atom.charge * PI / 6.0 : 0.0; // Slightly stronger charge effect
    double aromatic_shift = atom.is_aromatic ? PI / 6.0 : 0.0;
    double ring_shift = atom.in_ring ? PI / 10.0 : 0.0; // Effect if atom is in any ring
    
    return fmod(base_phase + en_factor * PI + valence_effect + 
                position_salt * PI / 4.0 + charge_effect + aromatic_shift + ring_shift, 
                2 * PI);
}

static double calculate_bond_phase_classical(BondSeg bond) {
    double order_phase = bond.order * PI / 4.0;
    double type_phase = 0.0;
    if (bond.type == BOND_AROMATIC) {
        type_phase = PI / 3.0;
    }
    // Accessing atom properties for the bond
    AtomPos atom_a = atoms[bond.a];
    AtomPos atom_b = atoms[bond.b];
    double charge_diff_effect = abs(atom_a.charge - atom_b.charge) * PI / 12.0;

    double ring_effect = bond.in_ring ? PI / 5.0 : 0.0;
    double length_effect = (bond.length > 0.1 ? (bond.length - 1.0) : 0.0) * PI / 10.0; 
    
    return fmod(order_phase + type_phase + ring_effect + length_effect + charge_diff_effect, 2 * PI);
}


void optimize_molecule_layout(int iterations, double k_spring, double k_repulsive, double damping_factor, double time_step_factor) {
    // Try to use CUDA if available
    if (check_cuda_available()) {
        optimize_molecule_layout_cuda(iterations, k_spring, k_repulsive, damping_factor, time_step_factor);
        return;
    }
    
    // Original CPU implementation if CUDA is not available
    if (atom_count == 0) return;

    double *forces_x = calloc(atom_count, sizeof(double));
    double *forces_y = calloc(atom_count, sizeof(double));
    double *forces_z = calloc(atom_count, sizeof(double)); // For 3D adjustments

    if (!forces_x || !forces_y || !forces_z) {
        fprintf(stderr, "Error: Memory allocation failed for layout forces.\n");
        free(forces_x); free(forces_y); free(forces_z);
        return;
    }

    for (int iter = 0; iter < iterations; iter++) {
        memset(forces_x, 0, atom_count * sizeof(double));
        memset(forces_y, 0, atom_count * sizeof(double));
        memset(forces_z, 0, atom_count * sizeof(double)); // Initialize Z forces

        // Repulsive forces (non-bonded atoms) - now in 3D
        for (int i = 0; i < atom_count; i++) {
            for (int j = i + 1; j < atom_count; j++) {
                double dx = atoms[j].x - atoms[i].x;
                double dy = atoms[j].y - atoms[i].y;
                double dz = atoms[j].z - atoms[i].z; // Z difference
                double dist_sq = dx*dx + dy*dy + dz*dz; // 3D distance squared
                
                if (dist_sq < 0.001) dist_sq = 0.001; 
                double dist = sqrt(dist_sq);
                
                double force_mag = k_repulsive / dist_sq; 
                
                forces_x[i] -= force_mag * (dx / dist);
                forces_y[i] -= force_mag * (dy / dist);
                forces_z[i] -= force_mag * (dz / dist); // Apply Z force
                forces_x[j] += force_mag * (dx / dist);
                forces_y[j] += force_mag * (dy / dist);
                forces_z[j] += force_mag * (dz / dist); // Apply Z force
            }
        }

        // Spring forces (bonded atoms) - now in 3D
        for (int i = 0; i < bond_count; i++) {
            int u = bonds[i].a;
            int v = bonds[i].b;
            
            double ideal_length = atoms[u].radius + atoms[v].radius;
            if (bonds[i].order == 2) ideal_length *= 0.85;
            else if (bonds[i].order == 3) ideal_length *= 0.75;
            if (bonds[i].type == BOND_AROMATIC) ideal_length *= 0.90; 

            if (ideal_length < 0.1) ideal_length = 1.0;

            double dx = atoms[v].x - atoms[u].x;
            double dy = atoms[v].y - atoms[u].y;
            double dz = atoms[v].z - atoms[u].z; // Z difference
            double dist_sq = dx*dx + dy*dy + dz*dz; // 3D distance squared

            if (dist_sq < 0.0001) dist_sq = 0.0001; 
            double dist = sqrt(dist_sq);
            
            double force_mag = k_spring * (dist - ideal_length); 
            
            double fx_comp = force_mag * (dx / dist);
            double fy_comp = force_mag * (dy / dist);
            double fz_comp = force_mag * (dz / dist); // Z component of spring force

            forces_x[u] += fx_comp;
            forces_y[u] += fy_comp;
            forces_z[u] += fz_comp; // Apply Z force
            forces_x[v] -= fx_comp;
            forces_y[v] -= fy_comp;
            forces_z[v] -= fz_comp; // Apply Z force
        }

        // Update positions (including Z)
        for (int i = 0; i < atom_count; i++) {
            double displacement_x = forces_x[i] * time_step_factor;
            double displacement_y = forces_y[i] * time_step_factor;
            double displacement_z = forces_z[i] * time_step_factor; // Z displacement
            double max_disp = 0.5; 
            
            if (fabs(displacement_x) > max_disp) displacement_x = copysign(max_disp, displacement_x);
            if (fabs(displacement_y) > max_disp) displacement_y = copysign(max_disp, displacement_y);
            if (fabs(displacement_z) > max_disp) displacement_z = copysign(max_disp, displacement_z); // Cap Z displacement

            atoms[i].x += displacement_x * damping_factor;
            atoms[i].y += displacement_y * damping_factor;
            atoms[i].z += displacement_z * damping_factor; // Update Z position
        }
    }
    free(forces_x);
    free(forces_y);
    free(forces_z);

    // Recalculate actual bond lengths after optimization (already 3D)
    for (int i = 0; i < bond_count; i++) {
        double dx = atoms[bonds[i].b].x - atoms[bonds[i].a].x;
        double dy = atoms[bonds[i].b].y - atoms[bonds[i].a].y;
        double dz = atoms[bonds[i].b].z - atoms[bonds[i].a].z;
        bonds[i].length = sqrt(dx*dx + dy*dy + dz*dz);
    }
     // Center the molecule (optional, now considers Z for centering if desired, but usually center X,Y)
    if (atom_count > 0) {
        double sum_x = 0, sum_y = 0, sum_z = 0;
        for (int i = 0; i < atom_count; i++) {
            sum_x += atoms[i].x;
            sum_y += atoms[i].y;
            sum_z += atoms[i].z;
        }
        double centroid_x = sum_x / atom_count;
        double centroid_y = sum_y / atom_count;
        double centroid_z = sum_z / atom_count; // Centroid Z
        for (int i = 0; i < atom_count; i++) {
            atoms[i].x -= centroid_x;
            atoms[i].y -= centroid_y;
            atoms[i].z -= centroid_z; // Center Z as well
        }
    }
}

// Define conversion functions for cuDoubleComplex <-> complex double
static complex double cuDoubleComplex_to_complex(cuDoubleComplex z) {
    return z.x + z.y * I;
}

static cuDoubleComplex complex_to_cuDoubleComplex(complex double z) {
    return make_cuDoubleComplex(creal(z), cimag(z));
}

// Modify draw_atom_on_grid and draw_bond_on_grid to use CUDA when available
void draw_atom_on_grid(complex double *aperture_grid, int grid_width, AtomPos atom, int atom_idx, bool use_quantum_model) {
    double phase = use_quantum_model ? 
        calculate_atom_phase_qm(atom, atom_idx) : 
        calculate_atom_phase_classical(atom, atom_idx);
    
    // Perspective projection: scale factor based on Z
    // Assume focal length is related to grid_width, e.g., grid_width or grid_width/2
    double focal_length = (double)grid_width; 
    double z_depth_effect = focal_length / (focal_length + atom.z * 5.0); // Adjust '5.0' for sensitivity to Z
    if (z_depth_effect <= 0.1) z_depth_effect = 0.1; // Prevent extreme scaling or inversion
    if (z_depth_effect >= 2.0) z_depth_effect = 2.0; // Cap max scaling

    double scale_factor = (grid_width / 8.0) * z_depth_effect; // Apply perspective scaling
    int center_x = (int)(atom.x * scale_factor + grid_width / 2.0);
    int center_y = (int)(atom.y * scale_factor + grid_width / 2.0);

    double base_amplitude;
    double atom_display_radius_px; 

    if (use_quantum_model) {
        base_amplitude = 1.0 + 0.1 * atom.effective_nuclear_charge + 0.1 * atom.valence;
        base_amplitude += 0.05 * (atom.isotope % 10); // Isotope effect on amplitude
        base_amplitude += 0.02 * atom.explicit_h_count;
        base_amplitude *= (1.0 + 0.1 * atom.charge); // Charge effect
        if(atom.in_ring) base_amplitude *= 1.05; // Slightly emphasize ring atoms

        atom_display_radius_px = (atom.radius * scale_factor) * (1.0 + 0.3 * atom.valence + 0.1 * atom.hybridization);
        atom_display_radius_px = min_double(5.0 * z_depth_effect, atom_display_radius_px); 
    } else {
        base_amplitude = 1.0 + 0.2 * atom.atomic_number / 10.0;
        base_amplitude += 0.1 * (atom.isotope % 10);
        base_amplitude += 0.05 * atom.explicit_h_count;
        base_amplitude *= (1.0 + 0.15 * atom.charge);
        if(atom.in_ring) base_amplitude *= 1.1;

        atom_display_radius_px = (3.0 + 5.0 * atom.radius) * (grid_width / 512.0) * z_depth_effect;
    }
    
    // Ensure radius isn't too small due to z_depth_effect
    if (atom_display_radius_px < 1.0) atom_display_radius_px = 1.0;

    double sigma_px = atom_display_radius_px / 2.5; 
    int effect_radius_px_int = (int)ceil(atom_display_radius_px * 1.5); 

    for (int dy_px = -effect_radius_px_int; dy_px <= effect_radius_px_int; dy_px++) {
        for (int dx_px = -effect_radius_px_int; dx_px <= effect_radius_px_int; dx_px++) {
            int current_grid_x = center_x + dx_px;
            int current_grid_y = center_y + dy_px;

            if (current_grid_x < 0 || current_grid_x >= grid_width || current_grid_y < 0 || current_grid_y >= grid_width) {
                continue;
            }

            double val;
            if (use_quantum_model) {
                double rel_x_angstrom = (double)dx_px / scale_factor;
                double rel_y_angstrom = (double)dy_px / scale_factor;
                // We could pass atom.z directly if electron_density takes absolute z
                // or calculate relative z if needed. Here, assume electron_density expects relative coords.
                double density_at_point = electron_density(atom, rel_x_angstrom, rel_y_angstrom, 0.0); 
                val = base_amplitude * density_at_point * 1e-2; 
            } else {
                double r_sq_px = dx_px*dx_px + dy_px*dy_px;
                if (sigma_px < 1e-6) sigma_px = 1.0; 
                val = base_amplitude * exp(-r_sq_px / (2.0 * sigma_px * sigma_px));
            }
            
            // Modulate phase slightly by Z-coordinate to create a subtle depth "shimmer"
            complex double z_phase_mod = cexp(I * atom.z * 0.1); // Adjust 0.1 for sensitivity
            int grid_idx = current_grid_y * grid_width + current_grid_x;
            aperture_grid[grid_idx] += val * cexp(I * phase) * z_phase_mod;
        }
    }
}

void draw_bond_on_grid(complex double *aperture_grid, int grid_width, BondSeg bond, bool use_quantum_model) {
    AtomPos atom_A = atoms[bond.a];
    AtomPos atom_B = atoms[bond.b];

    double focal_length = (double)grid_width;
    double z_A_effect = focal_length / (focal_length + atom_A.z * 5.0);
    double z_B_effect = focal_length / (focal_length + atom_B.z * 5.0);
    z_A_effect = clamp_double(z_A_effect, 0.1, 2.0);
    z_B_effect = clamp_double(z_B_effect, 0.1, 2.0);

    double scale_factor_A = (grid_width / 8.0) * z_A_effect;
    double scale_factor_B = (grid_width / 8.0) * z_B_effect;

    int x0 = (int)(atom_A.x * scale_factor_A + grid_width / 2.0);
    int y0 = (int)(atom_A.y * scale_factor_A + grid_width / 2.0);
    int x1 = (int)(atom_B.x * scale_factor_B + grid_width / 2.0);
    int y1 = (int)(atom_B.y * scale_factor_B + grid_width / 2.0);

    double phase_val; // Renamed to avoid conflict with `phase` from cexp(I*phase_val)
    double base_amp;

    if (use_quantum_model) {
        phase_val = calculate_bond_phase_qm(bond);
        double en_diff = fabs(atom_A.electronegativity - atom_B.electronegativity);
        double overlap_factor = 1.0 - 0.1 * en_diff; 
        base_amp = (0.4 + 0.1 * bond.order) * overlap_factor;
        if (bond.type == BOND_AROMATIC) base_amp = 0.6 * (1.0 + 0.1 * atom_A.is_aromatic * atom_B.is_aromatic);
        if (bond.in_ring) base_amp *= 1.05;

    } else {
        phase_val = calculate_bond_phase_classical(bond);
        base_amp = 0.5 + 0.15 * bond.order;
        if (bond.type == BOND_AROMATIC) base_amp = 0.7;
        if (bond.in_ring) base_amp *= 1.1;
    }
    complex double bond_phase_factor = cexp(I * phase_val);

    int dx_total = abs(x1 - x0);
    int dy_total = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx_total - dy_total;

    int bond_pixel_width = bond.order; 
    if (bond.type == BOND_AROMATIC) bond_pixel_width = 2; 
    // Dynamic bond width based on average Z depth effect of its atoms
    double avg_z_effect_for_bond = (z_A_effect + z_B_effect) / 2.0;
    bond_pixel_width = (int)fmax(1.0, bond_pixel_width * (grid_width / 512.0) * avg_z_effect_for_bond);


    int current_x = x0;
    int current_y = y0;
    double total_dist_px = sqrt(dx_total*dx_total + dy_total*dy_total);
    if (total_dist_px < 1.0) total_dist_px = 1.0;

    while (1) {
        double dist_from_start_px = sqrt(pow(current_x - x0, 2) + pow(current_y - y0, 2));
        double bond_pos_fraction = dist_from_start_px / total_dist_px;
        bond_pos_fraction = clamp_double(bond_pos_fraction, 0.0, 1.0);

        // Interpolate Z for phase effect along the bond
        double current_z = atom_A.z * (1.0 - bond_pos_fraction) + atom_B.z * bond_pos_fraction;
        complex double z_phase_mod_bond = cexp(I * current_z * 0.05); // Softer Z phase for bonds

        double current_amp = base_amp;
        if (use_quantum_model) {
            double sigma_shape = 1.0 + 0.2 * (1.0 - fabs(bond_pos_fraction - 0.5) * 2.0); 
            double pi_shape = 0.0;
            if (bond.order > 1 || bond.type == BOND_AROMATIC) {
                pi_shape = 0.15 * sin(bond_pos_fraction * (bond.order == 2 ? 2.0:3.0) * PI); 
            }
            current_amp *= (sigma_shape + pi_shape);
            double en_diff_val = atom_A.electronegativity - atom_B.electronegativity;
            current_amp *= (1.0 - 0.1 * en_diff_val * (bond_pos_fraction - 0.5)); 
        }
        
        for (int wy = -bond_pixel_width / 2; wy <= bond_pixel_width / 2; ++wy) {
            for (int wx = -bond_pixel_width / 2; wx <= bond_pixel_width / 2; ++wx) {
                int px = current_x, py = current_y;
                if (dx_total > dy_total) { py += wy; } 
                else { px += wx; } 

                if (px >= 0 && px < grid_width && py >= 0 && py < grid_width) {
                    aperture_grid[py * grid_width + px] += current_amp * bond_phase_factor * z_phase_mod_bond;
                }
            }
        }
        
        if (current_x == x1 && current_y == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy_total) { err -= dy_total; current_x += sx; }
        if (e2 < dx_total) { err += dx_total; current_y += sy; }
    }
}

// Add a new function that uses CUDA for drawing molecules
void draw_molecule_on_grid(complex double *aperture_grid, int grid_width, bool use_quantum_model) {
    if (check_cuda_available() && !use_quantum_model) {
        // Use CUDA implementation for classical model
        cuDoubleComplex *cu_aperture_grid = malloc(grid_width * grid_width * sizeof(cuDoubleComplex));
        if (!cu_aperture_grid) {
            fprintf(stderr, "Error: Failed to allocate memory for CUDA aperture grid\n");
            return;
        }
        
        // Initialize aperture grid
        for (int i = 0; i < grid_width * grid_width; i++) {
            cu_aperture_grid[i] = complex_to_cuDoubleComplex(aperture_grid[i]);
        }
        
        // Call CUDA implementation
        draw_molecule_on_grid_cuda(cu_aperture_grid, grid_width, atoms, atom_count, bonds, bond_count);
        
        // Copy back results
        for (int i = 0; i < grid_width * grid_width; i++) {
            aperture_grid[i] = cuDoubleComplex_to_complex(cu_aperture_grid[i]);
        }
        
        free(cu_aperture_grid);
    } else {
        // Use CPU implementation
        for (int i = 0; i < atom_count; ++i) {
            draw_atom_on_grid(aperture_grid, grid_width, atoms[i], i, use_quantum_model);
        }
        for (int i = 0; i < bond_count; ++i) {
            draw_bond_on_grid(aperture_grid, grid_width, bonds[i], use_quantum_model);
        }
    }
}

// Modify fft_2d to use CUDA when available
void fft_2d(complex double *data, int width, int height, int direction) {
    if (check_cuda_available() && direction == 1) { // Forward FFT
        // Use CUDA implementation for forward FFT and intensity calculation
        cuDoubleComplex *cu_data = malloc(width * height * sizeof(cuDoubleComplex));
        double *intensity = malloc(width * height * sizeof(double));
        
        if (!cu_data || !intensity) {
            fprintf(stderr, "Error: Failed to allocate memory for CUDA FFT\n");
            if (cu_data) free(cu_data);
            if (intensity) free(intensity);
            return;
        }
        
        // Initialize data
        for (int i = 0; i < width * height; i++) {
            cu_data[i] = complex_to_cuDoubleComplex(data[i]);
        }
        
        // Call CUDA implementation
        compute_diffraction_pattern_cuda(cu_data, intensity, width);
        
        // Convert intensity back to complex format
        for (int i = 0; i < width * height; i++) {
            data[i] = sqrt(intensity[i]); // Just put magnitude in real part
        }
        
        free(cu_data);
        free(intensity);
        
        // Apply fftshift
        fft_shift_2d(data, width, height);
    } else {
        // Use existing FFTW implementation
        fftw_complex *fftw_data = (fftw_complex*)data;
        fftw_plan plan = fftw_plan_dft_2d(width, height, fftw_data, fftw_data, 
                                          direction == 1 ? FFTW_FORWARD : FFTW_BACKWARD, 
                                          FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);

        // Normalize if it's an inverse FFT
        if (direction == -1) {
            for (int i = 0; i < width * height; i++) {
                data[i] /= (width * height);
            }
        }
        // Apply fftshift if it is a forward FFT and not handled by CUDA part
        if (direction == 1) {
            fft_shift_2d(data, width, height);
        }
    }
}

// Add the definition for fft_shift_2d here
void fft_shift_2d(complex double *data, int width, int height) {
    int half_width = width / 2;
    int half_height = height / 2;

    complex double *temp_quad = (complex double*)malloc(half_width * half_height * sizeof(complex double));
    if (!temp_quad) {
        fprintf(stderr, "Error: Failed to allocate memory for fft_shift_2d temp buffer.\n");
        return;
    }

    // Swap quadrants
    // Quadrant 1 (top-left) with Quadrant 3 (bottom-right)
    for (int r = 0; r < half_height; ++r) {
        for (int c = 0; c < half_width; ++c) {
            temp_quad[r * half_width + c] = data[r * width + c]; // Store Q1
            data[r * width + c] = data[(r + half_height) * width + (c + half_width)]; // Move Q3 to Q1
            data[(r + half_height) * width + (c + half_width)] = temp_quad[r * half_width + c]; // Move stored Q1 to Q3
        }
    }

    // Quadrant 2 (top-right) with Quadrant 4 (bottom-left)
    for (int r = 0; r < half_height; ++r) {
        for (int c = 0; c < half_width; ++c) {
            temp_quad[r * half_width + c] = data[r * width + (c + half_width)]; // Store Q2
            data[r * width + (c + half_width)] = data[(r + half_height) * width + c]; // Move Q4 to Q2
            data[(r + half_height) * width + c] = temp_quad[r * half_width + c]; // Move stored Q2 to Q4
        }
    }
    free(temp_quad);
}

// Add a new function for log scaling with CUDA
void apply_log_scale_intensity_cuda(double *intensity, double *scaled_intensity, 
                                  int width, int height, double max_intensity, double epsilon) {
    if (check_cuda_available()) {
        apply_log_scale_cuda(intensity, scaled_intensity, width, epsilon);
    } else {
        // CPU implementation
        for (int i = 0; i < width * height; i++) {
            double ratio = intensity[i] / max_intensity;
            scaled_intensity[i] = log10(ratio + epsilon) / log10(1.0 + epsilon);
        }
    }
}

void add_molecular_orbital_effects(complex double *aperture_grid, int grid_width) {
    if (atom_count == 0) return;

    int *conjugated_system_map = calloc(atom_count, sizeof(int));
    if (!conjugated_system_map) { /* ... error ... */ return; }
    int current_system_id = 0;

    // Identify conjugated systems (simplified)
    for (int i = 0; i < atom_count; i++) {
        if (atoms[i].is_aromatic && conjugated_system_map[i] == 0) {
            current_system_id++;
            int *queue = malloc(atom_count * sizeof(int));
            int head = 0, tail = 0;
            if (!queue) { free(conjugated_system_map); /* ... error ... */ return; }
            queue[tail++] = i;
            conjugated_system_map[i] = current_system_id;
            while(head < tail) {
                int u = queue[head++];
                for (int b_idx = 0; b_idx < bond_count; ++b_idx) {
                    if (bonds[b_idx].type == BOND_AROMATIC || 
                        ( (atoms[bonds[b_idx].a].is_aromatic || atoms[bonds[b_idx].b].is_aromatic) && bonds[b_idx].order >=1 ) ) {
                        int v = -1;
                        if (bonds[b_idx].a == u) v = bonds[b_idx].b;
                        else if (bonds[b_idx].b == u) v = bonds[b_idx].a;
                        if (v != -1 && atoms[v].is_aromatic && conjugated_system_map[v] == 0) {
                            conjugated_system_map[v] = current_system_id;
                            if (tail < atom_count) queue[tail++] = v;
                        }
                    }
                }
            }
            free(queue);
        }
    }

    double focal_length = (double)grid_width;

    for (int sys_id = 1; sys_id <= current_system_id; sys_id++) {
        double com_x = 0, com_y = 0, com_z = 0; // Include Z for system center
        int num_atoms_in_system = 0;
        double max_extent_sq = 0;

        for (int i = 0; i < atom_count; i++) {
            if (conjugated_system_map[i] == sys_id) {
                com_x += atoms[i].x;
                com_y += atoms[i].y;
                com_z += atoms[i].z; // Sum Z coordinates
                num_atoms_in_system++;
            }
        }

        if (num_atoms_in_system == 0) continue;
        com_x /= num_atoms_in_system;
        com_y /= num_atoms_in_system;
        com_z /= num_atoms_in_system; // Average Z for the system

        double system_z_depth_effect = focal_length / (focal_length + com_z * 5.0);
        system_z_depth_effect = clamp_double(system_z_depth_effect, 0.1, 2.0);

        for (int i = 0; i < atom_count; i++) {
            if (conjugated_system_map[i] == sys_id) {
                double dx = atoms[i].x - com_x;
                double dy = atoms[i].y - com_y;
                // Could also consider dz relative to com_z if needed for extent
                if (dx*dx + dy*dy > max_extent_sq) {
                    max_extent_sq = dx*dx + dy*dy;
                }
            }
        }
        double system_radius_angstrom = sqrt(max_extent_sq) + 1.0; 
        double system_radius_px = system_radius_angstrom * (grid_width / 8.0) * system_z_depth_effect; // Apply Z effect to radius

        int center_x_px = (int)(com_x * (grid_width / 8.0) * system_z_depth_effect + grid_width / 2.0); // Apply Z effect to center
        int center_y_px = (int)(com_y * (grid_width / 8.0) * system_z_depth_effect + grid_width / 2.0);
        int effect_radius_px = (int)ceil(system_radius_px * 1.2); 

        double mo_phase = sys_id * PI / (current_system_id + 1.0); 
        double mo_amplitude = 0.1 * num_atoms_in_system; 
        // Modulate MO amplitude by Z - closer systems appear brighter/stronger
        mo_amplitude *= system_z_depth_effect;


        for (int dy_px = -effect_radius_px; dy_px <= effect_radius_px; dy_px++) {
            for (int dx_px = -effect_radius_px; dx_px <= effect_radius_px; dx_px++) {
                int gx = center_x_px + dx_px;
                int gy = center_y_px + dy_px;
                if (gx < 0 || gx >= grid_width || gy < 0 || gy >= grid_width) continue;

                double r_px = sqrt(dx_px*dx_px + dy_px*dy_px);
                if (r_px > system_radius_px * 1.1) continue; 

                double val = mo_amplitude * exp(-(r_px*r_px) / (2.0 * system_radius_px * system_radius_px * 0.5));
                
                // Add a slight phase shift based on the system's average Z for depth feel
                complex double system_z_phase_mod = cexp(I * com_z * 0.03);

                aperture_grid[gy * grid_width + gx] += val * cexp(I * mo_phase) * system_z_phase_mod;
            }
        }
    }
    free(conjugated_system_map);
}

void condense_fingerprint_average(const double *input_fp, int input_w, int input_h, int block_size, double *output_fp, int *output_w, int *output_h) {
    if (block_size <= 1) { // No condensation or invalid block_size
        *output_w = input_w;
        *output_h = input_h;
        for(int i=0; i < input_w * input_h; ++i) output_fp[i] = input_fp[i];
        return;
    }

    *output_w = input_w / block_size;
    *output_h = input_h / block_size;

    for (int oy = 0; oy < *output_h; ++oy) {
        for (int ox = 0; ox < *output_w; ++ox) {
            double sum = 0.0;
            int count = 0;
            for (int by = 0; by < block_size; ++by) {
                for (int bx = 0; bx < block_size; ++bx) {
                    int ix = ox * block_size + bx;
                    int iy = oy * block_size + by;
                    if (ix < input_w && iy < input_h) { // Should always be true if input_w/h are divisible by block_size
                        sum += input_fp[iy * input_w + ix];
                        count++;
                    }
                }
            }
            if (count > 0) {
                output_fp[oy * (*output_w) + ox] = sum / count;
            } else {
                output_fp[oy * (*output_w) + ox] = 0.0; // Should not happen with perfect division
            }
        }
    }
}

void optimize_molecule_layout_batch(
    AtomPos **atoms_batch_ptr_array,
    int *atom_counts_batch,
    BondSeg **bonds_batch_ptr_array,
    int *bond_counts_batch,
    int num_molecules_in_batch,
    int iterations, double k_spring, double k_repulsive,
    double damping_factor, double time_step_factor
) {
    if (check_cuda_available()) { // check_cuda_available internally checks global use_cuda and actual device presence
        // Call the batched CUDA function declared in diffraction_engine_cuda.h
        optimize_molecule_layout_cuda_batched(
            atoms_batch_ptr_array, atom_counts_batch,
            bonds_batch_ptr_array, bond_counts_batch,
            num_molecules_in_batch, iterations, k_spring, k_repulsive,
            damping_factor, time_step_factor
        );
    } else {
        // CPU Batch processing:
        // The existing optimize_molecule_layout function contains the CPU logic
        // and operates on the global `atoms`, `bonds`, `atom_count`, `bond_count`.
        // We need to temporarily set these globals for each molecule in the batch.

        // Save original global pointers and counts
        AtomPos* original_global_atoms_ptr = atoms;
        BondSeg* original_global_bonds_ptr = bonds;
        int original_global_atom_count = atom_count;
        int original_global_bond_count = bond_count;

        for (int i = 0; i < num_molecules_in_batch; ++i) {
            // Set globals for the current molecule
            atoms = atoms_batch_ptr_array[i]; // atoms_batch_ptr_array[i] should point to the actual data
            atom_count = atom_counts_batch[i];
            bonds = bonds_batch_ptr_array[i]; // bonds_batch_ptr_array[i] should point to the actual data
            bond_count = bond_counts_batch[i];

            // Call the existing single-molecule optimize_molecule_layout.
            // Since check_cuda_available() returned false, this will execute its CPU path.
            optimize_molecule_layout(iterations, k_spring, k_repulsive, damping_factor, time_step_factor);
            // The results are written directly into the memory pointed to by atoms_batch_ptr_array[i]
            // because the global `atoms` was pointing to it.
        }

        // Restore original global pointers and counts
        atoms = original_global_atoms_ptr;
        bonds = original_global_bonds_ptr;
        atom_count = original_global_atom_count;
        bond_count = original_global_bond_count;
    }
}