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

// Wrapper for checking CUDA availability
int check_cuda_available(void) {
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
    // Base phase from atom properties - make more distinctive
    double base_phase = (atom.atomic_number * PI) / 18.0; // Increased sensitivity to atomic number
    
    // Add more diverse properties
    base_phase += (atom.isotope % 10) * PI / 15.0; // Stronger isotope effect
    base_phase += atom.explicit_h_count * PI / 20.0; // Enhanced H count effect
    base_phase += atom.ring_count * PI / 10.0; // Include ring membership information
    
    // Add pharmacophore information
    if (atom.is_pharmacophore) {
        base_phase += (atom.pharmacophore_type + 1) * PI / 12.0;
    }
    
    // Include hybridization information
    base_phase += atom.hybridization * PI / 8.0;
    
    // Electronegativity contribution - more sensitive
    double en_factor = atom.electronegativity / 3.0;
    
    // Valence contribution
    double valence_effect = (atom.valence / 6.0) * PI;
    
    // Position randomization - more sensitive to molecule structure
    unsigned int hash = hash_string(atom.atom, idx + atom.isotope + atom.n_bonds); 
    double position_salt = (hash % 1000) / 500.0;
    
    // Stronger charge effects
    double charge_effect = (atom.charge != 0) ? atom.charge * PI / 4.0 : 0.0;
    
    // Enhanced aromaticity and ring contributions
    double aromatic_shift = atom.is_aromatic ? PI / 4.0 : 0.0;
    double ring_shift = atom.in_ring ? PI / 8.0 : 0.0;
    
    // Add topological information
    double topo_effect = 0.0;
    if (atom.is_rotatable_bond_atom) topo_effect += PI / 6.0;
    if (atom.solvent_accessibility > 0.5) topo_effect += PI / 7.0;
    
    return fmod(base_phase + en_factor * PI + valence_effect + position_salt * PI / 2.0 + 
                charge_effect + aromatic_shift + ring_shift + topo_effect, 2 * PI);
}

static double calculate_bond_phase_classical(BondSeg bond) {
    // Enhance order-based phase
    double order_phase = bond.order * PI / 3.0;
    
    // More distinctive aromatic bond contributions
    double type_phase = 0.0;
    if (bond.type == BOND_AROMATIC) {
        type_phase = PI / 2.0;
    }
    
    // Access atom properties for more distinctive features
    AtomPos atom_a = atoms[bond.a];
    AtomPos atom_b = atoms[bond.b];
    
    // Enhanced charge difference effects
    double charge_diff_effect = abs(atom_a.charge - atom_b.charge) * PI / 8.0;
    
    // Add bond polarity based on electronegativity difference
    double en_diff = fabs(atom_a.electronegativity - atom_b.electronegativity);
    double polarity_effect = en_diff * PI / 7.0;
    
    // Enhanced ring effects
    double ring_effect = bond.in_ring ? PI / 4.0 : 0.0;
    
    // Bond conjugation effects
    double conjugation_effect = bond.is_conjugated ? PI / 5.0 : 0.0;
    
    // Rotatable bond information
    double rotatable_effect = bond.is_rotatable ? PI / 3.0 : 0.0;
    
    // Enhanced length effect - more sensitive to geometry
    double length_effect = (bond.length > 0.1 ? (bond.length - 1.0) * 2.0 : 0.0) * PI / 8.0;
    
    // Include bond energy for more chemical information
    double energy_effect = (bond.bond_energy / 100.0) * PI / 6.0;
    
    return fmod(order_phase + type_phase + ring_effect + length_effect + charge_diff_effect + 
                polarity_effect + conjugation_effect + rotatable_effect + energy_effect, 2 * PI);
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

#if defined(HAS_CUDA) || defined(__CUDACC__)
// When CUDA is available, use the make_cuDoubleComplex provided by CUDA headers
static cuDoubleComplex complex_to_cuDoubleComplex(complex double z) {
    return make_cuDoubleComplex(creal(z), cimag(z));
}
#else
// When CUDA is not available, use our own implementation
static cuDoubleComplex complex_to_cuDoubleComplex(complex double z) {
    cuDoubleComplex result;
    result.x = creal(z);
    result.y = cimag(z);
    return result;
}
#endif

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
        // Make amplitude more sensitive to atom-specific properties
        base_amplitude = 1.2 + 0.2 * atom.effective_nuclear_charge + 0.15 * atom.valence;
        base_amplitude += 0.08 * (atom.isotope % 10);
        base_amplitude += 0.05 * atom.explicit_h_count;
        base_amplitude *= (1.0 + 0.15 * atom.charge);
        
        // Enhance ring system differentiation
        if(atom.in_ring) base_amplitude *= (1.10 + 0.05 * atom.ring_count);
        
        // Add pharmacophore contribution
        if(atom.is_pharmacophore) base_amplitude *= (1.15 + 0.08 * atom.pharmacophore_type);
        
        // Add solvent accessibility contribution
        base_amplitude *= (1.0 + 0.12 * atom.solvent_accessibility);
        
        // More distinctive radius calculation
        atom_display_radius_px = (atom.radius * scale_factor) * 
                                (1.2 + 0.35 * atom.valence + 0.15 * atom.hybridization);
        if(atom.is_aromatic) atom_display_radius_px *= 1.15;
        
        // Cap max radius at a higher value for better discrimination
        atom_display_radius_px = min_double(6.0 * z_depth_effect, atom_display_radius_px);
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

// Modify the draw_molecule_on_grid function near line 429
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

    // Add 3D-specific phase modulations
    for (int i = 0; i < atom_count; i++) {
        // Create Z-dependent phase modulation
        double z_phase = atoms[i].z * 0.2; // Increased Z sensitivity
        
        // Calculate grid position of the atom with perspective scaling
        double focal_length = (double)grid_width;
        double z_depth_effect = focal_length / (focal_length + atoms[i].z * 5.0);
        z_depth_effect = clamp_double(z_depth_effect, 0.1, 2.0);
        double scale_factor = (grid_width / 8.0) * z_depth_effect;
        
        int center_x = (int)(atoms[i].x * scale_factor + grid_width / 2.0);
        int center_y = (int)(atoms[i].y * scale_factor + grid_width / 2.0);
        
        // Apply z-phase modulation to a small area around the atom
        int effect_radius = (int)(atoms[i].radius * scale_factor * 2.0);
        if (effect_radius < 1) effect_radius = 1;
        
        for (int dy = -effect_radius; dy <= effect_radius; dy++) {
            for (int dx = -effect_radius; dx <= effect_radius; dx++) {
                int px = center_x + dx;
                int py = center_y + dy;
                
                if (px >= 0 && px < grid_width && py >= 0 && py < grid_width) {
                    int idx = py * grid_width + px;
                    complex double z_modifier = cexp(I * z_phase);
                    aperture_grid[idx] *= z_modifier;
                }
            }
        }
    }

    // Add molecular orbital effects
    add_molecular_orbital_effects(aperture_grid, grid_width);
    
    // For quantum models, add enhanced electronic structure effects
    if (use_quantum_model) {
        // First, calculate partial charges if they haven't been calculated yet
        bool needs_charge_calculation = true;
        for (int i = 0; i < atom_count; i++) {
            if (atoms[i].partial_charge != 0.0) {
                needs_charge_calculation = false;
                break;
            }
        }
        
        if (needs_charge_calculation && atom_count > 0) {
            calculate_partial_charges(atoms, atom_count, bonds, bond_count);
            
            // Calculate atomic properties for quantum model
            for (int i = 0; i < atom_count; i++) {
                atoms[i].atomic_hardness = calculate_atomic_hardness(atoms[i]);
                atoms[i].atomic_softness = calculate_atomic_softness(atoms[i]);
                atoms[i].electronic_spatial_extent = calculate_electronic_spatial_extent(atoms[i]);
            }
        }
        
        // Add enhanced quantum electronic effects to the diffraction pattern
        enhance_diffraction_with_electronic_effects(aperture_grid, grid_width, atoms, atom_count, bonds, bond_count);
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

    // More advanced conjugated system detection
    int *conjugated_system_map = calloc(atom_count, sizeof(int));
    if (!conjugated_system_map) { return; }
    int current_system_id = 0;
    
    // Enhanced conjugated system detection
    // Include Pi systems that aren't necessarily aromatic
    for (int i = 0; i < atom_count; i++) {
        // Look for Pi systems (sp2 hybridized atoms, not just aromatic)
        if ((atoms[i].is_aromatic || 
            (atoms[i].hybridization > 1.5 && atoms[i].hybridization < 2.5)) &&
            conjugated_system_map[i] == 0) {
            current_system_id++;
            
            // Find all connected atoms in the Pi system through BFS
            int queue[atom_count];
            int front = 0, rear = 0;
            queue[rear++] = i;
            conjugated_system_map[i] = current_system_id;
            
            int num_atoms_in_system = 1; // Track atoms in this system
            
            while (front < rear) {
                int curr = queue[front++];
                
                // Check bonds to find connected atoms
                for (int b = 0; b < bond_count; b++) {
                    int next_atom = -1;
                    if (bonds[b].a == curr) next_atom = bonds[b].b;
                    else if (bonds[b].b == curr) next_atom = bonds[b].a;
                    
                    if (next_atom != -1 && conjugated_system_map[next_atom] == 0 &&
                        (atoms[next_atom].is_aromatic || 
                         (atoms[next_atom].hybridization > 1.5 && atoms[next_atom].hybridization < 2.5))) {
                        conjugated_system_map[next_atom] = current_system_id;
                        queue[rear++] = next_atom;
                        num_atoms_in_system++;
                    }
                }
            }
            
            // Calculate center of mass for this system
            double cm_x = 0, cm_y = 0, cm_z = 0;
            for (int a = 0; a < atom_count; a++) {
                if (conjugated_system_map[a] == current_system_id) {
                    cm_x += atoms[a].x;
                    cm_y += atoms[a].y;
                    cm_z += atoms[a].z;
                }
            }
            cm_x /= num_atoms_in_system;
            cm_y /= num_atoms_in_system;
            cm_z /= num_atoms_in_system;
            
            // Make amplitude more system-specific
            double mo_amplitude = 0.15 * num_atoms_in_system;
            mo_amplitude *= (1.0 + 0.1 * (current_system_id % 5)); // Vary by system ID
            
            // Use more distinctive phase patterns
            double mo_phase = current_system_id * PI / (current_system_id + 1.0);
            
            // Add additional phase modulation based on system properties
            // For example, count aromatic atoms in the system
            int aromatic_count = 0;
            for (int a = 0; a < atom_count; a++) {
                if (conjugated_system_map[a] == current_system_id && atoms[a].is_aromatic) {
                    aromatic_count++;
                }
            }
            
            mo_phase += (aromatic_count * PI) / (8.0 * num_atoms_in_system);
            
            // Apply orbital patterns to grid
            // Scale based on grid dimensions
            int center_x = (int)(cm_x * (grid_width / 8.0) + grid_width / 2.0);
            int center_y = (int)(cm_y * (grid_width / 8.0) + grid_width / 2.0);
            double radius = grid_width / 10.0 * sqrt(num_atoms_in_system);
            
            // Apply Z-depth effect to the pattern radius
            double z_depth_factor = 1.0 / (1.0 + 0.05 * fabs(cm_z));
            radius *= z_depth_factor;

            // Add Z-component to the phase for 3D effect
            mo_phase += cm_z * 0.05;

            for (int y = 0; y < grid_width; y++) {
                for (int x = 0; x < grid_width; x++) {
                    double dx = x - center_x;
                    double dy = y - center_y;
                    double dist = sqrt(dx*dx + dy*dy);
                    
                    if (dist < radius) {
                        // Create more distinctive orbital patterns
                        double pattern_val = sin((dist/radius) * PI * (1 + current_system_id % 3));
                        pattern_val *= mo_amplitude;
                        
                        int idx = y * grid_width + x;
                        complex double orbital_contrib = pattern_val * cexp(I * mo_phase);
                        aperture_grid[idx] += orbital_contrib;
                    }
                }
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

// Enhanced molecular orbital and quantum effects for creating more distinctive diffraction patterns
void enhance_diffraction_with_electronic_effects(complex double *aperture_grid, int grid_width, 
                                                AtomPos *atoms, int atom_count, 
                                                BondSeg *bonds, int bond_count) {
    if (atom_count == 0 || !aperture_grid) return;
    
    // Local grid for electronic effects
    complex double *electronic_grid = calloc(grid_width * grid_width, sizeof(complex double));
    if (!electronic_grid) {
        fprintf(stderr, "Error: Memory allocation failed for electronic effects grid.\n");
        return;
    }
    
    // 1. Process conjugated systems with enhanced wave patterns
    // First identify conjugated systems
    int *conjugated_systems = calloc(atom_count, sizeof(int));
    if (!conjugated_systems) {
        free(electronic_grid);
        fprintf(stderr, "Error: Memory allocation failed for conjugated system mapping.\n");
        return;
    }
    
    // Find and identify conjugated systems
    int system_count = 0;
    for (int i = 0; i < atom_count; i++) {
        if ((atoms[i].is_aromatic || 
            (atoms[i].hybridization > 1.5 && atoms[i].hybridization < 2.5)) && 
            conjugated_systems[i] == 0) {
            
            system_count++;
            
            // Find connected atoms in the conjugated system through BFS
            int *queue = calloc(atom_count, sizeof(int));
            if (!queue) {
                free(electronic_grid);
                free(conjugated_systems);
                fprintf(stderr, "Error: Queue allocation failed.\n");
                return;
            }
            
            int front = 0, rear = 0;
            queue[rear++] = i;
            conjugated_systems[i] = system_count;
            
            int system_atoms = 1;
            double cm_x = 0, cm_y = 0, cm_z = 0;
            
            // BFS to find connected conjugated atoms
            while (front < rear) {
                int curr = queue[front++];
                
                // Add to center of mass calculation
                cm_x += atoms[curr].x;
                cm_y += atoms[curr].y;
                cm_z += atoms[curr].z;
                
                // Find connected atoms through bonds
                for (int b = 0; b < bond_count; b++) {
                    int next_atom = -1;
                    if (bonds[b].a == curr) next_atom = bonds[b].b;
                    else if (bonds[b].b == curr) next_atom = bonds[b].a;
                    
                    if (next_atom != -1 && conjugated_systems[next_atom] == 0 &&
                        (atoms[next_atom].is_aromatic || 
                        (atoms[next_atom].hybridization > 1.5 && atoms[next_atom].hybridization < 2.5) ||
                        bonds[b].type == BOND_AROMATIC || bonds[b].is_conjugated)) {
                        
                        conjugated_systems[next_atom] = system_count;
                        queue[rear++] = next_atom;
                        system_atoms++;
                    }
                }
            }
            
            free(queue);
            
            if (system_atoms > 1) {
                // Calculate center of mass
                cm_x /= system_atoms;
                cm_y /= system_atoms;
                cm_z /= system_atoms;
                
                // Calculate the spatial extent of the system
                double max_dist = 0;
                for (int a = 0; a < atom_count; a++) {
                    if (conjugated_systems[a] == system_count) {
                        double dx = atoms[a].x - cm_x;
                        double dy = atoms[a].y - cm_y;
                        double dz = atoms[a].z - cm_z;
                        double dist = sqrt(dx*dx + dy*dy + dz*dz);
                        if (dist > max_dist) max_dist = dist;
                    }
                }
                
                // Create a distinctive orbital pattern for this conjugated system
                double scale_factor = grid_width / 8.0;
                int center_x = (int)(cm_x * scale_factor + grid_width / 2.0);
                int center_y = (int)(cm_y * scale_factor + grid_width / 2.0);
                double z_depth_factor = 1.0 / (1.0 + 0.1 * fabs(cm_z)); // Z-dependent scaling
                
                // Radius is proportional to system size, adjusted by z-depth
                double radius = (grid_width / 10.0) * sqrt(system_atoms) * z_depth_factor;
                
                // Generate a unique phase pattern for this conjugated system
                double base_phase = system_count * PI / 5.0; // Different phase per system
                
                // Distinctive amplitude based on system characteristics
                double amplitude = 0.2 * system_atoms;
                
                // Make systems with more aromatic character stronger
                int aromatic_count = 0;
                for (int a = 0; a < atom_count; a++) {
                    if (conjugated_systems[a] == system_count && atoms[a].is_aromatic) {
                        aromatic_count++;
                    }
                }
                double aromatic_factor = (double)aromatic_count / system_atoms;
                amplitude *= (1.0 + 0.5 * aromatic_factor);
                
                // Add distinctive nodal patterns based on system size and type
                int nodal_pattern = 2 + (system_atoms % 3); // Different node count
                
                // Apply the orbital pattern to grid points
                for (int y = 0; y < grid_width; y++) {
                    for (int x = 0; x < grid_width; x++) {
                        double dx = x - center_x;
                        double dy = y - center_y;
                        double dist = sqrt(dx*dx + dy*dy);
                        
                        if (dist < radius * 1.5) { // Extend beyond the core radius
                            // Nodal pattern based on system size
                            double pattern_val = 0.0;
                            double normalized_dist = dist / radius;
                            
                            // Radial part - creates nodal rings
                            double radial = sin(normalized_dist * PI * nodal_pattern) * exp(-normalized_dist);
                            
                            // Angular part - creates angular nodes
                            double theta = atan2(dy, dx);
                            int angular_nodes = 1 + (system_count % 5); // Different angular patterns
                            double angular = cos(angular_nodes * theta);
                            
                            // Combine radial and angular patterns
                            pattern_val = radial * angular * amplitude;
                            
                            // Attenuate by distance
                            double attenuation = exp(-normalized_dist * 0.5);
                            pattern_val *= attenuation;
                            
                            // Calculate phase - modulated by system-specific factors
                            double phase = base_phase + normalized_dist * 0.3 * PI;
                            
                            if (aromatic_factor > 0.5) {
                                // More complex phase patterns for highly aromatic systems
                                phase += cos(theta * angular_nodes) * 0.2 * PI;
                            }
                            
                            // Modulate by Z-position for 3D effect
                            phase += cm_z * 0.1;
                            
                            int idx = y * grid_width + x;
                            complex double contribution = pattern_val * cexp(I * phase);
                            electronic_grid[idx] += contribution;
                        }
                    }
                }
            }
        }
    }
    
    // 2. Add individual atomic orbital patterns (for non-conjugated atoms)
    for (int i = 0; i < atom_count; i++) {
        if (conjugated_systems[i] == 0) { // Skip atoms in conjugated systems
            double scale_factor = grid_width / 8.0;
            
            // Perspective projection
            double z_depth_effect = 1.0 / (1.0 + 0.1 * fabs(atoms[i].z));
            scale_factor *= z_depth_effect;
            
            int center_x = (int)(atoms[i].x * scale_factor + grid_width / 2.0);
            int center_y = (int)(atoms[i].y * scale_factor + grid_width / 2.0);
            
            // Set radius based on atomic properties
            double atom_radius = atoms[i].radius * 3.0 * scale_factor;
            if (atom_radius < 1.0) atom_radius = 1.0;
            
            // Set amplitude based on electronic properties
            double amplitude = 0.15;
            
            // Add electronegative atom effects
            if (atoms[i].electronegativity > 2.5) {
                amplitude *= (1.0 + 0.2 * (atoms[i].electronegativity - 2.5));
            }
            
            // Add effects for atoms with lone pairs or partial charges
            if (atoms[i].partial_charge != 0) {
                amplitude *= (1.0 + 0.3 * fabs(atoms[i].partial_charge));
            }
            
            // Generate phase based on atomic properties
            double phase = atoms[i].atomic_number * PI / 15.0;
            
            // Add hybridization effects
            phase += atoms[i].hybridization * PI / 10.0;
            
            // Add positional modulation
            phase += (atoms[i].x + atoms[i].y + atoms[i].z) * 0.1;
            
            // Apply atomic orbital pattern
            for (int y = 0; y < grid_width; y++) {
                for (int x = 0; x < grid_width; x++) {
                    double dx = x - center_x;
                    double dy = y - center_y;
                    double dist_sq = dx*dx + dy*dy;
                    double dist = sqrt(dist_sq);
                    
                    if (dist < atom_radius * 2.0) {
                        double normalized_dist = dist / atom_radius;
                        double orbital_val = 0.0;
                        
                        // Shape based on hybridization
                        if (atoms[i].hybridization > 2.5) { // sp3
                            // Spherical-like pattern
                            orbital_val = exp(-normalized_dist * normalized_dist) * amplitude;
                        } else if (atoms[i].hybridization > 1.5) { // sp2
                            // Planar pattern
                            double theta = atan2(dy, dx);
                            orbital_val = exp(-normalized_dist * normalized_dist) * 
                                        (1.0 + 0.3 * cos(3 * theta)) * amplitude;
                        } else { // sp or s
                            // Linear or spherical pattern
                            orbital_val = exp(-normalized_dist * normalized_dist) * amplitude;
                        }
                        
                        // Apply Z-position phase modulation
                        double z_phase = atoms[i].z * 0.1;
                        
                        int idx = y * grid_width + x;
                        complex double contribution = orbital_val * cexp(I * (phase + z_phase));
                        electronic_grid[idx] += contribution;
                    }
                }
            }
        }
    }
    
    // 3. Add bond effects, especially for key bonds like hydrogen bonds
    for (int i = 0; i < bond_count; i++) {
        int a1 = bonds[i].a;
        int a2 = bonds[i].b;
        
        // Skip bonds within the same conjugated system
        if (conjugated_systems[a1] != 0 && conjugated_systems[a1] == conjugated_systems[a2]) {
            continue;
        }
        
        // Process important non-conjugated bonds
        double scale_factor = grid_width / 8.0;
        
        // Get atom coordinates and adjust by Z-depth
        double x1 = atoms[a1].x;
        double y1 = atoms[a1].y;
        double z1 = atoms[a1].z;
        double x2 = atoms[a2].x;
        double y2 = atoms[a2].y;
        double z2 = atoms[a2].z;
        
        // Calculate midpoint and adjust for perspective
        double mid_x = (x1 + x2) / 2.0;
        double mid_y = (y1 + y2) / 2.0;
        double mid_z = (z1 + z2) / 2.0;
        
        double z_depth_effect = 1.0 / (1.0 + 0.1 * fabs(mid_z));
        scale_factor *= z_depth_effect;
        
        int center_x = (int)(mid_x * scale_factor + grid_width / 2.0);
        int center_y = (int)(mid_y * scale_factor + grid_width / 2.0);
        
        // Calculate bond vector
        double dx_bond = x2 - x1;
        double dy_bond = y2 - y1;
        double bond_length = sqrt(dx_bond*dx_bond + dy_bond*dy_bond);
        if (bond_length < 0.001) bond_length = 0.001;
        
        // Normalize bond vector
        double nx_bond = dx_bond / bond_length;
        double ny_bond = dy_bond / bond_length;
        
        // Amplitude based on bond type
        double amplitude = 0.1;
        if (bonds[i].order > 1) {
            amplitude += 0.05 * bonds[i].order;
        }
        
        // Special effects for important bond types
        if (bonds[i].type == BOND_HYDROGEN || 
            bonds[i].type == BOND_IONIC || 
            bonds[i].type == BOND_DATIVE) {
            amplitude *= 1.5;
        }
        
        // Bond-specific phase
        double bond_phase = bonds[i].order * PI / 4.0;
        
        // Add electronegativity difference effect
        double en_diff = fabs(atoms[a1].electronegativity - atoms[a2].electronegativity);
        bond_phase += en_diff * PI / 10.0;
        
        // Add 3D effect from Z-coordinate
        bond_phase += mid_z * 0.1;
        
        // Bond radius scaled by bond order and Z-depth
        double bond_radius = bonds[i].order * 2.0 * scale_factor;
        if (bond_radius < 1.0) bond_radius = 1.0;
        
        // Apply bond pattern
        for (int y = 0; y < grid_width; y++) {
            for (int x = 0; x < grid_width; x++) {
                double dx = x - center_x;
                double dy = y - center_y;
                
                // Project point onto bond axis to determine position along bond
                double proj = dx * nx_bond + dy * ny_bond;
                
                // Calculate perpendicular distance from bond axis
                double perp_dx = dx - proj * nx_bond;
                double perp_dy = dy - proj * ny_bond;
                double perp_dist = sqrt(perp_dx*perp_dx + perp_dy*perp_dy);
                
                // Bond pattern is stronger along axis and drops off perpendicular to it
                if (perp_dist < bond_radius && fabs(proj) < bond_length * scale_factor / 2.0) {
                    // Calculate position along bond (0 to 1)
                    double pos = (proj / scale_factor + bond_length / 2.0) / bond_length;
                    pos = fmax(0.0, fmin(1.0, pos));
                    
                    // Create bond-specific pattern
                    double pattern_val = 0.0;
                    
                    // Basic bond pattern decreases with perpendicular distance
                    pattern_val = exp(-perp_dist * perp_dist / (bond_radius * bond_radius)) * amplitude;
                    
                    // Multiple bond patterns have nodes
                    if (bonds[i].order == 2) {
                        pattern_val *= (1.0 + 0.3 * sin(pos * 2.0 * PI));
                    } else if (bonds[i].order == 3) {
                        pattern_val *= (1.0 + 0.3 * sin(pos * 3.0 * PI));
                    }
                    
                    // Polar bonds have asymmetric patterns
                    if (en_diff > 0.5) {
                        pattern_val *= (1.0 + 0.2 * (pos - 0.5) * en_diff);
                    }
                    
                    // Add Z-modulation for 3D effect
                    double z_pos = z1 * (1.0 - pos) + z2 * pos;
                    double z_phase_mod = z_pos * 0.05;
                    
                    int idx = y * grid_width + x;
                    complex double contribution = pattern_val * cexp(I * (bond_phase + z_phase_mod));
                    electronic_grid[idx] += contribution;
                }
            }
        }
    }
    
    // 4. Combine electronic grid with the main aperture grid
    for (int i = 0; i < grid_width * grid_width; i++) {
        aperture_grid[i] += electronic_grid[i] * 0.7; // Scale factor to control strength
    }
    
    // Clean up
    free(electronic_grid);
    free(conjugated_systems);
}

// Calculate molecular form factor - used for improved diffraction patterns
complex double calculate_molecular_form_factor(AtomPos atom, double q_magnitude) {
    // Form factor approximation based on atomic properties
    // q_magnitude is the scattering vector magnitude
    
    // Base scattering factor depends on electron count (atomic number)
    double base_factor = atom.atomic_number;
    
    // Approximate scattering falloff with q using exponential form
    // f(q) = sum(a_i * exp(-b_i * q^2))
    // Simplified to a single gaussian term for each atom
    double b_factor = pow(atom.radius, 2.0) * 0.5;
    double amplitude = base_factor * exp(-b_factor * q_magnitude * q_magnitude);
    
    // Calculate phase shift based on atomic properties
    double phase = 0.0;
    
    // Phase contribution from electronic structure
    phase += atom.electronegativity * 0.1;
    
    // Phase contribution from hybridization
    phase += atom.hybridization * 0.05;
    
    // Phase from partial charge
    phase += atom.partial_charge * 0.3;
    
    // Apply phase modulation
    complex double form_factor = amplitude * cexp(I * phase);
    
    return form_factor;
}