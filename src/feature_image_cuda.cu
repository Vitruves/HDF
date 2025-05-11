#include <cuda_runtime.h>
#include "molecule.h"
#include <stddef.h>

// Kernel to generate feature image
__global__ void generate_feature_image_kernel(
    float *d_data, int width, int height, int channels,
    AtomPos *d_atoms, int atom_count,
    int *d_channel_types,
    float grid_min_x, float grid_min_y, float grid_min_z,
    float grid_step_x, float grid_step_y, float grid_step_z
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Calculate grid point coordinates
    float grid_x = grid_min_x + x * grid_step_x;
    float grid_y = grid_min_y + y * grid_step_y;
    float grid_z = grid_min_z + 0.5f * grid_step_z;  // Middle of Z range
    
    // For each channel
    for (int c = 0; c < channels; c++) {
        int channel_type = d_channel_types[c];
        float sum = 0.0f;
        
        // Sum contributions from all atoms
        for (int a = 0; a < atom_count; a++) {
            AtomPos atom = d_atoms[a];
            
            float dx = grid_x - atom.x;
            float dy = grid_y - atom.y;
            float dz = grid_z - atom.z;
            float dist_sq = dx*dx + dy*dy + dz*dz;
            
            // Base falloff based on distance
            float radius = atom.radius * 2.0f;
            float sigma = radius / 3.0f;
            float falloff = expf(-dist_sq / (2.0f * sigma * sigma));
            
            // Channel-specific contributions
            float value = 0.0f;
            
            switch (channel_type) {
                case 0: // CHANNEL_ELECTRON_DENSITY
                    value = falloff * atom.electron_density_max;
                    break;
                    
                case 1: // CHANNEL_LIPOPHILICITY
                    // Enhanced lipophilicity based on atom type and environment
                    value = 0.0f;
                    if (atom.atom[0] == 'C' && atom.atom[1] == '\0') value = 1.0f;
                    else if (atom.atom[0] == 'H' && atom.atom[1] == '\0') value = 0.5f;
                    else if (atom.atom[0] == 'O' && atom.atom[1] == '\0') value = -0.5f;
                    else if (atom.atom[0] == 'N' && atom.atom[1] == '\0') value = -0.4f;
                    else if (atom.atom[0] == 'F' || atom.atom[0] == 'C') value = 0.7f;
                    else if (atom.atom[0] == 'S' && atom.atom[1] == '\0') value = 0.3f;
                    else if (atom.atom[0] == 'P' && atom.atom[1] == '\0') value = 0.2f;
                    
                    // Adjust based on environment
                    if (atom.is_aromatic) value *= 1.2f;
                    if (atom.in_ring) value *= 0.85f;

                    value *= falloff;
                    break;
                    
                case 2: // CHANNEL_HYDROGEN_DONOR
                    value = 0.0f;
                    if ((atom.atom[0] == 'N' || atom.atom[0] == 'O') && atom.atom[1] == '\0') {
                        if (atom.n_bonds < atom.valence) value = 1.0f;
                    }
                    // Include S-H as weak H-bond donor
                    if (atom.atom[0] == 'S' && atom.n_bonds < atom.valence) value = 0.4f;
                    
                    // Consider environment effects
                    if (atom.is_aromatic) value *= 0.7f;  // Aromatic atoms are weaker donors
                    if (atom.charge > 0) value *= 0.5f;   // Positive charge weakens donating
                    if (atom.charge < 0) value *= 1.3f;   // Negative charge strengthens donating
                    
                    value *= falloff;
                    break;
                    
                case 3: // CHANNEL_HYDROGEN_ACCEPTOR
                    value = 0.0f;
                    // Basic acceptors
                    if ((atom.atom[0] == 'N' || atom.atom[0] == 'O') && atom.atom[1] == '\0') {
                        value = 1.0f;
                    }
                    // Weaker acceptors
                    if ((atom.atom[0] == 'S' || atom.atom[0] == 'F') && atom.atom[1] == '\0') {
                        value = 0.6f;
                    }
                    
                    // Consider environment effects
                    if (atom.is_aromatic) value *= 0.8f;  // Aromatic atoms are weaker acceptors
                    if (atom.charge < 0) value *= 1.5f;   // Negative charge strengthens accepting
                    if (atom.charge > 0) value *= 0.3f;   // Positive charge weakens accepting
                    
                    value *= falloff;
                    break;
                    
                case 4: // CHANNEL_POSITIVE_CHARGE
                    value = falloff * fmaxf(0.0f, atom.charge);
                    break;
                    
                case 5: // CHANNEL_NEGATIVE_CHARGE
                    value = falloff * fmaxf(0.0f, -atom.charge);
                    break;
                    
                case 6: // CHANNEL_AROMATICITY
                    value = falloff * (atom.is_aromatic ? 1.0f : 0.0f);
                    break;
                    
                case 7: // CHANNEL_SP2_HYBRIDIZATION
                    value = falloff * (atom.hybridization > 1.5f && atom.hybridization < 2.5f ? 1.0f : 0.0f);
                    break;
                    
                case 8: // CHANNEL_SP3_HYBRIDIZATION
                    value = falloff * (atom.hybridization > 2.5f ? 1.0f : 0.0f);
                    break;
                    
                case 9: // CHANNEL_GASTEIGER_CHARGE
                    {
                        float charge_val = atom.charge;
                        if (charge_val > 1.0f) charge_val = 1.0f;
                        if (charge_val < -1.0f) charge_val = -1.0f;
                        value = falloff * (charge_val + 1.0f) / 2.0f;  // Scale to [0,1]
                    }
                    break;
                    
                case 10: // CHANNEL_RING_MEMBERSHIP
                    value = falloff * (atom.in_ring ? 1.0f : 0.0f);
                    break;
                
                case 11: // CHANNEL_AROMATIC_RING
                    value = falloff * (atom.in_ring && atom.is_aromatic ? 1.0f : 0.0f);
                    break;
                    
                case 12: // CHANNEL_ALIPHATIC_RING
                    value = falloff * (atom.in_ring && !atom.is_aromatic ? 1.0f : 0.0f);
                    break;
                    
                case 13: // CHANNEL_POLARIZABILITY
                    {
                        // Approximate atomic polarizability
                        float polar_val = 0.0f;
                        if (atom.atom[0] == 'C') polar_val = 1.0f;
                        else if (atom.atom[0] == 'N') polar_val = 0.7f;
                        else if (atom.atom[0] == 'O') polar_val = 0.5f;
                        else if (atom.atom[0] == 'S') polar_val = 1.5f;
                        else if (atom.atom[0] == 'P') polar_val = 1.8f;
                        else if (atom.atom[0] == 'F') polar_val = 0.3f;
                        else if (atom.atom[0] == 'C' && atom.atom[1] == 'l') polar_val = 1.2f;
                        else if (atom.atom[0] == 'B' && atom.atom[1] == 'r') polar_val = 1.8f;
                        else if (atom.atom[0] == 'I') polar_val = 2.5f;
                        
                        // Adjust for electronic effects
                        if (atom.is_aromatic) polar_val *= 1.2f;
                        if (atom.charge != 0) polar_val *= (1.0f - 0.2f * atom.charge);
                        
                        value = falloff * polar_val;
                    }
                    break;
                    
                case 14: // CHANNEL_VDWAALS_INTERACTION
                    {
                        // VDW interaction potential based on atom type
                        float vdw_val = atom.radius * 0.5f;
                        
                        // Distance dependence follows r^-6
                        float r = sqrtf(dist_sq);
                        if (r < 0.1f) r = 0.1f;  // Avoid singularity
                        
                        // Approximate with a smoother falloff than typical r^-6
                        float vdw_falloff = expf(-dist_sq / (4.0f * sigma * sigma));
                        value = vdw_val * vdw_falloff;
                    }
                    break;
                    
                case 15: // CHANNEL_ATOMIC_REFRACTIVITY
                    {
                        // Approximate atomic refractivity values (scaled relative values)
                        float refract_val = 0.0f;
                        if (atom.atom[0] == 'C') refract_val = 0.8f;
                        else if (atom.atom[0] == 'N') refract_val = 0.6f;
                        else if (atom.atom[0] == 'O') refract_val = 0.4f;
                        else if (atom.atom[0] == 'S') refract_val = 1.4f;
                        else if (atom.atom[0] == 'P') refract_val = 1.2f;
                        else if (atom.atom[0] == 'F') refract_val = 0.2f;
                        else if (atom.atom[0] == 'C' && atom.atom[1] == 'l') refract_val = 1.0f;
                        else if (atom.atom[0] == 'B' && atom.atom[1] == 'r') refract_val = 1.6f;
                        else if (atom.atom[0] == 'I') refract_val = 2.0f;
                        
                        // Adjust for electronic effects
                        if (atom.hybridization > 2.5f) refract_val *= 0.8f; // sp3 has lower refractivity
                        if (atom.hybridization > 1.5f && atom.hybridization < 2.5f) refract_val *= 1.1f; // sp2 has higher
                        
                        value = falloff * refract_val;
                    }
                    break;
                
                case 16: // CHANNEL_ELECTRONEGATIVITY
                    value = falloff * atom.electronegativity / 4.0f; // Normalize to approximately [0,1]
                    break;
                
                case 17: // CHANNEL_BOND_ORDER_INFLUENCE
                    {
                        // Represent the influence of bond orders
                        float bond_order_val = 0.0f;
                        
                        // Simple approximation: average n_bonds / valence
                        if (atom.valence > 0) {
                            bond_order_val = (float)atom.n_bonds / (float)atom.valence;
                            if (atom.is_aromatic) bond_order_val *= 1.2f; // Enhance for aromatic atoms
                        }
                        
                        value = falloff * bond_order_val;
                    }
                    break;
                    
                case 18: // CHANNEL_STEREOCHEMISTRY
                    // For now, just use the Z-coord as indication of stereochemistry
                    // More sophisticated approaches would use actual stereochemistry data
                    value = falloff * (0.5f + atom.z * 0.1f);
                    break;
                
                case 19: // CHANNEL_ROTATABLE_BOND_INFLUENCE
                    {
                        // Approximation for atoms that might be part of rotatable bonds
                        float rotatable_val = 0.0f;
                        
                        // Atoms with sp3 hybridization and not in rings are more likely part of rotatable bonds
                        if (atom.hybridization > 2.5f && !atom.in_ring) {
                            rotatable_val = 1.0f;
                        } else if (!atom.in_ring && atom.n_bonds > 1) {
                            rotatable_val = 0.7f;
                        }
                        
                        value = falloff * rotatable_val;
                    }
                    break;
                
                case 20: // CHANNEL_MOLECULAR_SHAPE
                    {
                        // Approximate molecular shape influence
                        // Use distance from closest axis as shape indicator
                        float axis_dist = fminf(fabsf(dx), fminf(fabsf(dy), fabsf(dz)));
                        value = falloff * (1.0f - axis_dist / (atom.radius + 1.0f));
                    }
                    break;

                case 21: // CHANNEL_SURFACE_ACCESSIBILITY
                    {
                        // Approximate solvent accessibility based on atom position
                        // Atoms on the periphery are more accessible
                        float peripheral_factor = sqrtf(atom.x*atom.x + atom.y*atom.y + atom.z*atom.z);
                        
                        // Scale by a typical factor to normalize
                        peripheral_factor = fminf(peripheral_factor / 5.0f, 1.0f);
                        
                        // Adjust for atom type (hydrophobic atoms tend to be less accessible)
                        if (atom.atom[0] == 'C' && !atom.is_aromatic) peripheral_factor *= 0.8f;
                        if (atom.atom[0] == 'O' || atom.atom[0] == 'N') peripheral_factor *= 1.2f;
                        
                        value = falloff * peripheral_factor;
                    }
                    break;
                
                case 22: // CHANNEL_PHARMACOPHORE_HYDROPHOBIC
                    {
                        // Hydrophobic feature - carbon atoms with low electronegativity
                        float hydrophobic_val = 0.0f;
                        
                        if (atom.atom[0] == 'C') {
                            hydrophobic_val = 1.0f - (atom.electronegativity / 4.0f);
                            
                            // Adjust for aromatic character
                            if (atom.is_aromatic) hydrophobic_val *= 0.7f;
                            
                            // Adjust for attached electronegative atoms (simplified)
                            if (atom.explicit_h_count < 2) hydrophobic_val *= 0.8f;
                        }
                        
                        value = falloff * hydrophobic_val;
                    }
                    break;
                    
                case 23: // CHANNEL_PHARMACOPHORE_AROMATIC
                    {
                        // Aromatic feature for pharmacophore
                        float aromatic_val = atom.is_aromatic ? 1.0f : 0.0f;
                        
                        // Enhanced value for atoms in fused ring systems
                        if (atom.ring_count > 1 && atom.is_aromatic) {
                            aromatic_val *= 1.3f;
                        }
                        
                        value = falloff * aromatic_val;
                    }
                    break;
                
                case 24: // CHANNEL_ISOTOPE_EFFECT
                    {
                        // Represent isotope effects - heavier isotopes change properties
                        float isotope_val = 0.0f;
                        
                        if (atom.isotope > 0) {
                            // Scale based on how much heavier the isotope is relative to common isotope
                            float relative_mass_increase = atom.isotope / 100.0f; // Just an approximation
                            isotope_val = relative_mass_increase;
                        }
                        
                        value = falloff * isotope_val;
                    }
                    break;
                
                case 25: // CHANNEL_QUANTUM_EFFECTS
                    {
                        // Simplified quantum mechanical effects
                        float qm_val = 0.0f;
                        
                        // Use atom electronegativity, hybridization and aromaticity for QM approximation
                        qm_val = (atom.electronegativity / 4.0f) * 
                                (0.5f + atom.hybridization / 6.0f) * 
                                (atom.is_aromatic ? 1.2f : 1.0f);
                        
                        // Consider orbital energy levels (highly simplified)
                        if (atom.orbital_config[1] > 0) qm_val *= 1.1f; // p orbitals contribution
                        
                        value = falloff * qm_val;
                    }
                    break;
                    
                default:
                    value = falloff;
                    break;
            }
            
            sum += value;
        }
        
        // Store the result
        d_data[(y * width + x) * channels + c] = sum;
    }
}

// Function to launch the CUDA kernel with proper C linkage for calling from C
extern "C" void launch_generate_feature_image_cuda(
    float *d_data, int width, int height, int channels,
    AtomPos *d_atoms, int atom_count,
    int *d_channel_types, double grid_min_x, double grid_min_y, double grid_min_z,
    double grid_step_x, double grid_step_y, double grid_step_z
) {
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                (height + blockDim.y - 1) / blockDim.y);
    
    generate_feature_image_kernel<<<gridDim, blockDim>>>(
        d_data, width, height, channels,
        d_atoms, atom_count,
        d_channel_types,
        (float)grid_min_x, (float)grid_min_y, (float)grid_min_z,
        (float)grid_step_x, (float)grid_step_y, (float)grid_step_z
    );
    
    cudaDeviceSynchronize();
} 