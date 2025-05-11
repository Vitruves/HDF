#include <float.h>
#include "feature_image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Macro for unused parameters to avoid compiler warnings
#ifdef __GNUC__
#define UNUSED __attribute__((unused))
#else
#define UNUSED
#endif

// CUDA-related includes
#if defined(HAS_CUDA) || defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cuda.h>
#endif

// Function declarations for CUDA implementation
void launch_generate_feature_image_cuda(
    float *d_data, int width, int height, int channels,
    AtomPos *d_atoms, int atom_count,
    int *d_channel_types, double grid_min_x, double grid_min_y, double grid_min_z,
    double grid_step_x, double grid_step_y, double grid_step_z);

// CUDA integration check
static int feature_cuda_available = -1;

// Check CUDA availability
static bool check_cuda_available() {
    if (feature_cuda_available == -1) {
        #if defined(HAS_CUDA) || defined(__CUDACC__)
        cudaError_t err = cudaFree(0);
        feature_cuda_available = (err == cudaSuccess) ? 1 : 0;
        #else
        feature_cuda_available = 0;
        #endif
    }
    return feature_cuda_available == 1;
}

// Channel information table
static const FeatureChannelInfo channel_info_table[] = {
    {CHANNEL_ELECTRON_DENSITY, "Electron Density", "Distribution of electron density", false},
    {CHANNEL_LIPOPHILICITY, "Lipophilicity", "Hydrophobic/lipophilic character", false},
    {CHANNEL_HYDROGEN_DONOR, "Hydrogen Donor", "Hydrogen bond donor capability", false},
    {CHANNEL_HYDROGEN_ACCEPTOR, "Hydrogen Acceptor", "Hydrogen bond acceptor capability", false},
    {CHANNEL_POSITIVE_CHARGE, "Positive Charge", "Positively charged regions", false},
    {CHANNEL_NEGATIVE_CHARGE, "Negative Charge", "Negatively charged regions", false},
    {CHANNEL_AROMATICITY, "Aromaticity", "Aromatic character", false},
    {CHANNEL_SP2_HYBRIDIZATION, "SP2 Hybridization", "SP2 hybridized regions", false},
    {CHANNEL_SP3_HYBRIDIZATION, "SP3 Hybridization", "SP3 hybridized regions", false},
    {CHANNEL_GASTEIGER_CHARGE, "Gasteiger Charge", "Partial charge distribution", false},
    {CHANNEL_RING_MEMBERSHIP, "Ring Membership", "Atoms in ring systems", false},
    {CHANNEL_AROMATIC_RING, "Aromatic Ring", "Atoms in aromatic rings", false},
    {CHANNEL_ALIPHATIC_RING, "Aliphatic Ring", "Atoms in aliphatic rings", false},
    {CHANNEL_POLARIZABILITY, "Polarizability", "Atomic polarizability", false},
    {CHANNEL_VDWAALS_INTERACTION, "VDW Interaction", "Van der Waals interaction potential", true},
    {CHANNEL_ATOMIC_REFRACTIVITY, "Refractivity", "Atomic refractivity", false},
    {CHANNEL_ELECTRONEGATIVITY, "Electronegativity", "Atomic electronegativity", false},
    {CHANNEL_BOND_ORDER_INFLUENCE, "Bond Order", "Influence of bond orders", false},
    {CHANNEL_STEREOCHEMISTRY, "Stereochemistry", "Stereochemical configuration", true},
    {CHANNEL_ROTATABLE_BOND_INFLUENCE, "Rotatable Bonds", "Influence of rotatable bonds", false},
    {CHANNEL_MOLECULAR_SHAPE, "Molecular Shape", "3D molecular shape", true},
    {CHANNEL_SURFACE_ACCESSIBILITY, "Surface Accessibility", "Solvent accessible surface", true},
    {CHANNEL_PHARMACOPHORE_HYDROPHOBIC, "Pharm. Hydrophobic", "Pharmacophore hydrophobic regions", false},
    {CHANNEL_PHARMACOPHORE_AROMATIC, "Pharm. Aromatic", "Pharmacophore aromatic regions", false},
    {CHANNEL_ISOTOPE_EFFECT, "Isotope Effect", "Influence of isotopic substitution", false},
    {CHANNEL_QUANTUM_EFFECTS, "Quantum Effects", "Simplified quantum mechanical effects", false},
    {-1, NULL, NULL, false}  // Terminator
};

// Create a new multi-channel image
MultiChannelImage* create_multi_channel_image(int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels <= 0 || channels > MAX_FEATURE_CHANNELS) {
        fprintf(stderr, "Error: Invalid dimensions for feature image (w=%d, h=%d, c=%d)\n", 
                width, height, channels);
        return NULL;
    }
    
    MultiChannelImage* image = (MultiChannelImage*)malloc(sizeof(MultiChannelImage));
    if (!image) {
        fprintf(stderr, "Error: Failed to allocate memory for MultiChannelImage\n");
        return NULL;
    }
    
    image->width = width;
    image->height = height;
    image->channels = channels;
    image->data = (float*)calloc(width * height * channels, sizeof(float));
    image->d_data = NULL;
    
    if (!image->data) {
        fprintf(stderr, "Error: Failed to allocate memory for image data\n");
        free(image);
        return NULL;
    }
    
    // Initialize channel types to default values
    for (int i = 0; i < channels && i < MAX_FEATURE_CHANNELS; i++) {
        image->channel_types[i] = i;  // Default to sequential channel types
    }
    
    // Allocate CUDA memory if available
    if (check_cuda_available()) {
        cudaError_t cuda_err;
        void *d_ptr = NULL;
        cuda_err = cudaMalloc(&d_ptr, width * height * channels * sizeof(float));
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDA error: Failed to allocate device memory for feature image: %s\n", 
                    cudaGetErrorString(cuda_err));
            image->d_data = NULL;
        } else {
            image->d_data = (float*)d_ptr;
            // Zero-initialize the CUDA memory
            cudaMemset(image->d_data, 0, width * height * channels * sizeof(float));
        }
    }
    
    return image;
}

// Free the multi-channel image
void free_multi_channel_image(MultiChannelImage* image) {
    if (!image) return;
    
    if (image->data) {
        free(image->data);
        image->data = NULL;
    }
    
    if (image->d_data) {
        cudaFree(image->d_data);
        image->d_data = NULL;
    }
    
    free(image);
}

// Calculate Gasteiger partial charges with enhanced approach
void calculate_gasteiger_charges(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count) {
    if (!atoms || atom_count <= 0) return;
    
    // Initialize parameters for Gasteiger-Marsili calculation
    // These are electronegativity parameters for each element
    // a, b, c for the equation chi = a + b*q + c*q^2
    double chi_params[10][3] = {
        {7.17, 19.04, 0.00},  // C
        {7.42, 20.57, 0.00},  // N
        {8.18, 22.35, 0.00},  // O
        {8.87, 21.48, 0.00},  // F
        {7.31, 19.87, 0.00},  // Si
        {6.95, 18.51, 0.00},  // P
        {6.60, 16.37, 0.00},  // S
        {8.21, 19.19, 0.00},  // Cl
        {7.34, 17.36, 0.00},  // Br
        {6.82, 14.86, 0.00}   // I
    };
    
    // Map atom types to index in the chi_params table
    int atom_param_idx[atom_count];
    for (int i = 0; i < atom_count; i++) {
        atom_param_idx[i] = -1;
        if (strcmp(atoms[i].atom, "C") == 0) atom_param_idx[i] = 0;
        else if (strcmp(atoms[i].atom, "N") == 0) atom_param_idx[i] = 1;
        else if (strcmp(atoms[i].atom, "O") == 0) atom_param_idx[i] = 2;
        else if (strcmp(atoms[i].atom, "F") == 0) atom_param_idx[i] = 3;
        else if (strcmp(atoms[i].atom, "Si") == 0) atom_param_idx[i] = 4;
        else if (strcmp(atoms[i].atom, "P") == 0) atom_param_idx[i] = 5;
        else if (strcmp(atoms[i].atom, "S") == 0) atom_param_idx[i] = 6;
        else if (strcmp(atoms[i].atom, "Cl") == 0) atom_param_idx[i] = 7;
        else if (strcmp(atoms[i].atom, "Br") == 0) atom_param_idx[i] = 8;
        else if (strcmp(atoms[i].atom, "I") == 0) atom_param_idx[i] = 9;
    }
    
    // Initialize all charges to 0
    for (int i = 0; i < atom_count; i++) {
        atoms[i].charge = 0.0;
    }
    
    // Perform the iterative charge calculation (simplified)
    const int max_iterations = 6;
    const double damping = 0.5;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        for (int b = 0; b < bond_count; b++) {
            int a1 = bonds[b].a;
            int a2 = bonds[b].b;
            
            if (atom_param_idx[a1] < 0 || atom_param_idx[a2] < 0) continue;
            
            // Calculate electronegativities
            double chi1 = chi_params[atom_param_idx[a1]][0] + 
                         chi_params[atom_param_idx[a1]][1] * atoms[a1].charge + 
                         chi_params[atom_param_idx[a1]][2] * atoms[a1].charge * atoms[a1].charge;
                         
            double chi2 = chi_params[atom_param_idx[a2]][0] + 
                         chi_params[atom_param_idx[a2]][1] * atoms[a2].charge + 
                         chi_params[atom_param_idx[a2]][2] * atoms[a2].charge * atoms[a2].charge;
            
            // Calculate charge transfer (simplified)
            double dq = 0;
            if (chi1 != chi2) {
                dq = (chi2 - chi1) / 
                     (chi_params[atom_param_idx[a1]][1] + chi_params[atom_param_idx[a2]][1]);
                
                // Adjust by bond order
                dq *= 0.05 * bonds[b].order;
                
                // Apply damping
                dq *= damping / (iter + 1);
            }
            
            // Update charges
            atoms[a1].charge += dq;
            atoms[a2].charge -= dq;
        }
    }
    
    // Apply corrections for special atom types and environments
    for (int i = 0; i < atom_count; i++) {
        // Nitrogen in amines often has a slight negative charge
        if (strcmp(atoms[i].atom, "N") == 0 && atoms[i].n_bonds == 3) {
            atoms[i].charge -= 0.2;
        }
        
        // Oxygen typically carries a negative charge
        if (strcmp(atoms[i].atom, "O") == 0) {
            atoms[i].charge -= 0.3;
            
            // Carboxylic acid or alcohol oxygen
            if (atoms[i].n_bonds == 1) {
            atoms[i].charge -= 0.2;
            }
        }
        
        // Carbonyl carbon
        if (strcmp(atoms[i].atom, "C") == 0 && atoms[i].n_bonds == 3 && 
            atoms[i].hybridization > 1.8 && atoms[i].hybridization < 2.2) {
            for (int b = 0; b < bond_count; b++) {
                if ((bonds[b].a == i || bonds[b].b == i) && bonds[b].order == 2) {
                    int other_idx = (bonds[b].a == i) ? bonds[b].b : bonds[b].a;
                    if (strcmp(atoms[other_idx].atom, "O") == 0) {
                        atoms[i].charge += 0.3;
                    }
                }
            }
        }
        
        // Halides are electronegative
        if (strcmp(atoms[i].atom, "F") == 0 || 
            strcmp(atoms[i].atom, "Cl") == 0 || 
            strcmp(atoms[i].atom, "Br") == 0 || 
            strcmp(atoms[i].atom, "I") == 0) {
            atoms[i].charge -= 0.15;
        }
        
        // Aromatic carbon has slight positive charge
        if (strcmp(atoms[i].atom, "C") == 0 && atoms[i].is_aromatic) {
            atoms[i].charge += 0.05;
        }
        
        // Aromatic nitrogen has slight negative charge
        if (strcmp(atoms[i].atom, "N") == 0 && atoms[i].is_aromatic) {
            atoms[i].charge -= 0.05;
        }
    }
    
    // Calculate bond dipoles and partial charge differences
    for (int b = 0; b < bond_count; b++) {
        int a1 = bonds[b].a;
        int a2 = bonds[b].b;
        
        bonds[b].partial_charge_diff = atoms[a2].charge - atoms[a1].charge;
        
        // Calculate bond dipole (simplified)
        double dx = atoms[a2].x - atoms[a1].x;
        double dy = atoms[a2].y - atoms[a1].y;
        double dz = atoms[a2].z - atoms[a1].z;
        double bond_length = sqrt(dx*dx + dy*dy + dz*dz);
        
        if (bond_length > 0.01) {
            bonds[b].bond_dipole = bonds[b].partial_charge_diff * bond_length;
        } else {
            bonds[b].bond_dipole = 0.0;
        }
    }
}

// Calculate various atomic properties
void calculate_atomic_properties(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count) {
    if (!atoms || atom_count <= 0) return;
    
    // Initialize atomic properties
    for (int i = 0; i < atom_count; i++) {
        // Set default values for additional properties
        atoms[i].polarizability = 0.0;
        atoms[i].refractivity = 0.0;
        atoms[i].is_chiral = false;
        atoms[i].chirality = '\0';
        atoms[i].surface_area = 0.0;
        atoms[i].solvent_accessibility = 0.0;
        atoms[i].is_pharmacophore = false;
        atoms[i].pharmacophore_type = 0;
        atoms[i].logP_contribution = 0.0;
        atoms[i].is_rotatable_bond_atom = false;
        
        // Element-specific polarizability values (approximate)
        if (strcmp(atoms[i].atom, "C") == 0) {
            atoms[i].polarizability = 1.76;
            atoms[i].refractivity = 2.42;
            atoms[i].logP_contribution = 0.18;
        } else if (strcmp(atoms[i].atom, "N") == 0) {
            atoms[i].polarizability = 1.09;
            atoms[i].refractivity = 1.82;
            atoms[i].logP_contribution = -0.17;
        } else if (strcmp(atoms[i].atom, "O") == 0) {
            atoms[i].polarizability = 0.80;
            atoms[i].refractivity = 1.64;
            atoms[i].logP_contribution = -0.32;
        } else if (strcmp(atoms[i].atom, "S") == 0) {
            atoms[i].polarizability = 2.90;
            atoms[i].refractivity = 7.69;
            atoms[i].logP_contribution = 0.25;
        } else if (strcmp(atoms[i].atom, "P") == 0) {
            atoms[i].polarizability = 3.60;
            atoms[i].refractivity = 6.92;
            atoms[i].logP_contribution = 0.20;
        } else if (strcmp(atoms[i].atom, "F") == 0) {
            atoms[i].polarizability = 0.56;
            atoms[i].refractivity = 0.92;
            atoms[i].logP_contribution = -0.18;
        } else if (strcmp(atoms[i].atom, "Cl") == 0) {
            atoms[i].polarizability = 2.18;
            atoms[i].refractivity = 5.84;
            atoms[i].logP_contribution = 0.06;
        } else if (strcmp(atoms[i].atom, "Br") == 0) {
            atoms[i].polarizability = 3.05;
            atoms[i].refractivity = 8.70;
            atoms[i].logP_contribution = 0.20;
        } else if (strcmp(atoms[i].atom, "I") == 0) {
            atoms[i].polarizability = 4.68;
            atoms[i].refractivity = 13.94;
            atoms[i].logP_contribution = 0.40;
        } else if (strcmp(atoms[i].atom, "H") == 0) {
            atoms[i].polarizability = 0.30;
            atoms[i].refractivity = 0.30;
            atoms[i].logP_contribution = 0.0;
        }
        
        // Hybridization effects
        if (atoms[i].hybridization > 2.5) {  // sp3
            atoms[i].polarizability *= 0.9;
            atoms[i].refractivity *= 0.95;
        } else if (atoms[i].hybridization > 1.5 && atoms[i].hybridization < 2.5) {  // sp2
            atoms[i].polarizability *= 1.1;
            atoms[i].refractivity *= 1.05;
            // sp2 carbon is more hydrophobic
            if (strcmp(atoms[i].atom, "C") == 0) {
                atoms[i].logP_contribution += 0.1;
            }
        } else if (atoms[i].hybridization > 0.8 && atoms[i].hybridization < 1.5) {  // sp
            atoms[i].polarizability *= 1.2;
            atoms[i].refractivity *= 1.1;
        }
        
        // Aromaticity effects
        if (atoms[i].is_aromatic) {
            atoms[i].polarizability *= 1.2;
            atoms[i].refractivity *= 1.1;
            // Aromatic carbons are more hydrophobic
            if (strcmp(atoms[i].atom, "C") == 0) {
                atoms[i].logP_contribution += 0.3;
            }
        }
        
        // Ring membership effects
        if (atoms[i].in_ring) {
            atoms[i].polarizability *= 1.05;
            atoms[i].refractivity *= 1.02;
        }
        
        // Charge effects
        atoms[i].polarizability *= (1.0 - 0.1 * abs((int)atoms[i].charge));
    }
    
    // Identify rotatable bonds
    identify_rotatable_bonds(atoms, atom_count, bonds, bond_count);
    
    // Calculate approximate bond energies
    for (int b = 0; b < bond_count; b++) {
        int a1 = bonds[b].a;
        int a2 = bonds[b].b;
        
        // Approximate bond energies in kcal/mol
        double bond_energy = 0.0;
        
        // C-C bonds
        if (strcmp(atoms[a1].atom, "C") == 0 && strcmp(atoms[a2].atom, "C") == 0) {
            if (bonds[b].order == 1) bond_energy = 83.0;
            else if (bonds[b].order == 2) bond_energy = 146.0;
            else if (bonds[b].order == 3) bond_energy = 200.0;
            else if (bonds[b].type == BOND_AROMATIC) bond_energy = 117.0;
        }
        // C-N bonds
        else if ((strcmp(atoms[a1].atom, "C") == 0 && strcmp(atoms[a2].atom, "N") == 0) ||
                 (strcmp(atoms[a1].atom, "N") == 0 && strcmp(atoms[a2].atom, "C") == 0)) {
            if (bonds[b].order == 1) bond_energy = 73.0;
            else if (bonds[b].order == 2) bond_energy = 147.0;
            else if (bonds[b].order == 3) bond_energy = 213.0;
            else if (bonds[b].type == BOND_AROMATIC) bond_energy = 109.0;
        }
        // C-O bonds
        else if ((strcmp(atoms[a1].atom, "C") == 0 && strcmp(atoms[a2].atom, "O") == 0) ||
                 (strcmp(atoms[a1].atom, "O") == 0 && strcmp(atoms[a2].atom, "C") == 0)) {
            if (bonds[b].order == 1) bond_energy = 85.0;
            else if (bonds[b].order == 2) bond_energy = 176.0;
            else if (bonds[b].type == BOND_AROMATIC) bond_energy = 112.0;
        }
        // Default bond energies for other atom pairs
        else {
            if (bonds[b].order == 1) bond_energy = 80.0;
            else if (bonds[b].order == 2) bond_energy = 150.0;
            else if (bonds[b].order == 3) bond_energy = 210.0;
            else if (bonds[b].type == BOND_AROMATIC) bond_energy = 110.0;
        }
        
        // Adjust based on electronegativity differences
        double en_diff = abs((int)atoms[a1].electronegativity - (int)atoms[a2].electronegativity);
        bond_energy += en_diff * 10.0;
        
        bonds[b].bond_energy = bond_energy;
    }
}

// Identify rotatable bonds in the molecule
void identify_rotatable_bonds(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count) {
    if (!atoms || !bonds || atom_count <= 0 || bond_count <= 0) return;
    
    // Initialize all bonds as non-rotatable
    for (int b = 0; b < bond_count; b++) {
        bonds[b].is_rotatable = false;
        bonds[b].is_conjugated = false;
        bonds[b].is_amide = false;
    }
    
    // First pass: identify conjugated and amide bonds
    for (int b = 0; b < bond_count; b++) {
        int a1 = bonds[b].a;
        int a2 = bonds[b].b;
        
        // Skip bonds that are not single bonds
        if (bonds[b].order != 1) continue;
        
        // Skip bonds in rings
        if (bonds[b].in_ring) continue;
        
        // Check if this is a conjugated bond (adjacent to double/triple bond or aromatic)
        bool a1_has_multiple = false;
        bool a2_has_multiple = false;
        
        for (int b2 = 0; b2 < bond_count; b2++) {
            if (b2 == b) continue;
            
            if (bonds[b2].a == a1 || bonds[b2].b == a1) {
                if (bonds[b2].order > 1 || bonds[b2].type == BOND_AROMATIC) {
                    a1_has_multiple = true;
                }
            }
            
            if (bonds[b2].a == a2 || bonds[b2].b == a2) {
                if (bonds[b2].order > 1 || bonds[b2].type == BOND_AROMATIC) {
                    a2_has_multiple = true;
                }
            }
        }
        
        if (a1_has_multiple && a2_has_multiple) {
            bonds[b].is_conjugated = true;
        }
        
        // Check if this is an amide bond (C-N with C=O)
        bool is_c_n_bond = (strcmp(atoms[a1].atom, "C") == 0 && strcmp(atoms[a2].atom, "N") == 0) ||
                           (strcmp(atoms[a1].atom, "N") == 0 && strcmp(atoms[a2].atom, "C") == 0);
        
        if (is_c_n_bond) {
            int c_idx = strcmp(atoms[a1].atom, "C") == 0 ? a1 : a2;
            
            // Look for C=O
            for (int b2 = 0; b2 < bond_count; b2++) {
                if (bonds[b2].order == 2 && 
                    (bonds[b2].a == c_idx || bonds[b2].b == c_idx)) {
                    int other_idx = bonds[b2].a == c_idx ? bonds[b2].b : bonds[b2].a;
                    if (strcmp(atoms[other_idx].atom, "O") == 0) {
                        bonds[b].is_amide = true;
                        break;
                    }
                }
            }
        }
    }
    
    // Second pass: identify rotatable bonds
    for (int b = 0; b < bond_count; b++) {
        int a1 = bonds[b].a;
        int a2 = bonds[b].b;
        
        // Skip bonds that are not single bonds
        if (bonds[b].order != 1) continue;
        
        // Skip bonds in rings
        if (bonds[b].in_ring) continue;
        
        // Skip bonds involving hydrogen (C-H, N-H, etc.)
        if (strcmp(atoms[a1].atom, "H") == 0 || strcmp(atoms[a2].atom, "H") == 0) continue;
        
        // Skip terminal bonds (to atoms with only one connection)
        if (atoms[a1].n_bonds <= 1 || atoms[a2].n_bonds <= 1) continue;
        
        // Skip amide bonds (they have restricted rotation)
        if (bonds[b].is_amide) continue;
        
        // Mark as rotatable
        bonds[b].is_rotatable = true;
        
        // Mark atoms as part of rotatable bonds
        atoms[a1].is_rotatable_bond_atom = true;
        atoms[a2].is_rotatable_bond_atom = true;
    }
}

// Calculate pharmacophore features
void calculate_pharmacophore_features(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count,
                                   PharmacophorePoint *points, int *point_count) {
    if (!atoms || atom_count <= 0 || !points || !point_count) return;
    
    *point_count = 0;
    
    // Identify hydrogen bond donors
    for (int i = 0; i < atom_count; i++) {
        // N-H and O-H groups are donors
        if ((strcmp(atoms[i].atom, "N") == 0 || strcmp(atoms[i].atom, "O") == 0) && 
            atoms[i].explicit_h_count > 0) {
            
            // Create pharmacophore point
            points[*point_count].x = atoms[i].x;
            points[*point_count].y = atoms[i].y;
            points[*point_count].z = atoms[i].z;
            points[*point_count].type = 0; // 0 = donor
            points[*point_count].radius = 1.5;
            points[*point_count].strength = 1.0;
            points[*point_count].associated_atom = i;
            
            (*point_count)++;
            
            // Mark the atom as a pharmacophore
            atoms[i].is_pharmacophore = true;
            atoms[i].pharmacophore_type = 0; // 0 = donor
        }
    }
    
    // Identify hydrogen bond acceptors
    for (int i = 0; i < atom_count; i++) {
        // N and O atoms are acceptors (unless positively charged)
        if ((strcmp(atoms[i].atom, "N") == 0 || strcmp(atoms[i].atom, "O") == 0) && 
            atoms[i].charge <= 0) {
            
            // Create pharmacophore point
            points[*point_count].x = atoms[i].x;
            points[*point_count].y = atoms[i].y;
            points[*point_count].z = atoms[i].z;
            points[*point_count].type = 1; // 1 = acceptor
            points[*point_count].radius = 1.5;
            points[*point_count].strength = 1.0;
            points[*point_count].associated_atom = i;
            
            (*point_count)++;
            
            // Mark the atom as a pharmacophore
            atoms[i].is_pharmacophore = true;
            atoms[i].pharmacophore_type = 1; // 1 = acceptor
        }
    }
    
    // Identify hydrophobic centers
    for (int i = 0; i < atom_count; i++) {
        // Carbon atoms with low electronegativity
        if (strcmp(atoms[i].atom, "C") == 0) {
            // Count neighboring carbons
            int carbon_neighbors = 0;
            for (int b = 0; b < bond_count; b++) {
                if (bonds[b].a == i || bonds[b].b == i) {
                    int neighbor = (bonds[b].a == i) ? bonds[b].b : bonds[b].a;
                    if (strcmp(atoms[neighbor].atom, "C") == 0) {
                        carbon_neighbors++;
                    }
                }
            }
            
            // If carbon has mostly carbon neighbors, it's hydrophobic
            if (carbon_neighbors >= 2) {
                // Create pharmacophore point
                points[*point_count].x = atoms[i].x;
                points[*point_count].y = atoms[i].y;
                points[*point_count].z = atoms[i].z;
                points[*point_count].type = 2; // 2 = hydrophobic
                points[*point_count].radius = 2.0;
                points[*point_count].strength = 0.8 + 0.2 * carbon_neighbors / atoms[i].n_bonds;
                points[*point_count].associated_atom = i;
                
                (*point_count)++;
                
                // Mark the atom as a pharmacophore
                atoms[i].is_pharmacophore = true;
                atoms[i].pharmacophore_type = 2; // 2 = hydrophobic
            }
        }
    }
    
    // Identify aromatic centers
    for (int i = 0; i < atom_count; i++) {
        if (atoms[i].is_aromatic) {
            // Create pharmacophore point
            points[*point_count].x = atoms[i].x;
            points[*point_count].y = atoms[i].y;
            points[*point_count].z = atoms[i].z;
            points[*point_count].type = 3; // 3 = aromatic
            points[*point_count].radius = 1.7;
            points[*point_count].strength = 1.0;
            points[*point_count].associated_atom = i;
            
            (*point_count)++;
            
            // Mark the atom as a pharmacophore
            atoms[i].is_pharmacophore = true;
            atoms[i].pharmacophore_type = 3; // 3 = aromatic
        }
    }
    
    // Identify positive ionizable centers
    for (int i = 0; i < atom_count; i++) {
        if (atoms[i].charge > 0 || 
            (strcmp(atoms[i].atom, "N") == 0 && atoms[i].n_bonds == 4)) {
            
            // Create pharmacophore point
            points[*point_count].x = atoms[i].x;
            points[*point_count].y = atoms[i].y;
            points[*point_count].z = atoms[i].z;
            points[*point_count].type = 4; // 4 = positive
            points[*point_count].radius = 1.5;
            points[*point_count].strength = atoms[i].charge > 0 ? abs((int)atoms[i].charge) : 0.7;
            points[*point_count].associated_atom = i;
            
            (*point_count)++;
            
            // Mark the atom as a pharmacophore
            atoms[i].is_pharmacophore = true;
            atoms[i].pharmacophore_type = 4; // 4 = positive
        }
    }
    
    // Identify negative ionizable centers
    for (int i = 0; i < atom_count; i++) {
        if (atoms[i].charge < 0) {
            // Create pharmacophore point
            points[*point_count].x = atoms[i].x;
            points[*point_count].y = atoms[i].y;
            points[*point_count].z = atoms[i].z;
            points[*point_count].type = 5; // 5 = negative
            points[*point_count].radius = 1.5;
            points[*point_count].strength = abs((int)atoms[i].charge);
            points[*point_count].associated_atom = i;
            
            (*point_count)++;
            
            // Mark the atom as a pharmacophore
            atoms[i].is_pharmacophore = true;
            atoms[i].pharmacophore_type = 5; // 5 = negative
        }
    }
}

// Get channel name from type
const char* get_channel_name(int channel_type) {
    for (int i = 0; channel_info_table[i].name != NULL; i++) {
        if (channel_info_table[i].type == channel_type) {
            return channel_info_table[i].name;
        }
    }
    return "Unknown";
}

// Get channel type from name
int get_channel_type_from_name(const char* name) {
    for (int i = 0; channel_info_table[i].name != NULL; i++) {
        if (strcmp(channel_info_table[i].name, name) == 0) {
            return channel_info_table[i].type;
        }
    }
    return -1;
}

// Get available channel names
void get_available_channel_names(char** names, int* count) {
    if (!names || !count) return;
    
    *count = 0;
    while (channel_info_table[*count].name != NULL) {
        names[*count] = (char*)channel_info_table[*count].name;
        (*count)++;
    }
}

// Get info about a specific channel
FeatureChannelInfo get_channel_info(int channel_type) {
    for (int i = 0; channel_info_table[i].name != NULL; i++) {
        if (channel_info_table[i].type == channel_type) {
            return channel_info_table[i];
        }
    }
    
    // Return default info for unknown channel
    FeatureChannelInfo info;
    info.type = channel_type;
    info.name = "Unknown";
    info.description = "Unknown channel type";
    info.requires_3d = false;
    return info;
}

// Normalize feature channels
void normalize_feature_channels(MultiChannelImage* image) {
    if (!image || !image->data) return;
    
    int size = image->width * image->height;
    
    for (int c = 0; c < image->channels; c++) {
        float min_val = FLT_MAX;
        float max_val = -FLT_MAX;
        
        // Find min and max
        for (int i = 0; i < size; i++) {
            float val = image->data[i * image->channels + c];
            min_val = fmin(min_val, val);
            max_val = fmax(max_val, val);
        }
        
        // Normalize
        float range = max_val - min_val;
        if (range > 1e-6) {
            for (int i = 0; i < size; i++) {
                int idx = i * image->channels + c;
                image->data[idx] = (image->data[idx] - min_val) / range;
            }
        }
    }
}

// Helper function to calculate atom contribution to a grid point
static float calculate_atom_contribution(AtomPos *atom, int channel, float x, float y, float z) {
    float dx = x - atom->x;
    float dy = y - atom->y;
    float dz = z - atom->z;
    float dist_sq = dx*dx + dy*dy + dz*dz;
    
    // Base falloff based on distance
    float radius = atom->radius * 2.0;
    float sigma = radius / 3.0;
    float falloff = expf(-dist_sq / (2.0f * sigma * sigma));
    
    // Channel-specific contributions (similar to CUDA kernel)
    switch (channel) {
        case CHANNEL_ELECTRON_DENSITY:
            return falloff * atom->electron_density_max;
            
        case CHANNEL_LIPOPHILICITY: {
            float lipo_val = 0.0f;
            if (strcmp(atom->atom, "C") == 0) lipo_val = 1.0f;
            else if (strcmp(atom->atom, "H") == 0) lipo_val = 0.5f;
            else if (strcmp(atom->atom, "O") == 0) lipo_val = -0.5f;
            else if (strcmp(atom->atom, "N") == 0) lipo_val = -0.4f;
            else if (strcmp(atom->atom, "F") == 0 || 
                     strcmp(atom->atom, "Cl") == 0 || 
                     strcmp(atom->atom, "Br") == 0) lipo_val = 0.7f;
            else if (strcmp(atom->atom, "S") == 0) lipo_val = 0.3f;
            else if (strcmp(atom->atom, "P") == 0) lipo_val = 0.2f;
            
            // Adjust based on environment
            if (atom->is_aromatic) lipo_val *= 1.2f;
            if (atom->in_ring) lipo_val *= 0.85f;
            
            return falloff * lipo_val;
        }
            
        case CHANNEL_HYDROGEN_DONOR: {
            float donor_val = 0.0f;
            if ((strcmp(atom->atom, "N") == 0 || strcmp(atom->atom, "O") == 0) && 
                atom->n_bonds < atom->valence) {
                donor_val = 1.0f;
            }
            
            // Include S-H as weak H-bond donor
            if (strcmp(atom->atom, "S") == 0 && atom->n_bonds < atom->valence) {
                donor_val = 0.4f;
            }
            
            // Consider environment effects
            if (atom->is_aromatic) donor_val *= 0.7f;  // Aromatic atoms are weaker donors
            if (atom->charge > 0) donor_val *= 0.5f;   // Positive charge weakens donating
            if (atom->charge < 0) donor_val *= 1.3f;   // Negative charge strengthens donating
            
            return falloff * donor_val;
        }
            
        case CHANNEL_HYDROGEN_ACCEPTOR: {
            float acceptor_val = 0.0f;
            
            // Basic acceptors
            if ((strcmp(atom->atom, "N") == 0 || strcmp(atom->atom, "O") == 0)) {
                acceptor_val = 1.0f;
            }
            
            // Weaker acceptors
            if ((strcmp(atom->atom, "S") == 0 || strcmp(atom->atom, "F") == 0)) {
                acceptor_val = 0.6f;
            }
            
            // Consider environment effects
            if (atom->is_aromatic) acceptor_val *= 0.8f;  // Aromatic atoms are weaker acceptors
            if (atom->charge < 0) acceptor_val *= 1.5f;   // Negative charge strengthens accepting
            if (atom->charge > 0) acceptor_val *= 0.3f;   // Positive charge weakens accepting
            
            return falloff * acceptor_val;
        }
            
        case CHANNEL_POSITIVE_CHARGE:
            return falloff * abs((int)atom->charge);
            
        case CHANNEL_NEGATIVE_CHARGE:
            return falloff * abs((int)-atom->charge);
            
        case CHANNEL_AROMATICITY:
            return falloff * (atom->is_aromatic ? 1.0f : 0.0f);
            
        case CHANNEL_SP2_HYBRIDIZATION:
            return falloff * (atom->hybridization > 1.5f && atom->hybridization < 2.5f ? 1.0f : 0.0f);
            
        case CHANNEL_SP3_HYBRIDIZATION:
            return falloff * (atom->hybridization > 2.5f ? 1.0f : 0.0f);
            
        case CHANNEL_GASTEIGER_CHARGE: {
            float charge_val = atom->charge;
            if (charge_val > 1.0f) charge_val = 1.0f;
            if (charge_val < -1.0f) charge_val = -1.0f;
            return falloff * (charge_val + 1.0f) / 2.0f;  // Scale to [0,1]
        }
            
        case CHANNEL_RING_MEMBERSHIP:
            return falloff * (atom->in_ring ? 1.0f : 0.0f);
            
        case CHANNEL_AROMATIC_RING:
            return falloff * (atom->in_ring && atom->is_aromatic ? 1.0f : 0.0f);
            
        case CHANNEL_ALIPHATIC_RING:
            return falloff * (atom->in_ring && !atom->is_aromatic ? 1.0f : 0.0f);
            
        case CHANNEL_POLARIZABILITY:
            return falloff * atom->polarizability / 5.0f;  // Normalize approx [0,1]
            
        case CHANNEL_VDWAALS_INTERACTION: {
            // VDW interaction potential based on atom type
            float vdw_val = atom->radius * 0.5f;
            
            // Distance dependence follows r^-6
            float r = sqrtf(dist_sq);
            if (r < 0.1f) r = 0.1f;  // Avoid singularity
            
            // Approximate with a smoother falloff than typical r^-6
            float vdw_falloff = expf(-dist_sq / (4.0f * sigma * sigma));
            return vdw_val * vdw_falloff;
        }
            
        case CHANNEL_ATOMIC_REFRACTIVITY:
            return falloff * atom->refractivity / 15.0f;  // Normalize approx [0,1]
            
        case CHANNEL_ELECTRONEGATIVITY:
            return falloff * atom->electronegativity / 4.0f;  // Normalize approx [0,1]
            
        case CHANNEL_BOND_ORDER_INFLUENCE: {
            // Represent the influence of bond orders
            float bond_order_val = 0.0f;
            
            // Simple approximation: average n_bonds / valence
            if (atom->valence > 0) {
                bond_order_val = (float)atom->n_bonds / (float)atom->valence;
                if (atom->is_aromatic) bond_order_val *= 1.2f; // Enhance for aromatic atoms
            }
            
            return falloff * bond_order_val;
        }
            
        case CHANNEL_STEREOCHEMISTRY:
            // For now, just use the Z-coord as indication of stereochemistry
            return falloff * (0.5f + atom->z * 0.1f);
            
        case CHANNEL_ROTATABLE_BOND_INFLUENCE: {
            // Approximation for atoms that might be part of rotatable bonds
            float rotatable_val = 0.0f;
            
            // Atoms with sp3 hybridization and not in rings are more likely part of rotatable bonds
            if (atom->hybridization > 2.5f && !atom->in_ring) {
                rotatable_val = 1.0f;
            } else if (!atom->in_ring && atom->n_bonds > 1) {
                rotatable_val = 0.7f;
            }
            
            // Use the explicit flag for more accurate detection
            if (atom->is_rotatable_bond_atom) {
                rotatable_val = 1.0f;
            }
            
            return falloff * rotatable_val;
        }
            
        case CHANNEL_MOLECULAR_SHAPE: {
            // Approximate molecular shape influence
            // Use distance from closest axis as shape indicator
            float axis_dist = fminf(fabsf(dx), fminf(fabsf(dy), fabsf(dz)));
            return falloff * (1.0f - axis_dist / (atom->radius + 1.0f));
        }
            
        case CHANNEL_SURFACE_ACCESSIBILITY: {
            // Approximate solvent accessibility based on solvent_accessibility property
            if (atom->solvent_accessibility > 0) {
                return falloff * atom->solvent_accessibility;
            }
            
            // Default approximation if not explicitly calculated
            // Atoms on the periphery are more accessible
            float peripheral_factor = sqrtf(atom->x*atom->x + atom->y*atom->y + atom->z*atom->z);
            
            // Scale by a typical factor to normalize
            peripheral_factor = fminf(peripheral_factor / 5.0f, 1.0f);
            
            // Adjust for atom type (hydrophobic atoms tend to be less accessible)
            if (strcmp(atom->atom, "C") == 0 && !atom->is_aromatic) peripheral_factor *= 0.8f;
            if (strcmp(atom->atom, "O") == 0 || strcmp(atom->atom, "N") == 0) peripheral_factor *= 1.2f;
            
            return falloff * peripheral_factor;
        }
            
        case CHANNEL_PHARMACOPHORE_HYDROPHOBIC: {
            // Hydrophobic feature - carbon atoms with low electronegativity
            float hydrophobic_val = 0.0f;
            
            if (strcmp(atom->atom, "C") == 0) {
                hydrophobic_val = 1.0f - (atom->electronegativity / 4.0f);
                
                // Adjust for aromatic character
                if (atom->is_aromatic) hydrophobic_val *= 0.7f;
                
                // Adjust for attached electronegative atoms (simplified)
                if (atom->explicit_h_count < 2) hydrophobic_val *= 0.8f;
            }
            
            // Use explicit pharmacophore flag if available
            if (atom->is_pharmacophore && atom->pharmacophore_type == 2) {
                hydrophobic_val = 1.0f;
            }
            
            return falloff * hydrophobic_val;
        }
            
        case CHANNEL_PHARMACOPHORE_AROMATIC: {
            // Aromatic feature for pharmacophore
            float aromatic_val = atom->is_aromatic ? 1.0f : 0.0f;
            
            // Enhanced value for atoms in fused ring systems
            if (atom->ring_count > 1 && atom->is_aromatic) {
                aromatic_val *= 1.3f;
            }
            
            // Use explicit pharmacophore flag if available
            if (atom->is_pharmacophore && atom->pharmacophore_type == 3) {
                aromatic_val = 1.0f;
            }
            
            return falloff * aromatic_val;
        }
            
        case CHANNEL_ISOTOPE_EFFECT: {
            // Represent isotope effects - heavier isotopes change properties
            float isotope_val = 0.0f;
            
            if (atom->isotope > 0) {
                // Scale based on how much heavier the isotope is relative to common isotope
                float relative_mass_increase = atom->isotope / 100.0f; // Just an approximation
                isotope_val = relative_mass_increase;
            }
            
            return falloff * isotope_val;
        }
            
        case CHANNEL_QUANTUM_EFFECTS: {
            // Simplified quantum mechanical effects
            float qm_val = 0.0f;
            
            // Use atom electronegativity, hybridization and aromaticity for QM approximation
            qm_val = (atom->electronegativity / 4.0f) * 
                    (0.5f + atom->hybridization / 6.0f) * 
                    (atom->is_aromatic ? 1.2f : 1.0f);
            
            // Consider orbital energy levels (highly simplified)
            if (atom->orbital_config[1] > 0) qm_val *= 1.1f; // p orbitals contribution
            
            return falloff * qm_val;
        }
            
        default:
            return falloff;
    }
}

// Generate feature image using CPU or CUDA
bool generate_feature_image(MultiChannelImage* image, AtomPos *atoms, int atom_count, 
                           BondSeg *bonds, int bond_count, ProgressCallback progress_callback) {
    // Mark unused parameters to suppress compiler warnings
    (void)bonds;
    (void)bond_count;
    
    if (!image || !image->data || !atoms || atom_count <= 0) {
        fprintf(stderr, "Error: Invalid parameters for feature image generation\n");
        return false;
    }
    
    if (progress_callback) {
        progress_callback(0.05f);
    }
    
    // Calculate molecule bounds for grid positioning
    double min_x, min_y, min_z, max_x, max_y, max_z;
    get_molecule_bounds(atoms, atom_count, &min_x, &min_y, &min_z, &max_x, &max_y, &max_z);
    
    // Add some padding
    double padding = 2.0;
    min_x -= padding; min_y -= padding; min_z -= padding;
    max_x += padding; max_y += padding; max_z += padding;
    
    // Calculate grid spacing
    double grid_step_x = (max_x - min_x) / image->width;
    double grid_step_y = (max_y - min_y) / image->height;
    double grid_step_z = (max_z - min_z) / 1.0;  // Z dimension is flattened
    
    if (progress_callback) {
        progress_callback(0.1f);
    }
    
    // Use CUDA if available
    bool using_cuda = check_cuda_available() && image->d_data != NULL;
    
    if (using_cuda) {
        #if defined(HAS_CUDA) || defined(__CUDACC__)
        // Allocate and copy atoms to device
        AtomPos *d_atoms = NULL;
        int *d_channel_types = NULL;
        cudaError_t cuda_err;
        void *d_atoms_ptr = NULL, *d_channel_types_ptr = NULL;
        
        cuda_err = cudaMalloc(&d_atoms_ptr, atom_count * sizeof(AtomPos));
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDA error: Failed to allocate device memory for atoms: %s\n", 
                    cudaGetErrorString(cuda_err));
            return false;
        }
        d_atoms = (AtomPos*)d_atoms_ptr;
        
        cuda_err = cudaMalloc(&d_channel_types_ptr, image->channels * sizeof(int));
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDA error: Failed to allocate device memory for channel types: %s\n", 
                    cudaGetErrorString(cuda_err));
            cudaFree(d_atoms);
            return false;
        }
        d_channel_types = (int*)d_channel_types_ptr;
        
        cuda_err = cudaMemcpy(d_atoms, atoms, atom_count * sizeof(AtomPos), cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDA error: Failed to copy atoms to device: %s\n", 
                    cudaGetErrorString(cuda_err));
            cudaFree(d_atoms);
            cudaFree(d_channel_types);
            return false;
        }
        
        cuda_err = cudaMemcpy(d_channel_types, image->channel_types, 
                             image->channels * sizeof(int), cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDA error: Failed to copy channel types to device: %s\n", 
                    cudaGetErrorString(cuda_err));
            cudaFree(d_atoms);
            cudaFree(d_channel_types);
            return false;
        }
        
        if (progress_callback) {
            progress_callback(0.2f);
        }
        
        // Launch CUDA kernel
        launch_generate_feature_image_cuda(
            image->d_data, image->width, image->height, image->channels,
            d_atoms, atom_count, d_channel_types,
            min_x, min_y, min_z, grid_step_x, grid_step_y, grid_step_z);
        
        if (progress_callback) {
            progress_callback(0.8f);
        }
        
        // Copy results back to host
        cuda_err = cudaMemcpy(image->data, image->d_data, 
                             image->width * image->height * image->channels * sizeof(float),
                             cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDA error: Failed to copy results from device: %s\n", 
                    cudaGetErrorString(cuda_err));
            cudaFree(d_atoms);
            cudaFree(d_channel_types);
            return false;
        }
        
        // Clean up
        cudaFree(d_atoms);
        cudaFree(d_channel_types);
        #endif
    } else {
        // CPU implementation
        // Zero initialize
        memset(image->data, 0, image->width * image->height * image->channels * sizeof(float));
        
        // Calculate contributions for each grid point
        for (int y = 0; y < image->height; y++) {
            if (progress_callback && y % 10 == 0) {
                progress_callback(0.1f + 0.8f * y / image->height);
            }
            
            for (int x = 0; x < image->width; x++) {
                // Calculate grid point coordinates
                float grid_x = min_x + x * grid_step_x;
                float grid_y = min_y + y * grid_step_y;
                float grid_z = min_z + 0.5 * grid_step_z;  // Middle of Z range
                
                // For each atom, calculate contribution to each channel
                for (int a = 0; a < atom_count; a++) {
                    for (int c = 0; c < image->channels; c++) {
                        int channel_type = image->channel_types[c];
                        float contribution = calculate_atom_contribution(
                            &atoms[a], channel_type, grid_x, grid_y, grid_z);
                        
                        // Add contribution to the grid point
                        int idx = (y * image->width + x) * image->channels + c;
                        image->data[idx] += contribution;
                    }
                }
            }
        }
    }
    
    // Normalize channels
    normalize_feature_channels(image);
    
    if (progress_callback) {
        progress_callback(1.0f);
    }
    
    return true;
}

// Save feature image to a file
bool save_feature_image(const MultiChannelImage* image, const char* filename) {
    if (!image || !image->data || !filename) {
        fprintf(stderr, "Error: Invalid parameters for saving feature image\n");
        return false;
    }
    
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return false;
    }
    
    // Write header
    fprintf(f, "FEATUREIMAGE\n");
    fprintf(f, "VERSION 1.0\n");
    fprintf(f, "WIDTH %d\n", image->width);
    fprintf(f, "HEIGHT %d\n", image->height);
    fprintf(f, "CHANNELS %d\n", image->channels);
    
    // Write channel types
    fprintf(f, "CHANNELTYPES");
    for (int c = 0; c < image->channels; c++) {
        fprintf(f, " %d", image->channel_types[c]);
    }
    fprintf(f, "\n");
    
    // Write channel names
    fprintf(f, "CHANNELNAMES");
    for (int c = 0; c < image->channels; c++) {
        fprintf(f, " %s", get_channel_name(image->channel_types[c]));
    }
    fprintf(f, "\n");
    
    // Write data
    fprintf(f, "DATA\n");
    fwrite(image->data, sizeof(float), image->width * image->height * image->channels, f);
    
    fclose(f);
    return true;
} 

// Get molecule bounds
void get_molecule_bounds(AtomPos *atoms, int atom_count, 
                       double *min_x, double *min_y, double *min_z,
                       double *max_x, double *max_y, double *max_z) {
    if (!atoms || atom_count <= 0) {
        *min_x = *min_y = *min_z = 0;
        *max_x = *max_y = *max_z = 0;
        return;
    }
    
    *min_x = *min_y = *min_z = 1e9;
    *max_x = *max_y = *max_z = -1e9;
    
    double padding = 1.5;  // Add padding based on atom radius
    
    for (int i = 0; i < atom_count; i++) {
        double rad = atoms[i].radius * padding;
        
        *min_x = fmin(*min_x, atoms[i].x - rad);
        *min_y = fmin(*min_y, atoms[i].y - rad);
        *min_z = fmin(*min_z, atoms[i].z - rad);
        
        *max_x = fmax(*max_x, atoms[i].x + rad);
        *max_y = fmax(*max_y, atoms[i].y + rad);
        *max_z = fmax(*max_z, atoms[i].z + rad);
    }
}

// Orient molecule along principal axes (PCA)
void orient_molecule_to_principal_axes(AtomPos *atoms, int atom_count) {
    if (!atoms || atom_count <= 0) return;
    
    // First center the molecule
    center_molecule(atoms, atom_count);
    
    // Calculate inertia tensor
    double inertia[3][3] = {{0}};
    // Double commented out to show it's intentionally not used
    // double total_mass = 0.0;
    
    for (int i = 0; i < atom_count; i++) {
        double mass = atoms[i].atomic_number;  // Use atomic number as approximation for mass
        // total_mass += mass;
        
        double x = atoms[i].x;
        double y = atoms[i].y;
        double z = atoms[i].z;
        
        inertia[0][0] += mass * (y*y + z*z);
        inertia[1][1] += mass * (x*x + z*z);
        inertia[2][2] += mass * (x*x + y*y);
        
        inertia[0][1] -= mass * x * y;
        inertia[0][2] -= mass * x * z;
        inertia[1][2] -= mass * y * z;
    }
    
    // Mirror the upper triangle to lower triangle
    inertia[1][0] = inertia[0][1];
    inertia[2][0] = inertia[0][2];
    inertia[2][1] = inertia[1][2];
    
    // Simplified approach: align with principal axes
    // In a full implementation, we would compute eigenvectors of inertia tensor
    
    // For now, do a simple approach - align longest dimension with x-axis, etc.
    double xx = 0, yy = 0, zz = 0;
    for (int i = 0; i < atom_count; i++) {
        xx += atoms[i].x * atoms[i].x;
        yy += atoms[i].y * atoms[i].y;
        zz += atoms[i].z * atoms[i].z;
    }
    
    // Simple rotation to ensure longest axis is on x, etc.
    if (xx < yy && xx < zz) {
        // Z is longest, rotate to make it X
        for (int i = 0; i < atom_count; i++) {
            double temp = atoms[i].x;
            atoms[i].x = atoms[i].z;
            atoms[i].z = -temp;
        }
    } else if (yy < xx && yy < zz) {
        // Y is longest, rotate to make it X
        for (int i = 0; i < atom_count; i++) {
            double temp = atoms[i].x;
            atoms[i].x = atoms[i].y;
            atoms[i].y = -temp;
        }
    }
    
    // Re-center after rotation
    center_molecule(atoms, atom_count);
}

// Center molecule at origin
void center_molecule(AtomPos *atoms, int atom_count) {
    if (!atoms || atom_count <= 0) return;
    
    double cx = 0, cy = 0, cz = 0;
    
    for (int i = 0; i < atom_count; i++) {
        cx += atoms[i].x;
        cy += atoms[i].y;
        cz += atoms[i].z;
    }
    
    cx /= atom_count;
    cy /= atom_count;
    cz /= atom_count;
    
    for (int i = 0; i < atom_count; i++) {
        atoms[i].x -= cx;
        atoms[i].y -= cy;
        atoms[i].z -= cz;
    }
}

// Calculate topological indices
void calculate_topological_indices(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count,
                                  double *wiener_index, double *balaban_index, double *randic_index) {
    if (!atoms || !bonds || atom_count <= 0 || bond_count <= 0) {
        if (wiener_index) *wiener_index = 0.0;
        if (balaban_index) *balaban_index = 0.0;
        if (randic_index) *randic_index = 0.0;
        return;
    }
    
    // Create adjacency matrix for distance calculation
    int *adjacency = (int*)calloc(atom_count * atom_count, sizeof(int));
    
    // Initialize to "infinity" (a large value)
    for (int i = 0; i < atom_count; i++) {
        for (int j = 0; j < atom_count; j++) {
            adjacency[i * atom_count + j] = (i == j) ? 0 : 999999;
        }
    }
    
    // Fill in known bond connections
    for (int b = 0; b < bond_count; b++) {
        int a1 = bonds[b].a;
        int a2 = bonds[b].b;
        
        // Use bond order as a weight (simple approach)
        adjacency[a1 * atom_count + a2] = bonds[b].order;
        adjacency[a2 * atom_count + a1] = bonds[b].order;
    }
    
    // Floyd-Warshall algorithm to compute all-pairs shortest paths
    for (int k = 0; k < atom_count; k++) {
        for (int i = 0; i < atom_count; i++) {
            for (int j = 0; j < atom_count; j++) {
                int direct = adjacency[i * atom_count + j];
                int indirect = adjacency[i * atom_count + k] + adjacency[k * atom_count + j];
                if (indirect < direct) {
                    adjacency[i * atom_count + j] = indirect;
                }
            }
        }
    }
    
    // Compute Wiener Index
    if (wiener_index) {
        *wiener_index = 0.0;
        for (int i = 0; i < atom_count; i++) {
            for (int j = i + 1; j < atom_count; j++) {
                *wiener_index += adjacency[i * atom_count + j];
            }
        }
    }
    
    // Compute Balaban Index (simplified)
    if (balaban_index) {
        *balaban_index = 0.0;
        
        // Calculate distance sums for each vertex
        double *distance_sums = (double*)calloc(atom_count, sizeof(double));
        
        for (int i = 0; i < atom_count; i++) {
            distance_sums[i] = 0.0;
            for (int j = 0; j < atom_count; j++) {
                distance_sums[i] += adjacency[i * atom_count + j];
            }
        }
        
        // Calculate Balaban Index
        double q = bond_count - atom_count + 1.0; // Cyclomatic number
        double sum = 0.0;
        
        for (int b = 0; b < bond_count; b++) {
            int a1 = bonds[b].a;
            int a2 = bonds[b].b;
            
            if (distance_sums[a1] > 0.0 && distance_sums[a2] > 0.0) {
                sum += 1.0 / sqrt(distance_sums[a1] * distance_sums[a2]);
            }
        }
        
        *balaban_index = (q / (q+1.0)) * sum;
        
        free(distance_sums);
    }
    
    // Compute Randic Index
    if (randic_index) {
        *randic_index = 0.0;
        
        // Calculate vertex degrees (number of connections)
        int *degrees = (int*)calloc(atom_count, sizeof(int));
        
        for (int b = 0; b < bond_count; b++) {
            degrees[bonds[b].a]++;
            degrees[bonds[b].b]++;
        }
        
        // Calculate Randic Index
        for (int b = 0; b < bond_count; b++) {
            int a1 = bonds[b].a;
            int a2 = bonds[b].b;
            
            if (degrees[a1] > 0 && degrees[a2] > 0) {
                *randic_index += 1.0 / sqrt(degrees[a1] * degrees[a2]);
            }
        }
        
        free(degrees);
    }
    
    free(adjacency);
}

// Calculate approximate solvent accessibility
void calculate_solvent_accessibility(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count) {
    // Mark parameters as unused to suppress warnings
    (void)bonds;
    (void)bond_count;
    
    if (!atoms || atom_count <= 0) return;
    
    // Simplified approach for estimating solvent accessibility 
    // Based on distance from center and number of neighboring atoms
    
    // Find molecular center
    double center_x = 0.0, center_y = 0.0, center_z = 0.0;
    for (int i = 0; i < atom_count; i++) {
        center_x += atoms[i].x;
        center_y += atoms[i].y;
        center_z += atoms[i].z;
    }
    center_x /= atom_count;
    center_y /= atom_count;
    center_z /= atom_count;
    
    // Calculate maximum distance from center (for normalization)
    double max_distance = 0.0;
    for (int i = 0; i < atom_count; i++) {
        double dx = atoms[i].x - center_x;
        double dy = atoms[i].y - center_y;
        double dz = atoms[i].z - center_z;
        double dist = sqrt(dx*dx + dy*dy + dz*dz);
        max_distance = fmax(max_distance, dist);
    }
    
    // Calculate neighbor counts for each atom
    int *neighbor_counts = (int*)calloc(atom_count, sizeof(int));
    for (int b = 0; b < bond_count; b++) {
        neighbor_counts[bonds[b].a]++;
        neighbor_counts[bonds[b].b]++;
    }
    
    // Find maximum neighbor count (for normalization)
    int max_neighbors = 0;
    for (int i = 0; i < atom_count; i++) {
        max_neighbors = fmax(max_neighbors, neighbor_counts[i]);
    }
    
    // Calculate solvent accessibility based on distance from center and neighbor count
    for (int i = 0; i < atom_count; i++) {
        double dx = atoms[i].x - center_x;
        double dy = atoms[i].y - center_y;
        double dz = atoms[i].z - center_z;
        double dist = sqrt(dx*dx + dy*dy + dz*dz);
        
        // Normalize distance to [0,1]
        double norm_dist = (max_distance > 0.0) ? dist / max_distance : 0.0;
        
        // Normalize neighbor count to [0,1] and invert (fewer neighbors = more accessible)
        double norm_neighbors = (max_neighbors > 0) ? 
            1.0 - (double)neighbor_counts[i] / max_neighbors : 1.0;
        
        // Combine factors (distance from center and inverse neighbor count)
        atoms[i].solvent_accessibility = 0.7 * norm_dist + 0.3 * norm_neighbors;
        
        // Adjust for atom type
        if (strcmp(atoms[i].atom, "C") == 0 && !atoms[i].is_aromatic) {
            atoms[i].solvent_accessibility *= 0.8;  // Hydrophobic atoms less accessible
        }
        if (strcmp(atoms[i].atom, "O") == 0 || strcmp(atoms[i].atom, "N") == 0) {
            atoms[i].solvent_accessibility *= 1.2;  // Polar atoms more accessible
        }
        
        // Surface area is proportional to radius squared
        atoms[i].surface_area = 4.0 * M_PI * atoms[i].radius * atoms[i].radius * atoms[i].solvent_accessibility;
    }
    
    free(neighbor_counts);
}

// Identify ring systems in the molecule
void identify_ring_systems(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count) {
    if (!atoms || !bonds || atom_count <= 0 || bond_count <= 0) return;
    
    // Initialize ring counts and ring sizes
    for (int i = 0; i < atom_count; i++) {
        atoms[i].ring_count = 0;
        for (int j = 0; j < 4; j++) {
            atoms[i].ring_sizes[j] = 0;
        }
    }
    
    // Initialize bond ring size
    for (int i = 0; i < bond_count; i++) {
        bonds[i].ring_size = 0;
    }
    
    // Create adjacency list for efficient traversal
    int *adjacency_starts = (int*)calloc(atom_count + 1, sizeof(int));
    int *adjacency_list = (int*)calloc(bond_count * 2, sizeof(int));
    int *adjacency_bond_idx = (int*)calloc(bond_count * 2, sizeof(int));
    
    // Count neighbors
    for (int b = 0; b < bond_count; b++) {
        adjacency_starts[bonds[b].a + 1]++;
        adjacency_starts[bonds[b].b + 1]++;
    }
    
    // Prefix sum to determine start positions
    for (int i = 1; i <= atom_count; i++) {
        adjacency_starts[i] += adjacency_starts[i - 1];
    }
    
    // Fill adjacency list
    int *cur_pos = (int*)calloc(atom_count, sizeof(int));
    for (int b = 0; b < bond_count; b++) {
        int a1 = bonds[b].a;
        int a2 = bonds[b].b;
        
        adjacency_list[adjacency_starts[a1] + cur_pos[a1]] = a2;
        adjacency_bond_idx[adjacency_starts[a1] + cur_pos[a1]] = b;
        cur_pos[a1]++;
        
        adjacency_list[adjacency_starts[a2] + cur_pos[a2]] = a1;
        adjacency_bond_idx[adjacency_starts[a2] + cur_pos[a2]] = b;
        cur_pos[a2]++;
    }
    
    // BFS for each atom to find rings
    for (int start = 0; start < atom_count; start++) {
        // Skip atoms already processed
        if (atoms[start].ring_count > 0) continue;
        
        int *queue = (int*)calloc(atom_count, sizeof(int));
        int *parent = (int*)calloc(atom_count, sizeof(int));
        int *distance = (int*)calloc(atom_count, sizeof(int));
        bool *visited = (bool*)calloc(atom_count, sizeof(bool));
        
        // Initialize
        for (int i = 0; i < atom_count; i++) {
            parent[i] = -1;
            distance[i] = -1;
        }
        
        // BFS
        int head = 0, tail = 0;
        queue[tail++] = start;
        visited[start] = true;
        distance[start] = 0;
        
        while (head < tail) {
            int current = queue[head++];
            
            // Process neighbors
            for (int i = adjacency_starts[current]; i < adjacency_starts[current + 1]; i++) {
                int neighbor = adjacency_list[i];
                
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    parent[neighbor] = current;
                    distance[neighbor] = distance[current] + 1;
                    queue[tail++] = neighbor;
                } 
                // Found a cycle if neighbor is visited but not the parent
                else if (neighbor != parent[current] && distance[neighbor] <= distance[current]) {
                    // Reconstruct the cycle
                    int cycle_size = distance[current] - distance[neighbor] + 1;
                    
                    // Only consider cycles of reasonable size (3-8 atoms)
                    if (cycle_size >= 3 && cycle_size <= 8) {
                        // Mark atoms and bonds in this cycle
                        int back_atom = current;
                        while (back_atom != neighbor) {
                            int p = parent[back_atom];
                            
                            // Find the bond between back_atom and p
                            for (int j = adjacency_starts[back_atom]; j < adjacency_starts[back_atom + 1]; j++) {
                                if (adjacency_list[j] == p) {
                                    int bond_idx = adjacency_bond_idx[j];
                                    bonds[bond_idx].in_ring = true;
                                    if (bonds[bond_idx].ring_size == 0 || 
                                        cycle_size < bonds[bond_idx].ring_size) {
                                        bonds[bond_idx].ring_size = cycle_size;
                                    }
                                    break;
                                }
                            }
                            
                            // Mark atom as in ring and update ring count and sizes
                            atoms[back_atom].in_ring = true;
                            if (atoms[back_atom].ring_count < 4) {
                                // Update ring_sizes if this is a new ring size or smaller than existing
                                bool found = false;
                                for (int r = 0; r < atoms[back_atom].ring_count; r++) {
                                    if (atoms[back_atom].ring_sizes[r] == cycle_size) {
                                        found = true;
                                        break;
                                    }
                                }
                                if (!found) {
                                    atoms[back_atom].ring_sizes[atoms[back_atom].ring_count++] = cycle_size;
                                }
                            }
                            
                            back_atom = p;
                        }
                        
                        // Mark the final atom (neighbor)
                        atoms[neighbor].in_ring = true;
                        if (atoms[neighbor].ring_count < 4) {
                            // Update ring_sizes if this is a new ring size or smaller than existing
                            bool found = false;
                            for (int r = 0; r < atoms[neighbor].ring_count; r++) {
                                if (atoms[neighbor].ring_sizes[r] == cycle_size) {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                atoms[neighbor].ring_sizes[atoms[neighbor].ring_count++] = cycle_size;
                            }
                        }
                    }
                }
            }
        }
        
        free(queue);
        free(parent);
        free(distance);
        free(visited);
    }
    
    // Clean up
    free(adjacency_starts);
    free(adjacency_list);
    free(adjacency_bond_idx);
    free(cur_pos);
}

// Analyze conformational flexibility
void analyze_conformational_flexibility(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count) {
    if (!atoms || !bonds || atom_count <= 0 || bond_count <= 0) return;
    
    // First identify rotatable bonds
    identify_rotatable_bonds(atoms, atom_count, bonds, bond_count);
    
    // Count rotatable bonds and calculate overall molecular flexibility
    double total_flexibility = 0.0;
    
    for (int b = 0; b < bond_count; b++) {
        if (bonds[b].is_rotatable) {
            int a1 = bonds[b].a;
            int a2 = bonds[b].b;
            
            // Count neighbors on each side of the rotatable bond
            int a1_neighbors = 0, a2_neighbors = 0;
            
            for (int b2 = 0; b2 < bond_count; b2++) {
                if (b2 == b) continue;
                
                if (bonds[b2].a == a1 || bonds[b2].b == a1) {
                    a1_neighbors++;
                }
                
                if (bonds[b2].a == a2 || bonds[b2].b == a2) {
                    a2_neighbors++;
                }
            }
            
            // Calculate rough energy barriers for rotation (approximate)
            // Based on number of neighbors and atom types
            double rotation_energy = 0.0;
            
            // More neighbors = higher barrier
            rotation_energy += (a1_neighbors + a2_neighbors) * 0.5;
            
            // Special case for specific atoms (e.g., amides, conjugated systems)
            if (bonds[b].is_conjugated) {
                rotation_energy += 3.0;  // Higher barrier for conjugated bonds
            }
            
            if (strcmp(atoms[a1].atom, "C") == 0 && strcmp(atoms[a2].atom, "N") == 0 &&
                bonds[b].is_amide) {
                rotation_energy += 5.0;  // Very high barrier for amide bonds
            }
            
            // Store as bond energy for now (not ideal but reusing existing field)
            bonds[b].bond_energy = rotation_energy;
            
            // Add to total flexibility (inverse of rotation energy)
            total_flexibility += 1.0 / (1.0 + rotation_energy);
        }
    }
    
    // Store the flexibility measure in molecules if a field is available
    // For now, just print it to indicate we're using the value
    if (atom_count > 0) {
        // Store in related existing field or simply comment out if no suitable field exists
        // Using polarizability as an approximate surrogate for flexibility
        atoms[0].polarizability += total_flexibility;
    }
}

// Detect stereocenter atoms
void detect_stereocenters(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count) {
    if (!atoms || atom_count <= 0) return;
    
    // Reset chirality for all atoms
    for (int i = 0; i < atom_count; i++) {
        atoms[i].is_chiral = false;
        atoms[i].chirality = '\0';
    }
    
    // Find potential stereocenters (atoms with 4 different substituents)
    for (int i = 0; i < atom_count; i++) {
        // Skip atoms with fewer than 4 connections
        if (atoms[i].n_bonds < 4) continue;
        
        // Carbon and silicon are common stereogenic atoms
        if (strcmp(atoms[i].atom, "C") != 0 && strcmp(atoms[i].atom, "Si") != 0) continue;
        
        // Find connected atoms
        int connected_atoms[4] = {-1, -1, -1, -1};
        int conn_count = 0;
        
        for (int b = 0; b < bond_count && conn_count < 4; b++) {
            if (bonds[b].a == i) {
                connected_atoms[conn_count++] = bonds[b].b;
            }
            else if (bonds[b].b == i) {
                connected_atoms[conn_count++] = bonds[b].a;
            }
        }
        
        // Skip if not 4 connections
        if (conn_count != 4) continue;
        
        // Check if substituents are different
        bool all_different = true;
        
        for (int j = 0; j < 3 && all_different; j++) {
            for (int k = j + 1; k < 4 && all_different; k++) {
                // Simple check - just compare atom types
                if (strcmp(atoms[connected_atoms[j]].atom, atoms[connected_atoms[k]].atom) == 0) {
                    all_different = false;
                }
            }
        }
        
        if (all_different) {
            atoms[i].is_chiral = true;
            
            // For demo purposes, assign R or S based on Z-coordinate
            // In a real implementation, would check CIP rules
            atoms[i].chirality = (atoms[i].z > 0.0) ? 'R' : 'S';
        }
    }
}

// Approximate resonance effects
void approximate_resonance_effects(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count) {
    if (!atoms || !bonds || atom_count <= 0 || bond_count <= 0) return;
    
    // First pass: identify conjugated systems
    for (int b = 0; b < bond_count; b++) {
        // Initialize bond as non-conjugated
        bonds[b].is_conjugated = false;
        
        // Aromatic bonds are already conjugated
        if (bonds[b].type == BOND_AROMATIC) {
            bonds[b].is_conjugated = true;
            continue;
        }
        
        // Check if this is a single bond between sp2 hybridized atoms
        int a1 = bonds[b].a;
        int a2 = bonds[b].b;
        
        if (bonds[b].order == 1 && 
            atoms[a1].hybridization > 1.5 && atoms[a1].hybridization < 2.5 && 
            atoms[a2].hybridization > 1.5 && atoms[a2].hybridization < 2.5) {
            
            bool adjacent_to_multiple_bond = false;
            
            // Check if adjacent to a double/triple bond
            for (int b2 = 0; b2 < bond_count; b2++) {
                if (b2 == b) continue;
                
                if ((bonds[b2].a == a1 || bonds[b2].b == a1 || 
                     bonds[b2].a == a2 || bonds[b2].b == a2) && 
                    (bonds[b2].order > 1 || bonds[b2].type == BOND_AROMATIC)) {
                    adjacent_to_multiple_bond = true;
                    break;
                }
            }
            
            if (adjacent_to_multiple_bond) {
                bonds[b].is_conjugated = true;
            }
        }
    }
    
    // Second pass: Update charges for atoms in conjugated systems
    for (int i = 0; i < atom_count; i++) {
        if (atoms[i].is_aromatic) {
            // Aromatic atoms have partial charges distributed according to electronegativity
            double charge_adjustment = 0.0;
            
            if (strcmp(atoms[i].atom, "C") == 0) {
                charge_adjustment = 0.05; // Slightly positive
            } else if (strcmp(atoms[i].atom, "N") == 0) {
                charge_adjustment = -0.1; // Slightly negative
            } else if (strcmp(atoms[i].atom, "O") == 0) {
                charge_adjustment = -0.15; // More negative
            } else if (strcmp(atoms[i].atom, "S") == 0) {
                charge_adjustment = -0.08; // Somewhat negative
            }
            
            atoms[i].charge += charge_adjustment;
        }
    }
    
    // Third pass: Update bond orders for conjugated systems
    for (int b = 0; b < bond_count; b++) {
        if (bonds[b].is_conjugated && bonds[b].order == 1) {
            // Single bonds in conjugated systems have partial double bond character
            // Represent this by setting a field (using bond_dipole as approximation)
            bonds[b].bond_dipole = 0.5; // Represents 1.5 effective bond order
        }
    }
}

// Estimate charge densities using a simplified molecular orbital approach
void estimate_charge_densities(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count) {
    if (!atoms || atom_count <= 0) return;
    
    // Initialize matrices for Hückel method (simplified)
    // For a real implementation, would use a full quantum chemistry approach
    
    // Reset electron density values
    for (int i = 0; i < atom_count; i++) {
        atoms[i].electron_density_max = 0.0;
    }
    
    // First, identify π systems (sp2 hybridized atoms)
    bool *in_pi_system = (bool*)calloc(atom_count, sizeof(bool));
    int pi_atom_count = 0;
    
    for (int i = 0; i < atom_count; i++) {
        if (atoms[i].hybridization > 1.5 && atoms[i].hybridization < 2.5) {
            in_pi_system[i] = true;
            pi_atom_count++;
        }
    }
    
    // If no π system, use simple electronegativity model
    if (pi_atom_count == 0) {
        for (int i = 0; i < atom_count; i++) {
            // Base electron density on electronegativity
            atoms[i].electron_density_max = atoms[i].electronegativity / 4.0 * 1.5;
            
            // Adjust for charge
            if (atoms[i].charge < 0) {
                atoms[i].electron_density_max += abs((int)atoms[i].charge) * 0.5;
            } else if (atoms[i].charge > 0) {
                atoms[i].electron_density_max -= atoms[i].charge * 0.3;
                if (atoms[i].electron_density_max < 0.2) {
                    atoms[i].electron_density_max = 0.2;
                }
            }
            
            // Adjust for lone pairs
            int val_electrons = atoms[i].valence;
            int used_electrons = atoms[i].n_bonds;
            int lone_pairs = (val_electrons - used_electrons) / 2;
            
            atoms[i].electron_density_max += lone_pairs * 0.3;
        }
        
        free(in_pi_system);
        return;
    }
    
    // For π systems, use a simplified Hückel-like approach
    double *alpha = (double*)calloc(atom_count, sizeof(double));
    double *beta = (double*)calloc(bond_count, sizeof(double));
    
    // Set Hückel parameters (simplified)
    for (int i = 0; i < atom_count; i++) {
        if (in_pi_system[i]) {
            // Alpha parameter based on element (ionization energy)
            if (strcmp(atoms[i].atom, "C") == 0) {
                alpha[i] = -11.4; // eV
            } else if (strcmp(atoms[i].atom, "N") == 0) {
                alpha[i] = -14.5;
            } else if (strcmp(atoms[i].atom, "O") == 0) {
                alpha[i] = -17.7;
            } else {
                alpha[i] = -10.0; // Default
            }
            
            // Correct for charge
            alpha[i] += atoms[i].charge * 2.0;
        }
    }
    
    // Strength of interactions (beta)
    for (int b = 0; b < bond_count; b++) {
        int a1 = bonds[b].a;
        int a2 = bonds[b].b;
        
        if (in_pi_system[a1] && in_pi_system[a2]) {
            if (bonds[b].order == 1) {
                beta[b] = -2.8; // eV, weaker for single bonds
            } else if (bonds[b].order >= 2 || bonds[b].type == BOND_AROMATIC) {
                beta[b] = -3.0; // eV, stronger for multiple bonds
            }
        }
    }
    
    // Very simplified calculation of π electron density
    for (int i = 0; i < atom_count; i++) {
        if (in_pi_system[i]) {
            double density = 0.0;
            
            // Contribution from alpha parameter (self-energy)
            density = -alpha[i] / 10.0; // Normalize to reasonable range
            
            // Contribution from neighboring atoms (resonance)
            for (int b = 0; b < bond_count; b++) {
                if ((bonds[b].a == i || bonds[b].b == i) && beta[b] != 0.0) {
                    int other = (bonds[b].a == i) ? bonds[b].b : bonds[b].a;
                    if (in_pi_system[other]) {
                        density += fabs(beta[b]) / 10.0 * atoms[other].electronegativity / 4.0;
                    }
                }
            }
            
            // Adjust for charge
            if (atoms[i].charge < 0) {
                density += abs((int)atoms[i].charge) * 0.5;
            } else if (atoms[i].charge > 0) {
                density -= atoms[i].charge * 0.3;
                if (density < 0.2) density = 0.2;
            }
            
            atoms[i].electron_density_max = density;
        } else {
            // For non-π atoms, use electronegativity model
            atoms[i].electron_density_max = atoms[i].electronegativity / 4.0 * 1.5;
            
            // Adjust for charge
            if (atoms[i].charge < 0) {
                atoms[i].electron_density_max += abs((int)atoms[i].charge) * 0.5;
            } else if (atoms[i].charge > 0) {
                atoms[i].electron_density_max -= atoms[i].charge * 0.3;
                if (atoms[i].electron_density_max < 0.2) {
                    atoms[i].electron_density_max = 0.2;
                }
            }
        }
    }
    
    // Clean up
    free(in_pi_system);
    free(alpha);
    free(beta);
}

// Calculate simplified orbital interactions
void calculate_simplified_orbital_interactions(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count) {
    if (!atoms || !bonds || atom_count <= 0 || bond_count <= 0) return;
    
    // Calculate orbital overlaps and energy levels (highly simplified)
    for (int b = 0; b < bond_count; b++) {
        int a1 = bonds[b].a;
        int a2 = bonds[b].b;
        
        // Calculate orbital energy based on bond type and atoms involved
        double orbital_energy = 0.0;
        
        // Base energy from bond order
        if (bonds[b].order == 1) {
            orbital_energy = -10.0; // sigma bond
        } else if (bonds[b].order == 2) {
            orbital_energy = -12.0; // sigma + pi bond
        } else if (bonds[b].order == 3) {
            orbital_energy = -15.0; // sigma + 2 pi bonds
        } else if (bonds[b].type == BOND_AROMATIC) {
            orbital_energy = -11.5; // aromatic bond
        }
        
        // Adjust for electronegativity difference
        double en_diff = fabs(atoms[a1].electronegativity - atoms[a2].electronegativity);
        orbital_energy -= en_diff * 2.0;
        
        // Store in bond energy field (reusing existing field)
        bonds[b].bond_energy = orbital_energy;
    }
    
    // Update electron density based on orbital interactions
    for (int i = 0; i < atom_count; i++) {
        double density = 0.0;
        
        // Base density from electronegativity
        density = atoms[i].electronegativity / 4.0 * 1.5;
        
        // Add contribution from bonds
        for (int b = 0; b < bond_count; b++) {
            if (bonds[b].a == i || bonds[b].b == i) {
                int other = (bonds[b].a == i) ? bonds[b].b : bonds[b].a;
                
                // Electron density shifts toward more electronegative atom
                if (atoms[i].electronegativity > atoms[other].electronegativity) {
                    density += 0.1 * (atoms[i].electronegativity - atoms[other].electronegativity);
                }
                
                // Bond order increases electron density
                density += 0.05 * bonds[b].order;
                
                // Aromatic bonds give special contribution
                if (bonds[b].type == BOND_AROMATIC) {
                    density += 0.1;
                }
            }
        }
        
        // Adjust for charge
        if (atoms[i].charge < 0) {
            density += abs((int)atoms[i].charge) * 0.5;
        } else if (atoms[i].charge > 0) {
            density -= atoms[i].charge * 0.3;
            if (density < 0.2) density = 0.2;
        }
        
        atoms[i].electron_density_max = density;
    }
}

// Predict binding hotspots on the molecule surface
void predict_binding_hotspots(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count,
                             double **hotspot_coords, int *hotspot_count) {
    if (!atoms || atom_count <= 0 || !hotspot_coords || !hotspot_count) {
        *hotspot_count = 0;
        return;
    }
    
    // First ensure solvent accessibility is calculated
    calculate_solvent_accessibility(atoms, atom_count, bonds, bond_count);
    
    // Identify potential hotspots (accessible pharmacophore features)
    const int max_hotspots = 10;
    double *hotspot_array = (double*)malloc(max_hotspots * 3 * sizeof(double));
    
    *hotspot_count = 0;
    
    // Check hydrogen bond donors
    for (int i = 0; i < atom_count && *hotspot_count < max_hotspots; i++) {
        if ((strcmp(atoms[i].atom, "N") == 0 || strcmp(atoms[i].atom, "O") == 0) && 
            atoms[i].explicit_h_count > 0 && atoms[i].solvent_accessibility > 0.5) {
            
            hotspot_array[*hotspot_count * 3] = atoms[i].x;
            hotspot_array[*hotspot_count * 3 + 1] = atoms[i].y;
            hotspot_array[*hotspot_count * 3 + 2] = atoms[i].z;
            (*hotspot_count)++;
        }
    }
    
    // Check hydrogen bond acceptors
    for (int i = 0; i < atom_count && *hotspot_count < max_hotspots; i++) {
        if ((strcmp(atoms[i].atom, "N") == 0 || strcmp(atoms[i].atom, "O") == 0) && 
            atoms[i].charge <= 0 && atoms[i].solvent_accessibility > 0.5) {
            
            hotspot_array[*hotspot_count * 3] = atoms[i].x;
            hotspot_array[*hotspot_count * 3 + 1] = atoms[i].y;
            hotspot_array[*hotspot_count * 3 + 2] = atoms[i].z;
            (*hotspot_count)++;
        }
    }
    
    // Check charged groups
    for (int i = 0; i < atom_count && *hotspot_count < max_hotspots; i++) {
        if (abs((int)atoms[i].charge) > 0.5 && atoms[i].solvent_accessibility > 0.5) {
            hotspot_array[*hotspot_count * 3] = atoms[i].x;
            hotspot_array[*hotspot_count * 3 + 1] = atoms[i].y;
            hotspot_array[*hotspot_count * 3 + 2] = atoms[i].z;
            (*hotspot_count)++;
        }
    }
    
    // Return the hotspot coordinates
    *hotspot_coords = hotspot_array;
}

// Optimize 3D geometry using a simplified force field
void optimize_3d_geometry(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count, 
                         double *energy, double convergence_threshold) {
    if (!atoms || atom_count <= 0 || !bonds || bond_count <= 0) {
        if (energy) *energy = 0.0;
        return;
    }
    
    // Allocate memory for forces
    double *forces_x = (double*)calloc(atom_count, sizeof(double));
    double *forces_y = (double*)calloc(atom_count, sizeof(double));
    double *forces_z = (double*)calloc(atom_count, sizeof(double));
    
    // Define force field parameters (very simplified)
    const double k_bond = 300.0;      // Bond stretching force constant
    const double k_angle = 100.0;     // Angle bending force constant
    const double k_vdw = 50.0;        // Van der Waals force constant
    const double k_elec = 80.0;       // Electrostatic force constant
    const double cutoff = 12.0;       // Cutoff distance for non-bonded interactions
    const double time_step = 0.001;   // Integration time step
    const int max_iterations = 500;   // Maximum number of iterations
    
    // Store ideal bond lengths
    double *ideal_bond_lengths = (double*)malloc(bond_count * sizeof(double));
    for (int b = 0; b < bond_count; b++) {
        int a1 = bonds[b].a;
        int a2 = bonds[b].b;
        
        // Basic ideal length based on atom types
        ideal_bond_lengths[b] = atoms[a1].radius + atoms[a2].radius;
        
        // Adjust for bond order
        if (bonds[b].order == 2) ideal_bond_lengths[b] *= 0.85;
        else if (bonds[b].order == 3) ideal_bond_lengths[b] *= 0.75;
        else if (bonds[b].type == BOND_AROMATIC) ideal_bond_lengths[b] *= 0.90;
    }
    
    // Create connectivity matrix to identify 1-3 and 1-4 interactions
    int *connectivity = (int*)calloc(atom_count * atom_count, sizeof(int));
    
    // Initialize with direct connections (1-2)
    for (int b = 0; b < bond_count; b++) {
        int a1 = bonds[b].a;
        int a2 = bonds[b].b;
        connectivity[a1 * atom_count + a2] = 1;
        connectivity[a2 * atom_count + a1] = 1;
    }
    
    // Calculate 1-3 connections (atoms connected through one other atom)
    for (int i = 0; i < atom_count; i++) {
        for (int j = 0; j < atom_count; j++) {
            if (connectivity[i * atom_count + j] == 1) {
                for (int k = 0; k < atom_count; k++) {
                    if (connectivity[j * atom_count + k] == 1 && i != k && connectivity[i * atom_count + k] == 0) {
                        connectivity[i * atom_count + k] = 2;
                        connectivity[k * atom_count + i] = 2;
                    }
                }
            }
        }
    }
    
    // Calculate 1-4 connections (atoms connected through two other atoms)
    for (int i = 0; i < atom_count; i++) {
        for (int j = 0; j < atom_count; j++) {
            if (connectivity[i * atom_count + j] == 2) {
                for (int k = 0; k < atom_count; k++) {
                    if (connectivity[j * atom_count + k] == 1 && i != k && 
                        connectivity[i * atom_count + k] == 0) {
                        connectivity[i * atom_count + k] = 3;
                        connectivity[k * atom_count + i] = 3;
                    }
                }
            }
        }
    }
    
    // Main optimization loop
    double prev_energy = 1e10;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // Calculate current energy and forces
        double total_energy = 0.0;
        
        // Reset forces
        memset(forces_x, 0, atom_count * sizeof(double));
        memset(forces_y, 0, atom_count * sizeof(double));
        memset(forces_z, 0, atom_count * sizeof(double));
        
        // Bond stretching energy and forces
        for (int b = 0; b < bond_count; b++) {
            int a1 = bonds[b].a;
            int a2 = bonds[b].b;
            
            double dx = atoms[a2].x - atoms[a1].x;
            double dy = atoms[a2].y - atoms[a1].y;
            double dz = atoms[a2].z - atoms[a1].z;
            double dist = sqrt(dx*dx + dy*dy + dz*dz);
            
            if (dist < 0.1) dist = 0.1; // Avoid division by zero
            
            double delta = dist - ideal_bond_lengths[b];
            double bond_energy = 0.5 * k_bond * delta * delta;
            total_energy += bond_energy;
            
            double force_mag = k_bond * delta;
            double fx = force_mag * dx / dist;
            double fy = force_mag * dy / dist;
            double fz = force_mag * dz / dist;
            
            forces_x[a1] += fx;
            forces_y[a1] += fy;
            forces_z[a1] += fz;
            forces_x[a2] -= fx;
            forces_y[a2] -= fy;
            forces_z[a2] -= fz;
        }
        
        // Angle bending energy and forces (simplified, only for 1-3 connections)
        for (int i = 0; i < atom_count; i++) {
            for (int j = 0; j < atom_count; j++) {
                if (connectivity[i * atom_count + j] == 2) {
                    // Find the common atom that connects i and j
                    for (int k = 0; k < atom_count; k++) {
                        if (connectivity[i * atom_count + k] == 1 && 
                            connectivity[j * atom_count + k] == 1) {
                            
                            // k is the middle atom of the angle i-k-j
                            double dx1 = atoms[i].x - atoms[k].x;
                            double dy1 = atoms[i].y - atoms[k].y;
                            double dz1 = atoms[i].z - atoms[k].z;
                            double dist1 = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1);
                            
                            double dx2 = atoms[j].x - atoms[k].x;
                            double dy2 = atoms[j].y - atoms[k].y;
                            double dz2 = atoms[j].z - atoms[k].z;
                            double dist2 = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2);
                            
                            if (dist1 < 0.1) dist1 = 0.1;
                            if (dist2 < 0.1) dist2 = 0.1;
                            
                            // Calculate angle
                            double cos_angle = (dx1*dx2 + dy1*dy2 + dz1*dz2) / (dist1*dist2);
                            if (cos_angle < -1.0) cos_angle = -1.0;
                            if (cos_angle > 1.0) cos_angle = 1.0;
                            
                            double angle = acos(cos_angle);
                            
                            // Ideal angle depends on hybridization (simplified)
                            double ideal_angle = 2.0944; // 120 degrees (sp2)
                            if (atoms[k].hybridization > 2.5) {
                                ideal_angle = 1.9112; // 109.5 degrees (sp3)
                            }
                            
                            double delta_angle = angle - ideal_angle;
                            double angle_energy = 0.5 * k_angle * delta_angle * delta_angle;
                            total_energy += angle_energy;
                            
                            // Force calculation for angle bending is complex
                            // This is a highly simplified version
                            double force_mag = k_angle * delta_angle * 0.1;
                            
                            // Apply forces perpendicular to the bonds
                            forces_x[i] += force_mag * (dy1*dz2 - dz1*dy2) / (dist1*dist2);
                            forces_y[i] += force_mag * (dz1*dx2 - dx1*dz2) / (dist1*dist2);
                            forces_z[i] += force_mag * (dx1*dy2 - dy1*dx2) / (dist1*dist2);
                            
                            forces_x[j] += force_mag * (dy2*dz1 - dz2*dy1) / (dist1*dist2);
                            forces_y[j] += force_mag * (dz2*dx1 - dx2*dz1) / (dist1*dist2);
                            forces_z[j] += force_mag * (dx2*dy1 - dy2*dx1) / (dist1*dist2);
                            
                            forces_x[k] -= forces_x[i] + forces_x[j];
                            forces_y[k] -= forces_y[i] + forces_y[j];
                            forces_z[k] -= forces_z[i] + forces_z[j];
                        }
                    }
                }
            }
        }
        
        // Non-bonded interactions (van der Waals and electrostatics)
        for (int i = 0; i < atom_count; i++) {
            for (int j = i + 1; j < atom_count; j++) {
                // Skip bonded atoms and 1-3, 1-4 interactions
                if (connectivity[i * atom_count + j] > 0 && connectivity[i * atom_count + j] <= 3) {
                    continue;
                }
                
                double dx = atoms[j].x - atoms[i].x;
                double dy = atoms[j].y - atoms[i].y;
                double dz = atoms[j].z - atoms[i].z;
                double dist_sq = dx*dx + dy*dy + dz*dz;
                
                // Apply cutoff
                if (dist_sq > cutoff*cutoff) continue;
                
                double dist = sqrt(dist_sq);
                if (dist < 0.1) dist = 0.1;
                
                // Van der Waals (simplified Lennard-Jones)
                double sigma = atoms[i].radius + atoms[j].radius;
                double vdw_energy = k_vdw * ((sigma/dist)*12 - 2.0*(sigma/dist)*6);
                
                // Electrostatics (Coulomb's law)
                double elec_energy = k_elec * atoms[i].charge * atoms[j].charge / dist;
                
                total_energy += vdw_energy + elec_energy;
                
                // VDW forces
                double vdw_force = k_vdw * (12.0*pow(sigma,12)/pow(dist,13) - 12.0*pow(sigma,6)/pow(dist,7));
                
                // Electrostatic forces
                double elec_force = k_elec * atoms[i].charge * atoms[j].charge / (dist_sq * dist);
                
                double total_force = vdw_force + elec_force;
                double fx = total_force * dx / dist;
                double fy = total_force * dy / dist;
                double fz = total_force * dz / dist;
                
                forces_x[i] += fx;
                forces_y[i] += fy;
                forces_z[i] += fz;
                forces_x[j] -= fx;
                forces_y[j] -= fy;
                forces_z[j] -= fz;
            }
        }
        
        // Update positions using calculated forces
        for (int i = 0; i < atom_count; i++) {
            atoms[i].x += forces_x[i] * time_step;
            atoms[i].y += forces_y[i] * time_step;
            atoms[i].z += forces_z[i] * time_step;
        }
        
        // Check for convergence
        if (fabs(total_energy - prev_energy) < convergence_threshold) {
            break;
        }
        
        prev_energy = total_energy;
    }
    
    // Return final energy if requested
    if (energy) {
        *energy = prev_energy;
    }
    
    // Clean up
    free(forces_x);
    free(forces_y);
    free(forces_z);
    free(ideal_bond_lengths);
    free(connectivity);
}
