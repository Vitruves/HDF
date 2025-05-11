#include "quantum_engine.h"
#include "globals.h"
#include "utils.h"   // Added for PI and potentially other utils
#include <math.h>    // For sqrt, exp, cos, acos, atan2, fabs, fmod, pow
#include <stdio.h>   // For fprintf and stderr
#include <stdlib.h>  // For malloc, calloc, and free
#include <stdbool.h> // For bool type
#include <complex.h> // For complex numbers

#ifdef __GNUC__
#define UNUSED __attribute__((unused))
#else
#define UNUSED
#endif

// The static definition of factorial has been removed.
// It is now assumed to be declared in "utils.h" and defined in "utils.c".

double hydrogen_1s(double r, double Z) {
    double a0 = BOHR_RADIUS;
    double rho = (2.0 * Z * r) / a0; // Note: r here should be in meters
    return sqrt(Z*Z*Z/(PI * a0*a0*a0)) * exp(-rho/2.0);
}

double hydrogen_2s(double r, double Z) {
    double a0 = BOHR_RADIUS;
    double rho = (2.0 * Z * r) / a0;
    return sqrt(Z*Z*Z/(32.0 * PI * a0*a0*a0)) * (2.0 - rho) * exp(-rho/2.0);
}

double hydrogen_2p(double r, double Z, double theta) {
    double a0 = BOHR_RADIUS;
    double rho = (2.0 * Z * r) / a0;
    return sqrt(Z*Z*Z/(32.0 * PI * a0*a0*a0)) * rho * exp(-rho/2.0) * cos(theta);
}

complex double slater_orbital(double r, double n, double zeta) {
    // Ensure n is treated as principal quantum number (integer part)
    // r should be in atomic units (Bohr radii) for typical STO zeta values
    // This function might need adjustment based on units of r and zeta
    return (pow(zeta, (int)n + 0.5) / sqrt(factorial(2 * (int)n))) *
           pow(2.0 * r, (int)n) * exp(-zeta * r);
}

double electron_density(AtomPos atom, double x_rel, double y_rel, double z_rel) {
    // Convert to meters for hydrogenic functions
    double dx_m = x_rel * ANGSTROM;
    double dy_m = y_rel * ANGSTROM;
    double dz_m = z_rel * ANGSTROM;
    
    double r_m = sqrt(dx_m*dx_m + dy_m*dy_m + dz_m*dz_m);
    if (r_m < 1e-12) r_m = 1e-12; // Avoid singularity

    double theta_polar = (r_m == 0) ? 0 : acos(dz_m / r_m);
    double phi_azimuthal = atan2(dy_m, dx_m);

    double Z_eff = atom.effective_nuclear_charge;
    double density = 0.0;

    // Core electrons (1s)
    int core_electrons = 2; // First shell
    if (atom.atomic_number > 2) {
        density += core_electrons * pow(hydrogen_1s(r_m, Z_eff * 1.2), 2.0); // Higher Z_eff for core
    }

    // Valence s-orbital contributions
    if (atom.orbital_config[0] > 0) {
        int n_principal = (atom.atomic_number <= 2) ? 1 : 2; // Principal quantum number
        if (n_principal == 1) {
            density += atom.orbital_config[0] * pow(hydrogen_1s(r_m, Z_eff), 2.0);
        } else {
            density += atom.orbital_config[0] * pow(hydrogen_2s(r_m, Z_eff), 2.0);
    }
    }

    // p-orbital contributions with proper angular dependence
    if (atom.orbital_config[1] > 0) {
        double p_electrons = atom.orbital_config[1];
        // Split between px, py, pz based on hybridization
        double px_contrib = 0.0, py_contrib = 0.0, pz_contrib = 0.0;
        
        switch ((int)atom.hybridization) {
            case 2: // sp2
                px_contrib = p_electrons / 3.0;
                py_contrib = p_electrons / 3.0;
                pz_contrib = p_electrons / 3.0;
                break;
            case 3: // sp3
                px_contrib = p_electrons / 4.0;
                py_contrib = p_electrons / 4.0;
                pz_contrib = p_electrons / 2.0;
                break;
            default: // sp or unhybridized
                px_contrib = py_contrib = pz_contrib = p_electrons / 3.0;
        }

        // Calculate p-orbital contributions with angular dependence
        double px = px_contrib * pow(hydrogen_2p(r_m, Z_eff, acos(dx_m/r_m)), 2.0);
        double py = py_contrib * pow(hydrogen_2p(r_m, Z_eff, acos(dy_m/r_m)), 2.0);
        double pz = pz_contrib * pow(hydrogen_2p(r_m, Z_eff, theta_polar), 2.0);
        
        density += px + py + pz;
    }

    // Hybridization effects
    if (atom.hybridization > 0) {
        double hybrid_factor = 1.0;
        switch ((int)atom.hybridization) {
            case 1: // sp
                hybrid_factor = 1.0 + 0.2 * cos(theta_polar);
                break;
            case 2: // sp2
                hybrid_factor = 1.0 + 0.3 * (cos(theta_polar) * cos(phi_azimuthal));
                break;
            case 3: // sp3
                hybrid_factor = 1.0 + 0.4 * pow(cos(theta_polar/2), 2);
                break;
        }
        density *= hybrid_factor;
    }

    // Aromatic effects
    if (atom.is_aromatic) {
        double pi_contribution = 0.2 * exp(-pow(dz_m/(0.5*ANGSTROM), 2)) * 
                               (1.0 + 0.5*cos(6.0*phi_azimuthal));
        density *= (1.0 + pi_contribution);
    }

    return density > 0 ? density : 0;
}

double calculate_hybridization(AtomPos atom) {
    // This is a rule-based heuristic. More accurate methods exist (e.g., VSEPR theory details).
    if (atom.is_aromatic) {
        return 2.0; // sp2 for aromatic atoms
    }

    // Count sigma bonds and lone pairs (approximated)
    // int sigma_bonds = 0;  // Remove if not used
    for(int i=0; i < bond_count; ++i) {
        if (bonds[i].a == (atom_count > 0 ? (&atom - &atoms[0]) : -1) || bonds[i].b == (atom_count > 0 ? (&atom - &atoms[0]) : -1) ) {
            // This way of getting atom index is unsafe if atom is not from global `atoms` array
            // Assuming atom is from the global array for now.
            // A better approach: pass atom index to this function or find it safely.
            // For now, rely on n_bonds if it correctly counts sigma-contributing connections.
            // sigma_bonds++; // Simplification: all bonds contribute to steric number for now
        }
    }
    // A better `n_bonds` would be `atom.num_neighbors`.
    // Let's use `atom.n_bonds` which should be set by parser.
    
    // int steric_number = atom.n_bonds;  // Remove if not used
    
    // Approximate lone pairs for common elements
    // Valence electrons - electrons used in bonds (crudely atom.n_bonds * bond_order)
    // This is complex. Let's use a simpler rule based on n_bonds for now.
    // TODO: Improve lone pair estimation for hybridization.

    switch (atom.n_bonds) { // Using n_bonds as a proxy for steric number components
        case 0: return 0.0; // No bonds, undefined hybridization
        case 1: // E.g. H in H2 (not typical for this), or terminal atom in linear
            return 1.0; // Could be sp, or unhybridized s
        case 2:
            // Could be sp (linear, e.g., CO2 central C) or sp2 (bent, e.g., H2O central O if lone pairs counted)
            // If terminal, likely sp if triple bonded, sp2 if double bonded.
            // For simplicity, if not aromatic:
            // Check bond orders of its two bonds if available.
            return 2.0; // Default to sp2 if ambiguous (common for C, N, O in planar)
        case 3:
            return 2.0; // sp2 (trigonal planar)
        case 4:
            return 3.0; // sp3 (tetrahedral)
        default: // More than 4 bonds, hypervalent (e.g. PCl5, SF6) - requires d-orbitals
            return 3.0; // Default to sp3 as a fallback, or handle hypervalency if model supports it
    }
}

void apply_quantum_corrections_to_atoms() {
    // First pass: Calculate basic properties
    for (int i = 0; i < atom_count; i++) {
        atoms[i].hybridization = calculate_hybridization(atoms[i]);
        
        // Initial Z_eff adjustment based on electronegativity
        double en_factor = (atoms[i].electronegativity - 2.0) / 4.0; // Normalized to common range
        atoms[i].effective_nuclear_charge *= (1.0 + en_factor);
    }

    // Second pass: Consider bonding environment
    for (int i = 0; i < atom_count; i++) {
        double charge_transfer = 0.0;
        int bonded_count = 0;

        // Calculate charge transfer effects
        for (int j = 0; j < bond_count; j++) {
            if (bonds[j].a == i || bonds[j].b == i) {
                int other_idx = (bonds[j].a == i) ? bonds[j].b : bonds[j].a;
                double en_diff = atoms[i].electronegativity - atoms[other_idx].electronegativity;
                
                // Charge transfer based on electronegativity difference and bond order
                double transfer = en_diff * 0.1 * bonds[j].order;
                if (bonds[j].type == BOND_AROMATIC) {
                    transfer *= 0.8; // Reduced effect for aromatic bonds
                }
                
                charge_transfer += transfer;
                bonded_count++;
            }
        }

        // Apply charge transfer effects
        if (bonded_count > 0) {
            double avg_charge_transfer = charge_transfer / bonded_count;
            atoms[i].effective_nuclear_charge *= (1.0 - avg_charge_transfer);
        }

        // Adjust for aromaticity
        if (atoms[i].is_aromatic) {
            atoms[i].effective_nuclear_charge *= 0.95; // Slight reduction due to electron delocalization
        }

        // Limit corrections to reasonable ranges
        if (atoms[i].effective_nuclear_charge < atoms[i].atomic_number * 0.5) {
            atoms[i].effective_nuclear_charge = atoms[i].atomic_number * 0.5;
        } else if (atoms[i].effective_nuclear_charge > atoms[i].atomic_number * 1.5) {
            atoms[i].effective_nuclear_charge = atoms[i].atomic_number * 1.5;
        }
    }
}

double calculate_atom_phase_qm(AtomPos atom, int idx) {
    double base_phase = fmod((atom.atomic_number * PI / 18.0) + (atom.isotope * PI / 90.0), 2.0 * PI); // Base on period and isotope
    
    // Effective nuclear charge and electronegativity influence
    double charge_density_effect = (atom.effective_nuclear_charge / (atom.radius + 1e-6)) * PI / 50.0; 
    double en_effect = atom.electronegativity * PI / 20.0;

    // Orbital contributions with more distinction
    double orbital_effect = 0.0;
    orbital_effect += atom.orbital_config[0] * 0.05 * PI; // s-electrons (valence)
    orbital_effect += atom.orbital_config[1] * 0.12 * PI; // p-electrons (valence)
    orbital_effect += atom.orbital_config[2] * 0.18 * PI; // d-electrons (if applicable)
    orbital_effect += atom.orbital_config[3] * 0.22 * PI; // f-electrons (if applicable)

    // Hybridization effect (more pronounced)
    double hybrid_effect = 0.0;
    if (atom.hybridization == 1) hybrid_effect = 0.1 * PI; // sp
    else if (atom.hybridization == 2) hybrid_effect = 0.2 * PI; // sp2
    else if (atom.hybridization == 3) hybrid_effect = 0.15 * PI; // sp3 (less phase shift than sp2 for some models)
    
    // Polarizability approximation (proportional to volume ~ radius^3)
    double polarizability_approx = pow(atom.radius, 3.0);
    double polarizability_effect = fmod(polarizability_approx * PI / 10.0, 2.0 * PI);

    // Reduced position salt for less randomness, more deterministic physics
    unsigned int hash = hash_string(atom.atom, idx + atom.atomic_number); // Salt with atomic number too
    double position_salt = (hash % 100) / 1000.0 * PI; // Smaller range, e.g., 0 to 0.1 PI
    
    double charge_effect = atom.charge * PI / 3.0; // Stronger charge effect
    
    double aromatic_shift = atom.is_aromatic ? 0.25 * PI : 0.0; // More distinct aromatic shift
    double ring_strain_effect = atom.in_ring ? 0.05 * PI * atom.n_bonds : 0.0; // Simple ring effect based on connectivity

    double total_phase = base_phase + charge_density_effect + en_effect + orbital_effect + 
                         hybrid_effect + polarizability_effect + position_salt + charge_effect + 
                         aromatic_shift + ring_strain_effect;
    
    // Phase contribution from Z-coordinate (pseudo depth cue)
    // Modulate based on how far from z=0, non-linearly
    double z_phase_contribution = atan(atom.z * 0.5) * 0.1 * PI; // arctan for bounded effect
    total_phase += z_phase_contribution;

    return fmod(total_phase, 2.0 * PI);
}

double calculate_bond_phase_qm(BondSeg bond) {
    AtomPos atom_a = atoms[bond.a];
    AtomPos atom_b = atoms[bond.b];
    
    // Bond order effect - more distinct steps
    double order_phase = 0.0;
    if (bond.order == 1) order_phase = 0.1 * PI;
    else if (bond.order == 2) order_phase = 0.3 * PI;
    else if (bond.order == 3) order_phase = 0.5 * PI;
    else order_phase = 0.05 * PI; // Fractional or other orders

    // Electronegativity difference (polarity) - non-linear response
    double en_diff = fabs(atom_a.electronegativity - atom_b.electronegativity);
    double polarity_effect = pow(en_diff / 4.0, 2.0) * 0.4 * PI; // Max EN diff is ~4.0
    
    // Resonance and aromaticity effect
    double resonance_effect = 0.0;
    if (bond.type == BOND_AROMATIC) {
        resonance_effect = 0.35 * PI; // Stronger phase for aromatic bonds
    } else if (atom_a.is_aromatic && atom_b.is_aromatic) {
        resonance_effect = 0.15 * PI; // Both atoms aromatic, bond itself might not be marked but is part of system
    }
    
    // Ring effects (strain and conjugation within rings)
    double ring_effect = 0.0;
    if (bond.in_ring) {
        // Could be more complex based on ring size, but a simple shift for now
        ring_effect = 0.1 * PI;
        if (bond.type == BOND_AROMATIC) ring_effect *= 1.5; // Enhance for aromatic rings
    }
        
    // Effect of average effective nuclear charge of bonded atoms
    double avg_Z_eff = (atom_a.effective_nuclear_charge + atom_b.effective_nuclear_charge) / 2.0;
    double avg_energy_effect = fmod((avg_Z_eff / 10.0) * PI / 4.0, 2.0 * PI);

    // Hybridization overlap effect - more specific contributions
    double hybrid_overlap_effect = 0.0;
    double h1 = atom_a.hybridization; // sp=1, sp2=2, sp3=3
    double h2 = atom_b.hybridization;
    // Sigma bonds base overlap
    if (h1 > 0 && h2 > 0) { // Both hybridized
        hybrid_overlap_effect = ( (4.0-h1) + (4.0-h2) ) / 6.0 * 0.15 * PI; // Higher s-character (lower h number) = better overlap phase shift
    }
    // Pi-bond contributions if applicable (sp2-sp2, sp-sp, sp-sp2)
    if ((h1 == 2 && h2 == 2 && bond.order >=2) || 
        (h1 == 1 && h2 == 1 && bond.order >=2) || 
        (((h1 == 1 && h2 == 2) || (h1 == 2 && h2 == 1)) && bond.order >=2)) {
        hybrid_overlap_effect += (bond.order -1) * 0.1 * PI; // Additional shift for pi bonds
    }

    // Bond length deviation from ideal (based on radii and order)
    double ideal_length = atom_a.radius + atom_b.radius;
    if (bond.order == 2) ideal_length *= 0.85; else if (bond.order == 3) ideal_length *= 0.75;
    if (bond.type == BOND_AROMATIC) ideal_length *= 0.92; // Slightly different factor for aromatic
    double length_deviation = bond.length - ideal_length;
    double length_effect = atan(length_deviation * 5.0) * 0.15 * PI; // arctan for bounded, sensitive effect

    // Interaction term: polarizability of atoms and bond polarity
    double avg_polarizability_a = pow(atom_a.radius, 3.0);
    double avg_polarizability_b = pow(atom_b.radius, 3.0);
    double polarizability_interaction_effect = (avg_polarizability_a + avg_polarizability_b)/2.0 * en_diff * PI / 50.0;

    double total_phase = order_phase + polarity_effect + resonance_effect + 
                         ring_effect + avg_energy_effect + hybrid_overlap_effect + 
                         length_effect + polarizability_interaction_effect;
                         
    return fmod(total_phase, 2.0 * PI);
}

// New function to calculate partial charges using an electronegativity equalization approach
double calculate_partial_charges(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count) {
    if (atom_count <= 0) return 0.0;
    
    // Allocate memory for partial charges
    double *partial_charges = (double*)calloc(atom_count, sizeof(double));
    double *prev_charges = (double*)calloc(atom_count, sizeof(double));
    
    if (!partial_charges || !prev_charges) {
        fprintf(stderr, "Error: Failed to allocate memory for partial charge calculation\n");
        if (partial_charges) free(partial_charges);
        if (prev_charges) free(prev_charges);
        return 0.0;
    }
    
    // Initialize based on electronegativity difference (simplified Gasteiger approach)
    const int MAX_ITERATIONS = 20;
    const double CONVERGENCE_THRESHOLD = 0.001;
    
    // First iteration: use basic electronegativity difference
    for (int b = 0; b < bond_count; b++) {
        int a1 = bonds[b].a;
        int a2 = bonds[b].b;
        
        if (a1 >= atom_count || a2 >= atom_count) continue;
        
        double en_diff = atoms[a1].electronegativity - atoms[a2].electronegativity;
        double bond_order_factor = bonds[b].order * 0.1;
        
        // More negative for more electronegative atom, scaled by bond order
        double charge_transfer = en_diff * bond_order_factor;
        
        // Special case for aromatic bonds
        if (bonds[b].type == BOND_AROMATIC) {
            charge_transfer *= 0.7;  // Reduced charge transfer in aromatic systems
        }
        
        partial_charges[a2] += charge_transfer;
        partial_charges[a1] -= charge_transfer;
    }
    
    // Iterative refinement
    bool converged = false;
    for (int iter = 0; iter < MAX_ITERATIONS && !converged; iter++) {
        // Copy current charges
        for (int i = 0; i < atom_count; i++) {
            prev_charges[i] = partial_charges[i];
        }
        
        // Update charges based on current distribution
        for (int b = 0; b < bond_count; b++) {
            int a1 = bonds[b].a;
            int a2 = bonds[b].b;
            
            if (a1 >= atom_count || a2 >= atom_count) continue;
            
            // Calculate electronegativities adjusted by current partial charges
            double en1 = atoms[a1].electronegativity * (1.0 - 0.3 * fabs(partial_charges[a1]));
            double en2 = atoms[a2].electronegativity * (1.0 - 0.3 * fabs(partial_charges[a2]));
            
            double en_diff = en1 - en2;
            
            // Dynamic bond order factor, reduced for highly charged atoms
            double bond_order_factor = bonds[b].order * 0.05 * 
                (1.0 - 0.2 * fabs(partial_charges[a1]) - 0.2 * fabs(partial_charges[a2]));
            
            double charge_transfer = en_diff * bond_order_factor;
            
            // Reduce charge transfer in aromatic and conjugated systems
            if (bonds[b].type == BOND_AROMATIC || bonds[b].is_conjugated) {
                charge_transfer *= 0.6;
            }
            
            partial_charges[a2] += charge_transfer;
            partial_charges[a1] -= charge_transfer;
        }
        
        // Check for convergence
        converged = true;
        for (int i = 0; i < atom_count; i++) {
            if (fabs(partial_charges[i] - prev_charges[i]) > CONVERGENCE_THRESHOLD) {
                converged = false;
                break;
            }
        }
    }
    
    // Apply charge damping based on atomic properties
    for (int i = 0; i < atom_count; i++) {
        // Atoms with more bonds distribute charge better
        double damping = 0.8 + 0.05 * atoms[i].n_bonds;
        
        // Aromatics distribute charge better through resonance
        if (atoms[i].is_aromatic) {
            damping *= 0.85;
        }
        
        // Cap extreme charges and scale by damping
        if (partial_charges[i] > 1.0) partial_charges[i] = 1.0;
        if (partial_charges[i] < -1.0) partial_charges[i] = -1.0;
        
        partial_charges[i] *= damping;
        
        // Store the partial charge in the atom structure
        atoms[i].partial_charge = partial_charges[i];
    }
    
    // Compute sum of squared charges as a measure of charge separation
    double charge_separation = 0.0;
    for (int i = 0; i < atom_count; i++) {
        charge_separation += partial_charges[i] * partial_charges[i];
    }
    
    free(partial_charges);
    free(prev_charges);
    
    return sqrt(charge_separation / atom_count);  // Root mean square charge
}

// Calculate atomic hardness (resistance to charge transfer)
double calculate_atomic_hardness(AtomPos atom) {
    // Hardness is related to ionization energy and electron affinity
    // Basic approximation based on electronegativity and atomic number
    double base_hardness = 0.0;
    
    // Empirical approach using electronegativity
    if (atom.atomic_number <= 2) {  // H, He
        base_hardness = 6.0;
    } else if (atom.atomic_number <= 10) {  // First row elements
        base_hardness = 5.0;
    } else if (atom.atomic_number <= 18) {  // Second row elements
        base_hardness = 4.0;
    } else {  // Heavier elements
        base_hardness = 3.0;
    }
    
    // Modify based on electronegativity - more electronegative atoms tend to be harder
    base_hardness *= (0.8 + 0.05 * atom.electronegativity);
    
    // Aromaticity decreases hardness due to electron delocalization
    if (atom.is_aromatic) {
        base_hardness *= 0.8;
    }
    
    // Hybridization effects - sp3 hybridized atoms are generally harder
    if (atom.hybridization > 2.5) {  // sp3
        base_hardness *= 1.1;
    } else if (atom.hybridization > 1.5) {  // sp2
        base_hardness *= 0.95;
    } else if (atom.hybridization > 0.5) {  // sp
        base_hardness *= 0.9;
    }
    
    // Charged species have modified hardness
    if (atom.charge > 0) {
        base_hardness *= (1.0 + 0.2 * atom.charge);  // Cations are harder
    } else if (atom.charge < 0) {
        base_hardness *= (1.0 - 0.15 * atom.charge);  // Anions are softer
    }
    
    return base_hardness;
}

// Calculate atomic softness (susceptibility to polarization), inverse of hardness
double calculate_atomic_softness(AtomPos atom) {
    double hardness = calculate_atomic_hardness(atom);
    
    // Avoid division by zero
    if (hardness < 0.001) {
        hardness = 0.001;
    }
    
    return 1.0 / hardness;
}

// Calculate ionization energy based on atomic properties
double calculate_ionization_energy(AtomPos atom) {
    // Base approximation using atomic number and Slater's rules
    double base_ionization = 0.0;
    
    // Very simplified version using element-specific values
    switch (atom.atomic_number) {
        case 1:  // H
            base_ionization = 13.6;
            break;
        case 6:  // C
            base_ionization = 11.3; 
            if (atom.is_aromatic) base_ionization -= 0.5;
            break;
        case 7:  // N
            base_ionization = 14.5;
            if (atom.is_aromatic) base_ionization -= 0.7;
            break;
        case 8:  // O
            base_ionization = 13.6;
            break;
        case 9:  // F
            base_ionization = 17.4;
            break;
        case 15: // P
            base_ionization = 10.5;
            break;
        case 16: // S
            base_ionization = 10.4;
            break;
        case 17: // Cl
            base_ionization = 13.0;
            break;
        default:
            // Approximate using electronegativity when specific data isn't available
            base_ionization = 9.0 + 1.5 * atom.electronegativity;
    }
    
    // Adjust based on charge state
    if (atom.charge > 0) {
        // Additional ionization requires more energy
        base_ionization += 10.0 * atom.charge;
    } else if (atom.charge < 0) {
        // Negative charge makes ionization easier
        base_ionization -= 3.0 * atom.charge;
        if (base_ionization < 3.0) base_ionization = 3.0;  // Minimum threshold
    }
    
    // Apply hybridization effects
    if (atom.hybridization > 2.5) {  // sp3
        base_ionization *= 1.05;  // sp3 slightly higher
    } else if (atom.hybridization > 1.5) {  // sp2
        base_ionization *= 0.98;  // sp2 slightly lower
    } else if (atom.hybridization > 0.5) {  // sp
        base_ionization *= 0.95;  // sp lower
    }
    
    return base_ionization;
}

// Calculate electron affinity
double calculate_electron_affinity(AtomPos atom) {
    // Base approximation
    double base_affinity = 0.0;
    
    // Element-specific values for common atoms (in eV)
    switch (atom.atomic_number) {
        case 1:  // H
            base_affinity = 0.75;
            break;
        case 6:  // C
            base_affinity = 1.2;
            if (atom.is_aromatic) base_affinity += 0.3;
            break;
        case 7:  // N
            base_affinity = 0.07;
            if (atom.is_aromatic) base_affinity += 0.5;
            break;
        case 8:  // O
            base_affinity = 1.46;
            break;
        case 9:  // F
            base_affinity = 3.4;
            break;
        case 15: // P
            base_affinity = 0.75;
            break;
        case 16: // S
            base_affinity = 2.08;
            break;
        case 17: // Cl
            base_affinity = 3.62;
            break;
        default:
            // Approximate using electronegativity
            base_affinity = 0.2 * atom.electronegativity;
    }
    
    // Charge effects
    if (atom.charge > 0) {
        // Positive charge increases electron affinity dramatically
        base_affinity += 2.0 * atom.charge;
    } else if (atom.charge < 0) {
        // Negative charge decreases electron affinity
        base_affinity *= (1.0 / (1.0 - 0.5 * atom.charge));
        if (base_affinity < 0) base_affinity = 0.0;
    }
    
    // Hybridization effects
    if (atom.hybridization > 2.5) {  // sp3
        base_affinity *= 0.9;  // sp3 lower electron affinity
    } else if (atom.hybridization > 1.5) {  // sp2
        base_affinity *= 1.1;  // sp2 higher
    } else if (atom.hybridization > 0.5) {  // sp
        base_affinity *= 1.2;  // sp higher
    }
    
    return base_affinity;
}

// Calculate electronic spatial extent - measure of electron cloud size
double calculate_electronic_spatial_extent(AtomPos atom) {
    // Base size scaled by atomic radius
    double extent = atom.radius * atom.radius * 5.0;  // r² scale
    
    // Adjust based on electron count (simplified)
    extent *= (0.5 + 0.05 * atom.atomic_number);
    
    // Charged atoms have altered extents
    if (atom.charge < 0) {
        // Anions have larger electron clouds
        extent *= (1.0 - 0.2 * atom.charge);  // Negative charge gives expansion
    } else if (atom.charge > 0) {
        // Cations have smaller electron clouds
        extent *= (1.0 - 0.15 * atom.charge);
    }
    
    // Hybridization effects
    if (atom.hybridization > 2.5) {  // sp3
        // Tetrahedral distribution
        extent *= 0.95;
    } else if (atom.hybridization > 1.5) {  // sp2
        // Planar distribution 
        extent *= 1.05;
    } else if (atom.hybridization > 0.5) {  // sp
        // Linear distribution
        extent *= 1.15;
    }
    
    // Aromatics have more extended π clouds
    if (atom.is_aromatic) {
        extent *= 1.2;
    }
    
    return extent;
}

// New enhanced electron density gradient calculation
double calculate_electron_density_gradient(AtomPos atom, double x, double y, double z) {
    // Calculate electron density at the central point
    double density_val = electron_density(atom, x, y, z);
    
    // Calculate gradient using small displacements
    const double h = 0.01;  // Step size in Angstroms
    
    // Use density_val as the reference for the central point
    double dx = (electron_density(atom, x+h, y, z) - density_val) - (density_val - electron_density(atom, x-h, y, z));
    double dy = (electron_density(atom, x, y+h, z) - density_val) - (density_val - electron_density(atom, x, y-h, z));
    double dz = (electron_density(atom, x, y, z+h) - density_val) - (density_val - electron_density(atom, x, y, z-h));
    
    // Calculate magnitude of gradient with improved accuracy using a 5-point stencil
    double gradient_magnitude = sqrt((dx*dx + dy*dy + dz*dz) / (4*h*h));
    
    return gradient_magnitude;
}

// Calculate orbital overlap between two atoms
double calculate_orbital_overlap(AtomPos atom1, AtomPos atom2, double distance) {
    if (distance < 0.001) distance = 0.001;  // Avoid division by zero
    
    // Base overlap decreases with distance
    double overlap = exp(-distance / 2.0);
    
    // Adjust based on atom types and orbital properties
    double s_character1 = 0.0, s_character2 = 0.0;
    
    // Calculate s orbital character based on hybridization
    if (atom1.hybridization <= 0.5) {
        s_character1 = 1.0;  // Pure s
    } else if (atom1.hybridization <= 1.5) {
        s_character1 = 0.5;  // sp (50% s)
    } else if (atom1.hybridization <= 2.5) {
        s_character1 = 0.33;  // sp2 (33% s)
    } else {
        s_character1 = 0.25;  // sp3 (25% s)
    }
    
    if (atom2.hybridization <= 0.5) {
        s_character2 = 1.0;
    } else if (atom2.hybridization <= 1.5) {
        s_character2 = 0.5;
    } else if (atom2.hybridization <= 2.5) {
        s_character2 = 0.33;
    } else {
        s_character2 = 0.25;
    }
    
    // s-s overlaps are stronger than p-p overlaps
    double s_overlap_factor = s_character1 * s_character2;
    double p_overlap_factor = (1.0 - s_character1) * (1.0 - s_character2);
    
    // Combine s and p overlap contributions
    overlap *= (1.5 * s_overlap_factor + 0.7 * p_overlap_factor);
    
    // Electronegativity difference reduces overlap
    double en_diff = fabs(atom1.electronegativity - atom2.electronegativity);
    overlap *= (1.0 - 0.1 * en_diff);
    
    // Aromatics have enhanced overlap due to π system
    if (atom1.is_aromatic && atom2.is_aromatic) {
        overlap *= 1.2;
    }
    
    return overlap;
}

// Identify conjugated systems in the molecule
int identify_conjugated_systems(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count, int *system_assignments) {
    if (!atoms || !bonds || !system_assignments || atom_count <= 0 || bond_count <= 0) {
        return 0;
    }
    
    // Initialize system assignments to 0 (no system)
    for (int i = 0; i < atom_count; i++) {
        system_assignments[i] = 0;
    }
    
    int system_count = 0;
    
    // First pass: identify seed atoms for conjugated systems
    for (int i = 0; i < atom_count; i++) {
        // Skip if already assigned
        if (system_assignments[i] != 0) continue;
        
        // Check if this atom could be part of a conjugated system
        if (atoms[i].is_aromatic || 
            (atoms[i].hybridization > 1.5 && atoms[i].hybridization < 2.5)) {
            
            // Found a potential seed atom for a new system
            system_count++;
            system_assignments[i] = system_count;
            
            // Queue for BFS
            int *queue = (int*)malloc(atom_count * sizeof(int));
            if (!queue) {
                fprintf(stderr, "Memory allocation failed for conjugated system BFS\n");
                return system_count;
            }
            
            int front = 0, rear = 0;
            queue[rear++] = i;
            
            // BFS to find connected atoms in the same conjugated system
            while (front < rear) {
                int curr = queue[front++];
                
                // Check all bonds for connections
                for (int b = 0; b < bond_count; b++) {
                    int next_atom = -1;
                    
                    // Find bonds connected to the current atom
                    if (bonds[b].a == curr) next_atom = bonds[b].b;
                    else if (bonds[b].b == curr) next_atom = bonds[b].a;
                    
                    // Skip if no connection or already processed
                    if (next_atom == -1 || system_assignments[next_atom] != 0) continue;
                    
                    // Check if the connected atom should be part of this conjugated system
                    bool add_to_system = false;
                    
                    if (atoms[next_atom].is_aromatic || 
                        (atoms[next_atom].hybridization > 1.5 && atoms[next_atom].hybridization < 2.5)) {
                        add_to_system = true;
                    }
                    
                    // Special bond types also create conjugation
                    if (bonds[b].type == BOND_AROMATIC || bonds[b].is_conjugated || bonds[b].order > 1) {
                        add_to_system = true;
                    }
                    
                    if (add_to_system) {
                        system_assignments[next_atom] = system_count;
                        queue[rear++] = next_atom;
                    }
                }
            }
            
            free(queue);
        }
    }
    
    // Store the conjugation path length for each atom
    for (int i = 0; i < atom_count; i++) {
        if (system_assignments[i] > 0) {
            int system_id = system_assignments[i];
            
            // Count atoms in this system
            int system_size = 0;
            for (int j = 0; j < atom_count; j++) {
                if (system_assignments[j] == system_id) {
                    system_size++;
                }
            }
            
            atoms[i].conjugation_path_length = system_size;
        } else {
            atoms[i].conjugation_path_length = 0;
        }
    }
    
    // Assign conjugation system IDs to bonds
    for (int b = 0; b < bond_count; b++) {
        int a1 = bonds[b].a;
        int a2 = bonds[b].b;
        
        if (a1 < 0 || a1 >= atom_count || a2 < 0 || a2 >= atom_count) {
            bonds[b].conjugation_system_id = 0;
            continue;
        }
        
        // If both atoms are in the same conjugated system, assign the bond
        if (system_assignments[a1] > 0 && system_assignments[a1] == system_assignments[a2]) {
            bonds[b].conjugation_system_id = system_assignments[a1];
            
            // Calculate bond pi character for conjugated bonds
            if (bonds[b].order > 1 || bonds[b].type == BOND_AROMATIC) {
                bonds[b].bond_pi_character = 0.8;
            } else {
                bonds[b].bond_pi_character = 0.3; // Smaller π character for single bonds in conjugated systems
            }
        } else {
            bonds[b].conjugation_system_id = 0;
            
            // Assign pi character based on bond order only
            if (bonds[b].order == 2) {
                bonds[b].bond_pi_character = 0.5;
            } else if (bonds[b].order == 3) {
                bonds[b].bond_pi_character = 0.7;
            } else {
                bonds[b].bond_pi_character = 0.0;
            }
        }
    }
    
    return system_count;
}

// Calculate conjugation contribution for an atom in a specific system
double calculate_conjugation_contribution(AtomPos atom, int conjugation_system_id, int *system_assignments) {
    // If not part of the specified system, no contribution
    int atom_idx = -1;
    
    // Find the atom's index
    for (int i = 0; i < atom_count; i++) {
        if (&atoms[i] == &atom) {
            atom_idx = i;
            break;
        }
    }
    
    if (atom_idx == -1 || system_assignments[atom_idx] != conjugation_system_id) {
        return 0.0;
    }
    
    double contribution = 0.0;
    
    // Base contribution by atom type
    if (atom.atom[0] == 'C') {
        if (atom.is_aromatic) {
            contribution = 1.0; // Aromatic carbon is key to conjugation
        } else if (atom.hybridization > 1.5 && atom.hybridization < 2.5) {
            contribution = 0.8; // sp2 carbon 
        }
    } else if (atom.atom[0] == 'N') {
        if (atom.is_aromatic) {
            contribution = 1.2; // Aromatic nitrogen contributes more
        } else if (atom.hybridization > 1.5 && atom.hybridization < 2.5) {
            contribution = 0.9; 
        }
    } else if (atom.atom[0] == 'O') {
        if (atom.is_aromatic) {
            contribution = 1.1;
        } else if (atom.hybridization > 1.5 && atom.hybridization < 2.5) {
            contribution = 0.85;
        }
    } else if (atom.atom[0] == 'S') {
        if (atom.is_aromatic) {
            contribution = 0.9;
        } else {
            contribution = 0.6;
        }
    } else {
        // Other atoms contribute less
        contribution = 0.3;
    }
    
    // If atom has a charge, it affects conjugation
    if (atom.charge != 0) {
        contribution *= (1.0 + 0.2 * abs(atom.charge));
    }
    
    // Adjust by connected bonds
    int conjugated_bonds = 0;
    for (int b = 0; b < bond_count; b++) {
        if ((bonds[b].a == atom_idx || bonds[b].b == atom_idx) && 
            bonds[b].conjugation_system_id == conjugation_system_id) {
            conjugated_bonds++;
            
            // Multiple bonds contribute more
            if (bonds[b].order > 1) {
                contribution *= (1.0 + 0.1 * (bonds[b].order - 1));
            }
        }
    }
    
    // Normalize by number of bonds
    if (conjugated_bonds > 0) {
        contribution *= sqrt((double)conjugated_bonds / 3.0); // Scale by sqrt of normalized bond count
    }
    
    return contribution;
}

// Calculate resonance energy for a conjugated system
double calculate_delocalization_energy(int *conjugated_system, int size) {
    if (size <= 1) return 0.0;
    
    double total_energy = 0.0;
    
    // Resonance energy increases with system size but not linearly
    // Hückel's rule approximation: 2*(n+1) electrons in a conjugated system
    // are especially stable when n = 4k+2 (6, 10, 14, etc. electrons)
    
    // Count contributing atoms (typically ones with p orbitals)
    int p_orbital_count = 0;
    int electron_count = 0;
    
    for (int i = 0; i < size; i++) {
        int atom_idx = conjugated_system[i];
        
        // Skip invalid indices
        if (atom_idx < 0 || atom_idx >= atom_count) continue;
        
        // Count p-orbital contributing atoms
        if (atoms[atom_idx].hybridization > 1.5 && atoms[atom_idx].hybridization < 2.5) {
            p_orbital_count++;
            
            // Estimate π electron contribution
            if (atoms[atom_idx].atom[0] == 'C') {
                electron_count += 1; // Carbon contributes 1 π electron
            } else if (atoms[atom_idx].atom[0] == 'N' || atoms[atom_idx].atom[0] == 'O') {
                if (atoms[atom_idx].is_aromatic) {
                    electron_count += 1; // Aromatic N/O contributes 1 π electron
                } else {
                    electron_count += 2; // Non-aromatic N/O can contribute 2 π electrons (lone pair)
                }
            }
            
            // Account for formal charges
            electron_count += atoms[atom_idx].charge;
        }
    }
    
    // Base energy calculation
    double base_resonance = 0.0;
    
    // Aromaticity check using Hückel's rule
    if (electron_count % 4 == 2) { // 4n+2 rule
        // Extra stability for aromatic systems
        base_resonance = 0.3 * electron_count;
    } else {
        // Less stability for non-aromatic conjugated systems
        base_resonance = 0.15 * electron_count;
    }
    
    // Scale by system size (not linear)
    total_energy = base_resonance * sqrt((double)p_orbital_count);
    
    // Enhance for cyclic systems (more stable)
    bool is_cyclic = false;
    int cycle_size = 0;
    
    // Check if this is a cyclic system by looking at the bonds
    // This is a simplified approximation
    for (int i = 0; i < size; i++) {
        int atom_idx = conjugated_system[i];
        if (atom_idx < 0 || atom_idx >= atom_count) continue;
        
        if (atoms[atom_idx].in_ring) {
            is_cyclic = true;
            // Find smallest ring size
            for (int r = 0; r < 4; r++) {
                if (atoms[atom_idx].ring_sizes[r] > 0) {
                    if (cycle_size == 0 || atoms[atom_idx].ring_sizes[r] < cycle_size) {
                        cycle_size = atoms[atom_idx].ring_sizes[r];
                    }
                }
            }
        }
    }
    
    if (is_cyclic) {
        // Adjust by ring size - 6-membered rings are most stable
        double cycle_factor = 1.0;
        if (cycle_size == 6) {
            cycle_factor = 1.5; // Benzene-like stability
        } else if (cycle_size == 5) {
            cycle_factor = 1.3; // 5-membered aromatic rings
        } else if (cycle_size > 6) {
            cycle_factor = 1.0 - 0.05 * (cycle_size - 6); // Decreasing stability for larger rings
        } else if (cycle_size > 0) {
            cycle_factor = 0.8; // Small rings have less resonance
        }
        
        total_energy *= cycle_factor;
    }
    
    return total_energy;
}

// Estimate resonance energy for the entire molecule
double estimate_resonance_energy(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count) {
    if (atom_count <= 0 || !atoms || !bonds) return 0.0;
    
    // First identify all conjugated systems
    int *system_assignments = (int*)calloc(atom_count, sizeof(int));
    if (!system_assignments) {
        fprintf(stderr, "Memory allocation failed for system assignments\n");
        return 0.0;
    }
    
    int system_count = identify_conjugated_systems(atoms, atom_count, bonds, bond_count, system_assignments);
    
    // No conjugated systems
    if (system_count == 0) {
        free(system_assignments);
        return 0.0;
    }
    
    // Now calculate resonance energy for each system
    double total_resonance_energy = 0.0;
    
    for (int sys = 1; sys <= system_count; sys++) {
        // Count atoms in this system
        int system_size = 0;
        for (int i = 0; i < atom_count; i++) {
            if (system_assignments[i] == sys) {
                system_size++;
            }
        }
        
        // Collect atoms in this system
        int *system_atoms = (int*)malloc(system_size * sizeof(int));
        if (!system_atoms) {
            fprintf(stderr, "Memory allocation failed for system atoms\n");
            continue;
        }
        
        int idx = 0;
        for (int i = 0; i < atom_count; i++) {
            if (system_assignments[i] == sys) {
                system_atoms[idx++] = i;
            }
        }
        
        // Calculate resonance energy for this system
        double system_energy = calculate_delocalization_energy(system_atoms, system_size);
        total_resonance_energy += system_energy;
        
        // Store contribution in each atom
        for (int i = 0; i < system_size; i++) {
            int atom_idx = system_atoms[i];
            atoms[atom_idx].resonance_energy_contribution = 
                system_energy * calculate_conjugation_contribution(atoms[atom_idx], sys, system_assignments) / system_size;
        }
        
        free(system_atoms);
    }
    
    free(system_assignments);
    return total_resonance_energy;
}

// Calculate ring strain based on ring geometry
double calculate_ring_strain(int *ring_atoms, int ring_size, AtomPos *atoms, BondSeg *bonds) {
    if (ring_size <= 2 || !ring_atoms || !atoms || !bonds) return 0.0;
    
    double total_strain = 0.0;
    
    // Base strain from ring size
    switch (ring_size) {
        case 3: // Highly strained
            total_strain = 30.0; // kcal/mol (approximate)
            break;
        case 4: // Significant strain
            total_strain = 25.0;
            break;
        case 5: // Moderate strain
            total_strain = 10.0;
            break;
        case 6: // Low strain for cyclohexane (chair)
            total_strain = 0.5;
            break;
        case 7: // Medium-large rings have strain due to transannular interactions
            total_strain = 7.0;
            break;
        case 8:
            total_strain = 10.0;
            break;
        default: // Larger rings
            total_strain = 12.0 - 0.5 * (ring_size - 8); // Decreases for very large rings
            if (total_strain < 3.0) total_strain = 3.0; // Minimum strain
    }
    
    // Adjust for aromaticity - aromatic rings have less strain
    bool is_aromatic = true;
    for (int i = 0; i < ring_size; i++) {
        int atom_idx = ring_atoms[i];
        if (atom_idx < 0 || atom_idx >= atom_count || !atoms[atom_idx].is_aromatic) {
            is_aromatic = false;
            break;
        }
    }
    
    if (is_aromatic) {
        // Aromatic rings have significantly reduced strain
        total_strain *= 0.3;
    }
    
    // Check for sp2 atoms in small rings (increases strain)
    if (ring_size < 6) {
        int sp2_count = 0;
        for (int i = 0; i < ring_size; i++) {
            int atom_idx = ring_atoms[i];
            if (atom_idx >= 0 && atom_idx < atom_count && 
                atoms[atom_idx].hybridization > 1.5 && atoms[atom_idx].hybridization < 2.5) {
                sp2_count++;
            }
        }
        
        // More sp2 atoms in small rings increases strain
        if (!is_aromatic) { // Only apply if not aromatic
            total_strain *= (1.0 + 0.1 * sp2_count);
        }
    }
    
    // Distribute strain to atoms
    double strain_per_atom = total_strain / ring_size;
    for (int i = 0; i < ring_size; i++) {
        int atom_idx = ring_atoms[i];
        if (atom_idx >= 0 && atom_idx < atom_count) {
            atoms[atom_idx].ring_strain_contribution += strain_per_atom;
        }
    }
    
    return total_strain;
}