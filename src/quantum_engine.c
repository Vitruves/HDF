#include "quantum_engine.h"
#include "globals.h"
#include "utils.h"   // Added for PI and potentially other utils
#include <math.h>    // For sqrt, exp, cos, acos, atan2, fabs, fmod, pow

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