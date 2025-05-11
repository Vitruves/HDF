#ifndef MOLECULE_H
#define MOLECULE_H

#include <stdbool.h>

// Constants
#define MAX_ATOMS 1000
#define MAX_BONDS 2000
#define MAX_FEATURE_CHANNELS 40  // Increased to support new channels

// Feature channel types
typedef enum {
    CHANNEL_ELECTRON_DENSITY = 0,
    CHANNEL_LIPOPHILICITY = 1,
    CHANNEL_HYDROGEN_DONOR = 2,
    CHANNEL_HYDROGEN_ACCEPTOR = 3,
    CHANNEL_POSITIVE_CHARGE = 4,
    CHANNEL_NEGATIVE_CHARGE = 5,
    CHANNEL_AROMATICITY = 6,
    CHANNEL_SP2_HYBRIDIZATION = 7,
    CHANNEL_SP3_HYBRIDIZATION = 8,
    CHANNEL_GASTEIGER_CHARGE = 9,
    CHANNEL_RING_MEMBERSHIP = 10,
    CHANNEL_AROMATIC_RING = 11,
    CHANNEL_ALIPHATIC_RING = 12,
    CHANNEL_POLARIZABILITY = 13,
    CHANNEL_VDWAALS_INTERACTION = 14,
    CHANNEL_ATOMIC_REFRACTIVITY = 15,
    CHANNEL_ELECTRONEGATIVITY = 16,
    CHANNEL_BOND_ORDER_INFLUENCE = 17,
    CHANNEL_STEREOCHEMISTRY = 18,
    CHANNEL_ROTATABLE_BOND_INFLUENCE = 19,
    CHANNEL_MOLECULAR_SHAPE = 20,
    CHANNEL_SURFACE_ACCESSIBILITY = 21,
    CHANNEL_PHARMACOPHORE_HYDROPHOBIC = 22,
    CHANNEL_PHARMACOPHORE_AROMATIC = 23,
    CHANNEL_ISOTOPE_EFFECT = 24,
    CHANNEL_QUANTUM_EFFECTS = 25,
    // New channels for enhanced molecular representation
    CHANNEL_ORBITAL_HYBRIDIZATION = 26,
    CHANNEL_CONJUGATION_EXTENT = 27,
    CHANNEL_RING_STRAIN = 28,
    CHANNEL_ATOMIC_PARTIAL_CHARGE = 29,
    CHANNEL_HYDROGEN_BOND_GEOMETRY = 30,
    CHANNEL_MOLECULAR_ORBITAL_ENERGY = 31,
    CHANNEL_RESONANCE_STABILIZATION = 32,
    CHANNEL_STERIC_HINDRANCE = 33,
    CHANNEL_ELECTRONIC_DELOCALIZATION = 34,
    CHANNEL_BOND_ANGLE_STRAIN = 35,
    CHANNEL_INTRAMOLECULAR_INTERACTIONS = 36,
    CHANNEL_FRAGMENT_CONTRIBUTIONS = 37,
    CHANNEL_SCAFFOLD_IDENTITY = 38,
    CHANNEL_FUNCTIONAL_GROUP_ENVIRONMENT = 39
} FeatureChannelType;

// Bond types
typedef enum {
    BOND_SINGLE = 0,
    BOND_DOUBLE = 1,
    BOND_TRIPLE = 2, 
    BOND_AROMATIC = 3,
    BOND_UNKNOWN = 4,
    BOND_ROTATABLE = 5,
    BOND_AMIDE = 6,
    BOND_CONJUGATED = 7,
    BOND_IONIC = 8,
    BOND_HYDROGEN = 9,
    BOND_DATIVE = 10,
    BOND_COORDINATED = 11,
    BOND_TORSIONALLY_STRAINED = 12
} BondType;

// Atom position and properties
typedef struct {
    char atom[4];               // Element symbol (C, H, O, etc.)
    double x, y, z;             // 3D coordinates
    double radius;              // Atomic radius
    int atomic_number;          // Element atomic number
    int isotope;                // Isotope number (0 for most common)
    int valence;                // Valence electron count
    double hybridization;       // Hybridization (1=sp, 2=sp2, 3=sp3)
    int explicit_h_count;       // Number of explicit hydrogens
    int charge;                 // Formal charge
    double electronegativity;   // Electronegativity value
    double effective_nuclear_charge; // For QM calculations
    bool is_aromatic;           // Aromatic flag
    bool in_ring;               // Is this atom part of any ring
    int ring_count;             // Number of rings this atom belongs to
    int ring_sizes[4];          // Sizes of rings (up to 4) this atom belongs to
    int orbital_config[4];      // s, p, d, f electron counts
    int n_bonds;                // Number of bonds connected to this atom
    double electron_density_max; // For QM visualization/scaling
    double polarizability;      // Atomic polarizability
    double refractivity;        // Atomic refractivity
    bool is_chiral;             // Chiral center flag
    char chirality;             // R/S chirality descriptor
    double surface_area;        // Approximate surface area contribution
    double solvent_accessibility; // Solvent accessibility score
    bool is_pharmacophore;      // Is this atom part of a pharmacophore
    int pharmacophore_type;     // Type of pharmacophore (hydrophobic, donor, etc.)
    double logP_contribution;   // Contribution to molecular logP
    bool is_rotatable_bond_atom; // Is this atom part of a rotatable bond
    // New properties for enhanced molecular representation
    double partial_charge;      // Partial charge from quantum calculations
    double atomic_hardness;     // Pearson hardness - resistance to charge transfer
    double atomic_softness;     // Inverse of hardness - susceptibility to polarization
    double atom_centered_fragments[5]; // Contribution to molecular fragments
    double molecular_orbital_coefficients[10]; // Coefficients in key molecular orbitals
    double vdw_interaction_strength;  // van der Waals interaction strength
    int conjugation_path_length; // Length of conjugation path this atom belongs to
    double ring_strain_contribution; // Contribution to ring strain energy
    double resonance_energy_contribution; // Contribution to resonance energy
    bool is_sterically_hindered; // Flag for steric hindrance
    double electronic_spatial_extent; // Spatial extent of electron density
    double local_ionization_energy; // Local ionization energy
    double bond_angles[4];      // Bond angles for geometry analysis
    double topological_torsion_values[4]; // Topological torsion descriptors
    double electron_donating_power; // Electron donating capability
    double electron_withdrawing_power; // Electron withdrawing capability
} AtomPos;

// Bond segment
typedef struct {
    int a, b;                   // Indices of atoms in the bond
    double length;              // Bond length
    int order;                  // Bond order (1, 2, 3)
    BondType type;              // Bond type
    bool in_ring;               // Is this bond part of a ring
    int ring_size;              // Size of smallest ring containing this bond (0 if not in ring)
    bool is_rotatable;          // Is this a rotatable bond
    bool is_conjugated;         // Is this a conjugated bond
    bool is_amide;              // Is this an amide bond
    double partial_charge_diff; // Difference in partial charges across bond
    double bond_dipole;         // Bond dipole moment
    double bond_energy;         // Approximate bond energy
    bool is_cis_trans;          // Has cis/trans geometry
    char cis_trans;             // E/Z or cis/trans descriptor
    // New properties for enhanced bond representation
    double bond_angle;          // Bond angle in 3D space
    double torsion_angle;       // Torsion/dihedral angle if part of a 4-atom system
    double bond_pi_character;   // Degree of pi character (0.0-1.0)
    double resonance_strength;  // Strength of resonance effects 
    double bond_dissociation_energy; // Energy required to break this bond
    double bond_stretching_constant; // Force constant for bond stretching
    double bond_strain_energy;  // Strain energy associated with this bond
    double orbital_overlap;     // Approximate orbital overlap integral
    double interaction_strength; // Strength of interaction (for non-covalent bonds)
    double electron_delocalization; // Extent of electron delocalization
    bool is_part_of_scaffold;   // Is this bond part of the core molecular scaffold
    int conjugation_system_id;  // ID of conjugation system this bond belongs to
    double bond_topology_index; // Topological index for this bond
} BondSeg;

// Colormap types
typedef enum {
    COLORMAP_GRAYSCALE = 0,     // Grayscale colormap
    COLORMAP_JET = 1,           // Jet colormap
    COLORMAP_VIRIDIS = 2,       // Viridis colormap
    COLORMAP_PLASMA = 3,        // Plasma colormap
    COLORMAP_INFERNO = 4,       // Inferno colormap
    COLORMAP_MAGMA = 5,         // Magma colormap
    COLORMAP_CIVIDIS = 6,       // Cividis colormap
    COLORMAP_TWILIGHT = 7,      // Twilight colormap
    COLORMAP_TURBO = 8,         // Turbo colormap
    COLORMAP_HEAT = 9           // Heat colormap
} ColormapType;

// Ring reference for SMILES parser
typedef struct {
    int idx;        // Atom index that opened the ring
    int pair_idx;   // Atom index that will close this specific ring instance (for multi-digit rings)
} RingRef;

// Pharmacophore point
typedef struct {
    double x, y, z;       // 3D coordinates
    int type;             // Pharmacophore type
    double radius;        // Interaction radius
    double strength;      // Interaction strength
    int associated_atom;  // Index of the atom this is associated with
    // New fields for enhanced pharmacophore representation
    double directionality[3]; // Directional vector for the pharmacophore
    double acceptance_angle;  // Angle of acceptance for directional pharmacophores
    double energy_well_depth; // Depth of the energy well for this point
    bool is_essential;        // Is this pharmacophore point essential for activity
    double volume_integral;   // Integrated volume contribution
} PharmacophorePoint;

// Topological feature
typedef struct {
    int type;             // Type of topological feature
    int atoms[10];        // Indices of atoms involved in this feature
    int atom_count;       // Number of atoms in this feature
    double score;         // Significance score for this feature
    // New fields for enhanced topological features
    double persistence;   // Topological persistence value
    int dimensionality;   // Topological dimensionality (0, 1, 2, etc.)
    double euler_characteristic; // Euler characteristic of this feature
    bool is_cyclic;       // Is this a cyclic feature
    double information_content; // Information content measure
} TopologicalFeature;

// Extended molecular structure
typedef struct {
    int atom_count;
    int bond_count;
    AtomPos *atoms;
    BondSeg *bonds;
    
    // Molecular properties
    double molecular_weight;
    double logP;
    double tpsa;
    double qed;
    int hba_count;
    int hbd_count;
    int rotatable_bond_count;
    
    // Topological features
    TopologicalFeature *topo_features;
    int topo_feature_count;
    
    // Pharmacophore features
    PharmacophorePoint *pharmacophores;
    int pharmacophore_count;
    
    // Ring information
    int ring_count;
    int ring_sizes[10];
    int ring_atom_indices[10][20];
    int ring_atom_counts[10];
    
    // Molecular connectivity indices
    double wiener_index;
    double balaban_index;
    double randic_index;
    
    // 3D structure quality metrics
    double energy;
    double rmsd;
    bool is_minimized;
    
    // New molecular properties for enhanced representation
    double molecular_orbital_energies[10]; // Energies of frontier molecular orbitals
    double homo_lumo_gap;                  // HOMO-LUMO energy gap
    double electron_affinity;              // Electron affinity
    double ionization_potential;           // Ionization potential
    double total_resonance_energy;         // Total resonance stabilization energy
    double total_strain_energy;            // Total strain energy
    double electronic_spatial_extent;      // Spatial extent of the electron density
    double principal_moments_of_inertia[3]; // Principal moments of inertia
    double shape_anisotropy;               // Measure of 3D shape anisotropy
    double scaffold_complexity;            // Complexity measure of the core scaffold
    double functional_group_count;         // Count of functional groups
    int stereocenter_count;                // Count of stereocenters
    double fragment_complexity;            // Complexity measure based on fragments
    double conformational_flexibility;     // Measure of conformational flexibility
    int conjugated_system_count;           // Count of conjugated systems
    double conjugated_system_sizes[10];    // Sizes of conjugated systems
} MolecularStructure;

#endif /* MOLECULE_H */