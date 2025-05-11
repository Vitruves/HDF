#ifndef MOLECULE_H
#define MOLECULE_H

#include <stdbool.h>

// Constants
#define MAX_ATOMS 1000
#define MAX_BONDS 2000

// Bond types
typedef enum {
    BOND_SINGLE = 0,
    BOND_DOUBLE = 1,
    BOND_TRIPLE = 2, 
    BOND_AROMATIC = 3,
    BOND_UNKNOWN = 4
} BondType;

// Atom position and properties
typedef struct {
    char atom[4];               // Element symbol (C, H, O, etc.)
    double x, y, z;             // 3D coordinates
    double radius;              // Atomic radius
    int atomic_number;          // Element atomic number
    int isotope;                // Isotope number (0 for most common)
    int valence;                // Valence electron count
    int hybridization;          // Hybridization (1=sp, 2=sp2, 3=sp3)
    int explicit_h_count;       // Number of explicit hydrogens
    int charge;                 // Formal charge
    double electronegativity;   // Electronegativity value
    double effective_nuclear_charge; // For QM calculations
    bool is_aromatic;           // Aromatic flag
    bool in_ring;               // Is this atom part of any ring
    int orbital_config[4];      // s, p, d, f electron counts
    int n_bonds;                // Number of bonds connected to this atom
    double electron_density_max; // For QM visualization/scaling
} AtomPos;

// Bond segment
typedef struct {
    int a, b;                   // Indices of atoms in the bond
    double length;              // Bond length
    int order;                  // Bond order (1, 2, 3)
    BondType type;              // Bond type
    bool in_ring;               // Is this bond part of a ring
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

#endif /* MOLECULE_H */