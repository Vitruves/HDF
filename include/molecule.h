#ifndef MOLECULE_H
#define MOLECULE_H

#include <stdbool.h>

#define MAX_ATOM_SYMBOL_LEN 4

/* Colormap options */
typedef enum {
    COLORMAP_GRAYSCALE,
    COLORMAP_JET,
    COLORMAP_VIRIDIS,
    COLORMAP_PLASMA,
    COLORMAP_HEAT
} ColormapType;

/* Bond types */
typedef enum {
    BOND_SINGLE = 1,
    BOND_DOUBLE = 2,
    BOND_TRIPLE = 3,
    BOND_AROMATIC = 4 // Should be 1.5 or similar in SMILES, but distinct type here
} BondType;

/* Atom properties lookup tables */
typedef struct {
    char symbol[MAX_ATOM_SYMBOL_LEN];
    int atomic_number;
    double electronegativity;  /* Pauling scale */
    double covalent_radius;    /* Angstroms */
    int valence_electrons;
    double mass;               /* Atomic mass units */
    /* Quantum mechanical properties */
    double effective_nuclear_charge;  /* Slater's rules */
    double ionization_energy;  /* eV */
    double electron_affinity;  /* eV */
    int orbital_config[4];    /* Number of electrons in s,p,d,f shells of valence */
} AtomProperties;

typedef struct {
    double x, y;               /* 2D coordinates */
    double z;                  /* Optional 3D coordinate from stereochemistry */
    char atom[MAX_ATOM_SYMBOL_LEN]; /* Atom symbol */
    int atomic_number;         /* Atomic number for quick reference */
    double electronegativity;  /* Electronegativity value */
    double radius;             /* Covalent radius */
    int valence;               /* Valence electrons */
    int charge;                /* Formal charge */
    int n_bonds;               /* Number of bonds connected to this atom */
    bool is_aromatic;           /* Is part of aromatic system */
    int isotope;               /* Isotope number (0 if not specified) */
    int explicit_h_count;      /* Count of explicit hydrogens (e.g., [CH3]) */
    bool in_ring;              /* True if this atom is part of any ring */
    /* Quantum mechanical properties */
    double effective_nuclear_charge;  /* Slater's effective Z */
    int orbital_config[4];     /* s,p,d,f electron configuration */
    double electron_density_max; /* Maximum electron density (placeholder) */
    double hybridization;      /* Hybridization state (1=sp, 2=sp2, 3=sp3, 0=unknown) */
} AtomPos;

typedef struct {
    int a, b;                  /* Indices of connected atoms */
    int order;                 /* Bond order (1, 2, 3) */
    BondType type;             /* Bond type (with aromatic) */
    double length;             /* Calculated bond length */
    bool in_ring;               /* Flag if bond is in a ring */
} BondSeg;

typedef struct {
    int idx;
    int pair_idx; // Which atom it pairs with for ring closure
    // int bond_order_to_ring_atom; // Store bond order for ring closure
} RingRef;


// External declaration of the periodic table
extern const AtomProperties element_properties[];

#endif // MOLECULE_H