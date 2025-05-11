#ifndef SMILES_PARSER_H
#define SMILES_PARSER_H

#include "molecule.h"
#include <stdbool.h>
#include <stdarg.h>

#define MAX_SMILES_TOKENS 1024
#define MAX_ATOM_SYMBOL_LEN 4

// RingRef is defined in molecule.h
extern RingRef rings[10];

// Define AtomProperties struct
typedef struct AtomProperties {
    char symbol[4];
    int atomic_number;
    double electronegativity; // Pauling scale
    double radius; // Covalent radius in Angstroms
    int valence; // Typical valence electrons
    double atomic_weight;
    double effective_nuclear_charge; // Slater's rules or similar
    double ionization_energy; // First ionization energy in eV
    double electron_affinity; // In eV, positive for energy released
    int orbital_config[4]; // s, p, d, f electron counts for valence shell or relevant shells
} AtomProperties;

// SMILES parser error codes
typedef enum {
    SMILES_SUCCESS = 0,
    SMILES_ERROR_INVALID_SYNTAX = -1,
    SMILES_ERROR_MEMORY_ALLOCATION = -2,
    SMILES_ERROR_RING_NUMBER_TOO_LARGE = -3,
    SMILES_ERROR_INVALID_ELEMENT = -4,
    SMILES_ERROR_MAX_ATOMS_EXCEEDED = -5,
    SMILES_ERROR_INVALID_BRACKET = -6,
    SMILES_ERROR_UNMATCHED_BRANCH = -7
} SmilesErrorCode;

// Public functions
bool initialize_molecule_data(void);
AtomProperties get_atom_properties(const char* symbol);
void set_atom_initial_properties(AtomPos *atom, const AtomProperties *props);
int parse_smiles(const char *s, bool *had_complex_feature_warning_flag);
int parse_smiles_with_progress(const char *smiles, bool *had_warnings, void (*progress_callback)(float));
void optimize_2d_coordinates(void);

// Helper functions
const char* get_aromatic_element(char aromatic_char);
bool is_valid_element(const char* symbol);
bool is_smiles_delimiter(char c);
double predict_bond_angle(AtomPos atom);

// Function declarations
bool read_mol_file(const char *filename, AtomPos *atoms, BondSeg *bonds);
bool read_pdb_file(const char *filename, AtomPos *atoms, BondSeg *bonds);

// Element property calculator
double calculate_hybridization(AtomPos atom);

// Configure parsing behavior
typedef struct {
    bool handle_stereochemistry;
    bool handle_aromatic;
    bool handle_ring_closures;
    int max_ring_size;
    bool verbose_warnings;
} ParserConfig;

extern ParserConfig parser_config;

// Check if a SMILES string has an unsupported ring size
bool has_unsupported_ring_size(const char *smiles, int max_supported);

#endif // SMILES_PARSER_H