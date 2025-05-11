#ifndef SMILES_PARSER_H
#define SMILES_PARSER_H

#include "molecule.h"
#include <stdbool.h>

#define MAX_SMILES_TOKENS 1024
#define MAX_ATOM_SYMBOL_LEN 4

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

// Public functions
void initialize_molecule_data();
AtomProperties get_atom_properties(const char* symbol);
void set_atom_initial_properties(AtomPos *atom, const AtomProperties *props);
int parse_smiles(const char *s, bool *had_complex_feature_warning_flag);

// Helper functions
const char* get_aromatic_element(char aromatic_char);
bool is_valid_element(const char* symbol);
bool is_smiles_delimiter(char c);

// Function declarations
bool read_mol_file(const char *filename, AtomPos *atoms, BondSeg *bonds);
bool read_pdb_file(const char *filename, AtomPos *atoms, BondSeg *bonds);

#endif // SMILES_PARSER_H