#ifndef SMILES_PARSER_H
#define SMILES_PARSER_H

#include "molecule.h"
#include <stdbool.h>

#define MAX_SMILES_TOKENS 1024

// Public functions
void initialize_molecule_data();
AtomProperties get_atom_properties(const char* symbol);
void set_atom_initial_properties(AtomPos *atom, const AtomProperties *props);
int parse_smiles(const char *s, bool *had_complex_feature_warning_flag);

// Helper functions
const char* get_aromatic_element(char aromatic_char);
bool is_valid_element(const char* symbol);
bool is_smiles_delimiter(char c);

#endif // SMILES_PARSER_H