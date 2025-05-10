#include "smiles_parser.h"
#include "globals.h" // For atoms, bonds, atom_count, bond_count, rings
#include "molecule.h"
#include "quantum_engine.h" // For calculate_hybridization
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h> // For sqrt, cos, sin
#include <stdbool.h>

// Define the element_properties array here
const AtomProperties element_properties[] = {
    {"H", 1, 2.20, 0.37, 1, 1.008, 1.0, 13.598, 0.754, {1,0,0,0}},
    {"He", 2, 0.00, 0.32, 0, 4.003, 1.7, 24.587, 0.000, {2,0,0,0}},
    {"Li", 3, 0.98, 1.34, 1, 6.941, 1.3, 5.392, 0.618, {1,0,0,0}},
    {"Be", 4, 1.57, 0.90, 2, 9.012, 1.95, 9.323, 0.000, {2,0,0,0}},
    {"B", 5, 2.04, 0.82, 3, 10.811, 2.6, 8.298, 0.277, {2,1,0,0}},
    {"C", 6, 2.55, 0.77, 4, 12.011, 3.25, 11.260, 1.263, {2,2,0,0}},
    {"N", 7, 3.04, 0.75, 5, 14.007, 3.9, 14.534, 0.000, {2,3,0,0}},
    {"O", 8, 3.44, 0.73, 6, 15.999, 4.55, 13.618, 1.461, {2,4,0,0}},
    {"F", 9, 3.98, 0.71, 7, 18.998, 5.2, 17.423, 3.339, {2,5,0,0}}, // Duplicated F, removed one below
    {"Ne", 10, 0.00, 0.69, 8, 20.180, 5.85, 21.565, 0.000, {2,6,0,0}},
    {"Na", 11, 0.93, 1.54, 1, 22.990, 2.2, 5.139, 0.548, {1,0,0,0}},
    {"Mg", 12, 1.31, 1.30, 2, 24.305, 2.85, 7.646, 0.000, {2,0,0,0}},
    {"Al", 13, 1.61, 1.18, 3, 26.982, 3.5, 5.986, 0.441, {2,1,0,0}},
    {"Si", 14, 1.90, 1.11, 4, 28.086, 4.15, 8.152, 1.385, {2,2,0,0}},
    {"P", 15, 2.19, 1.06, 5, 30.974, 4.8, 10.487, 0.746, {2,3,0,0}},
    {"S", 16, 2.58, 1.02, 6, 32.066, 5.45, 10.360, 2.077, {2,4,0,0}},
    {"Cl", 17, 3.16, 0.99, 7, 35.453, 6.1, 12.968, 3.617, {2,5,0,0}},
    {"Ar", 18, 0.00, 0.97, 8, 39.948, 6.75, 15.760, 0.000, {2,6,0,0}},
    {"K", 19, 0.82, 1.96, 1, 39.098, 2.2, 4.341, 0.501, {1,0,0,0}},
    {"Ca", 20, 1.00, 1.74, 2, 40.078, 2.85, 6.113, 0.025, {2,0,0,0}},
    {"Br", 35, 2.96, 1.14, 7, 79.904, 7.4, 11.814, 3.365, {2,5,0,0}}, // Corrected atomic number for F if it was a typo for Br
    {"I", 53, 2.66, 1.33, 7, 126.904, 7.0, 10.451, 3.059, {2,5,0,0}},
    // {"F", 9, 3.98, 0.71, 7, 18.998, 5.2, 17.423, 3.339, {2,5,0,0}}, // This was a duplicate F
    {"Se", 34, 2.55, 1.16, 6, 78.971, 6.5, 9.752, 2.020, {2,4,0,0}},
    {"Zn", 30, 1.65, 1.31, 2, 65.38, 3.0, 9.394, 0.000, {2,0,0,0}},
    {"", 0, 0.00, 0.00, 0, 0.000, 0.0, 0.000, 0.000, {0,0,0,0}}  /* Terminator */
};

// Aromatic atom mappings
typedef struct {
    char aromatic;
    const char *element;
} AromaticMapping;

static const AromaticMapping aromatic_atoms[] = {
    {'c', "C"},
    {'n', "N"},
    {'o', "O"},
    {'p', "P"},
    {'s', "S"},
    {'b', "B"},
    {'a', "C"}, // Sometimes used for aromatic carbon
    {'\0', NULL} // Terminator
};

// Valid two-letter elements
static const char *valid_elements[] = {
    "Br", "Cl", "Si", "Se", "As", "Li", "Be", "Na", "Mg", 
    "Al", "Ca", "Fe", "Zn", "He", "Ne", "Ar", "Kr", "Xe", 
    "Ga", "Ge", "Ru", "Pd", "Ag", "Cd", "In", "Sn", "Sb", 
    "Te", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", 
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", 
    "Hf", "Ta", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", 
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", 
    "Pa", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", 
    NULL // Terminator
};

// Parser configuration
typedef struct {
    bool handle_stereochemistry;
    bool handle_aromatic;
    bool handle_ring_closures;
    int max_ring_size;
    bool verbose_warnings;
} ParserConfig;

// Initialize with reasonable defaults
static ParserConfig parser_config = {
    .handle_stereochemistry = true,
    .handle_aromatic = true,
    .handle_ring_closures = true,
    .max_ring_size = 100,
    .verbose_warnings = false
};

void initialize_molecule_data() {
    atom_count = 0;
    bond_count = 0;
    memset(atoms, 0, sizeof(atoms));
    memset(bonds, 0, sizeof(bonds));
    for (int i = 0; i < 10; ++i) {
        rings[i].idx = -1;
        rings[i].pair_idx = -1;
    }
}

AtomProperties get_atom_properties(const char* symbol) {
    for (int i = 0; element_properties[i].atomic_number != 0; i++) {
        if (strcmp(element_properties[i].symbol, symbol) == 0) {
            return element_properties[i];
        }
    }
    fprintf(stderr, "Warning: Unknown element '%s', defaulting to Carbon properties.\n", symbol);
    return get_atom_properties("C"); // Default to Carbon
}

void set_atom_initial_properties(AtomPos *atom, const AtomProperties *props) {
    atom->atomic_number = props->atomic_number;
    atom->electronegativity = props->electronegativity;
    atom->radius = props->covalent_radius;
    atom->valence = props->valence_electrons;
    atom->effective_nuclear_charge = props->effective_nuclear_charge;
    memcpy(atom->orbital_config, props->orbital_config, sizeof(props->orbital_config));
    
    atom->electron_density_max = props->effective_nuclear_charge * 0.1; 
    atom->hybridization = 0.0;
    atom->charge = 0;
    atom->n_bonds = 0;
    atom->isotope = 0;             // Initialize new field
    atom->explicit_h_count = 0;    // Initialize new field
    atom->in_ring = false;         // Initialize new field
    atom->z = 0.0;                 // Initialize z coordinate
}

// Helper to check if a char is a valid SMILES delimiter
bool is_smiles_delimiter(char c) {
    return (c == '(' || c == ')' || c == '[' || c == ']' || 
            c == '.' || c == '=' || c == '#' || c == ':' || 
            c == '/' || c == '\\' || c == '@' || c == '+' || 
            c == '-' || isdigit(c) || c == '\0');
}

// Helper to get the element for an aromatic atom
const char* get_aromatic_element(char aromatic_char) {
    for (int i = 0; aromatic_atoms[i].element != NULL; i++) {
        if (aromatic_atoms[i].aromatic == aromatic_char) {
            return aromatic_atoms[i].element;
        }
    }
    return "C"; // Default to carbon
}

// Helper to check if a string is a valid two-letter element
bool is_valid_element(const char* symbol) {
    for (int i = 0; valid_elements[i] != NULL; i++) {
        if (strcasecmp(symbol, valid_elements[i]) == 0) {
            return true;
        }
    }
    return false;
}

// Helper function to precheck if a SMILES string has very large ring numbers
bool has_unsupported_ring_size(const char *smiles, int max_supported) {
    for (int i = 0; smiles[i]; i++) {
        if (smiles[i] == '%' && isdigit(smiles[i+1]) && isdigit(smiles[i+2])) {
            int ring_num = (smiles[i+1] - '0') * 10 + (smiles[i+2] - '0');
            if (ring_num >= max_supported) {
                fprintf(stderr, "Error: Ring number %d (from %%%c%c) too large (max %d supported).\n",
                        ring_num, smiles[i+1], smiles[i+2], max_supported);
                return true;
            }
             i += 2; // Advance past the two digits
        }
        // Check for two consecutive digits not preceded by %
        // This logic might be too aggressive if numbers appear in other contexts.
        // For now, relying on '%' for rings >= 10 is safer for SMILES.
        // Single digits 0-9 are handled directly.
    }
    return false;
}

// Parse SMILES string
int parse_smiles(const char *s, bool *had_complex_feature_warning_flag) {
    initialize_molecule_data();
    *had_complex_feature_warning_flag = false;

    if (has_unsupported_ring_size(s, parser_config.max_ring_size)) {
        // Error already printed by has_unsupported_ring_size
        return 0;
    }
    
    double angle = 0.0; // Current drawing angle
    double bond_length_step = 1.0; // Default bond length for 2D projection

    int current_atom_idx = -1;
    int pending_bond_order = 1;
    bool pending_bond_aromatic = false;

    int branch_atom_stack[MAX_SMILES_TOKENS];
    double branch_angle_stack[MAX_SMILES_TOKENS];
    int branch_stack_ptr = 0;

    int ring_closure_partners[100]; // Stores index of atom opening the ring
    for (int i = 0; i < 100; i++) {
        ring_closure_partners[i] = -1;
    }

    for (int i = 0; s[i]; ++i) {
        char token = s[i];
        
        if (isspace(token)) continue;

        if (isalpha(token)) {
            char element_symbol[MAX_ATOM_SYMBOL_LEN] = {0};
            bool is_explicitly_aromatic = islower(token);
            int char_consumed_for_element = 0;
            
            if (is_explicitly_aromatic) {
                const char* mapped_elem = get_aromatic_element(token);
                strncpy(element_symbol, mapped_elem, MAX_ATOM_SYMBOL_LEN - 1);
                char_consumed_for_element = 1;
            } else {
                element_symbol[0] = token;
                char_consumed_for_element = 1;
                if (s[i+1] && islower(s[i+1])) {
                    char two_letter_check[3] = {token, s[i+1], '\0'};
                    if (is_valid_element(two_letter_check)) {
                        element_symbol[1] = s[i+1];
                        char_consumed_for_element = 2;
                    }
                }
            }
            i += (char_consumed_for_element - 1); // Adjust main loop counter

            if (atom_count >= MAX_ATOMS) { /* ... error ... */ return 0; }
            AtomPos *new_atom = &atoms[atom_count];
            strcpy(new_atom->atom, element_symbol);
            new_atom->is_aromatic = is_explicitly_aromatic; // Initial aromaticity

            AtomProperties props = get_atom_properties(element_symbol);
            set_atom_initial_properties(new_atom, &props);
            
            if (current_atom_idx == -1) {
                new_atom->x = 0.0; new_atom->y = 0.0; new_atom->z = 0.0;
            } else {
                new_atom->x = atoms[current_atom_idx].x + bond_length_step * cos(angle);
                new_atom->y = atoms[current_atom_idx].y + bond_length_step * sin(angle);
                new_atom->z = atoms[current_atom_idx].z; // Keep same Z plane for simple chain extension

                if (bond_count >= MAX_ATOMS * 2) { /* ... error ... */ return 0; }
                bonds[bond_count].a = current_atom_idx;
                bonds[bond_count].b = atom_count;
                bonds[bond_count].order = pending_bond_order;
                
                if (pending_bond_aromatic || new_atom->is_aromatic || atoms[current_atom_idx].is_aromatic) {
                    bonds[bond_count].type = BOND_AROMATIC;
                    new_atom->is_aromatic = true; // Propagate aromaticity
                    atoms[current_atom_idx].is_aromatic = true;
                } else {
                    bonds[bond_count].type = (BondType)pending_bond_order;
                }
                
                atoms[current_atom_idx].n_bonds++;
                new_atom->n_bonds++;
                bond_count++;
            }
            current_atom_idx = atom_count;
            atom_count++;
            pending_bond_order = 1;
            pending_bond_aromatic = false;
            angle -= PI / 3.0;

        } else if (token == '=') { pending_bond_order = 2;
        } else if (token == '#') { pending_bond_order = 3;
        } else if (token == ':') { pending_bond_aromatic = true;
        } else if (token == '(') {
            if (branch_stack_ptr >= MAX_SMILES_TOKENS) { /* ... error ... */ return 0; }
            branch_atom_stack[branch_stack_ptr] = current_atom_idx;
            branch_angle_stack[branch_stack_ptr] = angle;
            branch_stack_ptr++;
            angle += PI / 3.0; // Adjust angle for branch
        } else if (token == ')') {
            if (branch_stack_ptr <= 0) { /* ... error ... */ return 0; }
            branch_stack_ptr--;
            current_atom_idx = branch_atom_stack[branch_stack_ptr];
            angle = branch_angle_stack[branch_stack_ptr];
            angle -= PI / 3.0; // Revert angle adjustment, continue main chain
            pending_bond_order = 1;
            pending_bond_aromatic = false;
        } else if (isdigit(token) || token == '%') {
            int ring_num;
            if (token == '%') {
                if (isdigit(s[i+1]) && isdigit(s[i+2])) {
                    ring_num = (s[i+1] - '0') * 10 + (s[i+2] - '0');
                    i += 2; // Consumed '%NN'
                } else { /* ... error: invalid % sequence ... */ return 0; }
            } else {
                ring_num = token - '0';
            }

            if (ring_num >= 100) { /* ... error: ring_num too large ... */ return 0;}

            if (ring_closure_partners[ring_num] == -1) { // Opening a ring
                ring_closure_partners[ring_num] = current_atom_idx;
            } else { // Closing a ring
                int partner_atom_idx = ring_closure_partners[ring_num];
                if (bond_count >= MAX_ATOMS * 2) { /* ... error ... */ return 0; }

                bonds[bond_count].a = partner_atom_idx;
                bonds[bond_count].b = current_atom_idx;
                bonds[bond_count].order = pending_bond_order;
                bonds[bond_count].in_ring = true;
                atoms[partner_atom_idx].in_ring = true;
                atoms[current_atom_idx].in_ring = true;

                if (pending_bond_aromatic || atoms[partner_atom_idx].is_aromatic || atoms[current_atom_idx].is_aromatic) {
                    bonds[bond_count].type = BOND_AROMATIC;
                    atoms[partner_atom_idx].is_aromatic = true; // Propagate aromaticity
                    atoms[current_atom_idx].is_aromatic = true;
                } else {
                    bonds[bond_count].type = (BondType)pending_bond_order;
                }

                atoms[partner_atom_idx].n_bonds++;
                atoms[current_atom_idx].n_bonds++;
                bond_count++;
                
                ring_closure_partners[ring_num] = -1; // Reset for potential reuse
                pending_bond_order = 1;
                pending_bond_aromatic = false;
            }
        } else if (token == '[') {
            i++; // Move past '['
            char element_symbol[MAX_ATOM_SYMBOL_LEN] = {0};
            int temp_isotope = 0;
            bool has_isotope = false;
            int temp_charge = 0;
            int h_count_val = -1; // -1 means not specified, use implicit
            bool is_explicitly_aromatic_in_bracket = false;
            double z_coord_from_stereo = 0.0; // For @, @@

            // 1. Isotope (digits at the beginning)
            if (isdigit(s[i])) {
                has_isotope = true;
                while(isdigit(s[i])) {
                    temp_isotope = temp_isotope * 10 + (s[i] - '0');
                    i++;
                }
            }

            // 2. Element symbol (must be present)
            if (!isalpha(s[i])) { /* ... error: element missing in bracket ... */ return 0; }
            
            if (islower(s[i])) { // Aromatic atom in bracket
                is_explicitly_aromatic_in_bracket = true;
                const char* mapped_elem = get_aromatic_element(s[i]);
                strncpy(element_symbol, mapped_elem, MAX_ATOM_SYMBOL_LEN - 1);
                i++;
            } else { // Non-aromatic or standard two-letter
                element_symbol[0] = s[i];
                i++;
                if (isalpha(s[i]) && islower(s[i])) {
                    char two_letter_check[3] = {element_symbol[0], s[i], '\0'};
                     if (is_valid_element(two_letter_check)) {
                        element_symbol[1] = s[i];
                        i++;
                    }
                }
            }

            // 3. Chirality, H count, charge
            int chiral_class = 0; // 0: none, 1: @, 2: @@

            while(s[i] && s[i] != ']') {
                if (s[i] == '@') {
                    i++;
                    if (s[i] == '@') { chiral_class = 2; i++; z_coord_from_stereo = -0.5; } // @@
                    else { chiral_class = 1; z_coord_from_stereo = 0.5;}              // @
                } else if (s[i] == 'H') {
                    i++;
                    if (isdigit(s[i])) {
                        h_count_val = s[i] - '0';
                        i++;
                    } else {
                        h_count_val = 1; // [CH] means one H
                    }
                } else if (s[i] == '+') {
                    i++;
                    if (isdigit(s[i])) { temp_charge += (s[i] - '0'); i++; }
                    else { temp_charge++; }
                } else if (s[i] == '-') {
                    i++;
                    if (isdigit(s[i])) { temp_charge -= (s[i] - '0'); i++; }
                    else { temp_charge--; }
                } else {
                    // Unknown char in bracket, could be class :N or other things
                    *had_complex_feature_warning_flag = true; // Flag unhandled bracket content
                    i++; // Skip it
                }
            }
            if (s[i] != ']') { /* ... error: unclosed bracket ... */ return 0; }
            // i is now on ']'

            if (atom_count >= MAX_ATOMS) { /* ... error ... */ return 0; }
            AtomPos *new_atom = &atoms[atom_count];
            strcpy(new_atom->atom, element_symbol);
            AtomProperties props = get_atom_properties(element_symbol);
            set_atom_initial_properties(new_atom, &props);

            new_atom->is_aromatic = is_explicitly_aromatic_in_bracket;
            if (has_isotope) new_atom->isotope = temp_isotope;
            new_atom->charge = temp_charge;
            if (h_count_val != -1) new_atom->explicit_h_count = h_count_val;
            new_atom->z = z_coord_from_stereo; // Apply z from @ or @@

            if (current_atom_idx == -1) {
                new_atom->x = 0.0; new_atom->y = 0.0;
                // new_atom->z is already set if chiral
            } else {
                new_atom->x = atoms[current_atom_idx].x + bond_length_step * cos(angle);
                new_atom->y = atoms[current_atom_idx].y + bond_length_step * sin(angle);
                // new_atom->z is already set if chiral, otherwise inherit from previous or keep 0
                if (chiral_class == 0) new_atom->z = atoms[current_atom_idx].z;


                if (bond_count >= MAX_ATOMS * 2) { /* ... error ... */ return 0; }
                bonds[bond_count].a = current_atom_idx;
                bonds[bond_count].b = atom_count;
                bonds[bond_count].order = pending_bond_order;

                if (pending_bond_aromatic || new_atom->is_aromatic || atoms[current_atom_idx].is_aromatic) {
                    bonds[bond_count].type = BOND_AROMATIC;
                     new_atom->is_aromatic = true; // Propagate aromaticity
                    atoms[current_atom_idx].is_aromatic = true;
                } else {
                    bonds[bond_count].type = (BondType)pending_bond_order;
                }
                
                atoms[current_atom_idx].n_bonds++;
                new_atom->n_bonds++;
                bond_count++;
            }
            current_atom_idx = atom_count;
            atom_count++;
            pending_bond_order = 1;
            pending_bond_aromatic = false;
            angle -= PI / 3.0;

        } else if (token == '.') { // Disconnected structure
            current_atom_idx = -1;
            pending_bond_order = 1;
            pending_bond_aromatic = false;
            // Reset angle for new component, or offset it significantly
            angle = 0.0; 
        } else if (token == '/' || token == '\\') {
            // Cis/trans isomerism - complex 3D, for now, flag and skip
            *had_complex_feature_warning_flag = true;
        } else if (token == '+' || token == '-') {
            // Charge on an atom not in brackets - this is unusual in standard SMILES
            // but could appear. If current_atom_idx is valid, apply charge.
            if(current_atom_idx != -1 && atom_count > 0){
                int charge_mod = (token == '+') ? 1 : -1;
                if(isdigit(s[i+1])){
                    charge_mod *= (s[i+1] - '0');
                    i++;
                }
                atoms[current_atom_idx].charge += charge_mod;
            } else {
                 *had_complex_feature_warning_flag = true; // Unattached charge
            }
        } else {
            // Unknown token
            // *had_complex_feature_warning_flag = true; // Temporarily disable for testing
        }
    }

    if (branch_stack_ptr != 0) { /* ... error: unmatched parens ... */ return 0; }
    
    bool unclosed_rings_exist = false;
    for (int r_idx = 0; r_idx < 100; ++r_idx) {
        if (ring_closure_partners[r_idx] != -1) {
            unclosed_rings_exist = true;
            break;
        }
    }
    if (unclosed_rings_exist) {
        *had_complex_feature_warning_flag = true;
        if (parser_config.verbose_warnings) {
            fprintf(stderr, "Warning: SMILES string contains unclosed ring(s).\n");
        }
    }
    
    // Final pass to ensure aromaticity is consistently propagated within identified rings
    for(int k=0; k < 5; ++k) { // Iterate a few times to propagate
        for (int j = 0; j < bond_count; ++j) {
            if (bonds[j].type == BOND_AROMATIC) {
                if (!atoms[bonds[j].a].is_aromatic) atoms[bonds[j].a].is_aromatic = true;
                if (!atoms[bonds[j].b].is_aromatic) atoms[bonds[j].b].is_aromatic = true;
            } else if (bonds[j].in_ring && (atoms[bonds[j].a].is_aromatic || atoms[bonds[j].b].is_aromatic)) {
                // If a bond is in a ring and one atom is aromatic, the bond might be considered aromatic
                // This heuristic can be refined by Kekulization or ring perception algorithms.
                // For now, if it's in a ring and connected to an aromatic atom, mark it aromatic.
                // This might be too aggressive.
                // bonds[j].type = BOND_AROMATIC;
                // atoms[bonds[j].a].is_aromatic = true;
                // atoms[bonds[j].b].is_aromatic = true;
            }
        }
    }


    // Update hybridization based on final n_bonds and aromaticity
    for (int i = 0; i < atom_count; ++i) {
        atoms[i].hybridization = calculate_hybridization(atoms[i]);
    }


    return atom_count;
}