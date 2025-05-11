#include "smiles_parser.h"
#include "globals.h" // For atoms, bonds, atom_count, bond_count, rings
#include "molecule.h"
#include "quantum_engine.h" // For calculate_hybridization
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h> // For sqrt, cos, sin
#include <stdbool.h>
#include <time.h> // For time

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
    {"Sc", 21, 1.36, 1.44, 3, 44.956, 3.1, 6.5615, 0.188, {2,1,0,0}},
    {"Tc", 43, 1.9, 1.35, 7, 98.9062, 5.3, 7.28, 0.55, {2,5,0,0}},
    {"Br", 35, 2.96, 1.14, 7, 79.904, 7.4, 11.814, 3.365, {2,5,0,0}}, // Corrected atomic number for F if it was a typo for Br
    {"I", 53, 2.66, 1.33, 7, 126.904, 7.0, 10.451, 3.059, {2,5,0,0}},
    // {"F", 9, 3.98, 0.71, 7, 18.998, 5.2, 17.423, 3.339, {2,5,0,0}}, // This was a duplicate F
    {"Se", 34, 2.55, 1.16, 6, 78.971, 6.5, 9.752, 2.020, {2,4,0,0}},
    {"Zn", 30, 1.65, 1.31, 2, 65.38, 3.0, 9.394, 0.000, {2,0,0,0}},
    // Add Rhenium
    {"Re", 75, 1.9, 1.28, 7, 186.207, 7.88, 7.88, 0.15, {2,5,0,0}},
    // Add Tritium (an isotope of hydrogen, not a distinct element)
    {"T", 1, 2.20, 0.37, 1, 3.0160, 1.0, 13.598, 0.754, {1,0,0,0}},
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
    "T", "Tc", "W", "U", "Rh", "Mo", "Mn", "Cr", "V", "Sc", "Y", "Zr", "Nb",
    NULL // Terminator
};

// Parser configuration
// typedef struct {
//     bool handle_stereochemistry;
//     bool handle_aromatic;
//     bool handle_ring_closures;
//     int max_ring_size;
//     bool verbose_warnings;
// } ParserConfig;

// Initialize with reasonable defaults
ParserConfig parser_config = {
    .handle_stereochemistry = true,
    .handle_aromatic = true,
    .handle_ring_closures = true,
    .max_ring_size = 100,
    .verbose_warnings = false
};

// Initialize molecule data structures
// Returns true on success, false on failure
bool initialize_molecule_data() {
    // Free any previously allocated memory
    if (atoms != NULL) {
        free(atoms);
        atoms = NULL;
    }
    if (bonds != NULL) {
        free(bonds);
        bonds = NULL;
    }

    // Allocate memory for atoms and bonds
    atoms = (AtomPos*)calloc(MAX_ATOMS, sizeof(AtomPos));
    if (atoms == NULL) {
        fprintf(stderr, "Critical Error: Failed to allocate memory for global atoms array in initialize_molecule_data.\n");
        return false;
    }
    
    bonds = (BondSeg*)calloc(MAX_BONDS, sizeof(BondSeg));
    if (bonds == NULL) {
        fprintf(stderr, "Critical Error: Failed to allocate memory for global bonds array in initialize_molecule_data.\n");
        free(atoms); // Clean up atoms if bonds allocation failed
        atoms = NULL;
        return false;
    }

    atom_count = 0;
    bond_count = 0;
    
    // No need for memset since we used calloc
    
    // Initialize rings array
    for (int i = 0; i < 10; ++i) {
        rings[i].idx = -1;
        rings[i].pair_idx = -1;
    }
    
    return true;
}

AtomProperties get_atom_properties(const char* symbol) {
    static char last_unknown_symbols[10][8] = {""}; // Track the last 10 unknown symbols
    static int unknown_counts[10] = {0};      // Count of each unknown symbol
    static int unknown_index = 0;             // Current index in the circular buffer
    static int total_unknown_count = 0;       // Total unknown elements seen
    static time_t last_warning_time = 0;      // Time of last warning message
    static bool warning_summary_shown = false; // Track if we've shown a warning summary for this run
    
    // First, try to find the element in our known elements
    for (int i = 0; element_properties[i].atomic_number != 0; i++) {
        if (strcmp(element_properties[i].symbol, symbol) == 0) {
            return element_properties[i];
        }
    }
    
    // Check if we've seen this unknown element before
    int found_idx = -1;
    for (int i = 0; i < 10; i++) {
        if (last_unknown_symbols[i][0] != '\0' && strcmp(last_unknown_symbols[i], symbol) == 0) {
            unknown_counts[i]++;
            found_idx = i;
            break;
        }
    }
    
    if (found_idx == -1) {
        // New unknown element, add to circular buffer
        strncpy(last_unknown_symbols[unknown_index], symbol, 7);
        last_unknown_symbols[unknown_index][7] = '\0';
        unknown_counts[unknown_index] = 1;
        found_idx = unknown_index;
        
        // Move to next position in circular buffer
        unknown_index = (unknown_index + 1) % 10;
        
        // Show a warning for first occurrence only if we haven't shown too many already
        if (total_unknown_count < 5) {
            fprintf(stderr, "Warning: Unknown element '%s', defaulting to Carbon properties.\n", symbol);
        } else if (total_unknown_count == 5 && !warning_summary_shown) {
            fprintf(stderr, "Warning: Additional unknown elements will be summarized at the end of processing.\n");
            warning_summary_shown = true;
        }
    }
    
    total_unknown_count++;
    
    // Only show periodic summary if we have many unknown elements and some time has passed
    time_t now = time(NULL);
    if (total_unknown_count > 5 && now - last_warning_time > 15) { // Show summary every 15 seconds
        last_warning_time = now;
        
        // Find the 3 most common unknown elements
        int top_indices[3] = {-1, -1, -1};
        int top_counts[3] = {0, 0, 0};
        
        for (int i = 0; i < 10; i++) {
            if (last_unknown_symbols[i][0] == '\0') continue;
            
            if (unknown_counts[i] > top_counts[0]) {
                top_indices[2] = top_indices[1]; top_counts[2] = top_counts[1];
                top_indices[1] = top_indices[0]; top_counts[1] = top_counts[0];
                top_indices[0] = i; top_counts[0] = unknown_counts[i];
            } else if (unknown_counts[i] > top_counts[1]) {
                top_indices[2] = top_indices[1]; top_counts[2] = top_counts[1];
                top_indices[1] = i; top_counts[1] = unknown_counts[i];
            } else if (unknown_counts[i] > top_counts[2]) {
                top_indices[2] = i; top_counts[2] = unknown_counts[i];
            }
        }
        
        // Print a compact summary
        fprintf(stderr, "Warning: %d unknown elements encountered. ", total_unknown_count);
        fprintf(stderr, "Most common: ");
        
        int shown = 0;
        for (int i = 0; i < 3; i++) {
            if (top_indices[i] >= 0 && top_counts[i] > 0) {
                if (shown > 0) fprintf(stderr, ", ");
                fprintf(stderr, "'%s'(%d)", last_unknown_symbols[top_indices[i]], top_counts[i]);
                shown++;
            }
        }
        fprintf(stderr, "\n");
    }
    
    // Default to Carbon properties for unknown elements
    for (int i = 0; element_properties[i].atomic_number != 0; i++) {
        if (strcmp(element_properties[i].symbol, "C") == 0) {
            return element_properties[i];
        }
    }
    
    // Fallback to first element if carbon not found (shouldn't happen)
    return element_properties[0];
}

void set_atom_initial_properties(AtomPos *atom, const AtomProperties *props) {
    atom->atomic_number = props->atomic_number;
    atom->electronegativity = props->electronegativity;
    atom->radius = props->radius;
    atom->valence = props->valence;
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

// Ring closure tracking structure
typedef struct {
    int atom_idx;
    int bond_order;
    bool is_aromatic;
} RingClosure;

// Parser state structure
typedef struct {
    const char *smiles;
    int position;
    int current_atom_idx;
    int pending_bond_order;
    bool pending_bond_aromatic;
    double angle;
    double bond_length;
    RingClosure *ring_closures;
    int max_ring_closures;
    int branch_stack[MAX_SMILES_TOKENS];
    double branch_angle_stack[MAX_SMILES_TOKENS];
    int branch_stack_ptr;
    bool *had_complex_feature_flag;
    char error_message[256];
} ParserState;

// Initialize parser state
static void init_parser_state(ParserState *state, const char *smiles, bool *warning_flag) {
    state->smiles = smiles;
    state->position = 0;
    state->current_atom_idx = -1;
    state->pending_bond_order = 1;
    state->pending_bond_aromatic = false;
    state->angle = 0.0;
    state->bond_length = 1.0;
    state->branch_stack_ptr = 0;
    state->had_complex_feature_flag = warning_flag;
    
    // Use a more conservative, reasonable estimate for max ring closures
    // Typically molecules rarely have more than 10-20 ring closures
    // Even for complex structures, 100 should be plenty
    const size_t DEFAULT_MAX_CLOSURES = 100;
    size_t smiles_len = strlen(smiles);
    
    // Instead of using the full SMILES length, count potential ring closure digits
    size_t digit_count = 0;
    for (size_t i = 0; i < smiles_len; i++) {
        if (isdigit(smiles[i]) || smiles[i] == '%') {
            digit_count++;
        }
    }
    
    // Set a reasonable limit - either digit count or default max, whichever is greater
    state->max_ring_closures = (digit_count > DEFAULT_MAX_CLOSURES) ? 
                              digit_count : DEFAULT_MAX_CLOSURES;
    
    // Add safety cap to prevent excessive allocation
    if (state->max_ring_closures > 1000) {
        state->max_ring_closures = 1000;
    }
    
    // Allocate ring closure array with error handling
    state->ring_closures = NULL;
    state->ring_closures = (RingClosure*)calloc(state->max_ring_closures, sizeof(RingClosure));
    if (!state->ring_closures) {
        fprintf(stderr, "Critical Error: Failed to allocate memory for ring closures\n");
        state->max_ring_closures = 0;
        // Don't exit - allow caller to handle the error
        strcpy(state->error_message, "Memory allocation failed for ring closures");
        return;
    }
    
    // Initialize ring closure array
    for (int i = 0; i < state->max_ring_closures; i++) {
        state->ring_closures[i].atom_idx = -1;
    }
    
    state->error_message[0] = '\0';
}

// Clean up parser state
static void cleanup_parser_state(ParserState *state) {
    if (state->ring_closures) {
        free(state->ring_closures);
        state->ring_closures = NULL;
    }
}

// Set error message
static void set_parser_error(ParserState *state, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(state->error_message, sizeof(state->error_message), format, args);
    va_end(args);
}

// Parse atom from SMILES string
static bool parse_atom(ParserState *state) {
    const char *s = state->smiles;
    int i = state->position;
    char token = s[i];
    
    char element_symbol[MAX_ATOM_SYMBOL_LEN] = {0};
    bool is_explicitly_aromatic = islower(token);
    int char_consumed = 0;
    
    if (is_explicitly_aromatic) {
        const char* mapped_elem = get_aromatic_element(token);
        strncpy(element_symbol, mapped_elem, MAX_ATOM_SYMBOL_LEN - 1);
        char_consumed = 1;
    } else {
        element_symbol[0] = token;
        char_consumed = 1;
        if (s[i+1] && islower(s[i+1])) {
            char two_letter_check[3] = {token, s[i+1], '\0'};
            if (is_valid_element(two_letter_check)) {
                element_symbol[1] = s[i+1];
                char_consumed = 2;
            }
        }
    }
    
    // Pre-check for array bounds
    if (atom_count >= MAX_ATOMS) {
        set_parser_error(state, "Maximum atom count (%d) exceeded", MAX_ATOMS);
        return false;
    }
    
    AtomPos *new_atom = &atoms[atom_count];
    strcpy(new_atom->atom, element_symbol);
    new_atom->is_aromatic = is_explicitly_aromatic;
    
    AtomProperties props = get_atom_properties(element_symbol);
    set_atom_initial_properties(new_atom, &props);
    
    // Set coordinates
    if (state->current_atom_idx == -1) {
        new_atom->x = 0.0; 
        new_atom->y = 0.0; 
        new_atom->z = 0.0;
    } else {
        new_atom->x = atoms[state->current_atom_idx].x + state->bond_length * cos(state->angle);
        new_atom->y = atoms[state->current_atom_idx].y + state->bond_length * sin(state->angle);
        new_atom->z = atoms[state->current_atom_idx].z;
        
        // Check bond array bounds
        if (bond_count >= MAX_BONDS) {
            set_parser_error(state, "Maximum bond count (%d) exceeded", MAX_BONDS);
            return false;
        }
        
        // Create bond to previous atom
        BondSeg *new_bond = &bonds[bond_count];
        new_bond->a = state->current_atom_idx;
        new_bond->b = atom_count;
        new_bond->order = state->pending_bond_order;
        
        if (state->pending_bond_aromatic || new_atom->is_aromatic || 
            atoms[state->current_atom_idx].is_aromatic) {
            new_bond->type = BOND_AROMATIC;
            new_atom->is_aromatic = true;
            atoms[state->current_atom_idx].is_aromatic = true;
        } else {
            new_bond->type = (BondType)state->pending_bond_order;
        }
        
        atoms[state->current_atom_idx].n_bonds++;
        new_atom->n_bonds++;
        bond_count++;
    }
    
    state->current_atom_idx = atom_count;
    atom_count++;
    state->pending_bond_order = 1;
    state->pending_bond_aromatic = false;
    state->angle -= M_PI / 3.0;
    
    state->position += char_consumed;
    return true;
}

// Parse a bracketed atom [...]
static bool parse_bracketed_atom(ParserState *state) {
    const char *s = state->smiles;
    int i = state->position + 1; // Skip the opening '['
    
    char element_symbol[MAX_ATOM_SYMBOL_LEN] = {0};
    int temp_isotope = 0;
    bool has_isotope = false;
    int temp_charge = 0;
    int h_count_val = -1;
    bool is_explicitly_aromatic = false;
    double z_coord_from_stereo = 0.0;
    int chiral_class = 0;
    
    // Parse isotope
    if (isdigit(s[i])) {
        has_isotope = true;
        while (isdigit(s[i])) {
            temp_isotope = temp_isotope * 10 + (s[i] - '0');
            i++;
        }
    }
    
    // Parse element symbol
    if (!isalpha(s[i])) {
        set_parser_error(state, "Element symbol missing in bracket at position %d", i);
        return false;
    }
    
    if (islower(s[i])) {
        is_explicitly_aromatic = true;
        const char* mapped_elem = get_aromatic_element(s[i]);
        strncpy(element_symbol, mapped_elem, MAX_ATOM_SYMBOL_LEN - 1);
        i++;
    } else {
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
    
    // Parse chirality, H count, charge, etc.
    while (s[i] && s[i] != ']') {
        if (s[i] == '@') {
            i++;
            if (s[i] == '@') {
                chiral_class = 2;
                z_coord_from_stereo = -0.5;
                i++;
            } else {
                chiral_class = 1;
                z_coord_from_stereo = 0.5;
            }
        } else if (s[i] == 'H') {
            i++;
            if (isdigit(s[i])) {
                h_count_val = s[i] - '0';
                i++;
            } else {
                h_count_val = 1;
            }
        } else if (s[i] == '+') {
            i++;
            if (isdigit(s[i])) {
                temp_charge += (s[i] - '0');
                i++;
            } else {
                temp_charge++;
            }
        } else if (s[i] == '-') {
            i++;
            if (isdigit(s[i])) {
                temp_charge -= (s[i] - '0');
                i++;
            } else {
                temp_charge--;
            }
        } else {
            // Skip unsupported features
            if (state->had_complex_feature_flag) *state->had_complex_feature_flag = true;
            i++;
        }
    }
    
    if (s[i] != ']') {
        set_parser_error(state, "Unclosed bracket at position %d", state->position);
        return false;
    }
    
    // Check atom array bounds
    if (atom_count >= MAX_ATOMS) {
        set_parser_error(state, "Maximum atom count (%d) exceeded", MAX_ATOMS);
        return false;
    }
    
    // Create the atom
    AtomPos *new_atom = &atoms[atom_count];
    strcpy(new_atom->atom, element_symbol);
    AtomProperties props = get_atom_properties(element_symbol);
    set_atom_initial_properties(new_atom, &props);
    
    new_atom->is_aromatic = is_explicitly_aromatic;
    if (has_isotope) new_atom->isotope = temp_isotope;
    new_atom->charge = temp_charge;
    if (h_count_val != -1) new_atom->explicit_h_count = h_count_val;
    new_atom->z = z_coord_from_stereo;
    
    // Set coordinates
    if (state->current_atom_idx == -1) {
        new_atom->x = 0.0;
        new_atom->y = 0.0;
    } else {
        new_atom->x = atoms[state->current_atom_idx].x + state->bond_length * cos(state->angle);
        new_atom->y = atoms[state->current_atom_idx].y + state->bond_length * sin(state->angle);
        if (chiral_class == 0) new_atom->z = atoms[state->current_atom_idx].z;
        
        // Check bond array bounds
        if (bond_count >= MAX_BONDS) {
            set_parser_error(state, "Maximum bond count (%d) exceeded", MAX_BONDS);
            return false;
        }
        
        // Create bond to previous atom
        BondSeg *new_bond = &bonds[bond_count];
        new_bond->a = state->current_atom_idx;
        new_bond->b = atom_count;
        new_bond->order = state->pending_bond_order;
        
        if (state->pending_bond_aromatic || new_atom->is_aromatic || 
            atoms[state->current_atom_idx].is_aromatic) {
            new_bond->type = BOND_AROMATIC;
            atoms[state->current_atom_idx].is_aromatic = true;
            atoms[state->current_atom_idx].is_aromatic = true;
        } else {
            new_bond->type = (BondType)state->pending_bond_order;
        }
        
        atoms[state->current_atom_idx].n_bonds++;
        new_atom->n_bonds++;
        bond_count++;
    }
    
    state->current_atom_idx = atom_count;
    atom_count++;
    state->pending_bond_order = 1;
    state->pending_bond_aromatic = false;
    state->angle -= M_PI / 3.0;
    
    state->position = i + 1; // Skip the closing ']'
    return true;
}

// Parse a ring closure (digit or %nn)
static bool parse_ring_closure(ParserState *state) {
    const char *s = state->smiles;
    int i = state->position;
    int ring_num;
    
    if (s[i] == '%') {
        if (isdigit(s[i+1]) && isdigit(s[i+2])) {
            ring_num = (s[i+1] - '0') * 10 + (s[i+2] - '0');
            state->position += 3;
        } else {
            set_parser_error(state, "Invalid ring number format %%nn at position %d", i);
            return false;
        }
    } else {
        ring_num = s[i] - '0';
        state->position++;
    }
    
    // Check if ring number is in range
    if (ring_num >= state->max_ring_closures) {
        if (state->ring_closures) {
            free(state->ring_closures);
            state->max_ring_closures = ring_num + 100;
            state->ring_closures = (RingClosure*)calloc(state->max_ring_closures, sizeof(RingClosure));
            if (!state->ring_closures) {
                set_parser_error(state, "Failed to allocate memory for ring closure %d", ring_num);
                return false;
            }
            for (int j = 0; j < state->max_ring_closures; j++) {
                state->ring_closures[j].atom_idx = -1;
            }
        }
    }
    
    if (state->ring_closures[ring_num].atom_idx == -1) {
        // Opening a ring
        state->ring_closures[ring_num].atom_idx = state->current_atom_idx;
        state->ring_closures[ring_num].bond_order = state->pending_bond_order;
        state->ring_closures[ring_num].is_aromatic = state->pending_bond_aromatic;
    } else {
        // Closing a ring
        int partner_idx = state->ring_closures[ring_num].atom_idx;
        int closure_bond_order = state->ring_closures[ring_num].bond_order;
        bool closure_is_aromatic = state->ring_closures[ring_num].is_aromatic;
        
        // Check bond array bounds
        if (bond_count >= MAX_BONDS) {
            set_parser_error(state, "Maximum bond count (%d) exceeded during ring closure", MAX_BONDS);
            return false;
        }
        
        // Create the bond
        BondSeg *new_bond = &bonds[bond_count];
        new_bond->a = partner_idx;
        new_bond->b = state->current_atom_idx;
        new_bond->order = closure_bond_order > state->pending_bond_order ? 
                           closure_bond_order : state->pending_bond_order;
        new_bond->in_ring = true;
        
        atoms[partner_idx].in_ring = true;
        atoms[state->current_atom_idx].in_ring = true;
        
        if (closure_is_aromatic || state->pending_bond_aromatic || 
            atoms[partner_idx].is_aromatic || atoms[state->current_atom_idx].is_aromatic) {
            new_bond->type = BOND_AROMATIC;
            atoms[partner_idx].is_aromatic = true;
            atoms[state->current_atom_idx].is_aromatic = true;
        } else {
            new_bond->type = (BondType)new_bond->order;
        }
        
        atoms[partner_idx].n_bonds++;
        atoms[state->current_atom_idx].n_bonds++;
        bond_count++;
        
        // Reset ring closure
        state->ring_closures[ring_num].atom_idx = -1;
    }
    
    state->pending_bond_order = 1;
    state->pending_bond_aromatic = false;
    return true;
}

// Parse branch start/end
static bool parse_branch(ParserState *state, bool is_branch_start) {
    if (is_branch_start) {
        if (state->branch_stack_ptr >= MAX_SMILES_TOKENS) {
            set_parser_error(state, "Branch nesting too deep at position %d", state->position);
            return false;
        }
        
        state->branch_stack[state->branch_stack_ptr] = state->current_atom_idx;
        state->branch_angle_stack[state->branch_stack_ptr] = state->angle;
        state->branch_stack_ptr++;
        state->angle += M_PI / 3.0;
    } else {
        if (state->branch_stack_ptr <= 0) {
            set_parser_error(state, "Unmatched branch closing at position %d", state->position);
            return false;
        }
        
        state->branch_stack_ptr--;
        state->current_atom_idx = state->branch_stack[state->branch_stack_ptr];
        state->angle = state->branch_angle_stack[state->branch_stack_ptr];
        state->angle -= M_PI / 3.0;
        state->pending_bond_order = 1;
        state->pending_bond_aromatic = false;
    }
    
    state->position++;
    return true;
}

// Rewritten parse_smiles function with better structure and error handling
int parse_smiles(const char *s, bool *had_complex_feature_warning_flag) {
    if (!s || !had_complex_feature_warning_flag) {
        fprintf(stderr, "Error: Invalid parameters for SMILES parsing\n");
        return 0;
    }

    // Initialize molecule data and handle any errors
    if (!initialize_molecule_data()) {
        fprintf(stderr, "Error: Failed to initialize molecule data structures\n");
        return 0;
    }
    
    *had_complex_feature_warning_flag = false;
    
    // Length check to prevent parsing extremely long SMILES strings
    size_t smiles_len = strlen(s);
    if (smiles_len > 4000) { // Reasonable limit for most molecules
        fprintf(stderr, "Error: SMILES string too long (length: %zu)\n", smiles_len);
        return 0;
    }
    
    // Pre-check for unsupported ring sizes
    if (has_unsupported_ring_size(s, parser_config.max_ring_size)) {
        return 0;
    }
    
    // Initialize parser state
    ParserState state;
    init_parser_state(&state, s, had_complex_feature_warning_flag);
    
    // Check if initialization was successful
    if (state.error_message[0] != '\0') {
        fprintf(stderr, "Error initializing SMILES parser: %s\n", state.error_message);
        cleanup_parser_state(&state);
        return 0;
    }
    
    if (!state.ring_closures) {
        fprintf(stderr, "Error: Failed to allocate memory for ring closures\n");
        return 0;
    }
    
    // Main parsing loop
    while (s[state.position]) {
        char token = s[state.position];
        
        if (isspace(token)) {
            state.position++;
            continue;
        }
        
        if (isalpha(token)) {
            if (!parse_atom(&state)) {
                cleanup_parser_state(&state);
                fprintf(stderr, "Error parsing atom at position %d: %s\n", 
                        state.position, state.error_message);
                return 0;
            }
        } else if (token == '[') {
            if (!parse_bracketed_atom(&state)) {
                cleanup_parser_state(&state);
                fprintf(stderr, "Error parsing bracketed atom at position %d: %s\n", 
                        state.position, state.error_message);
                return 0;
            }
        } else if (token == '=') {
            state.pending_bond_order = 2;
            state.position++;
        } else if (token == '#') {
            state.pending_bond_order = 3;
            state.position++;
        } else if (token == ':') {
            state.pending_bond_aromatic = true;
            state.position++;
        } else if (token == '(') {
            if (!parse_branch(&state, true)) {
                cleanup_parser_state(&state);
                fprintf(stderr, "Error parsing branch start at position %d: %s\n", 
                        state.position, state.error_message);
                return 0;
            }
        } else if (token == ')') {
            if (!parse_branch(&state, false)) {
                cleanup_parser_state(&state);
                fprintf(stderr, "Error parsing branch end at position %d: %s\n", 
                        state.position, state.error_message);
                return 0;
            }
        } else if (isdigit(token) || token == '%') {
            if (!parse_ring_closure(&state)) {
                cleanup_parser_state(&state);
                fprintf(stderr, "Error parsing ring closure at position %d: %s\n", 
                        state.position, state.error_message);
                return 0;
            }
        } else if (token == '.') {
            state.current_atom_idx = -1;
            state.pending_bond_order = 1;
            state.pending_bond_aromatic = false;
            state.angle = 0.0;
            state.position++;
        } else if (token == '/' || token == '\\') {
            *had_complex_feature_warning_flag = true;
            state.position++;
        } else if (token == '+' || token == '-') {
            if (state.current_atom_idx != -1 && atom_count > 0 && state.current_atom_idx < atom_count) {
                int charge_mod = (token == '+') ? 1 : -1;
                state.position++;
                if (s[state.position] && isdigit(s[state.position])) {
                    charge_mod *= (s[state.position] - '0');
                    state.position++;
                }
                atoms[state.current_atom_idx].charge += charge_mod;
            } else {
                *had_complex_feature_warning_flag = true;
                state.position++;
            }
        } else {
            // Skip unrecognized characters
            state.position++;
        }
        
        // Safety check - prevent infinite loops
        if (state.position > (int)smiles_len) {
            fprintf(stderr, "Error: Parser position exceeds SMILES length\n");
            cleanup_parser_state(&state);
            return 0;
        }
    }
    
    // Check for unmatched branches
    if (state.branch_stack_ptr != 0) {
        cleanup_parser_state(&state);
        fprintf(stderr, "Error: Unmatched branch openings in SMILES string\n");
        return 0;
    }
    
    // Check for unclosed rings
    bool unclosed_rings = false;
    for (int i = 0; i < state.max_ring_closures; i++) {
        if (state.ring_closures[i].atom_idx != -1) {
            unclosed_rings = true;
            break;
        }
    }
    
    if (unclosed_rings) {
        *had_complex_feature_warning_flag = true;
        if (parser_config.verbose_warnings) {
            fprintf(stderr, "Warning: SMILES string contains unclosed ring(s)\n");
        }
    }
    
    // Final atom and bond count checks
    if (atom_count <= 0) {
        cleanup_parser_state(&state);
        fprintf(stderr, "Error: No atoms found in SMILES string\n");
        return 0;
    }
    
    if (atom_count > MAX_ATOMS) {
        cleanup_parser_state(&state);
        fprintf(stderr, "Error: Too many atoms (%d) in molecule, maximum is %d\n", 
                atom_count, MAX_ATOMS);
        return 0;
    }
    
    if (bond_count > MAX_BONDS) {
        cleanup_parser_state(&state);
        fprintf(stderr, "Error: Too many bonds (%d) in molecule, maximum is %d\n", 
                bond_count, MAX_BONDS);
        return 0;
    }
    
    // Propagate aromaticity
    for (int k = 0; k < 5; k++) {
        for (int j = 0; j < bond_count; j++) {
            if (bonds[j].type == BOND_AROMATIC) {
                if (bonds[j].a < atom_count && bonds[j].b < atom_count) {
                    atoms[bonds[j].a].is_aromatic = true;
                    atoms[bonds[j].b].is_aromatic = true;
                }
            }
        }
    }
    
    // Calculate hybridization
    for (int i = 0; i < atom_count; i++) {
        atoms[i].hybridization = calculate_hybridization(atoms[i]);
    }
    
    cleanup_parser_state(&state);
    return atom_count;
}

// New helper functions for enhanced SMILES support

// Predict bond angles based on hybridization
double predict_bond_angle(AtomPos atom) {
    if (atom.hybridization >= 2.5) return 120.0 * M_PI / 180.0;  // sp2, ~120°
    if (atom.hybridization >= 1.5) return 109.5 * M_PI / 180.0;  // sp3, ~109.5°
    if (atom.n_bonds <= 1) return 180.0 * M_PI / 180.0;          // terminal, 180°
    return 120.0 * M_PI / 180.0;                                // default to 120°
}

// New function to improve 2D coordinates after parsing
void optimize_2d_coordinates() {
    // Simple force-directed layout for demonstration
    // In a real implementation, use a more sophisticated algorithm
    for (int iter = 0; iter < 50; iter++) {
        for (int i = 0; i < atom_count; i++) {
            double fx = 0.0, fy = 0.0;
            
            // Repulsive forces from all other atoms
            for (int j = 0; j < atom_count; j++) {
                if (i == j) continue;
                
                double dx = atoms[i].x - atoms[j].x;
                double dy = atoms[i].y - atoms[j].y;
                double dist = sqrt(dx*dx + dy*dy);
                
                // Avoid division by zero
                if (dist < 0.1) dist = 0.1;
                
                // Repulsive force inversely proportional to distance
                double force = 0.5 / (dist * dist);
                fx += force * dx / dist;
                fy += force * dy / dist;
            }
            
            // Spring forces from bonded atoms
            for (int j = 0; j < bond_count; j++) {
                if (bonds[j].a == i || bonds[j].b == i) {
                    int other = (bonds[j].a == i) ? bonds[j].b : bonds[j].a;
                    double dx = atoms[i].x - atoms[other].x;
                    double dy = atoms[i].y - atoms[other].y;
                    double dist = sqrt(dx*dx + dy*dy);
                    
                    // Spring force proportional to distance difference from ideal
                    double ideal_dist = 1.0;
                    double force = 0.1 * (dist - ideal_dist);
                    fx -= force * dx / dist;
                    fy -= force * dy / dist;
                }
            }
            
            // Apply forces with damping
            atoms[i].x += 0.1 * fx;
            atoms[i].y += 0.1 * fy;
        }
    }
}

// API function to parse SMILES with progress reporting
int parse_smiles_with_progress(const char *smiles, bool *had_warnings, void (*progress_callback)(float)) {
    // size_t len = strlen(smiles);  // Unused, commented out
    int result = parse_smiles(smiles, had_warnings);
    
    if (result > 0 && atom_count > 10) {
        // For larger molecules, optimize coordinates
        if (progress_callback) progress_callback(0.7);
        optimize_2d_coordinates();
        if (progress_callback) progress_callback(1.0);
    }
    
    return result;
}