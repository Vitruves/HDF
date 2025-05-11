#ifndef GLOBALS_H
#define GLOBALS_H

#include <stdbool.h>
#include <signal.h>  // For sig_atomic_t
#include "molecule.h"

// #define MAX_ATOMS 512 // Ensure this is consistent if used elsewhere before molecule.h include
#define MAX_SMILES_LEN 1024

// Physical Constants
// PI is defined in utils.h
#define BOHR_RADIUS 0.529177210903e-10 // meters
#define ANGSTROM 1.0e-10 // meters

// Global molecule data
extern AtomPos *atoms;
extern BondSeg *bonds;
extern int atom_count;
extern int bond_count;

// Global settings
extern int use_cuda;
extern int cuda_batch_size;
extern ColormapType colormap_global;
extern RingRef rings[10];

// Global flag for graceful shutdown (used by signal handlers to stop processing)
extern volatile sig_atomic_t keep_running;

// Other shared global variables as needed

#endif /* GLOBALS_H */