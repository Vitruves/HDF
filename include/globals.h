#ifndef GLOBALS_H
#define GLOBALS_H

#include "molecule.h"

#define MAX_ATOMS 512 // Ensure this is consistent if used elsewhere before molecule.h include
#define MAX_SMILES_LEN 1024

// Physical Constants
// PI is defined in utils.h
#define BOHR_RADIUS 0.529177210903e-10 // meters
#define ANGSTROM 1.0e-10 // meters

extern AtomPos atoms[MAX_ATOMS];
extern BondSeg bonds[MAX_ATOMS * 2];
extern RingRef rings[10]; // Max 10 concurrent ring closures
extern int atom_count;
extern int bond_count;
extern ColormapType colormap_global; // Renamed to avoid conflict with local variables

#endif // GLOBALS_H