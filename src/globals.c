#include "globals.h"
#include <string.h>
#include <stddef.h> // For NULL

// Global molecule data
AtomPos *atoms = NULL;
BondSeg *bonds = NULL;
int atom_count = 0;
int bond_count = 0;

// CUDA control flags
int use_cuda = 1;         // 0=off, 1=auto detect, 2=force CUDA
int cuda_batch_size = 512; // Default batch size
bool cuda_verbose = false; // Control verbosity of CUDA output

RingRef rings[10];
ColormapType colormap_global = COLORMAP_HEAT;