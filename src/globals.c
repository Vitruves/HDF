#include "globals.h"

AtomPos atoms[MAX_ATOMS];
BondSeg bonds[MAX_ATOMS * 2];
RingRef rings[10];
int atom_count = 0;
int bond_count = 0;
ColormapType colormap_global = COLORMAP_HEAT;