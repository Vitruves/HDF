#ifndef QUANTUM_ENGINE_H
#define QUANTUM_ENGINE_H

#include "molecule.h"
#include <complex.h>

double hydrogen_1s(double r, double Z);
double hydrogen_2s(double r, double Z);
double hydrogen_2p(double r, double Z, double theta);
complex double slater_orbital(double r, double n, double zeta);
double electron_density(AtomPos atom, double x, double y, double z_coord);
double calculate_hybridization(AtomPos atom);
void apply_quantum_corrections_to_atoms();
double calculate_atom_phase_qm(AtomPos atom, int idx);
double calculate_bond_phase_qm(BondSeg bond);

// TODO: Add more sophisticated QM functions as needed
// - More accurate effective nuclear charge calculation
// - Basis set considerations
// - Molecular orbital overlap integrals

#endif // QUANTUM_ENGINE_H