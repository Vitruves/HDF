#ifndef UTILS_H
#define UTILS_H

#include <math.h> // For M_PI if not defined elsewhere
#include <complex.h> // For complex double

#ifndef PI
#define PI M_PI
#endif

#define PLANCK_CONSTANT 6.62607015e-34  /* J⋅s */
#define ELECTRON_MASS 9.1093837015e-31  /* kg */
#define ATOMIC_MASS_UNIT 1.66053906660e-27  /* kg */
// Using ANGSTROM and BOHR_RADIUS from globals.h

double clamp_double(double value, double min, double max);
double min_double(double a, double b);
int factorial(int n);
unsigned int hash_string(const char* str, int salt);

#endif // UTILS_H