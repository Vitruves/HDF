#include "utils.h"
#include <string.h> // For strlen in hash_string if used, or *str++

double clamp_double(double value, double min, double max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

double min_double(double a, double b) {
    return (a < b) ? a : b;
}

int factorial(int n) {
    if (n < 0) return 0; // Or handle error
    if (n == 0 || n == 1) return 1;
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

unsigned int hash_string(const char* str, int salt) {
    unsigned int hash = salt * 5381; // Initialize with salt
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }
    return hash;
}