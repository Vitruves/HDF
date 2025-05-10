#ifndef CLI_H
#define CLI_H

#include <stdbool.h>

typedef struct {
    char *input_csv_path;
    char *output_csv_path;
    char *output_dir_images;
    char *smiles_column_name;  // New field for custom SMILES column name
    bool no_images;
    bool verbose;             // New field for verbose logging
    int resolution;
    char *colormap_name;
    char *output_format_images; // "ppm" or "pgm"
    int layout_iterations;
    bool use_quantum_model;
    bool use_mo_effects;
    // Parameters for layout optimization
    double k_spring;
    double k_repulsive;
    double damping_factor;
    double time_step_factor;
    int num_jobs;  // New parameter for parallel processing
    bool use_streaming;     // Whether to use streaming mode
    int csv_buffer_size;    // CSV buffer size in KB for streaming mode
    bool column_format;       // Output fingerprint as separate columns
    int condense_block_size;  // Block size for fingerprint condensation (N for NxN)

    // TODO: Add more parameters for diffraction profile customization
    // e.g., scaling factors, specific QM contributions toggle, etc.
} CliArgs;

int parse_cli_arguments(int argc, char **argv, CliArgs *args);
void print_cli_help(const char *prog_name);

#endif // CLI_H