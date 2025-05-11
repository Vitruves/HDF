#ifndef CLI_H
#define CLI_H

#include <stdbool.h>
#include "molecule.h"

// Command line arguments structure
typedef struct {
    // Basic file paths
    char *input_file;
    char *output_file;
    char *pdb_file;
    char *input_csv_path;
    char *output_csv_path;
    char *smiles_column_name;
    
    // Image output options
    int width;
    int height;
    int grid_size;
    int resolution;
    double scale_factor;
    double contrast;
    bool apply_log_scale;
    bool no_images;
    char *output_dir_images;
    char *output_format_images;
    char *colormap_name;
    
    // Processing options
    bool quiet;
    bool verbose;
    bool use_quantum_model;
    bool use_mo_effects;
    
    // Layout optimization parameters
    int iterations;
    int layout_iterations;
    double damping;
    double damping_factor;
    double k_spring;
    double k_repulsive;
    double time_step;
    double time_step_factor;
    int optimize_3d;
    
    // CUDA options
    int device;
    int cuda_batch_size;
    
    // Visualization options
    ColormapType colormap;
    
    // CSV and fingerprint options
    int num_jobs;
    bool use_streaming;
    int csv_buffer_size;
    bool column_format;
    int condense_block_size;
} CliArgs;

// Global variables for CUDA control
extern int use_cuda;            // 0=off, 1=auto, 2=force
extern int cuda_batch_size;     // Batch size for CUDA operations

// Function declarations
CliArgs parse_cli_args(int argc, char **argv);
void print_help(void);
void apply_cli_args(CliArgs args);

#endif /* CLI_H */