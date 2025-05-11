#define _GNU_SOURCE 
#include "cli.h"
#include "globals.h"
#include "molecule.h"
#include "smiles_parser.h"
#include "quantum_engine.h"
#include "diffraction_engine.h"
#include "render_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <complex.h>
#include <sys/stat.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>
#include <stdatomic.h>
#include <signal.h>  // For signal handling
#include <pthread.h> // For CPU affinity
#include <time.h>
#include <stdbool.h>
#ifdef __linux__
    #include <sched.h>
    #include <sys/types.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(x)
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

#define MAX_CSV_LINE_LENGTH 16384
#define MAX_CSV_COLUMNS 128
#define MAX_FINGERPRINT_STR_LEN (1024 * 1024)
#define BATCH_SIZE 50
#define CSV_BUFFER_SIZE (1024 * 1024)

// Structure to hold data for a single parsed molecule
typedef struct {
    AtomPos *atoms_data;         // Dynamically allocated array of atoms
    BondSeg *bonds_data;         // Dynamically allocated array of bonds
    int num_atoms;
    int num_bonds;
    char original_smiles[MAX_CSV_LINE_LENGTH];
    char original_line_for_output[MAX_CSV_LINE_LENGTH];
    int original_row_idx;        // 1-indexed for user display
    bool complex_feature_warning;
    // Any other per-molecule data needed by worker threads after layout can be added here
} ParsedMoleculeData;

pthread_mutex_t global_processing_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t output_mutex = PTHREAD_MUTEX_INITIALIZER;
atomic_int total_processed_count = 0;

// Global flag for graceful shutdown
volatile sig_atomic_t keep_running = 1;

// Global start time
static time_t start_time;

// Global processing variables
ParsedMoleculeData *parsed_molecules = NULL; // Array to hold parsed molecule data

// Signal handler
static void signal_handler(int signum) {
    if (signum == SIGINT || signum == SIGTERM) {
        fprintf(stderr, "\nReceived interrupt signal. Finishing current tasks and cleaning up...\n");
        keep_running = 0;
    }
}

// Get number of available CPU cores
static int get_cpu_count() {
    #ifdef _SC_NPROCESSORS_ONLN
        long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        return (ncpu > 0) ? (int)ncpu : 1;
    #else
        return 1;
    #endif
}

// Thread worker for parallel CSV processing
typedef struct {
    int thread_id;
    CliArgs *args;
    int status;  // 0 = idle, 1 = processing, 2 = done-success, -1 = error, -2 = invalid molecule
    bool has_new_task;
    bool please_exit;
    
    // Input
    char smiles_to_process[MAX_CSV_LINE_LENGTH];
    char original_line_for_output[MAX_CSV_LINE_LENGTH];
    int original_row_idx;
    
    // Whether to use pre-parsed molecule data
    bool use_parsed_molecule;
    int parsed_molecule_idx;
    
    // Output
    char *result_fingerprint;
    char *full_output_line;
    
    // Thread synchronization
    pthread_mutex_t mutex;
    pthread_cond_t condition_task;
    pthread_cond_t condition_done;
} WorkerThread;

typedef struct {
    FILE *infile;
    FILE *outfile;
    int csv_header_columns;
    int smiles_column_idx;
    char *csv_buffer;
    size_t buffer_size;
    size_t buffer_pos;
    size_t chars_in_buffer;
    bool eof_reached;
    long line_count;
    long processed_lines;
    char *header;
} CsvProcessor;

// Function prototypes
void* process_molecule_thread(void *arg);
int ensure_dir_exists(const char *path);
void print_cli_help(const char *prog_name);
int parse_cli_arguments(int argc, char **argv, CliArgs *args);
char** parse_csv_line(char* line, int* num_cols);
int init_csv_processor(CsvProcessor *processor, const char *input_path, const char *output_path, const char *smiles_column_name, CliArgs *args);
char* read_csv_line(CsvProcessor *processor);
int get_line_count(const char *file_path);
int process_csv_streaming(CliArgs *args);
int process_csv_parallel(CliArgs *args);

// Function declarations for external functions
extern void initialize_molecule_data(void);
extern int parse_smiles(const char *smiles, bool *complex_feature_warning);
extern void apply_quantum_corrections_to_atoms(void);
extern void optimize_molecule_layout(int iterations, double k_spring, double k_repulsive, 
                                   double damping, double time_step);
extern void optimize_molecule_layout_batch(
    AtomPos **atoms_batch_ptr_array,
    int *atom_counts_batch,
    BondSeg **bonds_batch_ptr_array,
    int *bond_counts_batch,
    int num_molecules_in_batch,
    int iterations, double k_spring, double k_repulsive,
    double damping_factor, double time_step_factor
);
extern void draw_atom_on_grid(complex double *aperture_grid, int grid_width, 
                            AtomPos atom, int atom_idx, bool use_quantum_model);
extern void draw_bond_on_grid(complex double *aperture_grid, int grid_width, 
                            BondSeg bond, bool use_quantum_model);
extern void add_molecular_orbital_effects(complex double *grid, int size);
extern void fft_2d(complex double *data, int width, int height, int direction);
extern double apply_log_scale_intensity(double intensity, double max_intensity, double epsilon);
extern void set_render_colormap(ColormapType type);
extern void output_diffraction_image(FILE *fp, double *intensity, int width, int height, 
                                   bool color, const char *format);

// Add these external variable declarations
extern AtomPos *atoms;
extern BondSeg *bonds;
extern int atom_count;
extern int bond_count;

// Function prototype for condensation, if not already in diffraction_engine.h include
extern void condense_fingerprint_average(const double *input_fp, int input_w, int input_h, int block_size, double *output_fp, int *output_w, int *output_h);

// Forward declarations for new helper functions
static double* generate_diffraction_for_molecule_cli(CliArgs *args, int *fp_width_out, int *fp_height_out, double *max_intensity_out);
static void format_fingerprint_for_csv_cli(CliArgs *args, const char* smiles, const char* original_line, 
                                      int fp_width, int fp_height, const double* fingerprint_data, 
                                      char* fingerprint_str_out, int max_fp_str_len, 
                                      char* full_output_line_out, int max_full_line_len, 
                                      double max_intensity);
static void save_diffraction_image_for_molecule_cli(CliArgs *args, const char* smiles, int original_row_idx,
                                               const double* fingerprint_data, int fp_width, int fp_height);

// Thread function for parallel molecule processing
void *process_molecule_thread(void *arg) {
    WorkerThread *worker = (WorkerThread *)arg;
    
    while (1) {
        // Wait for a new task
        pthread_mutex_lock(&worker->mutex);
        while (!worker->has_new_task && !worker->please_exit) {
            pthread_cond_wait(&worker->condition_task, &worker->mutex);
        }
        
        if (worker->please_exit) {
            pthread_mutex_unlock(&worker->mutex);
            break;
        }
        
        worker->status = 1; // processing
        worker->has_new_task = false;
        pthread_mutex_unlock(&worker->mutex);
        
        CliArgs *args = worker->args;
        int row_idx = worker->original_row_idx;
        extern AtomPos *atoms;     // Use global atoms
        extern BondSeg *bonds;     // Use global bonds
        extern int atom_count;
        extern int bond_count;
        extern ParsedMoleculeData *parsed_molecules; // Access parsed molecule data
        
        if (args->verbose) {
            pthread_mutex_lock(&output_mutex);
            printf("\nThread %d: Processing SMILES (row %d): %s\n", 
                   worker->thread_id, row_idx, worker->smiles_to_process);
            pthread_mutex_unlock(&output_mutex);
        }
        
        // Reset globals (important if pre-parsed molecule isn't used)
        initialize_molecule_data();
        
        // This flag tracks whether the molecule was processed successfully
        bool success = false;
        
        // Use the pre-parsed molecule data if available
        if (worker->use_parsed_molecule) {
            success = true;
            
            // Temporarily update globals to point to our pre-processed molecule data
            atom_count = parsed_molecules[worker->parsed_molecule_idx].num_atoms;
            bond_count = parsed_molecules[worker->parsed_molecule_idx].num_bonds;
            atoms = parsed_molecules[worker->parsed_molecule_idx].atoms_data;
            bonds = parsed_molecules[worker->parsed_molecule_idx].bonds_data;
        } else {
            // Parse SMILES (this will use and update globals)
            bool complex_feature_warning = false;
            int parsed_atom_count = parse_smiles(worker->smiles_to_process, &complex_feature_warning);
            
            if (parsed_atom_count > 0) {
                // Applies to the global atoms/bonds
                success = true;
                
                if (args->use_quantum_model || args->use_mo_effects) {
                    apply_quantum_corrections_to_atoms();
                }
                
                // Optimize layout
                optimize_molecule_layout(
                    args->layout_iterations,
                    args->k_spring,
                    args->k_repulsive,
                    args->damping_factor,
                    args->time_step_factor
                );
            }
        }
        
        if (success) {
            // Generate diffraction pattern using the existing process_molecule function
            // which uses the global arrays
            int W = args->resolution;
            int total_grid_points = W * W;
            complex double *aperture_grid = calloc(total_grid_points, sizeof(complex double));
            double *intensity = malloc(total_grid_points * sizeof(double));
            
            if (!aperture_grid || !intensity) {
                pthread_mutex_lock(&output_mutex);
                fprintf(stderr, "\nError: Worker %d failed to allocate memory for grid arrays\n", worker->thread_id);
                pthread_mutex_unlock(&output_mutex);
                
                if (aperture_grid) free(aperture_grid);
                if (intensity) free(intensity);
                
                worker->status = -1; // error
                continue;
            }
            
            // Draw atoms and bonds on grid
            for (int i = 0; i < atom_count; i++) {
                draw_atom_on_grid(aperture_grid, W, atoms[i], i, args->use_quantum_model);
            }
            
            for (int i = 0; i < bond_count; i++) {
                draw_bond_on_grid(aperture_grid, W, bonds[i], args->use_quantum_model);
            }
            
            if (args->use_mo_effects) {
                add_molecular_orbital_effects(aperture_grid, W);
            }
            
            // Calculate diffraction pattern
            fft_2d(aperture_grid, W, W, 1);
            
            // Calculate intensity
            double max_intensity = 0.0;
            for (int i = 0; i < total_grid_points; i++) {
                intensity[i] = cabs(aperture_grid[i]) * cabs(aperture_grid[i]);
                if (intensity[i] > max_intensity) max_intensity = intensity[i];
            }
            
            // Apply log scaling
            double epsilon = max_intensity * 1.0e-7;
            if (epsilon < 1.0e-10) epsilon = 1.0e-10;
            
            // Generate output line
            if (args->column_format) {
                worker->full_output_line[0] = '\0';
                strcat(worker->full_output_line, worker->original_line_for_output);
                
                // Use a more efficient approach for building the string - direct writes instead of strcat
                int offset = strlen(worker->full_output_line);
                int remaining = MAX_CSV_LINE_LENGTH + MAX_FINGERPRINT_STR_LEN - offset - 1;
                char *ptr = worker->full_output_line + offset;
                
                // For column format, add one value per column
                for (int i = 0; i < total_grid_points && remaining > 10; i++) {
                    double scaled_value = apply_log_scale_intensity(intensity[i], max_intensity, epsilon);
                    int written = snprintf(ptr, remaining, ",%.4f", scaled_value);
                    if (written < 0 || written >= remaining) {
                        // Won't fit, stop here
                        break;
                    }
                    ptr += written;
                    remaining -= written;
                }
            } else {
                // For row format, use fixed width for each value to calculate if it will fit
                int est_chars_per_value = 8; // Approx ",0.1234" or ",12.34" length
                int max_values = (MAX_CSV_LINE_LENGTH + MAX_FINGERPRINT_STR_LEN - 
                                 strlen(worker->original_line_for_output) - 32) / est_chars_per_value;
                
                if (max_values < total_grid_points) {
                    // Will truncate, so use a condensed format - only include every Nth point
                    int stride = (total_grid_points + max_values - 1) / max_values;
                    
                    // Copy header first
                    snprintf(worker->full_output_line, MAX_CSV_LINE_LENGTH + MAX_FINGERPRINT_STR_LEN,
                            "%s,", worker->original_line_for_output);
                    
                    // Append values with stride
                    char *ptr = worker->full_output_line + strlen(worker->full_output_line);
                    int remaining = MAX_CSV_LINE_LENGTH + MAX_FINGERPRINT_STR_LEN - strlen(worker->full_output_line) - 1;
                    
                    for (int i = 0; i < total_grid_points; i += stride) {
                        double scaled_value = apply_log_scale_intensity(intensity[i], max_intensity, epsilon);
                        int written;
                        if (i > 0) {
                            written = snprintf(ptr, remaining, " %.4f", scaled_value);
                        } else {
                            written = snprintf(ptr, remaining, "%.4f", scaled_value);
                        }
                        
                        if (written < 0 || written >= remaining) break;
                        
                        ptr += written;
                        remaining -= written;
                    }
                    
                    if (stride > 1) {
                        pthread_mutex_lock(&output_mutex);
                        fprintf(stderr, "\nNote: Fingerprint for row %d condensed by factor of %d due to size constraints\n", 
                                row_idx, stride);
                        pthread_mutex_unlock(&output_mutex);
                    }
                } else {
                    // Build the string directly with all points
                    worker->result_fingerprint[0] = '\0';
                    char *ptr = worker->result_fingerprint;
                    int remaining = MAX_FINGERPRINT_STR_LEN - 1;
                    
                    for (int i = 0; i < total_grid_points && remaining > 10; i++) {
                        double scaled_value = apply_log_scale_intensity(intensity[i], max_intensity, epsilon);
                        int written;
                        if (i > 0) {
                            written = snprintf(ptr, remaining, " %.4f", scaled_value);
                        } else {
                            written = snprintf(ptr, remaining, "%.4f", scaled_value);
                        }
                        
                        if (written < 0 || written >= remaining) break;
                        
                        ptr += written;
                        remaining -= written;
                    }
                    
                    snprintf(worker->full_output_line, MAX_CSV_LINE_LENGTH + MAX_FINGERPRINT_STR_LEN,
                            "%s,%s", worker->original_line_for_output, worker->result_fingerprint);
                }
            }
            
            // Save diffraction image if requested
            if (!args->no_images && args->output_dir_images) {
                save_diffraction_image_for_molecule_cli(args, worker->smiles_to_process, row_idx, intensity, W, W);
            }
            
            free(aperture_grid);
            free(intensity);
            worker->status = 2; // success
        } else {
            // SMILES parsing failed
            pthread_mutex_lock(&output_mutex);
            fprintf(stderr, "\nError: Failed to process SMILES '%s' on row %d\n", 
                    worker->smiles_to_process, row_idx);
            pthread_mutex_unlock(&output_mutex);
            
            snprintf(worker->full_output_line, MAX_CSV_LINE_LENGTH + MAX_FINGERPRINT_STR_LEN,
                     "%s,PARSE_ERROR", worker->original_line_for_output);
            worker->status = -2; // invalid molecule
        }
        
        // Signal that the task is done
        pthread_mutex_lock(&worker->mutex);
        pthread_cond_signal(&worker->condition_done);
        pthread_mutex_unlock(&worker->mutex);
    }
    
    return NULL;
}

int init_csv_processor(CsvProcessor *processor, const char *input_path, const char *output_path, const char *smiles_column_name, CliArgs *args) {
    processor->infile = fopen(input_path, "r");
    if (!processor->infile) {
        perror("Error opening input CSV");
        return 0;
    }

    processor->outfile = fopen(output_path, "w");
    if (!processor->outfile) {
        perror("Error opening output CSV");
        fclose(processor->infile);
        return 0;
    }

    processor->csv_buffer = malloc(CSV_BUFFER_SIZE);
    if (!processor->csv_buffer) {
        fprintf(stderr, "Error: Failed to allocate CSV buffer\n");
        fclose(processor->infile);
        fclose(processor->outfile);
        return 0;
    }

    processor->buffer_size = CSV_BUFFER_SIZE;
    processor->buffer_pos = 0;
    processor->chars_in_buffer = 0;
    processor->eof_reached = false;
    processor->line_count = 0;
    processor->processed_lines = 0;

    // Read first chunk
    processor->chars_in_buffer = fread(processor->csv_buffer, 1, processor->buffer_size, processor->infile);
    if (processor->chars_in_buffer < processor->buffer_size) {
        if (feof(processor->infile)) {
            processor->eof_reached = true;
        } else if (ferror(processor->infile)) {
            perror("Error reading from input CSV");
            fclose(processor->infile);
            fclose(processor->outfile);
            free(processor->csv_buffer);
            return 0;
        }
    }

    // Read and process header
    char *header_line = read_csv_line(processor);
    if (!header_line) {
        fprintf(stderr, "Error: Failed to read CSV header\n");
        fclose(processor->infile);
        fclose(processor->outfile);
        free(processor->csv_buffer);
        return 0;
    }

    processor->header = strdup(header_line);
    
    // Find SMILES column index
    char header_copy[MAX_CSV_LINE_LENGTH];
    strncpy(header_copy, header_line, sizeof(header_copy) - 1);
    header_copy[sizeof(header_copy) - 1] = '\0';
    
    int num_header_cols;
    char **header_cols = parse_csv_line(header_copy, &num_header_cols);
    
    processor->csv_header_columns = num_header_cols;
    processor->smiles_column_idx = -1;
    
    if (header_cols) {
        for (int i = 0; i < num_header_cols; i++) {
            if (strcmp(header_cols[i], smiles_column_name) == 0) {
                processor->smiles_column_idx = i;
                break;
            }
        }
        free(header_cols);
    }
    
    if (processor->smiles_column_idx == -1) {
        fprintf(stderr, "Error: Column '%s' not found in input CSV header.\n", smiles_column_name);
        fclose(processor->infile);
        fclose(processor->outfile);
        free(processor->csv_buffer);
        free(processor->header);
        return 0;
    }
    
    // Write header with new fingerprint column(s)
    int final_fp_width = args->resolution;
    int final_fp_height = args->resolution;
    if (args->condense_block_size > 1 && args->resolution % args->condense_block_size == 0) {
        final_fp_width /= args->condense_block_size;
        final_fp_height /= args->condense_block_size;
    } else if (args->condense_block_size > 1) {
        // This case should be caught by CLI parsing, but as a fallback:
        fprintf(stderr, "Warning: Resolution not perfectly divisible by condense_block_size. Header might be inaccurate for column format.\n");
    }
    int grid_size_for_header = final_fp_width * final_fp_height;

    if (args->column_format) {
        fprintf(processor->outfile, "%s", processor->header); // Print original header
        for (int i = 0; i < grid_size_for_header; i++) {
            fprintf(processor->outfile, ",FP_%d", i);
        }
        fprintf(processor->outfile, "\n");
    } else {
        fprintf(processor->outfile, "%s,Fingerprint\n", processor->header);
    }
    
    // Count total lines for progress
    processor->line_count = get_line_count(input_path);
    
    return 1;
}

void free_csv_processor(CsvProcessor *processor) {
    if (processor->infile) fclose(processor->infile);
    if (processor->outfile) fclose(processor->outfile);
    if (processor->csv_buffer) free(processor->csv_buffer);
    if (processor->header) free(processor->header);
}

char* read_csv_line(CsvProcessor *processor) {
    static char line_buffer[MAX_CSV_LINE_LENGTH];
    int pos = 0;
    bool line_end = false;
    
    while (!line_end && pos < MAX_CSV_LINE_LENGTH - 1) {
        if (processor->buffer_pos >= processor->chars_in_buffer) {
            // Need to read more data
            if (processor->eof_reached) return pos > 0 ? line_buffer : NULL;
            
            processor->chars_in_buffer = fread(processor->csv_buffer, 1, processor->buffer_size, processor->infile);
            processor->buffer_pos = 0;
            
            if (processor->chars_in_buffer == 0) {
                processor->eof_reached = true;
                return pos > 0 ? line_buffer : NULL;
            }
        }
        
        char c = processor->csv_buffer[processor->buffer_pos++];
        
        if (c == '\n') {
            line_end = true;
        } else if (c != '\r') { // Skip CR characters
            line_buffer[pos++] = c;
        }
    }
    
    line_buffer[pos] = '\0';
    return line_buffer;
}

int get_line_count(const char *file_path) {
    FILE *f = fopen(file_path, "r");
    if (!f) return 0;
    
    char buffer[8192];
    int count = 0;
    
    // Skip header
    if (fgets(buffer, sizeof(buffer), f)) {
        count = 0;
    }
    
    while (fgets(buffer, sizeof(buffer), f)) {
        count++;
    }
    
    fclose(f);
    return count;
}

int process_csv_streaming(CliArgs *args) {
    CsvProcessor processor;
    if (!init_csv_processor(&processor, args->input_csv_path, args->output_csv_path, args->smiles_column_name, args)) {
        return 1;
    }
    
    int W = args->resolution;
    int total_grid_points = W * W;
    complex double *aperture_grid = calloc(total_grid_points, sizeof(complex double));
    double *intensity = malloc(total_grid_points * sizeof(double));
    char *fingerprint_str = malloc(MAX_FINGERPRINT_STR_LEN);
    
    if (!aperture_grid || !intensity || !fingerprint_str) {
        fprintf(stderr, "Memory allocation failed for processing buffers.\n");
        free_csv_processor(&processor);
        if (aperture_grid) free(aperture_grid);
        if (intensity) free(intensity);
        if (fingerprint_str) free(fingerprint_str);
        return 1;
    }

    printf("Processing %ld molecules...\n", processor.line_count);
    
    char *line;
    while ((line = read_csv_line(&processor)) != NULL) {
        processor.processed_lines++;
        
        // Progress display
        if (!args->verbose && processor.line_count > 0) {
            float progress = (float)processor.processed_lines / (float)processor.line_count * 100.0f;
            printf("\rProgress: %.1f%% (%ld/%ld) ", progress, processor.processed_lines, processor.line_count);
            fflush(stdout);
        }
        
        char original_line_copy[MAX_CSV_LINE_LENGTH];
        strncpy(original_line_copy, line, sizeof(original_line_copy) - 1);
        original_line_copy[sizeof(original_line_copy) - 1] = '\0';
        
        char line_copy_for_parse[MAX_CSV_LINE_LENGTH];
        strcpy(line_copy_for_parse, line);
        
        int num_cols;
        char **cols = parse_csv_line(line_copy_for_parse, &num_cols);
        
        if (!cols || num_cols <= processor.smiles_column_idx) {
            fprintf(stderr, "\nWarning: Skipping malformed CSV line %ld or not enough columns.\n", processor.processed_lines);
            fprintf(processor.outfile, "%s,CSV_PARSE_ERROR\n", original_line_copy);
            if (cols) free(cols);
            continue;
        }
        
        const char *smiles = cols[processor.smiles_column_idx];
        if (args->verbose) {
            printf("\nProcessing SMILES (row %ld): %s\n", processor.processed_lines, smiles);
        }
        
        initialize_molecule_data();
        bool complex_feature_warning = false;
        int parsed_atom_count = parse_smiles(smiles, &complex_feature_warning);
        
        if (complex_feature_warning) {
            fprintf(stderr, "\nNote: SMILES '%s' (row %ld) contained complex features ([...]) or unclosed rings that were skipped/partially handled.\n", 
                    smiles, processor.processed_lines);
        }
        
        if (parsed_atom_count == 0) {
            fprintf(stderr, "\nWarning: Failed to parse SMILES '%s' on line %ld. Skipping.\n", smiles, processor.processed_lines);
            fprintf(processor.outfile, "%s,PARSE_ERROR\n", original_line_copy);
            free(cols);
            continue;
        }
        
        if (args->use_quantum_model || args->use_mo_effects) {
            apply_quantum_corrections_to_atoms();
        }
        
        optimize_molecule_layout(args->layout_iterations, args->k_spring, args->k_repulsive, args->damping_factor, args->time_step_factor);
        
        memset(aperture_grid, 0, total_grid_points * sizeof(complex double));
        
        for (int i = 0; i < atom_count; ++i) {
            draw_atom_on_grid(aperture_grid, W, atoms[i], i, args->use_quantum_model);
        }
        
        for (int i = 0; i < bond_count; ++i) {
            draw_bond_on_grid(aperture_grid, W, bonds[i], args->use_quantum_model);
        }
        
        if (args->use_mo_effects) {
            add_molecular_orbital_effects(aperture_grid, W);
        }
        
        fft_2d(aperture_grid, W, W, 1);
        
        double max_raw_intensity = 0.0;
        for (int i = 0; i < total_grid_points; ++i) {
            intensity[i] = cabs(aperture_grid[i]) * cabs(aperture_grid[i]);
            if (intensity[i] > max_raw_intensity) max_raw_intensity = intensity[i];
        }
        
        double epsilon_log_fp = max_raw_intensity * 1e-7;
        if (epsilon_log_fp < 1e-10) epsilon_log_fp = 1e-10;
        
        fingerprint_str[0] = '\0';
        char temp_val_str[32];
        
        for (int i = 0; i < total_grid_points; ++i) {
            double scaled_intensity = apply_log_scale_intensity(intensity[i], max_raw_intensity, epsilon_log_fp);
            sprintf(temp_val_str, "%.4f", scaled_intensity);
            strcat(fingerprint_str, temp_val_str);
            
            if (i < total_grid_points - 1) {
                strcat(fingerprint_str, " ");
            }
            
            if (strlen(fingerprint_str) > MAX_FINGERPRINT_STR_LEN - 32) {
                fprintf(stderr, "\nWarning: Fingerprint string truncated for SMILES %s due to length.\n", smiles);
                break;
            }
        }
        
        // Write output
        if (args->column_format) {
            fprintf(processor.outfile, "%s", original_line_copy);
            
            char temp_val_str_out[32];
            for (int i = 0; i < total_grid_points; ++i) {
                double val_to_write = apply_log_scale_intensity(intensity[i], max_raw_intensity, epsilon_log_fp);
                sprintf(temp_val_str_out, ",%.4f", val_to_write);
                fprintf(processor.outfile, "%s", temp_val_str_out);
            }
            fprintf(processor.outfile, "\n");
        } else {
            fprintf(processor.outfile, "%s,%s\n", original_line_copy, fingerprint_str);
        }
        
        // Save image if needed
        if (!args->no_images && args->output_dir_images) {
            save_diffraction_image_for_molecule_cli(args, smiles, processor.processed_lines, intensity, W, W);
        }
        
        free(cols);
    }
    
    printf("\nProcessing complete. Output written to %s\n", args->output_csv_path);
    
    free_csv_processor(&processor);
    free(aperture_grid);
    free(intensity);
    free(fingerprint_str);
    
    return 0;
}

int process_csv_parallel(CliArgs *args) {
    // Set up signal handling
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    CsvProcessor processor;
    if (!init_csv_processor(&processor, args->input_csv_path, args->output_csv_path, args->smiles_column_name, args)) {
        return 1;
    }
    
    char **input_lines_buffer = malloc(processor.line_count * sizeof(char*));
    char **smiles_strings_buffer = malloc(processor.line_count * sizeof(char*));
    
    if (!input_lines_buffer || !smiles_strings_buffer) {
        fprintf(stderr, "Error: Failed to allocate memory for input buffers\n");
        free_csv_processor(&processor);
        if (input_lines_buffer) free(input_lines_buffer);
        if (smiles_strings_buffer) free(smiles_strings_buffer);
        return 1;
    }
    
    int data_line_idx = 0;
    char *line;
    while ((line = read_csv_line(&processor)) != NULL && data_line_idx < processor.line_count) {
        char original_line_copy[MAX_CSV_LINE_LENGTH];
        strncpy(original_line_copy, line, sizeof(original_line_copy) - 1);
        original_line_copy[sizeof(original_line_copy) - 1] = '\0';
        input_lines_buffer[data_line_idx] = strdup(original_line_copy);
        
        char line_copy_for_parse[MAX_CSV_LINE_LENGTH];
        strcpy(line_copy_for_parse, line);
        int num_cols;
        char **cols = parse_csv_line(line_copy_for_parse, &num_cols);
        if (cols && num_cols > processor.smiles_column_idx) {
            smiles_strings_buffer[data_line_idx] = strdup(cols[processor.smiles_column_idx]);
        } else {
            smiles_strings_buffer[data_line_idx] = strdup("INVALID_SMILES_IN_CSV_ROW");
            fprintf(stderr, "\nWarning: Malformed CSV line %d or SMILES column not found. Will mark as PARSE_ERROR.\n", data_line_idx +1 );
        }
        if (cols) free(cols);
        data_line_idx++;
    }
    
    parsed_molecules = malloc(data_line_idx * sizeof(ParsedMoleculeData));
    if (!parsed_molecules) {
        fprintf(stderr, "Error: Failed to allocate memory for parsed molecule data\n");
        for (int i = 0; i < data_line_idx; i++) { free(input_lines_buffer[i]); free(smiles_strings_buffer[i]); }
        free(input_lines_buffer); free(smiles_strings_buffer);
        free_csv_processor(&processor);
        return 1;
    }
    
    printf("Parsing %d molecules...\n", data_line_idx);
    AtomPos **atoms_batch_ptrs = malloc(data_line_idx * sizeof(AtomPos*));
    BondSeg **bonds_batch_ptrs = malloc(data_line_idx * sizeof(BondSeg*));
    int *atom_counts_batch_arr = malloc(data_line_idx * sizeof(int)); // Renamed from atom_counts_batch
    int *bond_counts_batch_arr = malloc(data_line_idx * sizeof(int)); // Renamed from bond_counts_batch
    
    if (!atoms_batch_ptrs || !bonds_batch_ptrs || !atom_counts_batch_arr || !bond_counts_batch_arr) {
        fprintf(stderr, "Error: Failed to allocate memory for batch arrays\n");
        for (int i = 0; i < data_line_idx; i++) { free(input_lines_buffer[i]); free(smiles_strings_buffer[i]); }
        free(input_lines_buffer); free(smiles_strings_buffer); free(parsed_molecules);
        if (atoms_batch_ptrs) free(atoms_batch_ptrs); if (bonds_batch_ptrs) free(bonds_batch_ptrs);
        if (atom_counts_batch_arr) free(atom_counts_batch_arr); if (bond_counts_batch_arr) free(bond_counts_batch_arr);
        free_csv_processor(&processor);
        return 1;
    }
    
    int valid_molecule_count = 0;
    for (int i = 0; i < data_line_idx; i++) {
        parsed_molecules[i].atoms_data = NULL; parsed_molecules[i].bonds_data = NULL;
        parsed_molecules[i].num_atoms = 0; parsed_molecules[i].num_bonds = 0;
        parsed_molecules[i].complex_feature_warning = false;
        parsed_molecules[i].original_row_idx = i + 1;
        strcpy(parsed_molecules[i].original_smiles, smiles_strings_buffer[i]); // Store original SMILES
        strcpy(parsed_molecules[i].original_line_for_output, input_lines_buffer[i]); // Store original line

        if (!keep_running) { printf("\nParsing interrupted.\n"); break;}
        float progress = (float)(i + 1) / (float)data_line_idx * 100.0f;
        printf("\rParsing progress: %.1f%% (%d/%d)", progress, i + 1, data_line_idx); fflush(stdout);
        
        initialize_molecule_data(); // Resets global atoms, bonds, counts
        bool complex_feature_warning = false;
        int parsed_atom_count_val = parse_smiles(smiles_strings_buffer[i], &complex_feature_warning);
        
        if (parsed_atom_count_val > 0) {
            parsed_molecules[i].atoms_data = malloc(atom_count * sizeof(AtomPos));
            parsed_molecules[i].bonds_data = malloc(bond_count * sizeof(BondSeg));
            if (!parsed_molecules[i].atoms_data || !parsed_molecules[i].bonds_data) {
                fprintf(stderr, "\nError: Failed to allocate memory for molecule %d atoms/bonds\n", i);
                if (parsed_molecules[i].atoms_data) free(parsed_molecules[i].atoms_data);
                if (parsed_molecules[i].bonds_data) free(parsed_molecules[i].bonds_data);
                parsed_molecules[i].atoms_data = NULL; parsed_molecules[i].bonds_data = NULL;
                continue;
            }
            memcpy(parsed_molecules[i].atoms_data, atoms, atom_count * sizeof(AtomPos));
            memcpy(parsed_molecules[i].bonds_data, bonds, bond_count * sizeof(BondSeg));
            parsed_molecules[i].num_atoms = atom_count;
            parsed_molecules[i].num_bonds = bond_count;
            parsed_molecules[i].complex_feature_warning = complex_feature_warning;
            
            if (args->use_quantum_model || args->use_mo_effects) {
                // apply_quantum_corrections needs to operate on the specific molecule's data
                // Temporarily point globals, apply, then copy back (if it modifies in place)
                AtomPos* temp_global_atoms = atoms; BondSeg* temp_global_bonds = bonds;
                int temp_global_atom_count = atom_count; int temp_global_bond_count = bond_count;
                atoms = parsed_molecules[i].atoms_data; atom_count = parsed_molecules[i].num_atoms;
                bonds = parsed_molecules[i].bonds_data; bond_count = parsed_molecules[i].num_bonds;
                apply_quantum_corrections_to_atoms();
                // Data is modified in parsed_molecules[i].atoms_data directly
                atoms = temp_global_atoms; bonds = temp_global_bonds;
                atom_count = temp_global_atom_count; bond_count = temp_global_bond_count;
            }
            
            atoms_batch_ptrs[valid_molecule_count] = parsed_molecules[i].atoms_data;
            bonds_batch_ptrs[valid_molecule_count] = parsed_molecules[i].bonds_data;
            atom_counts_batch_arr[valid_molecule_count] = parsed_molecules[i].num_atoms;
            bond_counts_batch_arr[valid_molecule_count] = parsed_molecules[i].num_bonds;
            valid_molecule_count++;
        } else {
            fprintf(stderr, "\nWarning: Failed to parse SMILES '%s' on line %d. Skipping for layout.\n", smiles_strings_buffer[i], i + 1);
            // Mark as parse error for final output
            snprintf(parsed_molecules[i].original_line_for_output, MAX_CSV_LINE_LENGTH, "%s,PARSE_ERROR", input_lines_buffer[i]);
        }
    }
    printf("\nParsed %d valid molecules out of %d total for layout optimization.\n", valid_molecule_count, data_line_idx);

    atomic_store(&total_processed_count, 0); // Reset for output progress

    if (valid_molecule_count > 0) {
        printf("Optimizing molecular layouts with %d iterations...\n", args->layout_iterations);
        int layout_batch_size = cuda_batch_size > 0 ? cuda_batch_size : 100; // Use cuda_batch_size for layout batching too

        for (int batch_start_idx = 0; batch_start_idx < valid_molecule_count; batch_start_idx += layout_batch_size) {
            if (!keep_running) { printf("\nLayout optimization interrupted.\n"); break; }
            
            int current_layout_batch_size = valid_molecule_count - batch_start_idx;
            if (current_layout_batch_size > layout_batch_size) current_layout_batch_size = layout_batch_size;

            printf("\rOptimizing layout for batch: molecules %d-%d of %d valid molecules... ",
                   batch_start_idx + 1, batch_start_idx + current_layout_batch_size, valid_molecule_count);
            fflush(stdout);

            optimize_molecule_layout_batch(
                &atoms_batch_ptrs[batch_start_idx],
                &atom_counts_batch_arr[batch_start_idx],
                &bonds_batch_ptrs[batch_start_idx],
                &bond_counts_batch_arr[batch_start_idx],
                current_layout_batch_size,
                args->layout_iterations, args->k_spring, args->k_repulsive,
                args->damping_factor, args->time_step_factor
            );
            
            // After layout for this batch, generate diffraction patterns and output
            printf("\rGenerating diffraction & output for layout batch %d-%d... ",
                   batch_start_idx + 1, batch_start_idx + current_layout_batch_size);
            fflush(stdout);

            for (int k = 0; k < current_layout_batch_size; ++k) {
                if (!keep_running) { printf("\nOutput generation interrupted for layout batch.\n"); break; }

                // Find the original index in parsed_molecules array
                // This relies on atoms_batch_ptrs being populated sequentially from valid parsed_molecules
                int original_molecule_array_idx = -1;
                int count_valid = 0;
                for(int scan_idx = 0; scan_idx < data_line_idx; ++scan_idx) {
                    if(parsed_molecules[scan_idx].atoms_data && parsed_molecules[scan_idx].num_atoms > 0) { // valid and was processed for layout
                        if(count_valid == batch_start_idx + k) {
                            original_molecule_array_idx = scan_idx;
                            break;
                        }
                        count_valid++;
                    }
                }
                
                if (original_molecule_array_idx != -1) {
                    ParsedMoleculeData* current_mol_data = &parsed_molecules[original_molecule_array_idx];
                    
                    // Set globals for the diffraction engine functions
                    AtomPos* prev_atoms = atoms; BondSeg* prev_bonds = bonds;
                    int prev_atom_count = atom_count; int prev_bond_count = bond_count;
                    atoms = current_mol_data->atoms_data; atom_count = current_mol_data->num_atoms;
                    bonds = current_mol_data->bonds_data; bond_count = current_mol_data->num_bonds;

                    if (atom_count > 0) { // Check if molecule is valid after all
                        int fp_width = 0, fp_height = 0;
                        double max_intensity_val = 0.0;
                        double* fingerprint_val = generate_diffraction_for_molecule_cli(args, &fp_width, &fp_height, &max_intensity_val);

                        char fingerprint_str_buffer[MAX_FINGERPRINT_STR_LEN];
                        char full_output_line_buffer[MAX_CSV_LINE_LENGTH + MAX_FINGERPRINT_STR_LEN + 2];

                        if (fingerprint_val) {
                            format_fingerprint_for_csv_cli(args, current_mol_data->original_smiles, current_mol_data->original_line_for_output,
                                                      fp_width, fp_height, fingerprint_val,
                                                      fingerprint_str_buffer, MAX_FINGERPRINT_STR_LEN,
                                                      full_output_line_buffer, sizeof(full_output_line_buffer), max_intensity_val);
                            
                            save_diffraction_image_for_molecule_cli(args, current_mol_data->original_smiles, current_mol_data->original_row_idx, fingerprint_val, fp_width, fp_height);
                            free(fingerprint_val);
                        } else {
                            snprintf(full_output_line_buffer, sizeof(full_output_line_buffer), "%s,DIFFRACTION_ERROR_BATCH", current_mol_data->original_line_for_output);
                        }
                        pthread_mutex_lock(&output_mutex);
                        fprintf(processor.outfile, "%s\n", full_output_line_buffer);
                        fflush(processor.outfile);
                        pthread_mutex_unlock(&output_mutex);
                    } else { // Molecule became invalid or was marked as PARSE_ERROR earlier
                         pthread_mutex_lock(&output_mutex);
                         // If original_line_for_output already contains PARSE_ERROR, just print that
                         fprintf(processor.outfile, "%s\n", current_mol_data->original_line_for_output);
                         fflush(processor.outfile);
                         pthread_mutex_unlock(&output_mutex);
                    }
                    // Restore globals
                    atoms = prev_atoms; bonds = prev_bonds;
                    atom_count = prev_atom_count; bond_count = prev_bond_count;
                }
                 atomic_fetch_add(&total_processed_count, 1);
                 if (!args->verbose && processor.line_count > 0) {
                    float current_progress = (float)atomic_load(&total_processed_count) / (float)data_line_idx * 100.0f;
                    float current_speed = (float)atomic_load(&total_processed_count) / ((float)(time(NULL) - start_time) + 0.001f);
                     printf("\rOverall Progress: %.1f%% (%d/%d) - %.1f mol/sec", current_progress, atomic_load(&total_processed_count), data_line_idx, current_speed);
                     fflush(stdout);
                 }
            } // End loop over molecules in current_layout_batch
            if (!keep_running) break;
        } // End loop over all layout batches
    } else { // No valid molecules
        // Write out any molecules that had parse errors during initial scan
        for(int i=0; i < data_line_idx; ++i) {
            if (strstr(parsed_molecules[i].original_line_for_output, "PARSE_ERROR")) {
                pthread_mutex_lock(&output_mutex);
                fprintf(processor.outfile, "%s\n", parsed_molecules[i].original_line_for_output);
                fflush(processor.outfile);
                pthread_mutex_unlock(&output_mutex);
                atomic_fetch_add(&total_processed_count, 1);
            }
        }
    }
    printf("\nProcessing complete or interrupted. Total processed: %d / %d\n", atomic_load(&total_processed_count), data_line_idx);

    // Clean up resources
    for (int i = 0; i < data_line_idx; i++) {
        if (input_lines_buffer[i]) free(input_lines_buffer[i]);
        if (smiles_strings_buffer[i]) free(smiles_strings_buffer[i]);
        if (parsed_molecules && parsed_molecules[i].atoms_data) free(parsed_molecules[i].atoms_data);
        if (parsed_molecules && parsed_molecules[i].bonds_data) free(parsed_molecules[i].bonds_data);
    }
    free(input_lines_buffer);
    free(smiles_strings_buffer);
    if (parsed_molecules) free(parsed_molecules); parsed_molecules = NULL;
    if (atoms_batch_ptrs) free(atoms_batch_ptrs);
    if (bonds_batch_ptrs) free(bonds_batch_ptrs);
    if (atom_counts_batch_arr) free(atom_counts_batch_arr);
    if (bond_counts_batch_arr) free(bond_counts_batch_arr);
    
    free_csv_processor(&processor);
    return 0;
}

int ensure_dir_exists(const char *path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        if (mkdir(path, 0700) != 0 && errno != EEXIST) {
            perror("mkdir failed");
            return -1;
        }
    } else if (!S_ISDIR(st.st_mode)) {
        fprintf(stderr, "Error: '%s' exists but is not a directory.\n", path);
        return -1;
    }
    return 0;
}

void print_cli_help(const char *prog_name) {
    fprintf(stderr, "Usage: %s -i <input.csv> -o <output.csv> [options]\n\n", prog_name);
    fprintf(stderr, "Generates holographic diffraction fingerprints from SMILES strings.\n\n");
    fprintf(stderr, "Required arguments:\n");
    fprintf(stderr, "  -i, --input-csv <filepath>    Path to input CSV file containing a 'SMILES' column.\n");
    fprintf(stderr, "  -o, --output-csv <filepath>   Path to output CSV file for processed fingerprints.\n\n");
    fprintf(stderr, "Image generation options (optional):\n");
    fprintf(stderr, "  --output-dir <dirpath>        Directory to save generated diffraction images.\n");
    fprintf(stderr, "                                (If not provided and -n is not set, images won't be saved).\n");
    fprintf(stderr, "  -n, --no-images               Suppress generation and saving of diffraction images.\n");
    fprintf(stderr, "  --output-format <fmt>         Image output format ('ppm' (color) or 'pgm' (grayscale)). Default: ppm.\n");
    fprintf(stderr, "  -r, --resolution <size>       Resolution of the diffraction grid (power of 2). Default: 512.\n");
    fprintf(stderr, "  -c, --colormap <name>         Colormap for images (gray, jet, viridis, plasma, heat). Default: heat.\n\n");
    fprintf(stderr, "Simulation options:\n");
    fprintf(stderr, "  --layout-iterations <num>     Number of iterations for force-directed layout. Default: 100.\n");
    fprintf(stderr, "  -q, --quantum-model           Enable more detailed quantum mechanical effects for simulation. Slower.\n");
    fprintf(stderr, "  -m, --mo-effects              Include simplified molecular orbital effects in simulation.\n\n");
    fprintf(stderr, "Layout parameters (advanced):\n");
    fprintf(stderr, "  --k-spring <val>              Spring constant for bonds in layout. Default: 1.0.\n");
    fprintf(stderr, "  --k-repulsive <val>           Repulsive force constant between atoms. Default: 0.5.\n");
    fprintf(stderr, "  --damping <val>               Damping factor for layout stability. Default: 0.8.\n");
    fprintf(stderr, "  --time-step <val>             Time step for layout integration. Default: 0.1.\n\n");
    fprintf(stderr, "CSV processing options:\n");
    fprintf(stderr, "  --smiles-col <column>         Name of the column containing SMILES strings. Default: 'SMILES'.\n\n");
    fprintf(stderr, "Other options:\n");
    fprintf(stderr, "  -v, --verbose                 Enable detailed logging.\n");
    fprintf(stderr, "  -h, --help                    Show this help message and exit.\n");
    fprintf(stderr, "  -j, --jobs <num>              Number of jobs for parallel processing. Default: 1.\n\n");
    fprintf(stderr, "Performance options:\n");
    fprintf(stderr, "  --streaming                   Use streaming processing mode (lower memory usage).\n");
    fprintf(stderr, "                                Default for single-threaded processing, optional for parallel.\n");
    fprintf(stderr, "  --buffer-size <size>          CSV buffer size in KB for streaming mode. Default: 1024 (1MB).\n\n");
    fprintf(stderr, "Fingerprint options:\n");
    fprintf(stderr, "  --column-format               Save fingerprint values as separate CSV columns.\n");
    fprintf(stderr, "  --space-format                Save fingerprint as space-separated values (default).\n");
    fprintf(stderr, "  --condense-block-size <N>     Condense fingerprint by averaging NxN blocks. N=1 for no condensation. Default: 1.\n");
    fprintf(stderr, "                                Resolution must be divisible by N if N > 1.\n\n");
    fprintf(stderr, "Input CSV format:\n");
    fprintf(stderr, "  The input CSV must have a header row. Column with SMILES strings is identified by name.\n");
    fprintf(stderr, "  Other columns will be copied to the output CSV.\n\n");
    fprintf(stderr, "Output CSV format:\n");
    fprintf(stderr, "  The output CSV will contain all original columns plus a new 'Fingerprint' column (space-separated)\n");
    fprintf(stderr, "  or multiple FP_N columns (column-format).\n");
    fprintf(stderr, "  Fingerprint values are log-scaled intensities from the diffraction pattern.\n");
    fprintf(stderr, "  --device <device>      Compute device (cuda, cpu, auto). Default: auto\n");
    fprintf(stderr, "  --cuda-batch-size <n>  Batch size for CUDA processing. Default: 256\n");
}

int parse_cli_arguments(int argc, char **argv, CliArgs *args) {
    // Set defaults
    args->input_csv_path = NULL;
    args->output_csv_path = NULL;
    args->output_dir_images = NULL;
    args->smiles_column_name = strdup("SMILES");
    args->no_images = false;
    args->verbose = false;
    args->resolution = 512;
    args->colormap_name = strdup("heat");
    args->output_format_images = strdup("ppm");
    args->layout_iterations = 100;
    args->use_quantum_model = false;
    args->use_mo_effects = false;
    args->k_spring = 1.0;
    args->k_repulsive = 0.5;
    args->damping_factor = 0.8;
    args->time_step_factor = 0.1;
    args->num_jobs = 1;
    args->use_streaming = true;
    args->csv_buffer_size = 1024; // Default 1MB buffer for streaming
    args->column_format = false; // Default to space-separated fingerprint
    args->condense_block_size = 1; // Default to no condensation

    if (argc == 1) {
        print_cli_help(argv[0]);
        return 0;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_cli_help(argv[0]);
            return 0;
        } else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input-csv") == 0) && i + 1 < argc) {
            args->input_csv_path = argv[++i];
        } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output-csv") == 0) && i + 1 < argc) {
            args->output_csv_path = argv[++i];
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            args->output_dir_images = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--no-images") == 0) {
            args->no_images = true;
        } else if ((strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--resolution") == 0) && i + 1 < argc) {
            args->resolution = atoi(argv[++i]);
            if (args->resolution <= 0 || (args->resolution & (args->resolution - 1)) != 0) {
                fprintf(stderr, "Error: Resolution must be a positive power of 2.\n");
                return -1;
            }
        } else if ((strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--colormap") == 0) && i + 1 < argc) {
            free(args->colormap_name);
            args->colormap_name = strdup(argv[++i]);
        } else if (strcmp(argv[i], "--output-format") == 0 && i + 1 < argc) {
            free(args->output_format_images);
            args->output_format_images = strdup(argv[++i]);
            if (strcmp(args->output_format_images, "ppm") != 0 && strcmp(args->output_format_images, "pgm") != 0) {
                fprintf(stderr, "Error: Invalid image output format '%s'. Must be 'ppm' or 'pgm'.\n", args->output_format_images);
                return -1;
            }
        } else if (strcmp(argv[i], "--layout-iterations") == 0 && i + 1 < argc) {
            args->layout_iterations = atoi(argv[++i]);
            if (args->layout_iterations < 0) args->layout_iterations = 0;
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quantum-model") == 0) {
            args->use_quantum_model = true;
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--mo-effects") == 0) {
            args->use_mo_effects = true;
        } else if (strcmp(argv[i], "--k-spring") == 0 && i+1 < argc) {
            args->k_spring = atof(argv[++i]);
        } else if (strcmp(argv[i], "--k-repulsive") == 0 && i+1 < argc) {
            args->k_repulsive = atof(argv[++i]);
        } else if (strcmp(argv[i], "--damping") == 0 && i+1 < argc) {
            args->damping_factor = atof(argv[++i]);
        } else if (strcmp(argv[i], "--time-step") == 0 && i+1 < argc) {
            args->time_step_factor = atof(argv[++i]);
        } else if (strcmp(argv[i], "--smiles-col") == 0 && i+1 < argc) {
            free(args->smiles_column_name);
            args->smiles_column_name = strdup(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            args->verbose = true;
        } else if ((strcmp(argv[i], "-j") == 0 || strcmp(argv[i], "--jobs") == 0) && i + 1 < argc) {
            args->num_jobs = atoi(argv[++i]);
            if (args->num_jobs < 1) {
                args->num_jobs = 1;
            } else if (args->num_jobs > get_cpu_count() * 2) {
                fprintf(stderr, "Warning: Number of jobs (%d) exceeds twice the number of CPU cores (%d).\n", 
                        args->num_jobs, get_cpu_count());
            }
            // By default, disable streaming for multi-threaded mode
            if (args->num_jobs > 1) {
                args->use_streaming = false;
            }
        } else if (strcmp(argv[i], "--streaming") == 0) {
            args->use_streaming = true;
        } else if (strcmp(argv[i], "--buffer-size") == 0 && i + 1 < argc) {
            args->csv_buffer_size = atoi(argv[++i]);
            if (args->csv_buffer_size < 16) {
                fprintf(stderr, "Error: Buffer size must be at least 16 KB.\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--column-format") == 0) {
            args->column_format = true;
        } else if (strcmp(argv[i], "--space-format") == 0) {
            args->column_format = false;
        } else if (strcmp(argv[i], "--condense-block-size") == 0 && i + 1 < argc) {
            args->condense_block_size = atoi(argv[++i]);
            if (args->condense_block_size < 1) {
                fprintf(stderr, "Error: Condense block size must be at least 1.\n");
                return -1;
            }
            // Resolution check will be done after resolution is parsed, or here if resolution is always parsed before.
            // For safety, check it later or ensure parsing order.
            // For now, we'll assume resolution is known or check it when both are parsed.
        } else if (strcmp(argv[i], "--device") == 0 || strcmp(argv[i], "-d") == 0) {
            if (i + 1 < argc) {
                const char* device = argv[++i];
                if (strcmp(device, "cuda") == 0) {
                    use_cuda = 2;  // Force CUDA
                } else if (strcmp(device, "cpu") == 0) {
                    use_cuda = 0;  // Force CPU
                } else if (strcmp(device, "auto") == 0) {
                    use_cuda = 1;  // Auto-select
                } else {
                    fprintf(stderr, "Error: Unknown device '%s'. Use 'cuda', 'cpu', or 'auto'.\n", device);
                    return 0;
                }
            } else {
                fprintf(stderr, "Error: --device option requires an argument.\n");
                return 0;
            }
        } else if (strcmp(argv[i], "--cuda-batch-size") == 0) {
            if (i + 1 < argc) {
                cuda_batch_size = atoi(argv[++i]);
                if (cuda_batch_size <= 0) {
                    fprintf(stderr, "Error: CUDA batch size must be positive.\n");
                    return 0;
                }
            } else {
                fprintf(stderr, "Error: --cuda-batch-size option requires an argument.\n");
                return 0;
            }
        } else {
            fprintf(stderr, "Error: Unknown or incomplete option '%s'.\n", argv[i]);
            print_cli_help(argv[0]);
            return -1;
        }
    }

    // Check required arguments
    if (!args->input_csv_path) {
        fprintf(stderr, "Error: Input CSV path (-i or --input-csv) is required.\n");
        return -1;
    }
    if (!args->output_csv_path) {
        fprintf(stderr, "Error: Output CSV path (-o or --output-csv) is required.\n");
        return -1;
    }

    // Create output directory if needed
    if (!args->no_images && args->output_dir_images) {
        if (ensure_dir_exists(args->output_dir_images) != 0) {
            fprintf(stderr, "Error: Could not create or access image output directory '%s'.\n", args->output_dir_images);
            return -1;
        }
    }
    
    // Validate colormap name
    if (strcmp(args->colormap_name, "gray") == 0) set_render_colormap(COLORMAP_GRAYSCALE);
    else if (strcmp(args->colormap_name, "jet") == 0) set_render_colormap(COLORMAP_JET);
    else if (strcmp(args->colormap_name, "viridis") == 0) set_render_colormap(COLORMAP_VIRIDIS);
    else if (strcmp(args->colormap_name, "plasma") == 0) set_render_colormap(COLORMAP_PLASMA);
    else if (strcmp(args->colormap_name, "heat") == 0) set_render_colormap(COLORMAP_HEAT);
    else {
        fprintf(stderr, "Error: Unknown colormap '%s'.\n", args->colormap_name);
        return -1;
    }

    // Validate condensation block size against resolution
    if (args->condense_block_size > 1 && args->resolution % args->condense_block_size != 0) {
        fprintf(stderr, "Error: Resolution (%d) must be divisible by condense_block_size (%d).\n", args->resolution, args->condense_block_size);
        return -1;
    }

    return 1; // Success
}

char** parse_csv_line(char* line, int* num_cols) {
    char** cols = malloc(MAX_CSV_COLUMNS * sizeof(char*));
    if (!cols) { *num_cols = 0; return NULL; }

    int count = 0;
    char* token = strtok(line, ",\n\r");
    while (token != NULL && count < MAX_CSV_COLUMNS) {
        cols[count++] = token;
        token = strtok(NULL, ",\n\r");
    }
    *num_cols = count;
    return cols;
}

int main(int argc, char **argv) {
    CliArgs args;
    
    int parse_result = parse_cli_arguments(argc, argv, &args);
    if (parse_result <= 0) {
        // Args cleanup is handled by parse_cli_arguments or here if early exit
        if(args.colormap_name) free(args.colormap_name);
        if(args.output_format_images) free(args.output_format_images);
        if(args.smiles_column_name) free(args.smiles_column_name);
        return (parse_result == 0) ? 0 : 1;
    }
    
    start_time = time(NULL);
    
    int result;
    // The choice between process_csv_parallel and process_csv_streaming might need to be re-evaluated.
    // Forcing parallel for now as it's been heavily modified.
    if (args.num_jobs > 0) { // Modified to always use this path for now
        result = process_csv_parallel(&args);
    } else { // Fallback or explicit streaming
        result = process_csv_streaming(&args);
    }
    
    if(args.colormap_name) free(args.colormap_name);
    if(args.output_format_images) free(args.output_format_images);
    if(args.smiles_column_name) free(args.smiles_column_name);
    
    if (atoms != NULL) { free(atoms); atoms = NULL; }
    if (bonds != NULL) { free(bonds); bonds = NULL; }
    
    return result;
}

// Implementation of previously forward-declared helper functions
static double* generate_diffraction_for_molecule_cli(CliArgs *args, int *fp_width_out, int *fp_height_out, double *max_intensity_out) {
    int W = args->resolution;
    int total_grid_points = W * W;
    
    // Initialize grid for molecule drawing
    complex double *aperture_grid = calloc(total_grid_points, sizeof(complex double));
    double *intensity = malloc(total_grid_points * sizeof(double));
    
    if (!aperture_grid || !intensity) {
        fprintf(stderr, "Error: Failed to allocate memory for grid arrays\n");
        if (aperture_grid) free(aperture_grid);
        if (intensity) free(intensity);
        return NULL;
    }
    
    // Draw atoms and bonds on grid
    for (int i = 0; i < atom_count; i++) {
        draw_atom_on_grid(aperture_grid, W, atoms[i], i, args->use_quantum_model);
    }
    
    for (int i = 0; i < bond_count; i++) {
        draw_bond_on_grid(aperture_grid, W, bonds[i], args->use_quantum_model);
    }
    
    // Add molecular orbital effects if enabled
    if (args->use_mo_effects) {
        add_molecular_orbital_effects(aperture_grid, W);
    }
    
    // Calculate diffraction pattern
    fft_2d(aperture_grid, W, W, 1);
    
    // Calculate intensity
    double max_intensity = 0.0;
    for (int i = 0; i < total_grid_points; i++) {
        intensity[i] = cabs(aperture_grid[i]) * cabs(aperture_grid[i]);
        if (intensity[i] > max_intensity) max_intensity = intensity[i];
    }
    
    // Apply condensation if requested
    double *final_intensity = intensity;
    int final_width = W;
    int final_height = W;
    
    if (args->condense_block_size > 1) {
        int condensed_width = W / args->condense_block_size;
        int condensed_height = W / args->condense_block_size;
        double *condensed_intensity = malloc(condensed_width * condensed_height * sizeof(double));
        
        if (condensed_intensity) {
            condense_fingerprint_average(intensity, W, W, args->condense_block_size, 
                                         condensed_intensity, &condensed_width, &condensed_height);
            free(intensity);
            final_intensity = condensed_intensity;
            final_width = condensed_width;
            final_height = condensed_height;
        } else {
            fprintf(stderr, "Warning: Failed to allocate memory for condensed fingerprint, using full resolution\n");
        }
    }
    
    // Set output parameters
    *fp_width_out = final_width;
    *fp_height_out = final_height;
    *max_intensity_out = max_intensity;
    
    // Free temporary resources
    free(aperture_grid);
    
    return final_intensity;
}

static void format_fingerprint_for_csv_cli(CliArgs *args, const char* smiles, const char* original_line, 
                                      int fp_width, int fp_height, const double* fingerprint_data, 
                                      char* fingerprint_str_out, int max_fp_str_len, 
                                      char* full_output_line_out, int max_full_line_len, 
                                      double max_intensity) {
    int total_grid_points = fp_width * fp_height;
    double epsilon = max_intensity * 1.0e-7;
    if (epsilon < 1.0e-10) epsilon = 1.0e-10;
    
    if (args->column_format) {
        // For column format, we'll write the original line followed by fingerprint values as separate columns
        snprintf(full_output_line_out, max_full_line_len, "%s", original_line);
        
        int offset = strlen(full_output_line_out);
        int remaining = max_full_line_len - offset - 1;
        char *ptr = full_output_line_out + offset;
        
        for (int i = 0; i < total_grid_points && remaining > 10; i++) {
            double scaled_value = apply_log_scale_intensity(fingerprint_data[i], max_intensity, epsilon);
            int written = snprintf(ptr, remaining, ",%.4f", scaled_value);
            if (written < 0 || written >= remaining) {
                // Won't fit, stop here
                break;
            }
            ptr += written;
            remaining -= written;
        }
    } else {
        // For row format, we need to check if all values will fit
        int est_chars_per_value = 8; // Approximate length for ",0.1234" or ",12.34"
        int max_values = (max_full_line_len - strlen(original_line) - 32) / est_chars_per_value;
        
        if (max_values < total_grid_points) {
            // Will truncate, so use a condensed format - only include every Nth point
            int stride = (total_grid_points + max_values - 1) / max_values;
            
            // First format the fingerprint as a separate string
            fingerprint_str_out[0] = '\0';
            char *ptr = fingerprint_str_out;
            int remaining = max_fp_str_len - 1;
            
            for (int i = 0; i < total_grid_points; i += stride) {
                double scaled_value = apply_log_scale_intensity(fingerprint_data[i], max_intensity, epsilon);
                int written;
                if (i > 0) {
                    written = snprintf(ptr, remaining, " %.4f", scaled_value);
                } else {
                    written = snprintf(ptr, remaining, "%.4f", scaled_value);
                }
                
                if (written < 0 || written >= remaining) break;
                
                ptr += written;
                remaining -= written;
            }
            
            // Now combine with the original line
            snprintf(full_output_line_out, max_full_line_len, "%s,%s", original_line, fingerprint_str_out);
        } else {
            // All values will fit, format them directly
            fingerprint_str_out[0] = '\0';
            char *ptr = fingerprint_str_out;
            int remaining = max_fp_str_len - 1;
            
            for (int i = 0; i < total_grid_points && remaining > 10; i++) {
                double scaled_value = apply_log_scale_intensity(fingerprint_data[i], max_intensity, epsilon);
                int written;
                if (i > 0) {
                    written = snprintf(ptr, remaining, " %.4f", scaled_value);
                } else {
                    written = snprintf(ptr, remaining, "%.4f", scaled_value);
                }
                
                if (written < 0 || written >= remaining) break;
                
                ptr += written;
                remaining -= written;
            }
            
            snprintf(full_output_line_out, max_full_line_len, "%s,%s", original_line, fingerprint_str_out);
        }
    }
}

static void save_diffraction_image_for_molecule_cli(CliArgs *args, const char* smiles, int original_row_idx,
                                               const double* fingerprint_data, int fp_width, int fp_height) {
    // Skip if no image output is requested
    if (args->no_images || !args->output_dir_images) {
        return;
    }
    
    // Create sanitized SMILES for filename (replace non-alphanumeric chars with underscore)
    char sanitized_smiles[64];
    strncpy(sanitized_smiles, smiles, 60);
    sanitized_smiles[60] = '\0';
    
    for (char *p = sanitized_smiles; *p; p++) {
        if (!isalnum(*p) && *p != '_' && *p != '-') *p = '_';
    }
    
    // Build image filename
    char image_filename[512];
    snprintf(image_filename, sizeof(image_filename), "%s/smi_r%d_%s.%s", args->output_dir_images, 
            original_row_idx, sanitized_smiles, args->output_format_images);
    
    // Open file for writing
    FILE *img_file = fopen(image_filename, "wb");
    if (!img_file) {
        fprintf(stderr, "Warning: Could not open file to save image: %s\n", image_filename);
        return;
    }
    
    // Write image data
    bool is_color = (strcmp(args->output_format_images, "ppm") == 0);
    output_diffraction_image(img_file, fingerprint_data, fp_width, fp_height, 
                           is_color, args->output_format_images);
    fclose(img_file);
    
    if (args->verbose) {
        printf("Saved image: %s\n", image_filename);
    }
}