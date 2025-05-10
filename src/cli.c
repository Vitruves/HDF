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
    #define _GNU_SOURCE
    #include <sched.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(x)
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

#define MAX_CSV_LINE_LENGTH 2048
#define MAX_CSV_COLUMNS 128
#define MAX_FINGERPRINT_STR_LEN (4096 * 4096 * 8)
#define BATCH_SIZE 50
#define CSV_BUFFER_SIZE (1024 * 1024)

pthread_mutex_t global_processing_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t output_mutex = PTHREAD_MUTEX_INITIALIZER;
atomic_int total_processed_count = 0;

// Global flag for graceful shutdown
static volatile sig_atomic_t keep_running = 1;

// Global start time
static time_t start_time;

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

// Set thread affinity
static void set_thread_affinity(int thread_id) {
    #if defined(__linux__) && defined(_OPENMP)
        int cpu_count = get_cpu_count();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id % cpu_count, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    #elif defined(__APPLE__) && defined(_OPENMP)
        // macOS doesn't support CPU affinity in the same way
        // Could use thread_policy_set() but it's not recommended
        (void)thread_id; // Suppress unused parameter warning
    #else
        (void)thread_id; // Suppress unused parameter warning
    #endif
}

typedef struct {
    int thread_id;
    CliArgs *args;
    char smiles_to_process[MAX_CSV_LINE_LENGTH];
    char original_line_for_output[MAX_CSV_LINE_LENGTH];
    int original_row_idx;
    
    char *result_fingerprint;
    char *full_output_line;
    
    int status; // 0=idle, 1=processing, 2=done(clean), -1=error(hard), -2=done(warning)
    pthread_mutex_t mutex;
    pthread_cond_t condition_task;
    pthread_cond_t condition_done;
    bool has_new_task;
    bool please_exit;
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
int process_csv_parallel(CliArgs *args, int num_threads);

// Function declarations for external functions
extern void initialize_molecule_data(void);
extern int parse_smiles(const char *smiles, bool *complex_feature_warning);
extern void apply_quantum_corrections_to_atoms(void);
extern void optimize_molecule_layout(int iterations, double k_spring, double k_repulsive, 
                                   double damping, double time_step);
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
extern int atom_count;
extern AtomPos atoms[MAX_ATOMS];
extern int bond_count;
extern BondSeg bonds[MAX_ATOMS * 2];

// Function prototype for condensation, if not already in diffraction_engine.h include
extern void condense_fingerprint_average(const double *input_fp, int input_w, int input_h, int block_size, double *output_fp, int *output_w, int *output_h);

void* process_molecule_thread(void *arg) {
    WorkerThread *worker = (WorkerThread*)arg;
    CliArgs *args = worker->args;

    // Set thread affinity
    set_thread_affinity(worker->thread_id);

    int W = args->resolution;
    int total_grid_points = W * W;
    complex double *aperture_grid = calloc(total_grid_points, sizeof(complex double));
    double *intensity = malloc(total_grid_points * sizeof(double));
    
    if (!aperture_grid || !intensity) {
        pthread_mutex_lock(&worker->mutex);
        worker->status = -1;
        worker->has_new_task = false;
        pthread_mutex_unlock(&worker->mutex);
        if (aperture_grid) free(aperture_grid);
        if (intensity) free(intensity);
        return NULL;
    }

    while (keep_running) {  // Check global flag for graceful shutdown
        pthread_mutex_lock(&worker->mutex);
        while (!worker->has_new_task && !worker->please_exit && keep_running) {
            pthread_cond_wait(&worker->condition_task, &worker->mutex);
        }

        if (!keep_running || worker->please_exit) {
            worker->status = 0;
            pthread_mutex_unlock(&worker->mutex);
            break;
        }

        worker->status = 1;
        char current_smiles[MAX_CSV_LINE_LENGTH];
        char current_original_line[MAX_CSV_LINE_LENGTH];
        int current_row_idx = worker->original_row_idx;
        strcpy(current_smiles, worker->smiles_to_process);
        strcpy(current_original_line, worker->original_line_for_output);
        
        worker->has_new_task = false;
        pthread_mutex_unlock(&worker->mutex);

        int parsed_atom_count_result = 0;
        bool complex_feature_warning = false;

        pthread_mutex_lock(&global_processing_mutex);
        initialize_molecule_data();
        parsed_atom_count_result = parse_smiles(current_smiles, &complex_feature_warning);
        int num_parsed_atoms = parsed_atom_count_result;

        if (complex_feature_warning) {
            pthread_mutex_lock(&output_mutex);
            fprintf(stderr, "\nNote: SMILES '%s' (row %d) contained complex features ([...]) or unclosed rings that were skipped/partially handled.\n", current_smiles, current_row_idx);
            fflush(stderr);
            pthread_mutex_unlock(&output_mutex);
        }

        if (num_parsed_atoms > 0) {
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
                if(intensity[i] > max_raw_intensity) max_raw_intensity = intensity[i];
            }
            
            double epsilon_log_fp = max_raw_intensity * 1e-7;
            if(epsilon_log_fp < 1e-10) epsilon_log_fp = 1e-10;

            // Buffer for log-scaled intensity values
            double *scaled_intensity_buffer = malloc(total_grid_points * sizeof(double));
            if (!scaled_intensity_buffer) {
                pthread_mutex_lock(&output_mutex);
                fprintf(stderr, "\nError: Worker %d failed to alloc scaled_intensity_buffer for SMILES '%s'\n", worker->thread_id, current_smiles);
                pthread_mutex_unlock(&output_mutex);
                // Set error status and cleanup (simplified here)
                pthread_mutex_unlock(&global_processing_mutex);
                pthread_mutex_lock(&worker->mutex);
                worker->status = -1;
                pthread_mutex_unlock(&worker->mutex);
                if(aperture_grid) free(aperture_grid);
                if(intensity) free(intensity);
                return NULL; // Critical error
            }

            for(int i = 0; i < total_grid_points; ++i) {
                scaled_intensity_buffer[i] = apply_log_scale_intensity(intensity[i], max_raw_intensity, epsilon_log_fp);
            }
            
            double *fingerprint_data_to_write = NULL;
            int num_fp_values_to_write = 0;

            if (args->condense_block_size > 1) {
                int condensed_w, condensed_h;
                double *condensed_buffer = malloc((W / args->condense_block_size) * (W / args->condense_block_size) * sizeof(double));
                if (!condensed_buffer) {
                     pthread_mutex_lock(&output_mutex);
                     fprintf(stderr, "\nError: Worker %d failed to alloc condensed_buffer for SMILES '%s'\n", worker->thread_id, current_smiles);
                     pthread_mutex_unlock(&output_mutex);
                     free(scaled_intensity_buffer);
                     // Set error status and cleanup (similar to above)
                    pthread_mutex_unlock(&global_processing_mutex);
                    pthread_mutex_lock(&worker->mutex);
                    worker->status = -1;
                    pthread_mutex_unlock(&worker->mutex);
                    if(aperture_grid) free(aperture_grid);
                    if(intensity) free(intensity);
                    return NULL;
                }
                condense_fingerprint_average(scaled_intensity_buffer, W, W, args->condense_block_size, condensed_buffer, &condensed_w, &condensed_h);
                free(scaled_intensity_buffer);
                scaled_intensity_buffer = NULL;
                fingerprint_data_to_write = condensed_buffer;
                num_fp_values_to_write = condensed_w * condensed_h;
            } else {
                fingerprint_data_to_write = scaled_intensity_buffer;
                num_fp_values_to_write = total_grid_points;
            }

            worker->result_fingerprint[0] = '\0';
            char temp_val_str[32];
            for (int i = 0; i < num_fp_values_to_write; ++i) {
                sprintf(temp_val_str, "%.4f", fingerprint_data_to_write[i]);
                strcat(worker->result_fingerprint, temp_val_str);
                if (i < num_fp_values_to_write - 1) {
                    strcat(worker->result_fingerprint, " ");
                }
                if (strlen(worker->result_fingerprint) >= MAX_FINGERPRINT_STR_LEN - 32) {
                    pthread_mutex_lock(&output_mutex);
                    fprintf(stderr, "\nWarning: Fingerprint string truncated for SMILES '%s' (row %d) due to length.\n", current_smiles, current_row_idx);
                    pthread_mutex_unlock(&output_mutex);
                    break;
                }
            }

            // Prepare full output line
            if (args->column_format) {
                worker->full_output_line[0] = '\0'; 
                strcat(worker->full_output_line, current_original_line); 
                
                char temp_fp_val_str[32];
                for (int i = 0; i < num_fp_values_to_write; ++i) {
                    sprintf(temp_fp_val_str, ",%.4f", fingerprint_data_to_write[i]);
                    strcat(worker->full_output_line, temp_fp_val_str);
                    
                    if (strlen(worker->full_output_line) >= MAX_CSV_LINE_LENGTH + MAX_FINGERPRINT_STR_LEN - 32) {
                        pthread_mutex_lock(&output_mutex);
                        fprintf(stderr, "\nWarning: Fingerprint columns truncated for SMILES '%s' (row %d) due to length.\n", 
                                current_smiles, current_row_idx);
                        pthread_mutex_unlock(&output_mutex);
                        break;
                    }
                }
            } else {
                // Space-separated format
                worker->result_fingerprint[0] = '\0';
                 char temp_fp_val_str[32];
                for (int i = 0; i < num_fp_values_to_write; ++i) {
                    sprintf(temp_fp_val_str, "%.4f", fingerprint_data_to_write[i]);
                    strcat(worker->result_fingerprint, temp_fp_val_str);
                    if (i < num_fp_values_to_write - 1) {
                        strcat(worker->result_fingerprint, " ");
                    }
                    if (strlen(worker->result_fingerprint) >= MAX_FINGERPRINT_STR_LEN - 32) {
                        pthread_mutex_lock(&output_mutex);
                        fprintf(stderr, "\nWarning: Fingerprint string truncated for SMILES '%s' (row %d) due to length.\n", current_smiles, current_row_idx);
                        pthread_mutex_unlock(&output_mutex);
                        break;
                    }
                }
                snprintf(worker->full_output_line, MAX_CSV_LINE_LENGTH + MAX_FINGERPRINT_STR_LEN + 1,
                         "%s,%s", current_original_line, worker->result_fingerprint);
            }
            
            // Free the buffer that holds the final fingerprint data
            if (fingerprint_data_to_write) {
                free(fingerprint_data_to_write);
                fingerprint_data_to_write = NULL;
            }

            if (!args->no_images && args->output_dir_images) {
                char image_filename[512];
                char sanitized_smiles[64];
                strncpy(sanitized_smiles, current_smiles, 60);
                sanitized_smiles[60] = '\0';
                for(char *p_san = sanitized_smiles; *p_san; ++p_san) {
                    if (!isalnum(*p_san) && *p_san != '_' && *p_san != '-') *p_san = '_';
                }
                sprintf(image_filename, "%s/smi_r%d_%s.%s", args->output_dir_images, current_row_idx, sanitized_smiles, args->output_format_images);
                FILE *img_file = fopen(image_filename, "wb");
                if (img_file) {
                    bool is_color = (strcmp(args->output_format_images, "ppm") == 0);
                    output_diffraction_image(img_file, intensity, W, W, is_color, args->output_format_images);
                    fclose(img_file);
                    if(args->verbose){
                        pthread_mutex_lock(&output_mutex);
                        printf("\nSaved image: %s (Worker %d)\n", image_filename, worker->thread_id);
                        pthread_mutex_unlock(&output_mutex);
                    }
                } else {
                    pthread_mutex_lock(&output_mutex);
                    fprintf(stderr, "\nWarning: Worker %d could not open file to save image: %s for SMILES %s (row %d)\n", worker->thread_id, image_filename, current_smiles, current_row_idx);
                    pthread_mutex_unlock(&output_mutex);
                }
            }
        } else {
            pthread_mutex_lock(&output_mutex);
            fprintf(stderr, "\nWarning: Worker %d failed to parse SMILES '%s' (row %d). Skipping.\n", worker->thread_id, current_smiles, current_row_idx);
            fflush(stderr);
            pthread_mutex_unlock(&output_mutex);
            snprintf(worker->full_output_line, MAX_CSV_LINE_LENGTH + MAX_FINGERPRINT_STR_LEN + 1,
                     "%s,PARSE_ERROR", current_original_line);
        }
        
        pthread_mutex_unlock(&global_processing_mutex);

        pthread_mutex_lock(&worker->mutex);
        if (num_parsed_atoms > 0) {
            worker->status = complex_feature_warning ? -2 : 2;
        } else {
            worker->status = -1;
        }
        pthread_mutex_unlock(&worker->mutex);
    }

    free(aperture_grid);
    free(intensity);
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
            char image_filename[512];
            char sanitized_smiles[64];
            strncpy(sanitized_smiles, smiles, 60);
            sanitized_smiles[60] = '\0';
            
            for (char *p_san = sanitized_smiles; *p_san; ++p_san) {
                if (!isalnum(*p_san) && *p_san != '_' && *p_san != '-') *p_san = '_';
            }
            
            sprintf(image_filename, "%s/smi_r%ld_%s.%s", args->output_dir_images, 
                    processor.processed_lines, sanitized_smiles, args->output_format_images);
            
            FILE *img_file = fopen(image_filename, "wb");
            if (img_file) {
                bool is_color = (strcmp(args->output_format_images, "ppm") == 0);
                output_diffraction_image(img_file, intensity, W, W, is_color, args->output_format_images);
                fclose(img_file);
                
                if (args->verbose) {
                    printf("\nSaved image: %s\n", image_filename);
                }
            } else {
                fprintf(stderr, "\nWarning: Could not open file to save image: %s for SMILES %s (row %ld)\n", 
                        image_filename, smiles, processor.processed_lines);
            }
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

int process_csv_parallel(CliArgs *args, int num_threads) {
    // Set up signal handling
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    // Initialize processor
    CsvProcessor processor;
    if (!init_csv_processor(&processor, args->input_csv_path, args->output_csv_path, args->smiles_column_name, args)) {
        return 1;
    }
    
    // Only read all lines into memory in parallel mode
    char **input_lines_buffer = malloc(processor.line_count * sizeof(char*));
    char **smiles_strings_buffer = malloc(processor.line_count * sizeof(char*));
    
    if (!input_lines_buffer || !smiles_strings_buffer) {
        fprintf(stderr, "Error: Failed to allocate memory for input buffers\n");
        free_csv_processor(&processor);
        if (input_lines_buffer) free(input_lines_buffer);
        if (smiles_strings_buffer) free(smiles_strings_buffer);
        return 1;
    }
    
    // Read all data lines into memory first
    int data_line_idx = 0;
    char *line;
    while ((line = read_csv_line(&processor)) != NULL && data_line_idx < processor.line_count) {
        char original_line_copy[MAX_CSV_LINE_LENGTH];
        strncpy(original_line_copy, line, sizeof(original_line_copy) - 1);
        original_line_copy[sizeof(original_line_copy) - 1] = '\0';
        
        input_lines_buffer[data_line_idx] = strdup(original_line_copy);
        
        // Extract SMILES
        char line_copy_for_parse[MAX_CSV_LINE_LENGTH];
        strcpy(line_copy_for_parse, line);
        
        int num_cols;
        char **cols = parse_csv_line(line_copy_for_parse, &num_cols);
        
        if (cols && num_cols > processor.smiles_column_idx) {
            smiles_strings_buffer[data_line_idx] = strdup(cols[processor.smiles_column_idx]);
        } else {
            smiles_strings_buffer[data_line_idx] = strdup("INVALID_SMILES_IN_CSV_ROW");
            fprintf(stderr, "\nWarning: Malformed CSV line %d or SMILES column not found. Will mark as PARSE_ERROR.\n", data_line_idx);
        }
        
        if (cols) free(cols);
        data_line_idx++;
    }
    
    // Prepare worker threads
    WorkerThread *workers = malloc(num_threads * sizeof(WorkerThread));
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    
    if (!workers || !threads) {
        fprintf(stderr, "Error: Failed to allocate memory for worker threads\n");
        free_csv_processor(&processor);
        
        // Clean up
        for (int i = 0; i < data_line_idx; i++) {
            free(input_lines_buffer[i]);
            free(smiles_strings_buffer[i]);
        }
        free(input_lines_buffer);
        free(smiles_strings_buffer);
        if (workers) free(workers);
        if (threads) free(threads);
        
        return 1;
    }
    
    // Create output buffers
    char **output_lines_buffer = calloc(processor.line_count, sizeof(char*));
    bool *output_ready_flags = calloc(processor.line_count, sizeof(bool));
    
    if (!output_lines_buffer || !output_ready_flags) {
        fprintf(stderr, "Error: Failed to allocate memory for output buffers\n");
        free_csv_processor(&processor);
        
        // Clean up
        for (int i = 0; i < data_line_idx; i++) {
            free(input_lines_buffer[i]);
            free(smiles_strings_buffer[i]);
        }
        free(input_lines_buffer);
        free(smiles_strings_buffer);
        free(workers);
        free(threads);
        if (output_lines_buffer) free(output_lines_buffer);
        if (output_ready_flags) free(output_ready_flags);
        
        return 1;
    }
    
    // Initialize and create worker threads
    for (int i = 0; i < num_threads; i++) {
        workers[i].thread_id = i;
        workers[i].args = args;
        workers[i].result_fingerprint = malloc(MAX_FINGERPRINT_STR_LEN);
        workers[i].full_output_line = malloc(MAX_CSV_LINE_LENGTH + MAX_FINGERPRINT_STR_LEN + 2);
        
        if (!workers[i].result_fingerprint || !workers[i].full_output_line) {
            fprintf(stderr, "Error: Failed to allocate memory for worker thread %d\n", i);
            // Clean up previously allocated workers
            for (int j = 0; j < i; j++) {
                free(workers[j].result_fingerprint);
                free(workers[j].full_output_line);
                pthread_mutex_destroy(&workers[j].mutex);
                pthread_cond_destroy(&workers[j].condition_task);
                pthread_cond_destroy(&workers[j].condition_done);
            }
            // Clean up other resources
            free_csv_processor(&processor);
            for (int j = 0; j < data_line_idx; j++) {
                free(input_lines_buffer[j]);
                free(smiles_strings_buffer[j]);
            }
            free(input_lines_buffer);
            free(smiles_strings_buffer);
            free(output_lines_buffer);
            free(output_ready_flags);
            free(workers);
            free(threads);
            return 1;
        }
        
        workers[i].status = 0;
        workers[i].has_new_task = false;
        workers[i].please_exit = false;
        pthread_mutex_init(&workers[i].mutex, NULL);
        pthread_cond_init(&workers[i].condition_task, NULL);
        pthread_cond_init(&workers[i].condition_done, NULL);
        
        pthread_create(&threads[i], NULL, process_molecule_thread, &workers[i]);
    }
    
    // Process molecules in parallel
    int tasks_dispatched = 0;
    int tasks_written_to_file = 0;
    printf("Processing %ld molecules using %d threads on %d CPU cores...\n", 
           processor.line_count, num_threads, get_cpu_count());
    
    while (keep_running && tasks_written_to_file < processor.line_count) {
        // Dispatch tasks to idle workers
        if (tasks_dispatched < processor.line_count) {
            for (int i = 0; i < num_threads; i++) {
                if (tasks_dispatched >= processor.line_count) break;

                pthread_mutex_lock(&workers[i].mutex);
                if (workers[i].status == 0 || workers[i].status == 2 || workers[i].status == -1) {
                    // Assign new task
                    strcpy(workers[i].smiles_to_process, smiles_strings_buffer[tasks_dispatched]);
                    strcpy(workers[i].original_line_for_output, input_lines_buffer[tasks_dispatched]);
                    workers[i].original_row_idx = tasks_dispatched + 1; // 1-indexed for user-friendly display
                    
                    workers[i].status = 0;
                    workers[i].has_new_task = true;
                    pthread_cond_signal(&workers[i].condition_task);
                    tasks_dispatched++;
                }
                pthread_mutex_unlock(&workers[i].mutex);
            }
        }

        // Check for completed tasks and write them in order
        bool wrote_something_this_cycle = false;
        for (int i = 0; i < num_threads; i++) {
            pthread_mutex_lock(&workers[i].mutex);
            if ((workers[i].status == 2 || workers[i].status == -1 || workers[i].status == -2) && 
                workers[i].original_row_idx == tasks_written_to_file + 1) {
                // This worker has finished the task we're waiting to write next
                output_lines_buffer[tasks_written_to_file] = strdup(workers[i].full_output_line);
                output_ready_flags[tasks_written_to_file] = true;
                
                workers[i].status = 0; // Mark as idle again
            }
            pthread_mutex_unlock(&workers[i].mutex);
        }

        // Try to write any contiguous block of ready results
        while (tasks_written_to_file < processor.line_count && output_ready_flags[tasks_written_to_file]) {
            fprintf(processor.outfile, "%s\n", output_lines_buffer[tasks_written_to_file]);
            free(output_lines_buffer[tasks_written_to_file]);
            output_lines_buffer[tasks_written_to_file] = NULL; 
            tasks_written_to_file++;
            atomic_fetch_add(&total_processed_count, 1);
            wrote_something_this_cycle = true;
        }

        // Progress display with more info
        if (!args->verbose && processor.line_count > 0) {
            float progress = (float)atomic_load(&total_processed_count) / (float)processor.line_count * 100.0f;
            float speed = (float)atomic_load(&total_processed_count) / 
                         ((float)(time(NULL) - start_time) + 0.001);
            
            pthread_mutex_lock(&output_mutex);
            printf("\rProgress: %.1f%% (%d/%ld) - %.1f molecules/sec", 
                   progress, atomic_load(&total_processed_count), 
                   processor.line_count, speed);
            fflush(stdout);
            pthread_mutex_unlock(&output_mutex);
        }
        
        // Adaptive sleep based on load
        if (!wrote_something_this_cycle) {
            if (tasks_dispatched == processor.line_count) {
                usleep(1000); // Shorter sleep when waiting for final tasks
            } else {
                usleep(100);  // Very short sleep when actively processing
            }
        }
    }
    
    // Signal threads to exit
    for (int i = 0; i < num_threads; i++) {
        pthread_mutex_lock(&workers[i].mutex);
        workers[i].please_exit = true;
        pthread_cond_signal(&workers[i].condition_task);
        pthread_mutex_unlock(&workers[i].mutex);
    }

    // Wait for threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Clean up all the other resources
    for (int i = 0; i < processor.line_count; i++) {
        if (input_lines_buffer[i]) free(input_lines_buffer[i]);
        if (smiles_strings_buffer[i]) free(smiles_strings_buffer[i]);
        if (output_lines_buffer[i]) free(output_lines_buffer[i]);
    }
    
    free(input_lines_buffer);
    free(smiles_strings_buffer);
    free(output_lines_buffer);
    free(output_ready_flags);
    free(workers);
    free(threads);
    
    if (!keep_running) {
        printf("\nProcessing interrupted. Processed %d/%ld molecules.\n",
               atomic_load(&total_processed_count), processor.line_count);
    } else {
        printf("\nProcessing complete. Output written to %s\n", args->output_csv_path);
    }
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
        free(args.colormap_name);
        free(args.output_format_images);
        free(args.smiles_column_name);
        return (parse_result == 0) ? 0 : 1;
    }
    
    // Add this line in main() before the if statement that checks args.num_jobs
    start_time = time(NULL);
    
    int result;
    if (args.num_jobs > 1 && !args.use_streaming) {
        // Multi-threaded parallel processing
        result = process_csv_parallel(&args, args.num_jobs);
    } else {
        // Single-threaded or multi-threaded streaming processing
        result = process_csv_streaming(&args);
    }
    
    free(args.colormap_name);
    free(args.output_format_images);
    free(args.smiles_column_name);
    
    return result;
}