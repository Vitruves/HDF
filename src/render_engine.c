#include "render_engine.h"
#include "globals.h" // For AtomPos etc. if needed, or general constants
#include "utils.h"   // Added for clamp_double
#include <stdio.h>   // For FILE, fopen, fclose, fprintf, fputc
#include <math.h>    // For log10, fmax, fmin, pow, ceil, floor
#include <stdlib.h>  // Added back for malloc, free
#include <string.h>  // Added back for strcmp

void set_render_colormap(ColormapType type) {
    colormap_global = type;
}

ColormapType get_render_colormap(void) {
    return colormap_global;
}

void apply_colormap_to_pixel(double intensity_normalized, int *r_out, int *g_out, int *b_out) {
    double val = clamp_double(intensity_normalized, 0.0, 1.0);
    int r=0, g=0, b=0;

    switch (colormap_global) {
        case COLORMAP_GRAYSCALE:
            r = g = b = (int)(255.0 * val);
            break;
        case COLORMAP_JET: // Approximate Jet
            if (val < 0.375) {        // Dark blue to blue
                r = 0; g = 0; b = (int)(255 * (0.5 + val / 0.375 * 0.5));
            } else if (val < 0.625) { // Blue to cyan to green
                r = 0; g = (int)(255 * ((val - 0.375) / 0.25)); b = 255;
            } else if (val < 0.875) { // Green to yellow to red
                r = (int)(255 * ((val - 0.625) / 0.25)); g = 255; b = (int)(255 * (1.0 - (val - 0.625) / 0.25));
            } else {                  // Red to dark red
                r = 255; g = (int)(255 * (1.0 - (val - 0.875) / 0.125)); b = 0;
            }
            // Clamp final values
            r = (int)clamp_double(r,0,255); g = (int)clamp_double(g,0,255); b = (int)clamp_double(b,0,255);
            break;
        case COLORMAP_VIRIDIS: // Simplified Viridis approximation
            r = (int)(255.0 * sqrt(val));
            g = (int)(255.0 * pow(val, 1.5)); // Emphasize green a bit more in mid-range
            b = (int)(255.0 * (val < 0.5 ? (1.0 - val * 1.5) : (0.25 * (1.0 - val))));
             r = (int)clamp_double(r*0.8 + 0.2*val*255, 0, 255); // Mix to get purple->blue start
             g = (int)clamp_double(g*0.9, 0, 255);
             b = (int)clamp_double(b*0.7 + 0.3*(1-val)*255, 0, 255);
             // Viridis: #440154 -> #21908d -> #fde725
            // r = (int)(255.0 * (0.267004*val*val*val - 0.129193*val*val + 1.031060*val + 0.200498)); // These are poly approx.
            // g = (int)(255.0 * (-1.366149*val*val*val + 2.809703*val*val - 0.090472*val + 0.059897));
            // b = (int)(255.0 * (0.750097*val*val*val - 1.123993*val*val + 0.105150*val + 0.301029));
            // Simplified based on visual progression:
            r = (int)(255 * (0.27 + 0.78 * val - 0.1 * val*val));  // Adjusted for Viridis feel
            g = (int)(255 * (0.0 + 1.12 * val - 0.35 * val*val));
            b = (int)(255 * (0.33 + 0.2 * val - 0.7 * val*val + 0.3 * val*val*val));
            break;
        case COLORMAP_PLASMA: // Simplified Plasma approximation
            // Plasma: #0d0887 -> #f0f921
            r = (int)(255.0 * pow(val, 0.8));
            g = (int)(255.0 * pow(val, 1.5));
            b = (int)(255.0 * (1.0 - val) * 0.8 + 0.2 * pow(1.0-val, 0.5)*255 );
            r = (int)(255 * (0.05 + 0.95*val -0.3*val*val + 0.3*pow(val,0.5)));
            g = (int)(255 * (0.02 + 0.3*val + 0.6*pow(val,2.5) - 0.2*pow(val,0.5)));
            b = (int)(255 * (0.52 - 0.4*val - 0.1*pow(val,0.5) + 0.1*pow(val,2.0)));

            break;
        case COLORMAP_HEAT:
            r = (int)(255.0 * clamp_double(3.0 * val, 0.0, 1.0));
            g = (int)(255.0 * clamp_double(3.0 * val - 1.0, 0.0, 1.0));
            b = (int)(255.0 * clamp_double(3.0 * val - 2.0, 0.0, 1.0));
            break;
        default:
            r = g = b = (int)(255.0 * val);
    }
    *r_out = (int)clamp_double(r, 0, 255);
    *g_out = (int)clamp_double(g, 0, 255);
    *b_out = (int)clamp_double(b, 0, 255);
}

double apply_log_scale_intensity(double value, double max_value, double min_value_for_log) {
    if (max_value <= min_value_for_log) return 0.0; // Avoid log of non-positive or log(1) if max is small
    // Ensure value + min_value_for_log is positive
    double val_to_log = value;
    if (val_to_log < 0) val_to_log = 0; // Ensure non-negative before adding epsilon

    double log_val = log(val_to_log + min_value_for_log);
    double log_max = log(max_value + min_value_for_log);
    double log_min_epsilon = log(min_value_for_log); // Log of the "floor"

    if (log_max <= log_min_epsilon) return 0.0; // Avoid division by zero or negative result
    
    double scaled = (log_val - log_min_epsilon) / (log_max - log_min_epsilon);
    return clamp_double(scaled, 0.0, 1.0);
}


void output_diffraction_image(FILE *outfile, double *intensity_data, int width, int height, bool is_color_output, const char* output_format_str) {
    if (!outfile) {
        fprintf(stderr, "Error: Output file stream is NULL in output_diffraction_image.\n");
        return;
    }
    
    double max_intensity = 0.0;
    double min_intensity = intensity_data[0]; // For dynamic range in log scaling
    for (int i = 0; i < width * height; i++) {
        if (intensity_data[i] > max_intensity) max_intensity = intensity_data[i];
        if (intensity_data[i] < min_intensity) min_intensity = intensity_data[i];
    }
    // Small epsilon for log scaling, relative to max intensity or a fixed small number
    double epsilon_log = max_intensity * 1e-7; 
    if (epsilon_log < 1e-10) epsilon_log = 1e-10;


    // Determine if actual color output is PPM based on format_str
    bool use_ppm = is_color_output && (strcmp(output_format_str, "ppm") == 0);

    if (use_ppm) {
        fprintf(outfile, "P6\n%d %d\n255\n", width, height); // PPM header
    } else { // PGM output
        fprintf(outfile, "P5\n%d %d\n255\n", width, height); // PGM header
    }

    unsigned char *byte_buffer = malloc(use_ppm ? width * 3 : width);
    if(!byte_buffer){
        fprintf(stderr, "Error: Failed to allocate byte buffer for image output.\n");
        return;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double val = intensity_data[y * width + x];
            double scaled_val = apply_log_scale_intensity(val, max_intensity, epsilon_log);
            
            if (use_ppm) {
                int r, g, b;
                apply_colormap_to_pixel(scaled_val, &r, &g, &b);
                byte_buffer[x * 3 + 0] = (unsigned char)r;
                byte_buffer[x * 3 + 1] = (unsigned char)g;
                byte_buffer[x * 3 + 2] = (unsigned char)b;
            } else {
                byte_buffer[x] = (unsigned char)(255.0 * scaled_val);
            }
        }
        fwrite(byte_buffer, use_ppm ? 3 : 1, width, outfile);
    }
    free(byte_buffer);
}