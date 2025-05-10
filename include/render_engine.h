#ifndef RENDER_ENGINE_H
#define RENDER_ENGINE_H

#include "molecule.h" // For ColormapType
#include <stdio.h>    // For FILE*

void set_render_colormap(ColormapType type);
ColormapType get_render_colormap(void);
void apply_colormap_to_pixel(double intensity_normalized, int *r, int *g, int *b);
double apply_log_scale_intensity(double value, double max_value, double min_value_for_log);
void output_diffraction_image(FILE *outfile, double *intensity_data, int width, int height, bool is_color_ppm, const char* output_format_str);

// TODO: Add functions for other output formats or visualizations if needed
// - Save to PNG, JPG (would require external libraries like libpng, libjpeg)
// - Vector graphics output (SVG)

#endif // RENDER_ENGINE_H