#ifndef FEATURE_IMAGE_H
#define FEATURE_IMAGE_H

#include "molecule.h"
#include <stdbool.h>

// Use the MAX_FEATURE_CHANNELS defined in molecule.h instead of redefining it
// #define MAX_FEATURE_CHANNELS 50

// Only define CUDA types if not already defined elsewhere
#if !defined(CUDA_TYPES_DEFINED) && !defined(HAS_CUDA) && !defined(__CUDACC__)
#define CUDA_TYPES_DEFINED
typedef enum {
    cudaSuccess = 0
} cudaError_t;
#endif

// A multi-channel image representing molecular features
typedef struct {
    int width;
    int height;
    int channels;
    float *data;       // Host data (width * height * channels)
    float *d_data;     // Device data (CUDA)
    int channel_types[MAX_FEATURE_CHANNELS];
} MultiChannelImage;

// Feature channel information
typedef struct {
    int type;
    const char* name;
    const char* description;
    bool requires_3d;
} FeatureChannelInfo;

// Progress callback function type
typedef void (*ProgressCallback)(float progress);

// Core image functions
MultiChannelImage* create_multi_channel_image(int width, int height, int channels);
void free_multi_channel_image(MultiChannelImage* image);
bool generate_feature_image(MultiChannelImage* image, AtomPos *atoms, int atom_count, 
                           BondSeg *bonds, int bond_count, ProgressCallback progress_callback);
bool save_feature_image(const MultiChannelImage* image, const char* filename);

// Molecular property calculation functions
void calculate_gasteiger_charges(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void calculate_atomic_properties(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void calculate_pharmacophore_features(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count,
                                   PharmacophorePoint *points, int *point_count);
void calculate_topological_features(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count,
                                 TopologicalFeature *features, int *feature_count);
void identify_rotatable_bonds(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void orient_molecule_to_principal_axes(AtomPos *atoms, int atom_count);
void get_molecule_bounds(AtomPos *atoms, int atom_count, 
                        double *min_x, double *min_y, double *min_z,
                        double *max_x, double *max_y, double *max_z);
void center_molecule(AtomPos *atoms, int atom_count);
void normalize_feature_channels(MultiChannelImage* image);

// Channel information functions
const char* get_channel_name(int channel_type);
int get_channel_type_from_name(const char* name);
void get_available_channel_names(char** names, int* count);
FeatureChannelInfo get_channel_info(int channel_type);

// Advanced molecular property functions
void estimate_3d_coordinates(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void detect_stereocenters(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void apply_3d_constraints(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void calculate_solvent_accessibility(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void calculate_topological_indices(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count,
                                  double *wiener_index, double *balaban_index, double *randic_index);
void identify_ring_systems(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void analyze_conformational_flexibility(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void optimize_3d_geometry(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count, 
                         double *energy, double convergence_threshold);
void calculate_molecular_surfaces(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count,
                                 double probe_radius, double *molecular_surface, 
                                 double *solvent_accessible_surface);
void predict_binding_hotspots(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count,
                             double **hotspot_coords, int *hotspot_count);
void calculate_simplified_orbital_interactions(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void estimate_charge_densities(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void approximate_resonance_effects(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void calculate_hybridization_state(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void calculate_atom_contributions(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void calculate_bond_contributions(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void calculate_intramolecular_interactions(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void identify_pharmacophore_patterns(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void calculate_scaffold_decomposition(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void calculate_atom_centered_fragments(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void calculate_three_dimensional_descriptors(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);
void identify_functional_groups(AtomPos *atoms, int atom_count, BondSeg *bonds, int bond_count);

#endif // FEATURE_IMAGE_H 