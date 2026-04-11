#ifndef LKF_ASM_H
#define LKF_ASM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns the size in bytes needed for the LKF state struct */
size_t lkf_sizeof(void);

/* Initializes the LKF struct with a given time step (dt) */
void lkf_init(void *lkf, double dt);

/* Sets the initial position for a specific joint (0-22) */
void lkf_set_initial_state(void *lkf, int joint_idx, const double *pos3);

/* Performs the prediction step: x = Fx, P = FPF^T + Q */
void lkf_predict(void *lkf);

/* Performs the update step using a flat measurement vector (69 elements) */
void lkf_update(void *lkf, const double *z_flat);

/* Extracts only the position components (69 doubles) from the state */
void lkf_get_positions(const void *lkf, double *pos_out);

/* Extracts the full state vector (276 doubles) */
void lkf_get_full_state(const void *lkf, double *out);

#ifdef __cplusplus
}
#endif

#endif /* LKF_ASM_H */