#ifndef EKF_VECTOR_H
#define EKF_VECTOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t ekf_vec_sizeof(void);
void ekf_vec_init(void *ekf, double dt);
void ekf_vec_set_initial_state(void *ekf, int joint_idx, const double *pos3);
void ekf_vec_predict(void *ekf);
void ekf_vec_update(void *ekf, const double *z_flat);
void ekf_vec_get_positions(const void *ekf, double *pos_out);
void ekf_vec_get_full_state(const void *ekf, double *out);

#ifdef __cplusplus
}
#endif

#endif /* EKF_VECTOR_H */
