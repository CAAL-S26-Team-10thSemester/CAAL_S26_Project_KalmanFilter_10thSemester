#ifndef LKF_VECTOR_H
#define LKF_VECTOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t lkf_vec_sizeof(void);
void lkf_vec_init(void *lkf, double dt);
void lkf_vec_set_initial_state(void *lkf, int joint_idx, const double *pos3);
void lkf_vec_predict(void *lkf);
void lkf_vec_update(void *lkf, const double *z_flat);
void lkf_vec_get_positions(const void *lkf, double *pos_out);
void lkf_vec_get_full_state(const void *lkf, double *out);

#ifdef __cplusplus
}
#endif

#endif /* LKF_VECTOR_H */
