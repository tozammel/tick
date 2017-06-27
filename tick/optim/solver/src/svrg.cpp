#include <prox_separable.h>
#include <model_labels_features.h>
#include "svrg.h"

SVRG::SVRG(ulong epoch_size,
           double tol,
           RandType rand_type,
           double step,
           int seed,
           VarianceReductionMethod variance_reduction,
           DelayedUpdatesMethod delayed_updates
)
    : StoSolver(epoch_size, tol, rand_type, seed),
      step(step), variance_reduction(variance_reduction),
      delayed_updates(delayed_updates) {}

void SVRG::solve() {
  if (model->is_sparse()) {
    if (prox->is_separable()) {
      if (delayed_updates == DelayedUpdatesMethod::Exact) {
        solve_sparse_with_separable_prox_exact_delayed_updates();
      }
      if (delayed_updates == DelayedUpdatesMethod::Proba) {
        solve_sparse_with_separable_prox_proba_updates();
      }
    } else {
      if (delayed_updates == DelayedUpdatesMethod::Exact) {
        solve_sparse_without_separable_prox_exact_delayed_updates();
      }
      if (delayed_updates == DelayedUpdatesMethod::Proba) {
        solve_sparse_with_separable_prox_proba_updates();
      }
    }
  } else {
    solve_dense();
  }
}

void SVRG::prepare_solve() {
  full_gradient = ArrayDouble(iterate.size());
  fixed_w = next_iterate;
  model->grad(fixed_w, full_gradient);
  rand_index = 0;

  if (model->is_sparse()) {
    if (delayed_updates == DelayedUpdatesMethod::Proba) {
      ulong n_features = model->get_n_features();
      ulong n_samples = model->get_n_samples();

      std::shared_ptr<ModelLabelsFeatures> casted_model;
      casted_model = std::dynamic_pointer_cast<ModelLabelsFeatures>(model);

      ArrayULong columns_non_zeros(n_features);
      casted_model->compute_columns_non_zeros(columns_non_zeros);
      steps_lazy = ArrayDouble();
      for (ulong j = 0; j < n_features; ++j) {
        steps_lazy[j] = n_samples / columns_non_zeros[j];
      }
    }
  }

  if (variance_reduction == VarianceReductionMethod::Random ||
      variance_reduction == VarianceReductionMethod::Average) {
    next_iterate.init_to_zero();
  }

  if (variance_reduction == VarianceReductionMethod::Random) {
    rand_index = rand_unif(epoch_size);
  }
}

void SVRG::solve_dense() {
  prepare_solve();

  ArrayDouble grad_i(iterate.size());
  ArrayDouble grad_i_fixed_w(iterate.size());

  for (ulong t = 0; t < epoch_size; ++t) {
    ulong i = get_next_i();
    model->grad_i(i, iterate, grad_i);
    model->grad_i(i, fixed_w, grad_i_fixed_w);
    for (ulong j = 0; j < iterate.size(); ++j) {
      iterate[j] = iterate[j] - step * (grad_i[j] - grad_i_fixed_w[j] + full_gradient[j]);
    }
    prox->call(iterate, step, iterate);

    if (variance_reduction == VarianceReductionMethod::Random && t == rand_index)
      next_iterate = iterate;

    if (variance_reduction == VarianceReductionMethod::Average)
      next_iterate.mult_incr(iterate, 1.0 / epoch_size);
  }
  if (variance_reduction == VarianceReductionMethod::Last)
    next_iterate = iterate;

  t += epoch_size;
}

void SVRG::solve_sparse_without_separable_prox_proba_updates() {
  // Data is sparse and prox is not separable.
  // This means that model is a child of ModelGeneralizedLinear.
  // The strategy used here uses non-delayed probabilistic updates
  prepare_solve();

  ulong n_features = model->get_n_features();
  bool use_intercept = model->use_intercept();

  for (t = 0; t < epoch_size; ++t) {
    // Get next sample index
    ulong i = get_next_i();

    // Sparse features vector
    BaseArrayDouble x_i = model->get_features(i);

    // Gradients factors (model is a GLM)
    // TODO: a grad_i_factor(i, array1, array2) to loop once on the features (once again, it requires new array functions
    double alpha_i_iterate = model->grad_i_factor(i, iterate);
    double alpha_i_fixed_w = model->grad_i_factor(i, fixed_w);
    double delta = alpha_i_iterate - alpha_i_fixed_w;

    // We update the iterate within the support of the features vector, with the probabilistic correction
    for (ulong idx_nnz = 0; idx_nnz < x_i.size_sparse(); ++idx_nnz) {
      // Get the index of the idx-th sparse feature of x_i
      ulong j = x_i.indices()[idx_nnz];
      double full_gradient_j = full_gradient[j];
      // Apply gradient descent to the model weights in the support of x_i
      iterate[j] -= step * steps_lazy[j] * (x_i.data()[idx_nnz] * delta + full_gradient_j);
    }
    // Apply the prox in a non-delayed fashion, since this prox is not separable (too bad for him)
    prox->call(iterate, step, iterate);

    // And let's not forget to update the intercept as well
    if (use_intercept) {
      iterate[n_features] -= step * (delta + full_gradient[n_features]);
      // NB: no lazy-updating for the intercept, and no prox applied on it
    }

    // TODO: Random and Average options for variance reduction are very bad when using lazy updates
    if (variance_reduction == VarianceReductionMethod::Random && t == rand_index) {
      next_iterate = iterate;
    }

    if (variance_reduction == VarianceReductionMethod::Average) {
      next_iterate.mult_incr(iterate, 1.0 / epoch_size);
    }
  }
  t += epoch_size;
  // This is a VERY bad idea when using lazy updates
  if (variance_reduction == VarianceReductionMethod::Last)
    next_iterate = iterate;
}

void SVRG::solve_sparse_without_separable_prox_exact_delayed_updates() {
  // Data is sparse and prox is not separable.
  // This means that model is a child of ModelGeneralizedLinear.
  // The strategy used here uses exact delayed updates, but we cannot delay the penalization
  prepare_solve();

  ulong n_features = model->get_n_features();
  bool use_intercept = model->use_intercept();

  // The array will contain the iteration index of the last update of each
  // coefficient (model-weights and intercept)
  ArrayULong last_time(n_features);
  last_time.fill(0);

  for (t = 0; t < epoch_size; ++t) {
    // Get next sample index
    ulong i = get_next_i();

    // Sparse features vector
    BaseArrayDouble x_i = model->get_features(i);

    // Gradients factors (model is a GLM)
    // TODO: a grad_i_factor(i, array1, array2) to loop once on the features (once again, it requires new array functions
    double alpha_i_iterate = model->grad_i_factor(i, iterate);
    double alpha_i_fixed_w = model->grad_i_factor(i, fixed_w);
    double delta = alpha_i_iterate - alpha_i_fixed_w;

    // We update the iterate within the support of the features vector
    for (ulong idx_nnz = 0; idx_nnz < x_i.size_sparse(); ++idx_nnz) {
      // Get the index of the idx-th sparse feature of x_i
      ulong j = x_i.indices()[idx_nnz];

      // How many iterations since the last update of feature j ?
      ulong delay_j = 0;
      if (t > last_time[j] + 1) {
        delay_j = t - last_time[j] - 1;
      }
      double full_gradient_j = full_gradient[j];

      if (delay_j > 0) {
        // If there is delay, then we need to update coordinate j of the iterate because doing the new update
        // We need to apply the delayed gradient steps for variance reduction
        iterate[j] -= step * delay_j * full_gradient_j;
      }
      // Apply gradient descent to the model weights in the support of x_i
      iterate[j] -= step * (x_i.data()[idx_nnz] * delta + full_gradient_j);

      // Update last_time
      last_time[j] = t;
    }
    // Apply the prox in a non-delayed fashion, since this prox is not separable (too bad for him)
    prox->call(iterate, step, iterate);

    // And let's not forget to update the intercept as well
    if (use_intercept) {
      iterate[n_features] -= step * (delta + full_gradient[n_features]);
      // NB: no lazy-updating for the intercept, and no prox applied on it
    }

    // TODO: Random and Average options for variance reduction are very bad when using lazy updates
    if (variance_reduction == VarianceReductionMethod::Random && t == rand_index) {
      next_iterate = iterate;
    }

    if (variance_reduction == VarianceReductionMethod::Average) {
      next_iterate.mult_incr(iterate, 1.0 / epoch_size);
    }
  }
  // Now we need to fully update the iterate (not the intercept),
  // since we reached the end of the epoch
  for (ulong j = 0; j < n_features; ++j) {
    ulong delay_j = 0;
    if (t > last_time[j] + 1) {
      delay_j = t - last_time[j] - 1;
    }
    if (delay_j > 0) {
      // If there is delay, then we need to update coordinate j of the iterate first
      // We need to apply the delayed gradient steps for variance reduction
      iterate[j] -= step * delay_j * full_gradient[j];
    }
  }
  t += epoch_size;
  // This is a VERY bad idea when using lazy updates
  if (variance_reduction == VarianceReductionMethod::Last)
    next_iterate = iterate;
}

void SVRG::solve_sparse_with_separable_prox_proba_updates() {
}

void SVRG::solve_sparse_with_separable_prox_exact_delayed_updates() {
  // Data is sparse and prox is separable.
  // This means that model is a child of ModelGeneralizedLinear.
  // The strategy used here uses the exact delayed updates and
  // penalization trick: with such a model and prox, we can work only inside the current support
  // (non-zero values) of the sampled vector of features
  prepare_solve();

  ulong n_features = model->get_n_features();
  bool use_intercept = model->use_intercept();

  // We need the call_single method of ProxSeparable
  std::shared_ptr<ProxSeparable> casted_prox;
  casted_prox = std::static_pointer_cast<ProxSeparable>(prox);

  // The array will contain the iteration index of the last update of each
  // coefficient (model-weights and intercept)
  ArrayULong last_time(n_features);
  last_time.fill(0);

  for (t = 0; t < epoch_size; ++t) {
    // Get next sample index
    ulong i = get_next_i();

    // Sparse features vector
    BaseArrayDouble x_i = model->get_features(i);

    // Gradients factors (model is a GLM)
    // TODO: a grad_i_factor(i, array1, array2) to loop once on the features (once again, it requires new array functions
    double alpha_i_iterate = model->grad_i_factor(i, iterate);
    double alpha_i_fixed_w = model->grad_i_factor(i, fixed_w);
    double delta = alpha_i_iterate - alpha_i_fixed_w;

    // We update the iterate within the support of the features vector
    for (ulong idx_nnz = 0; idx_nnz < x_i.size_sparse(); ++idx_nnz) {
      // Get the index of the idx-th sparse feature of x_i
      ulong j = x_i.indices()[idx_nnz];

      // How many iterations since the last update of feature j ?
      ulong delay_j = 0;
      if (t > last_time[j] + 1) {
        delay_j = t - last_time[j] - 1;
      }
      double full_gradient_j = full_gradient[j];

      if (delay_j > 0) {
        // If there is delay, then we need to update coordinate j of the iterate because doing the new update
        // We need to apply the delayed gradient steps for variance reduction
        iterate[j] -= step * delay_j * full_gradient_j;

        // And we need to apply the delayed regularization
        iterate[j] = casted_prox->call_single(iterate[j], step, delay_j);
      }
      // Apply gradient descent to the model weights in the support of x_i
      iterate[j] -= step * (x_i.data()[idx_nnz] * delta + full_gradient_j);

      // Regularize the features of the model weights in the support of x_i
      iterate[j] = casted_prox->call_single(iterate[j], step);

      // Update last_time
      last_time[j] = t;
    }
    // And let's not forget to update the intercept as well
    if (use_intercept) {
      iterate[n_features] -= step * (delta + full_gradient[n_features]);
      // NB: no lazy-updating for the intercept, and no prox applied on it
    }

    // TODO: Random and Average options for variance reduction are very bad when using lazy updates
    if (variance_reduction == VarianceReductionMethod::Random && t == rand_index) {
      next_iterate = iterate;
    }

    if (variance_reduction == VarianceReductionMethod::Average) {
      next_iterate.mult_incr(iterate, 1.0 / epoch_size);
    }
  }
  // Now we need to fully update the iterate (not the intercept),
  // since we reached the end of the epoch
  for (ulong j = 0; j < n_features; ++j) {
    ulong delay_j = 0;
    if (t > last_time[j] + 1) {
      delay_j = t - last_time[j] - 1;
    }
    if (delay_j > 0) {
      // If there is delay, then we need to update coordinate j of the iterate first
      // We need to apply the delayed gradient steps for variance reduction
      iterate[j] -= step * delay_j * full_gradient[j];
      // And we need to apply the delayed regularization
      iterate[j] = casted_prox->call_single(iterate[j], step, delay_j);
    }
  }
  t += epoch_size;
  // This is a VERY bad idea when using lazy updates
  if (variance_reduction == VarianceReductionMethod::Last)
    next_iterate = iterate;
}

void SVRG::set_starting_iterate(ArrayDouble &new_iterate) {
  StoSolver::set_starting_iterate(new_iterate);
  next_iterate = iterate;
}
