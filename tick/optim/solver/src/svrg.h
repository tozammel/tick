//
// Created by Martin Bompaire on 23/10/15.
//

#ifndef TICK_OPTIM_SOLVER_SRC_SVRG_H_
#define TICK_OPTIM_SOLVER_SRC_SVRG_H_

#include "array.h"
#include "sgd.h"
#include "../../prox/src/prox.h"

class SVRG : public StoSolver {
 public:
  enum class VarianceReductionMethod {
    Last = 1,
    Average = 2,
    Random = 3,
  };

  enum class DelayedUpdatesMethod {
    Exact = 1,
    Proba = 2,
  };

 private:
  double step;
  VarianceReductionMethod variance_reduction;
  DelayedUpdatesMethod delayed_updates;
  ArrayDouble next_iterate;
  // An array to save the full gradient used for variance reduction
  ArrayDouble full_gradient;

  ArrayDouble fixed_w;

  ulong rand_index;

  // Probabilistic correction of the step-sizes of all model weights,
  // given by the inverse proportion of non-zero entries in each feature column
  ArrayDouble steps_lazy;

  void prepare_solve();

  void solve_dense();

  void solve_sparse_with_separable_prox_exact_delayed_updates();

  void solve_sparse_with_separable_prox_proba_updates();

  void solve_sparse_without_separable_prox_exact_delayed_updates();

  void solve_sparse_without_separable_prox_proba_updates();

 public:
  SVRG(ulong epoch_size,
       double tol,
       RandType rand_type,
       double step,
       int seed = -1,
       VarianceReductionMethod variance_reduction = VarianceReductionMethod::Last,
       DelayedUpdatesMethod delayed_updates = DelayedUpdatesMethod::Exact);

  void solve() override;

  double get_step() const {
    return step;
  }

  void set_step(double step) {
    SVRG::step = step;
  }

  VarianceReductionMethod get_variance_reduction() const {
    return variance_reduction;
  }

  DelayedUpdatesMethod get_delayed_updates() const {
    return delayed_updates;
  }

  void set_variance_reduction(VarianceReductionMethod variance_reduction) {
    SVRG::variance_reduction = variance_reduction;
  }

  void set_delayed_updates(DelayedUpdatesMethod delayed_updates) {
    this->delayed_updates = delayed_updates;
  }

  void set_starting_iterate(ArrayDouble &new_iterate) override;
};

#endif  // TICK_OPTIM_SOLVER_SRC_SVRG_H_
