//
// Created by Stéphane GAIFFAS on 06/12/2015.
//

#include "model_labels_features.h"

ModelLabelsFeatures::ModelLabelsFeatures(SBaseArrayDouble2dPtr features,
                                         SArrayDoublePtr labels)
    : n_samples(labels.get() ? labels->size() : 0),
      n_features(features.get() ? features->n_cols() : 0),
      labels(labels),
      features(features),
      ready_columns_sparsity(false) {
  if (labels.get() && labels->size() != features->n_rows()) {
    std::stringstream ss;
    ss << "In ModelLabelsFeatures, number of labels is " << labels->size();
    ss << " while the features matrix has " << features->n_rows() << " rows.";
    throw std::invalid_argument(ss.str());
  }
}


void ModelLabelsFeatures::compute_columns_non_zeros(ArrayULong &columns_sparsity) {
  if (!ready_columns_sparsity) {
    if (features->is_sparse()) {
      if (columns_sparsity.size() != n_samples) {
        TICK_ERROR("given `columns_sparsity` vector must match `n_samples`")
      }
      columns_sparsity.fill(0.);
      for (ulong i = 0; i < n_samples; ++i) {
        BaseArrayDouble features_i = get_features(i);
        for (ulong j = 0; j < features_i.size_sparse(); ++j) {
          // If the entry is indeed non-zero (nothing forbids to store zeros...) increment
          // the number of non-zeros of the columns
          if (std::abs(features_i.data()[j]) > 0) {
            columns_sparsity[features_i.indices()[j]] += 1;
          }
        }
      }
    } else {
      TICK_ERROR("The features matrix is not sparse.")
    }
  }
}
