#pragma once

#include <array>
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <SmurffCpp/Configs/TensorConfig.h>
#include <SmurffCpp/Configs/MatrixConfig.h>

namespace smurff { namespace tensor_utils {

Eigen::MatrixXd dense_to_eigen(const smurff::TensorConfig& tensorConfig);

Eigen::MatrixXd dense_to_eigen(smurff::TensorConfig& tensorConfig);

template<typename Tensor>
Eigen::SparseMatrix<double> sparse_to_eigen(Tensor& Y);

// Conversion of TensorConfig to sparse eigen matrix

template<>
Eigen::SparseMatrix<double> sparse_to_eigen<const smurff::TensorConfig>(const smurff::TensorConfig& tensorConfig);

template<>
Eigen::SparseMatrix<double> sparse_to_eigen<smurff::TensorConfig>(smurff::TensorConfig& tensorConfig);

template<typename EigenType>
EigenType config_to_eigen(const smurff::TensorConfig& tensorConfig); 

template<>
inline Eigen::SparseMatrix<double> config_to_eigen(const smurff::TensorConfig& tensorConfig) {
   return sparse_to_eigen(tensorConfig);
}

template<>
inline Eigen::MatrixXd config_to_eigen(const smurff::TensorConfig& tensorConfig) {
   return dense_to_eigen(tensorConfig);
}

// Conversion of tensor config to matrix config

smurff::MatrixConfig tensor_to_matrix(const smurff::TensorConfig& tensorConfig);

// Print tensor config to console

std::ostream& operator << (std::ostream& os, const TensorConfig& tc);

// Take a matrix slice of tensor by fixing specific dimensions

Eigen::MatrixXd slice(const TensorConfig& tensorConfig
   , const std::array<std::uint64_t, 2>& fixedDims
   , const std::unordered_map<std::uint64_t, std::uint32_t>& dimCoords
    );

}}
