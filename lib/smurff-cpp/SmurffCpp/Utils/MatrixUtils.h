#pragma once

#include <limits>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <SmurffCpp/Configs/MatrixConfig.h>

#include <SmurffCpp/Utils/Error.h>

namespace smurff { namespace matrix_utils {
   // Conversion of MatrixConfig to sparse eigen matrix

   Eigen::SparseMatrix<double> sparse_to_eigen(const smurff::MatrixConfig& matrixConfig);

   // Conversion of dense data to dense eigen matrix - do we need it? (sparse eigen matrix can be converted to dense eigen matrix with = operator)

   Eigen::MatrixXd dense_to_eigen(const smurff::MatrixConfig& matrixConfig);

   Eigen::MatrixXd dense_to_eigen(smurff::MatrixConfig& matrixConfig);

   std::ostream& operator << (std::ostream& os, const MatrixConfig& mc);

   bool equals(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2, double precision = std::numeric_limits<double>::epsilon());

   bool equals_vector(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2, double precision = std::numeric_limits<double>::epsilon() * 100);
}}
