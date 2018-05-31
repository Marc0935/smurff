#include "catch.hpp"

#include <SmurffCpp/Utils/linop.h>
#include <SmurffCpp/Utils/Distribution.h>

using namespace smurff;

static NoiseConfig fixed_ncfg(NoiseTypes::fixed);

TEST_CASE( "SparseMatrix/solve_blockcg", "BlockCG solver (1rhs)" ) 
{
   std::vector<uint32_t> rows = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
   std::vector<uint32_t> cols = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
   MatrixConfig sf(6, 4, rows, cols);
   Eigen::SparseMatrix<double> F = matrix_utils::sparse_to_eigen(sf);
   Eigen::SparseMatrix<double> Ft = F.transpose();
   
   Eigen::MatrixXd B(1, 4), X(1, 4), X_true(1, 4);
 
   B << 0.56,  0.55,  0.3 , -1.78;
   X_true << 0.35555556,  0.40709677, -0.16444444, -0.87483871;

   auto op = [&F, &Ft](const Eigen::MatrixXd &X) -> Eigen::MatrixXd
   {
      return (Ft * (F * X.transpose())).transpose() + 0.5 * X;
   }; 
   
   int niter = smurff::linop::solve_blockcg(X, op, B, 1e-6, 1000);
   for (int i = 0; i < X.rows(); i++) {
     for (int j = 0; j < X.cols(); j++) {
       REQUIRE( X(i,j) == Approx(X_true(i,j)) );
     }
   }
   REQUIRE( niter <= 4);
}

TEST_CASE( "SparseFeat/solve_blockcg_1_0", "BlockCG solver (3rhs separately)" ) 
{
   std::vector<uint32_t> rows = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
   std::vector<uint32_t> cols = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
   MatrixConfig sf(6, 4, rows, cols);
   Eigen::SparseMatrix<double> F = matrix_utils::sparse_to_eigen(sf);
   Eigen::SparseMatrix<double> Ft = F.transpose();

   Eigen::MatrixXd B(3, 4), X(3, 4), X_true(3, 4);
 
   B << 0.56,  0.55,  0.3 , -1.78,
        0.34,  0.05, -1.48,  1.11,
        0.09,  0.51, -0.63,  1.59;
 
   X_true << 0.35555556,  0.40709677, -0.16444444, -0.87483871,
             1.69333333, -0.12709677, -1.94666667,  0.49483871,
             0.66      , -0.04064516, -0.78      ,  0.65225806;

   auto op = [&F, &Ft](const Eigen::MatrixXd &X) -> Eigen::MatrixXd
   {
      return (Ft * (F * X.transpose())).transpose() + 0.5 * X;
   };
 
   smurff::linop::solve_blockcg(X, op, B, 1e-6, 1000, 1, 0);
   for (int i = 0; i < X.rows(); i++) {
     for (int j = 0; j < X.cols(); j++) {
       REQUIRE( X(i,j) == Approx(X_true(i,j)) );
     }
   }
}

TEST_CASE( "linop/solve_blockcg_dense/ok", "BlockCG solver for dense (3rhs separately)" ) 
{
   double reg = 0.5;

   Eigen::MatrixXd KK(6, 6);
   KK <<  1.7488399 , -1.92816395, -1.39618642, -0.2769755 , -0.52815529, 0.24624319,
        -1.92816395,  3.34435465,  2.07258617,  0.4417173 ,  0.84673143, -0.35075244,
        -1.39618642,  2.07258617,  2.1623261 ,  0.25923918,  0.64428255, -0.2329581,
        -0.2769755 ,  0.4417173 ,  0.25923918,  0.6147927 ,  0.15112057, -0.00692033,
        -0.52815529,  0.84673143,  0.64428255,  0.15112057,  0.80141217, -0.19682322,
         0.24624319, -0.35075244, -0.2329581 , -0.00692033, -0.19682322, 0.56240547;

   Eigen::MatrixXd K = KK.llt().matrixU();

   REQUIRE(((K.transpose() * K) - KK).norm() < 1e-3);

   Eigen::MatrixXd X_true(3, 6);
   X_true << 0.35555556,  0.40709677, -0.16444444, -0.87483871, -0.16444444, -0.87483871,
             1.69333333, -0.12709677, -1.94666667,  0.49483871, -1.94666667,  0.49483871,
             0.66      , -0.04064516, -0.78      ,  0.65225806, -0.78      ,  0.65225806;

   Eigen::MatrixXd B = ((K.transpose() * K + Eigen::MatrixXd::Identity(6,6) * reg) * X_true.transpose()).transpose();
   Eigen::MatrixXd X(3, 6);

   auto op = [&K](const Eigen::MatrixXd &X) -> Eigen::MatrixXd
   {
      return (K.transpose() * (K * X.transpose())).transpose() + 0.5 * X;
   };

   //-- Solves the system (K' * K + reg * I) * X = B for X for m right-hand sides
   smurff::linop::solve_blockcg(X, op, B, 1e-6, 1000, true);

   for (int i = 0; i < X.rows(); i++) {
     for (int j = 0; j < X.cols(); j++) {
       REQUIRE( X(i,j) == Approx(X_true(i,j)) );
     }
   }
}
