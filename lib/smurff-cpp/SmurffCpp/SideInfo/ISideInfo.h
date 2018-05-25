#pragma once

#include <iostream>

#include <Eigen/Dense>

namespace smurff {

   class ISideInfo
   {
   public:
      virtual ~ISideInfo() {}

      virtual int cols() const = 0;

      virtual int rows() const = 0;

      virtual std::ostream& print(std::ostream &os) const = 0;

      virtual bool is_dense() const = 0;

   public:
      //linop

      virtual void compute_uhat(Eigen::MatrixXd& uhat, Eigen::MatrixXd& beta) = 0;

      virtual void At_mul_A(Eigen::MatrixXd& out) = 0;

      virtual Eigen::MatrixXd A_mul_B(Eigen::MatrixXd& A) = 0;

      virtual void solve_blockcg(Eigen::MatrixXd& X, double reg, Eigen::MatrixXd& B, double tol, int max_iter, const int blocksize, const int excess, bool throw_on_cholesky_error = false) = 0;

      virtual Eigen::VectorXd col_square_sum() = 0;

      virtual void At_mul_Bt(Eigen::VectorXd& Y, const int col, Eigen::MatrixXd& B) = 0;

      virtual void add_Acol_mul_bt(Eigen::MatrixXd& Z, const int col, Eigen::VectorXd& b) = 0;
   };

}
