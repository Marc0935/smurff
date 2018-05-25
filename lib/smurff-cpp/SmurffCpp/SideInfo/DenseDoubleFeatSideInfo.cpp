#include "DenseDoubleFeatSideInfo.h"

#include <SmurffCpp/Utils/linop.h>

using namespace smurff;

DenseDoubleFeatSideInfo::DenseDoubleFeatSideInfo(std::shared_ptr<Eigen::MatrixXd> side_info)
   : m_side_info(side_info)
{
}

int DenseDoubleFeatSideInfo::cols() const
{
   return m_side_info->cols();
}

int DenseDoubleFeatSideInfo::rows() const
{
   return m_side_info->rows();
}

std::ostream& DenseDoubleFeatSideInfo::print(std::ostream &os) const
{
   os << "DenseDouble [" << m_side_info->rows() << ", " << m_side_info->cols() << "]" << std::endl;
   return os;
}

bool DenseDoubleFeatSideInfo::is_dense() const
{
   return true;
}

void DenseDoubleFeatSideInfo::compute_uhat(Eigen::MatrixXd& uhat, Eigen::MatrixXd& beta)
{
   smurff::linop::compute_uhat(uhat, *m_side_info, beta);
}

void DenseDoubleFeatSideInfo::At_mul_A(Eigen::MatrixXd& out)
{
   smurff::linop::At_mul_A(out, *m_side_info);
}

Eigen::MatrixXd DenseDoubleFeatSideInfo::A_mul_B(Eigen::MatrixXd& A)
{
   return smurff::linop::A_mul_B(A, *m_side_info);
}

void DenseDoubleFeatSideInfo::solve_blockcg(Eigen::MatrixXd& X, double reg, Eigen::MatrixXd& B, double tol, int max_iter, const int blocksize, const int excess, bool throw_on_cholesky_error)
{
   smurff::linop::solve_blockcg(X, *m_side_info, reg, B, tol, max_iter, blocksize, excess, throw_on_cholesky_error);
}

Eigen::VectorXd DenseDoubleFeatSideInfo::col_square_sum()
{
   return smurff::linop::col_square_sum(*m_side_info);
}

void DenseDoubleFeatSideInfo::At_mul_Bt(Eigen::VectorXd& Y, const int col, Eigen::MatrixXd& B)
{
   smurff::linop::At_mul_Bt(Y, *m_side_info, col, B);
}

void DenseDoubleFeatSideInfo::add_Acol_mul_bt(Eigen::MatrixXd& Z, const int col, Eigen::VectorXd& b)
{
   smurff::linop::add_Acol_mul_bt(Z, *m_side_info, col, b);
}

std::shared_ptr<Eigen::MatrixXd> DenseDoubleFeatSideInfo::get_features()
{
   return m_side_info;
}
