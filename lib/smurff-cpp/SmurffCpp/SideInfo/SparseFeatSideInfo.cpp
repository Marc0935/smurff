#include "SparseFeatSideInfo.h"

#include <SmurffCpp/Utils/linop.h>

using namespace smurff;

SparseFeatSideInfo::SparseFeatSideInfo(std::shared_ptr<SparseFeat> side_info)
   : m_side_info(side_info)
{
}

int SparseFeatSideInfo::cols() const
{
   return m_side_info->cols();
}

int SparseFeatSideInfo::rows() const
{
   return m_side_info->rows();
}

std::ostream& SparseFeatSideInfo::print(std::ostream &os) const
{
   double percent = 100.8 * (double)m_side_info->nnz() / (double)m_side_info->rows() / (double)m_side_info->cols();
   os << "SparseBinary " << m_side_info->nnz() << " [" << m_side_info->rows() << ", " << m_side_info->cols() << "] ("
      << percent << "%)" << std::endl;
   return os;
}

bool SparseFeatSideInfo::is_dense() const
{
   return false;
}

void SparseFeatSideInfo::compute_uhat(Eigen::MatrixXd& uhat, Eigen::MatrixXd& beta)
{
   smurff::linop::compute_uhat(uhat, *m_side_info, beta);
}

void SparseFeatSideInfo::At_mul_A(Eigen::MatrixXd& out)
{
   smurff::linop::At_mul_A(out, *m_side_info);
}

Eigen::MatrixXd SparseFeatSideInfo::A_mul_B(Eigen::MatrixXd& A)
{
   return smurff::linop::A_mul_B(A, *m_side_info);
}

void SparseFeatSideInfo::solve_blockcg(Eigen::MatrixXd& X, double reg, Eigen::MatrixXd& B, double tol, int max_iter, const int blocksize, const int excess, bool throw_on_cholesky_error)
{
   smurff::linop::solve_blockcg(X, *m_side_info, reg, B, tol, max_iter, blocksize, excess, throw_on_cholesky_error);
}

Eigen::VectorXd SparseFeatSideInfo::col_square_sum()
{
   return smurff::linop::col_square_sum(*m_side_info);
}

void SparseFeatSideInfo::At_mul_Bt(Eigen::VectorXd& Y, const int col, Eigen::MatrixXd& B)
{
   smurff::linop::At_mul_Bt(Y, *m_side_info, col, B);
}

void SparseFeatSideInfo::add_Acol_mul_bt(Eigen::MatrixXd& Z, const int col, Eigen::VectorXd& b)
{
   smurff::linop::add_Acol_mul_bt(Z, *m_side_info, col, b);
}

std::shared_ptr<SparseFeat> SparseFeatSideInfo::get_features()
{
   return m_side_info;
}
