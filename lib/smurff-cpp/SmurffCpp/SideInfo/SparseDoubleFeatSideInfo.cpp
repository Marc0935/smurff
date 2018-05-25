#include "SparseDoubleFeatSideInfo.h"

#include <SmurffCpp/Utils/linop.h>

using namespace smurff;

SparseDoubleFeatSideInfo::SparseDoubleFeatSideInfo(std::shared_ptr<SparseDoubleFeat> side_info)
   : m_side_info(side_info)
{
}


int SparseDoubleFeatSideInfo::cols() const
{
   return m_side_info->cols();
}

int SparseDoubleFeatSideInfo::rows() const
{
   return m_side_info->rows();
}

std::ostream& SparseDoubleFeatSideInfo::print(std::ostream &os) const
{
   double percent = 100.8 * (double)m_side_info->nnz() / (double)m_side_info->rows() / (double) m_side_info->cols();
   os << "SparseDouble " << m_side_info->nnz() << " [" << m_side_info->rows() << ", " << m_side_info->cols() << "] ("
      << percent << "%)" << std::endl;
   return os;
}

bool SparseDoubleFeatSideInfo::is_dense() const
{
   return false;
}

void SparseDoubleFeatSideInfo::compute_uhat(Eigen::MatrixXd& uhat, Eigen::MatrixXd& beta)
{
   smurff::linop::compute_uhat(uhat, *m_side_info, beta);
}

void SparseDoubleFeatSideInfo::At_mul_A(Eigen::MatrixXd& out)
{
   smurff::linop::At_mul_A(out, *m_side_info);
}

Eigen::MatrixXd SparseDoubleFeatSideInfo::A_mul_B(Eigen::MatrixXd& A)
{
   return smurff::linop::A_mul_B(A, *m_side_info);
}

void SparseDoubleFeatSideInfo::solve_blockcg(Eigen::MatrixXd& X, double reg, Eigen::MatrixXd& B, double tol, int max_iter, const int blocksize, const int excess, bool throw_on_cholesky_error)
{
   smurff::linop::solve_blockcg(X, *m_side_info, reg, B, tol, max_iter, blocksize, excess, throw_on_cholesky_error);
}

Eigen::VectorXd SparseDoubleFeatSideInfo::col_square_sum()
{
   return smurff::linop::col_square_sum(*m_side_info);
}

void SparseDoubleFeatSideInfo::At_mul_Bt(Eigen::VectorXd& Y, const int col, Eigen::MatrixXd& B)
{
   smurff::linop::At_mul_Bt(Y, *m_side_info, col, B);
}

void SparseDoubleFeatSideInfo::add_Acol_mul_bt(Eigen::MatrixXd& Z, const int col, Eigen::VectorXd& b)
{
   smurff::linop::add_Acol_mul_bt(Z, *m_side_info, col, b);
}

std::shared_ptr<SparseDoubleFeat> SparseDoubleFeatSideInfo::get_features()
{
   return m_side_info;
}
