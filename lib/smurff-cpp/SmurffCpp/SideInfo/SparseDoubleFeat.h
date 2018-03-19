#pragma once

#include "LibFastSparseDependency.h"

#include "ISideInfo.h"

#include <memory>

namespace smurff {

class SparseDoubleFeat
{
public:
   CSR M;
   CSR Mt;

   SparseDoubleFeat() {}

   SparseDoubleFeat(int nrow, int ncol, long nnz, int* rows, int* cols, double* vals)
   {
      new_csr(&M, nnz, nrow, ncol, rows, cols, vals);
      new_csr(&Mt, nnz, ncol, nrow, cols, rows, vals);
   }

   virtual ~SparseDoubleFeat()
   {
      free_csr(&M);
      free_csr(&Mt);
   }

   int nfeat() const
   {
      return M.ncol;
   }

   int cols() const
   {
      return M.ncol;
   }

   int nsamples() const
   {
      return M.nrow;
   }

   int rows() const
   {
      return M.nrow;
   }
};

class SparseDoubleFeatSideInfo : public ISideInfo
{
private:
   std::shared_ptr<SparseDoubleFeat> m_side_info;

public:
   SparseDoubleFeatSideInfo(std::shared_ptr<SparseDoubleFeat> side_info)
      : m_side_info(side_info)
   {
   }

public:
   int cols() const override
   {
      return m_side_info->cols();
   }

   int rows() const override
   {
      return m_side_info->rows();
   }

public:
   std::ostream& print(std::ostream &os) const override
   {
      os << "SparseDouble [" << m_side_info->rows() << ", " << m_side_info->cols() << "]" << std::endl;
      return os;
   }

   bool is_dense() const
   {
      return false;
   }
};

}