#include "ScarceMatrixData.h"

#include <SmurffCpp/Utils/counters.h>
#include <SmurffCpp/VMatrixExprIterator.hpp>
#include <SmurffCpp/ConstVMatrixExprIterator.hpp>

#include <SmurffCpp/Utils/ThreadVector.hpp>

using namespace smurff;
using namespace Eigen;

ScarceMatrixData::ScarceMatrixData(SparseMatrix<double> Y)
   : MatrixDataTempl<SparseMatrix<double> >(Y)
{
   name = "ScarceMatrixData [with NAs]";
}

void ScarceMatrixData::init_pre()
{
   MatrixDataTempl<SparseMatrix<double> >::init_pre();

   // check no rows, nor cols withouth data
   for(std::uint64_t mode = 0; mode < nmode(); ++mode)
   {
      auto& m = this->Y(mode);
      auto& count = num_empty[mode];
      for (int j = 0; j < m.cols(); j++)
      {
         if (m.col(j).nonZeros() == 0) 
            count++;
      }
   }
}

double ScarceMatrixData::train_rmse(const SubModel& model) const 
{
   return std::sqrt(sumsq(model) / this->nnz());
}

std::ostream& ScarceMatrixData::info(std::ostream& os, std::string indent)
{
    MatrixDataTempl<SparseMatrix<double> >::info(os, indent);
    if (num_empty[0]) os << indent << "  Warning: " << num_empty[0] << " empty rows\n";
    if (num_empty[1]) os << indent << "  Warning: " << num_empty[1] << " empty cols\n";
    return os;
}

void ScarceMatrixData::getMuLambda(const SubModel& model, std::uint32_t mode, int n, VectorXd& rr, MatrixXd& MM) const
{
   COUNTER("getMuLambda");

   auto &Y = this->Y(mode);
   const int num_latent = model.nlatent();
   const std::int64_t local_nnz = Y.col(n).nonZeros();
   const std::int64_t total_nnz = Y.nonZeros();
   auto from = Y.outerIndexPtr()[n];
   auto to = Y.outerIndexPtr()[n+1];

   auto getMuLambdaBasic = [&model, this, mode, n](int from, int to, VectorXd& rr, MatrixXd& MM) -> void
   {
       auto &Y = this->Y(mode);
       auto Vf = *model.CVbegin(mode);
       auto &ns = noise();

       for(int i = from; i < to; ++i)
       {
           auto val = Y.valuePtr()[i];
           auto idx = Y.innerIndexPtr()[i];
           const auto &col = Vf.col(idx);
           auto pos = this->pos(mode, n, idx);
           double noisy_val = ns.sample(model, pos, val);
           rr.noalias() += col * noisy_val;
           MM.triangularView<Lower>() +=  ns.getAlpha() * col * col.transpose();
       }

       // make MM complete
       MM.triangularView<Upper>() = MM.transpose();
    };

   

   bool in_parallel = (local_nnz >10000) || ((double)local_nnz > (double)total_nnz / 100.);
   if (in_parallel) 
   {
       const int task_size = ceil(local_nnz / 100.0);
       thread_vector<VectorXd> rrs(VectorXd::Zero(num_latent));
       thread_vector<MatrixXd> MMs(MatrixXd::Zero(num_latent, num_latent));

       for(int j = from; j < to; j += task_size) 
       {
           #pragma omp task shared(rrs, MMs)
           getMuLambdaBasic(j, std::min(j + task_size, to), rrs.local(), MMs.local());
       }
       #pragma omp taskwait
       
       // accumulate 
       MM += MMs.combine();
       rr += rrs.combine();
   } 
   else 
   {
      VectorXd my_rr = VectorXd::Zero(num_latent);
      MatrixXd my_MM = MatrixXd::Zero(num_latent, num_latent);

      getMuLambdaBasic(from, to, my_rr, my_MM);

      // add to global
      rr += my_rr;
      MM += my_MM;
   }
}

void ScarceMatrixData::update_pnm(const SubModel &, std::uint32_t mode)
{
   //can not cache VV because of scarceness
}

std::uint64_t ScarceMatrixData::nna() const
{
   return size() - this->nnz(); //nrows * ncols - nnz
}

double ScarceMatrixData::var_total() const
{
   double cwise_mean = this->sum() / this->nnz();
   double se = 0.0;

   #pragma omp parallel for schedule(guided) reduction(+:se)
   for (int k = 0; k < Y().outerSize(); ++k)
   {
      for (SparseMatrix<double>::InnerIterator it(Y(), k); it; ++it)
      {
         se += std::pow(it.value() - cwise_mean, 2);
      }
   }

   double var = se / this->nnz();
   if (var <= 0.0 || std::isnan(var))
   {
      // if var cannot be computed using 1.0
      var = 1.0;
   }

   return var;
}

double ScarceMatrixData::sumsq(const SubModel& model) const
{
   double sumsq = 0.0;

   #pragma omp parallel for schedule(guided) reduction(+:sumsq)
   for (int j = 0; j < Y().outerSize(); j++) 
   {
      for (SparseMatrix<double>::InnerIterator it(Y(), j); it; ++it) 
      {
         sumsq += std::pow(model.predict({static_cast<int>(it.row()), static_cast<int>(it.col())})- it.value(), 2);
      }
   }

   return sumsq;
}
