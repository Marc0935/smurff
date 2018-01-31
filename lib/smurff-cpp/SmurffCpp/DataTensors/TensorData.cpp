#include "TensorData.h"

#include <iostream>
#include <sstream>

#include <SmurffCpp/ConstVMatrixExprIterator.hpp>

using namespace Eigen;
using namespace smurff;

//convert array of coordinates to [nnz x nmodes] matrix
MatrixXui32 toMatrixNew(const std::vector<std::uint32_t>& columns, std::uint64_t nnz, std::uint64_t nmodes) 
{
   MatrixXui32 idx(nnz, nmodes);
   for (std::uint64_t row = 0; row < nnz; row++) 
   {
      for (std::uint64_t col = 0; col < nmodes; col++) 
      {
         idx(row, col) = columns[col * nnz + row];
      }
   }
   return idx;
}

TensorData::TensorData(const smurff::TensorConfig& tc) 
   : m_dims(tc.getDims()),
     m_nnz(tc.getNNZ()),
     m_Y(std::make_shared<std::vector<std::shared_ptr<SparseMode> > >())
{
   //combine coordinates into [nnz x nmodes] matrix
   MatrixXui32 idx = toMatrixNew(tc.getColumns(), tc.getNNZ(), tc.getNModes());

   for (std::uint64_t mode = 0; mode < tc.getNModes(); mode++) 
   {
      m_Y->push_back(std::make_shared<SparseMode>(idx, tc.getValues(), mode, m_dims[mode]));
   }
}

std::shared_ptr<SparseMode> TensorData::Y(std::uint64_t mode) const
{
   return m_Y->operator[](mode);
}


void TensorData::init_pre()
{
   //no logic here
}

double TensorData::sum() const
{
   double esum = 0.0;

   std::shared_ptr<SparseMode> sview = Y(0);

   #pragma omp parallel for schedule(dynamic, 4) reduction(+:esum)
   for(std::uint64_t n = 0; n < sview->getNPlanes(); n++) //go through each hyperplane
   {
      for(std::uint64_t j = sview->beginPlane(n); j < sview->endPlane(n); j++) //go through each item in the plane
      {
         esum += sview->getValues()[j];
      }
   }

   return esum;
}

std::uint64_t TensorData::nmode() const
{
   return m_dims.size();
}

std::uint64_t TensorData::nnz() const
{
   return m_nnz;
}

std::uint64_t TensorData::nna() const
{
   return size() - this->nnz();
}

PVec<> TensorData::dim() const
{
   std::vector<int> pvec_dims;
   for(auto& d : m_dims)
      pvec_dims.push_back(static_cast<int>(d));
   return PVec<>(pvec_dims);
}

double TensorData::train_rmse(const SubModel& model) const
{
   return std::sqrt(sumsq(model) / this->nnz());
}

//d is an index of column in U matrix
//this function selects d'th hyperplane from mode`th SparseMode
//it does j multiplications
//where each multiplication is a cwiseProduct of columns from each V matrix
void TensorData::getMuLambda(const SubModel& model, uint32_t mode, int d, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) const
{
   std::shared_ptr<SparseMode> sview = Y(mode); //get tensor rotation for mode
   
   auto V0 = model.CVbegin(mode); //get first V matrix
   for (std::uint64_t j = sview->beginPlane(d); j < sview->endPlane(d); j++) //go through hyperplane in tensor rotation
   {
      VectorXd col = (*V0).col(sview->getIndices()(j, 0)); //create a copy of m'th column from V (m = 0)
      auto V = model.CVbegin(mode); //get V matrices for mode      
      for (std::uint64_t m = 1; m < sview->getNCoords(); m++) //go through each coordinate of value
      {
         ++V; //inc iterator prior to access since we are starting from m = 1
         col.noalias() = col.cwiseProduct((*V).col(sview->getIndices()(j, m))); //multiply by m'th column from V
      }
      MM.triangularView<Eigen::Lower>() += noise()->getAlpha() * col * col.transpose(); // MM = MM + (col * colT) * alpha (where col = product of columns in each V)
      
      auto pos = sview->pos(d, j);
      double noisy_val = noise()->sample(model, pos, sview->getValues()[j]);
      rr.noalias() += col * noisy_val; // rr = rr + (col * value) * alpha (where value = j'th value of Y)
   }
}

void TensorData::update_pnm(const SubModel& model, uint32_t mode)
{
   //do not need to cache VV here
}

double TensorData::sumsq(const SubModel& model) const
{
   double sumsq = 0.0;

   std::shared_ptr<SparseMode> sview = Y(0);

   #pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
   for(std::uint64_t h = 0; h < sview->getNPlanes(); h++) //go through each hyperplane
   {
      for(std::uint64_t n = 0; n < sview->nItemsOnPlane(h); n++) //go through each item in the hyperplane
      {
         auto item = sview->item(h, n);
         double pred = model.predict(item.first);
         sumsq += std::pow(pred - item.second, 2);
      }
   }

   return sumsq;
}

double TensorData::var_total() const
{
   double cwise_mean = this->sum() / this->nnz();
   double se = 0.0;

   std::shared_ptr<SparseMode> sview = Y(0);

   #pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
   for(std::uint64_t h = 0; h < sview->getNPlanes(); h++) //go through each hyperplane
   {
      for(std::uint64_t n = 0; n < sview->nItemsOnPlane(h); n++) //go through each item in the hyperplane
      {
         auto item = sview->item(h, n);
         se += std::pow(item.second - cwise_mean, 2);
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

std::pair<PVec<>, double> TensorData::item(std::uint64_t mode, std::uint64_t hyperplane, std::uint64_t item) const
{
   if(mode >= m_Y->size())
   {
      THROWERROR("Invalid mode");
   }

   return m_Y->at(mode)->item(hyperplane, item);
}

PVec<> TensorData::pos(std::uint64_t mode, std::uint64_t hyperplane, std::uint64_t item) const
{
   if(mode >= m_Y->size())
   {
      THROWERROR("Invalid mode");
   }

   return m_Y->at(mode)->pos(hyperplane, item);
}