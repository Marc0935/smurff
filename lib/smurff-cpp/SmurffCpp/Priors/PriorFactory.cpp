#include "PriorFactory.h"

#include <Eigen/Core>

#include <SmurffCpp/Priors/NormalPrior.h>
#include <SmurffCpp/Priors/NormalOnePrior.h>
#include <SmurffCpp/Priors/SpikeAndSlabPrior.h>

#include <SmurffCpp/Utils/Error.h>

#include <SmurffCpp/SideInfo/DenseDoubleSideInfo.h>
#include <SmurffCpp/SideInfo/SparseDoubleSideInfo.h>
#include <SmurffCpp/SideInfo/SparseBinarySideinfo.h>

#include <SmurffCpp/Utils/MatrixUtils.h>

using namespace smurff;
using namespace Eigen;

//create macau prior features

std::shared_ptr<ISideInfo> PriorFactory::side_info_config_to_dense_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode)
{
   auto side_info_ptr = std::make_shared<Eigen::MatrixXd>(matrix_utils::dense_to_eigen(*sideinfoConfig));
   return std::make_shared<DenseDoubleSideInfo>(side_info_ptr);
}

std::shared_ptr<ISideInfo> PriorFactory::side_info_config_to_sparse_binary_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode)
{
   std::uint64_t nrow = sideinfoConfig->getNRow();
   std::uint64_t ncol = sideinfoConfig->getNCol();
   std::uint64_t nnz = sideinfoConfig->getNNZ();

   const std::uint32_t* rows = sideinfoConfig->getRows().data();
   const std::uint32_t* cols = sideinfoConfig->getCols().data();

   auto side_info_ptr = std::make_shared<SparseFeat>(nrow, ncol, nnz, (int *)rows,  (int *)cols);
   return std::make_shared<SparseBinarySideinfo>(side_info_ptr);
}

std::shared_ptr<ISideInfo> PriorFactory::side_info_config_to_sparse_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode)
{
   std::uint64_t nrow = sideinfoConfig->getNRow();
   std::uint64_t ncol = sideinfoConfig->getNCol();
   std::uint64_t nnz = sideinfoConfig->getNNZ();

   const std::uint32_t* rows = sideinfoConfig->getRows().data();
   const std::uint32_t* cols = sideinfoConfig->getCols().data();
   const double*        vals = sideinfoConfig->getValues().data();

   auto side_info_ptr = std::make_shared<SparseDoubleFeat>(nrow, ncol, nnz, (int *)rows, (int *)cols, (double *)vals);
   return std::make_shared<SparseDoubleSideInfo>(side_info_ptr);
}

//-------

std::shared_ptr<ILatentPrior> PriorFactory::create_macau_prior(std::shared_ptr<Session> session, PriorTypes prior_type,
   const std::vector<std::shared_ptr<ISideInfo> >& side_infos,
   const std::vector<std::shared_ptr<SideInfoConfig> >& config_items)
{
   if(prior_type == PriorTypes::macau || prior_type == PriorTypes::default_prior)
   {
      return create_macau_prior<MacauPrior>(session, side_infos, config_items);
   }
   else if(prior_type == PriorTypes::macauone)
   {
      return create_macau_prior<MacauOnePrior>(session, side_infos, config_items);
   }
   else
   {
      THROWERROR("Unknown prior with side info: " + priorTypeToString(prior_type));
   }
}

std::shared_ptr<ILatentPrior> PriorFactory::create_prior(std::shared_ptr<Session> session, int mode)
{
   PriorTypes priorType = session->getConfig().getPriorTypes().at(mode);

   switch(priorType)
   {
   case PriorTypes::normal:
   case PriorTypes::default_prior:
      return std::shared_ptr<NormalPrior>(new NormalPrior(session, -1));
   case PriorTypes::spikeandslab:
      return std::shared_ptr<SpikeAndSlabPrior>(new SpikeAndSlabPrior(session, -1));
   case PriorTypes::normalone:
      return std::shared_ptr<NormalOnePrior>(new NormalOnePrior(session, -1));
   case PriorTypes::macau:
   case PriorTypes::macauone:
      return create_macau_prior<PriorFactory>(session, mode, priorType, session->getConfig().getSideInfoConfigs(mode));
   default:
      {
         THROWERROR("Unknown prior: " + priorTypeToString(priorType));
      }
   }
}
