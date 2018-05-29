#include "PriorFactory.h"

#include <Eigen/Core>

#include <SmurffCpp/Priors/NormalPrior.h>
#include <SmurffCpp/Priors/NormalOnePrior.h>
#include <SmurffCpp/Priors/SpikeAndSlabPrior.h>

#include <SmurffCpp/Utils/Error.h>

#include <SmurffCpp/Utils/MatrixUtils.h>

using namespace smurff;
using namespace Eigen;

//create macau prior features

//-------

std::shared_ptr<ILatentPrior> PriorFactory::create_macau_prior(std::shared_ptr<Session> session, PriorTypes prior_type,
   const std::vector<std::shared_ptr<SideInfoConfig> >& config_items)
{
   if(prior_type == PriorTypes::macau || prior_type == PriorTypes::default_prior)
   {
      return create_macau_prior<MacauPrior>(session, config_items);
   }
   else if(prior_type == PriorTypes::macauone)
   {
      return create_macau_prior<MacauOnePrior>(session, config_items);
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
