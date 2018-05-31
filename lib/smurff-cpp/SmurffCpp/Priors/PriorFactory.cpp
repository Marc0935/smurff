#include "PriorFactory.h"

#include <Eigen/Core>

#include <SmurffCpp/Priors/NormalPrior.h>
#include <SmurffCpp/Priors/NormalOnePrior.h>
#include <SmurffCpp/Priors/SpikeAndSlabPrior.h>

#include <SmurffCpp/Utils/Error.h>

#include <SmurffCpp/Utils/MatrixUtils.h>

using namespace smurff;
using namespace Eigen;

std::shared_ptr<ILatentPrior> PriorFactory::create_prior(std::shared_ptr<Session> session, int mode)
{
   PriorTypes priorType = session->getConfig().getPriorTypes().at(mode);
   std::shared_ptr<ILatentPrior> prior;

   switch(priorType)
   {
   case PriorTypes::normal:
   case PriorTypes::default_prior:
      prior = std::shared_ptr<NormalPrior>(new NormalPrior(session, -1));
      break;
   case PriorTypes::spikeandslab:
      prior = std::shared_ptr<SpikeAndSlabPrior>(new SpikeAndSlabPrior(session, -1));
      break;
   case PriorTypes::normalone:
      prior = std::shared_ptr<NormalOnePrior>(new NormalOnePrior(session, -1));
      break;
   case PriorTypes::macau:
      prior = std::shared_ptr<MacauPrior>(new MacauPrior(session, -1));
      break;
   case PriorTypes::macauone:
      prior = std::shared_ptr<MacauOnePrior>(new MacauOnePrior(session, -1));
      break;
   default:
      THROWERROR("Unknown prior: " + priorTypeToString(priorType));
   }

   for (auto side : session->getConfig().getSideInfoConfigs(mode))
   {
      prior->addSideInfo(*side);
   }

   return prior;
}
