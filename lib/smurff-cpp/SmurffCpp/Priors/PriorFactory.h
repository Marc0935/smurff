#pragma once

#include <SmurffCpp/Priors/IPriorFactory.h>
#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffCpp/Priors/MacauOnePrior.h>
#include <SmurffCpp/Priors/MacauPrior.h>
#include <SmurffCpp/Sessions/Session.h>
#include <SmurffCpp/Configs/MatrixConfig.h>
#include <SmurffCpp/Configs/SideInfoConfig.h>

#include <SmurffCpp/SideInfo/ISideInfo.h>

namespace smurff {

class PriorFactory : public IPriorFactory
{
public:
    template<class MacauPrior>
    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session,
                                                     const std::vector<std::shared_ptr<SideInfoConfig> >& config_items);

    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session, PriorTypes prior_type, 
                                                     const std::vector<std::shared_ptr<SideInfoConfig> >& config_items);

    template<class Factory>
    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<Session> session, int mode, PriorTypes prior_type,
            const std::vector<std::shared_ptr<SideInfoConfig> >& config_items);

    std::shared_ptr<ILatentPrior> create_prior(std::shared_ptr<Session> session, int mode) override;
};

//-------

template <class MacauPrior>
std::shared_ptr<ILatentPrior> PriorFactory::create_macau_prior(std::shared_ptr<Session> session,
                                                               const std::vector<std::shared_ptr<SideInfoConfig>> &config_items)
{
   std::shared_ptr<MacauPrior> prior(new MacauPrior(session, -1));
   for (auto ci : config_items)
   {
      prior->addSideInfo(*ci);
   }

   return prior;
}

}
