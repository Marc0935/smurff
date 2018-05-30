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
    std::shared_ptr<ILatentPrior> create_prior(std::shared_ptr<Session> session, int mode) override;
};

}
