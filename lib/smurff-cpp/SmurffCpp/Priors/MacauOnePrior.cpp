#include "MacauOnePrior.h"

#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/IO/GenericIO.h>

#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffCpp/SideInfo/ISideInfo.h>

using namespace smurff;

MacauOnePrior::MacauOnePrior(std::shared_ptr<BaseSession> session, uint32_t mode)
   : NormalOnePrior(session, mode, "MacauOnePrior")
{
}

void MacauOnePrior::init()
{
   NormalOnePrior::init();

   for(auto &f : features)
   {
      f->init();
      THROWERROR_ASSERT_MSG(f->num_item() == num_cols(), 
         "Number of rows in train must be equal to number of rows in features");
   }
}

void MacauOnePrior::update_prior()
{
   sample_mu_lambda();

   // update beta, uhat and beta_precision
   for (auto f : features)
      f->sample_beta_one();
}

const Eigen::VectorXd MacauOnePrior::getMu(int n) const
{
   Eigen::VectorXd ret = this->mu;
   for (auto &f : features)
      ret += f->getUhat().col(n);

   return ret;
}

void MacauOnePrior::addSideInfo(const SideInfoConfig &c)
{
   features.push_back(ISideInfo::create_side_info(c, *this));
}

void MacauOnePrior::sample_mu_lambda()
{
   Eigen::MatrixXd WI(num_latent(), num_latent());
   WI.setIdentity();

   Eigen::MatrixXd Udelta = U();
   for (auto f : features)
   {
      Udelta.noalias() -= f->getUhat();
   }

   std::tie(mu, Lambda) = CondNormalWishart(Udelta, 
      Eigen::VectorXd::Constant(num_latent(), 0.0), 2.0, WI, num_latent());
}



void MacauOnePrior::save(std::shared_ptr<const StepFile> sf) const
{
   NormalOnePrior::save(sf);
   for (auto &f : features)
      f->save(sf);
}

void MacauOnePrior::restore(std::shared_ptr<const StepFile> sf)
{
   NormalOnePrior::restore(sf);
   for (auto &f : features)
      f->restore(sf);
}

std::ostream& MacauOnePrior::status(std::ostream &os, std::string indent) const
{
   //os << indent << "  " << m_name << ": Beta = " << beta.norm() << std::endl;
   return os;
}
