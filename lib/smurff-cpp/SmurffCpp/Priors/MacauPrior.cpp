#include "MacauPrior.h"

#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/IO/GenericIO.h>

#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/Error.h>

#include <SmurffCpp/Utils/linop.h>

#include <ios>

using namespace smurff;

MacauPrior::MacauPrior()
   : NormalPrior() 
{
}

MacauPrior::MacauPrior(std::shared_ptr<BaseSession> session, uint32_t mode)
   : NormalPrior(session, mode, "MacauPrior")
{
}

MacauPrior::~MacauPrior()
{
}

void MacauPrior::init()
{
   NormalPrior::init();

   for(auto &f : features)
   {
      THROWERROR_ASSERT_MSG(f->rows() == num_cols(), "Number of rows in train must be equal to number of rows in features");
      f->init();
   }
}

void MacauPrior::update_prior()
{
   // sample hyper params
   {
      Eigen::MatrixXd BBt = Eigen::MatrixXd::Zero(num_latent(), num_latent());
      Eigen::MatrixXd Ures = U();
      std::uint64_t cols = 0;
      
      for(auto f : features) {
         BBt.noalias() += f->getBBt();
         Ures.noalias() -= f->getUhat();
         cols += f->cols();
      }

      std::tie(mu, Lambda) = CondNormalWishart(Ures, mu0, b0, WI + BBt, df + cols);
   }

   // update beta, uhat and beta_precision
   for (auto f : features)
      f->sample_beta();
}

const Eigen::VectorXd MacauPrior::getMu(int n) const
{
   Eigen::VectorXd ret = this->mu;
   for (auto &f : features)
      ret += f->getUhat().col(n);

   return ret;
}

void MacauPrior::addSideInfo(const SideInfoConfig &c)
{
   features.push_back(std::make_shared<ISideInfo>(c, *this));
}

void MacauPrior::save(std::shared_ptr<const StepFile> sf) const
{
   NormalPrior::save(sf);
   for (auto &f : features)
      f->save(sf);
}

void MacauPrior::restore(std::shared_ptr<const StepFile> sf)
{
   NormalPrior::restore(sf);
   for (auto &f : features)
      f->restore(sf);
}

std::ostream& MacauPrior::info(std::ostream &os, std::string indent)
{
   NormalPrior::info(os, indent);
   for (auto &f : features)
      f->info(os, indent + "  ");
   return os;
}

std::ostream& MacauPrior::status(std::ostream &os, std::string indent) const
{
   os << indent << m_name << ": " << std::endl;

   for (auto &f : features)
      f->status(os, indent + "  ");

   return os;
}