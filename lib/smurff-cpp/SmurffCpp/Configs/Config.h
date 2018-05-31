#pragma once

#include <string>
#include <vector>
#include <memory>

#include <SmurffCpp/Utils/PVec.hpp>
#include <SmurffCpp/Utils/Error.h>
#include "MatrixConfig.h"
#include "SideInfoConfig.h"

#define PRIOR_NAME_DEFAULT "default"
#define PRIOR_NAME_MACAU "macau"
#define PRIOR_NAME_MACAU_ONE "macauone"
#define PRIOR_NAME_SPIKE_AND_SLAB "spikeandslab"
#define PRIOR_NAME_NORMAL "normal"
#define PRIOR_NAME_NORMALONE "normalone"

#define MODEL_INIT_NAME_RANDOM "random"
#define MODEL_INIT_NAME_ZERO "zero"

namespace smurff {

enum class PriorTypes
{
   default_prior,
   macau,
   macauone,
   spikeandslab,
   normal,
   normalone,
   mpi
};

enum class ModelInitTypes
{
   random,
   zero
};

PriorTypes stringToPriorType(std::string name);

std::string priorTypeToString(PriorTypes type);

ModelInitTypes stringToModelInitType(std::string name);

std::string modelInitTypeToString(ModelInitTypes type);

struct Config
{
public:

   //config
   static int BURNIN_DEFAULT_VALUE;
   static int NSAMPLES_DEFAULT_VALUE;
   static int NUM_LATENT_DEFAULT_VALUE;
   static int NUM_THREADS_DEFAULT_VALUE;
   static ModelInitTypes INIT_MODEL_DEFAULT_VALUE;
   static const char* SAVE_PREFIX_DEFAULT_VALUE;
   static const char* SAVE_EXTENSION_DEFAULT_VALUE;
   static int SAVE_FREQ_DEFAULT_VALUE;
   static int CHECKPOINT_FREQ_DEFAULT_VALUE;
   static int VERBOSE_DEFAULT_VALUE;
   static const char* STATUS_DEFAULT_VALUE;
   static bool ENABLE_BETA_PRECISION_SAMPLING_DEFAULT_VALUE;
   static double THRESHOLD_DEFAULT_VALUE;
   static int RANDOM_SEED_DEFAULT_VALUE;

private:
   //-- train and test
   std::shared_ptr<TensorConfig> m_train;
   std::shared_ptr<TensorConfig> m_test;

   //-- aux_data (contains pos)
   std::vector<std::shared_ptr<TensorConfig> > m_auxData; //set of aux data matrices for normal and spikeandslab priors

   //-- sideinfo per mode
   std::map<int, std::vector<std::shared_ptr<SideInfoConfig> > > m_sideInfoConfigs;

   // -- priors
   std::vector<PriorTypes> m_prior_types;

   //-- init model
   ModelInitTypes m_model_init_type;

   //-- save
   std::string m_save_prefix;
   std::string m_save_extension;
   int m_save_freq;
   int m_checkpoint_freq;

   //-- general
   bool m_random_seed_set;
   int m_random_seed;
   int m_verbose;
   std::string m_csv_status;
   int m_burnin;
   int m_nsamples;
   int m_num_latent;
   int m_num_threads; 


   //-- binary classification
   bool m_classify;
   double m_threshold;

public:
   Config();

public:
   bool validate() const;

   void save(std::string fname) const;

   bool restore(std::string fname);

   static bool restoreSaveInfo(std::string fname, std::string& save_prefix, std::string& save_extension);

public:
   std::shared_ptr<TensorConfig> getTrain() const
   {
      return m_train;
   }

   void setTrain(std::shared_ptr<TensorConfig> value)
   {
      m_train = value;
   }

   std::shared_ptr<TensorConfig> getTest() const
   {
      return m_test;
   }

   void setTest(std::shared_ptr<TensorConfig> value)
   {
      m_test = value;
   }

   const std::vector< std::shared_ptr<TensorConfig> >& getAuxData() const
   {
      return m_auxData;
   }

   const std::vector< std::shared_ptr<TensorConfig> >& addAuxData(std::shared_ptr<TensorConfig> c)
   {
      m_auxData.push_back(c);
      return m_auxData;
   }

   const std::map<int, std::vector<std::shared_ptr<SideInfoConfig> > >& getSideInfoConfigs() const
   {
      return m_sideInfoConfigs;
   }

   const std::vector<std::shared_ptr<SideInfoConfig> > getSideInfoConfigs(int mode) const;

   const std::map<int, std::vector<std::shared_ptr<SideInfoConfig> > >& addSideInfoConfig(int mode, std::shared_ptr<SideInfoConfig> c);

   bool hasSideInfo(int mode) const
   {
       return m_sideInfoConfigs.find(mode) != m_sideInfoConfigs.end();
   }

   std::vector< std::shared_ptr<TensorConfig> > getData() const
   {
       auto data = m_auxData;
       data.push_back(m_train);
       return data;
   }

   const std::vector<PriorTypes> getPriorTypes() const
   {
      if (m_prior_types.empty())
      {
          THROWERROR_ASSERT(getTrain())
          return std::vector<PriorTypes>(getTrain()->getNModes(), PriorTypes::default_prior);
      }
      return m_prior_types;
   }

   const std::vector<PriorTypes>& setPriorTypes(std::vector<PriorTypes> values)
   {
      m_prior_types = values;
      return m_prior_types;
   }

   const std::vector<PriorTypes>& setPriorTypes(std::vector<std::string> values)
   {
      m_prior_types.clear();
      for(auto &value : values)
      {
          m_prior_types.push_back(stringToPriorType(value));
      }
      return m_prior_types;
   }

   ModelInitTypes getModelInitType() const
   {
      return m_model_init_type;
   }

   void setModelInitType(ModelInitTypes value)
   {
      m_model_init_type = value;
   }

   std::string getModelInitTypeAsString() const
   {
      return modelInitTypeToString(m_model_init_type);
   }

   void setModelInitType(std::string value)
   {
      m_model_init_type = stringToModelInitType(value);
   }

   std::string getSavePrefix() const
   {
      return m_save_prefix;
   }

   void setSavePrefix(std::string value)
   {
      m_save_prefix = value;
   }

   std::string getSaveExtension() const
   {
      return m_save_extension;
   }

   void setSaveExtension(std::string value)
   {
      m_save_extension = value;
   }

   int getSaveFreq() const
   {
      return m_save_freq;
   }

   void setSaveFreq(int value)
   {
      m_save_freq = value;
   }

   int getCheckpointFreq() const
   {
      return m_checkpoint_freq;
   }

   void setCheckpointFreq(int value)
   {
      m_checkpoint_freq = value;
   }

   bool getRandomSeedSet() const
   {
      return m_random_seed_set;
   }

   int getRandomSeed() const
   {
      THROWERROR_ASSERT_MSG(getRandomSeedSet(), "Random seed is unset");
      return m_random_seed;
   }

   void setRandomSeed(int value)
   {
      m_random_seed_set = true;
      m_random_seed = value;
   }

   int getVerbose() const
   {
      return m_verbose;
   }

   void setVerbose(int value)
   {
      if (value < 0) value = 0;
      m_verbose = value;
   }

   std::string getCsvStatus() const
   {
      return m_csv_status;
   }

   void setCsvStatus(std::string value)
   {
      m_csv_status = value;
   }

   int getBurnin() const
   {
      return m_burnin;
   }

   void setBurnin(int value)
   {
      m_burnin = value;
   }

   int getNSamples() const
   {
      return m_nsamples;
   }

   void setNSamples(int value)
   {
      m_nsamples = value;
   }

   int getNumLatent() const
   {
      return m_num_latent;
   }

   void setNumLatent(int value)
   {
      m_num_latent = value;
   }

   bool getClassify() const
   {
      return m_classify;
   }

   void setClassify(bool value)
   {
      m_classify = value;
   }

   double getThreshold() const
   {
      return m_threshold;
   }

   void setThreshold(double value)
   {
      m_threshold = value;
   }

   int getNumThreads() const
   {
       return m_num_threads;
   }

   void setNumThreads(int value)
   {
       m_num_threads = value;
   }
};

}

