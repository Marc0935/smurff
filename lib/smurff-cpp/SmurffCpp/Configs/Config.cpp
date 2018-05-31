#include "Config.h"

#include <set>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <string>
#include <memory>

#include <SmurffCpp/Version.h>
#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/TensorUtils.h>
#include <SmurffCpp/IO/INIFile.h>
#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/IO/GenericIO.h>
#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/Utils/StringUtils.h>

#define GLOBAL_SECTION_TAG "global"
#define TRAIN_SECTION_TAG "train"
#define TEST_SECTION_TAG "test"

#define NUM_PRIORS_TAG "num_priors"
#define PRIOR_PREFIX "prior"
#define NUM_AUX_DATA_TAG "num_aux_data"
#define AUX_DATA_PREFIX "aux_data"
#define SAVE_PREFIX_TAG "save_prefix"
#define SAVE_EXTENSION_TAG "save_extension"
#define SAVE_FREQ_TAG "save_freq"
#define CHECKPOINT_FREQ_TAG "checkpoint_freq"
#define VERBOSE_TAG "verbose"
#define BURNING_TAG "burnin"
#define NSAMPLES_TAG "nsamples"
#define NUM_LATENT_TAG "num_latent"
#define NUM_THREADS_TAG "num_threads"
#define RANDOM_SEED_SET_TAG "random_seed_set"
#define RANDOM_SEED_TAG "random_seed"
#define CSV_STATUS_TAG "csv_status"
#define INIT_MODEL_TAG "init_model"
#define CLASSIFY_TAG "classify"
#define THRESHOLD_TAG "threshold"

using namespace smurff;

PriorTypes smurff::stringToPriorType(std::string name)
{
   if(name == PRIOR_NAME_DEFAULT)
      return PriorTypes::default_prior;
   else if(name == PRIOR_NAME_MACAU)
      return PriorTypes::macau;
   else if(name == PRIOR_NAME_MACAU_ONE)
      return PriorTypes::macauone;
   else if(name == PRIOR_NAME_SPIKE_AND_SLAB)
      return PriorTypes::spikeandslab;
   else if(name == PRIOR_NAME_NORMALONE)
      return PriorTypes::normalone;
   else if(name == PRIOR_NAME_NORMAL)
      return PriorTypes::normal;
   else
   {
      THROWERROR("Invalid prior type");
   }
}

std::string smurff::priorTypeToString(PriorTypes type)
{
   switch(type)
   {
      case PriorTypes::default_prior:
         return PRIOR_NAME_DEFAULT;
      case PriorTypes::macau:
         return PRIOR_NAME_MACAU;
      case PriorTypes::macauone:
         return PRIOR_NAME_MACAU_ONE;
      case PriorTypes::spikeandslab:
         return PRIOR_NAME_SPIKE_AND_SLAB;
      case PriorTypes::normal:
         return PRIOR_NAME_NORMAL;
      case PriorTypes::normalone:
         return PRIOR_NAME_NORMALONE;
      default:
      {
         THROWERROR("Invalid prior type");
      }
   }
}

ModelInitTypes smurff::stringToModelInitType(std::string name)
{
   if(name == MODEL_INIT_NAME_RANDOM)
      return ModelInitTypes::random;
   else if (name == MODEL_INIT_NAME_ZERO)
      return ModelInitTypes::zero;
   else
   {
      THROWERROR("Invalid model init type " + name);
   }
}

std::string smurff::modelInitTypeToString(ModelInitTypes type)
{
   switch(type)
   {
      case ModelInitTypes::random:
         return MODEL_INIT_NAME_RANDOM;
      case ModelInitTypes::zero:
         return MODEL_INIT_NAME_ZERO;
      default:
      {
         THROWERROR("Invalid model init type");
      }
   }
}

//config
int Config::BURNIN_DEFAULT_VALUE = 200;
int Config::NSAMPLES_DEFAULT_VALUE = 800;
int Config::NUM_LATENT_DEFAULT_VALUE = 96;
int Config::NUM_THREADS_DEFAULT_VALUE = 0; // as many as you want
ModelInitTypes Config::INIT_MODEL_DEFAULT_VALUE = ModelInitTypes::zero;
const char* Config::SAVE_PREFIX_DEFAULT_VALUE = "save";
const char* Config::SAVE_EXTENSION_DEFAULT_VALUE = ".ddm";
int Config::SAVE_FREQ_DEFAULT_VALUE = 0;
int Config::CHECKPOINT_FREQ_DEFAULT_VALUE = 0;
int Config::VERBOSE_DEFAULT_VALUE = 1;
const char* Config::STATUS_DEFAULT_VALUE = "";
bool Config::ENABLE_BETA_PRECISION_SAMPLING_DEFAULT_VALUE = true;
double Config::THRESHOLD_DEFAULT_VALUE = 0.0;
int Config::RANDOM_SEED_DEFAULT_VALUE = 0;

Config::Config()
{
   m_model_init_type = Config::INIT_MODEL_DEFAULT_VALUE;

   m_save_prefix = Config::SAVE_PREFIX_DEFAULT_VALUE;
   m_save_extension = Config::SAVE_EXTENSION_DEFAULT_VALUE;
   m_save_freq = Config::SAVE_FREQ_DEFAULT_VALUE;
   m_checkpoint_freq = Config::CHECKPOINT_FREQ_DEFAULT_VALUE;

   m_random_seed_set = false;
   m_random_seed = Config::RANDOM_SEED_DEFAULT_VALUE;

   m_verbose = Config::VERBOSE_DEFAULT_VALUE;
   m_csv_status = Config::STATUS_DEFAULT_VALUE;
   m_burnin = Config::BURNIN_DEFAULT_VALUE;
   m_nsamples = Config::NSAMPLES_DEFAULT_VALUE;
   m_num_latent = Config::NUM_LATENT_DEFAULT_VALUE;
   m_num_threads = Config::NUM_THREADS_DEFAULT_VALUE;

   m_threshold = Config::THRESHOLD_DEFAULT_VALUE;
   m_classify = false;
}

const std::vector<std::shared_ptr<SideInfoConfig> > Config::getSideInfoConfigs(int mode) const
{
  auto iter = m_sideInfoConfigs.find(mode);
  if (iter == m_sideInfoConfigs.end()) {
     // return empty vector
     return std::vector<std::shared_ptr<SideInfoConfig> >();
  }
  return iter->second;
}

const std::map<int, std::vector<std::shared_ptr<SideInfoConfig> > >& Config::addSideInfoConfig(int mode, std::shared_ptr<SideInfoConfig> c)
{
    m_sideInfoConfigs[mode].push_back(c);

    // automagically update prior type 
    // normal(one) prior -> macau(one) prior
    if ((int)m_prior_types.size() > mode)
    {
      PriorTypes &pt = m_prior_types[mode];
           if (pt == PriorTypes::normal) pt = PriorTypes::macau;
      else if (pt == PriorTypes::normalone) pt = PriorTypes::macauone;
    }

    return m_sideInfoConfigs;
}

bool Config::validate() const
{
   if (!m_train || !m_train->getNNZ())
   {
      THROWERROR("Missing train data");
   }

   auto train_pos = PVec<>(m_train->getNModes());
   if (!m_train->hasPos())
   {
       m_train->setPos(train_pos);
   }
   else if (m_train->getPos() != train_pos)
   {
       THROWERROR("Train should be at upper position (all zeros)");
   }

   if (m_test && !m_test->getNNZ())
   {
      THROWERROR("Missing test data");
   }

   if (m_test && m_test->getDims() != m_train->getDims())
   {
      THROWERROR("Train and test data should have the same dimensions");
   }

   if(getPriorTypes().size() != m_train->getNModes())
   {
      THROWERROR("Number of priors should equal to number of dimensions in train data");
   }

   if (m_train->getNModes() > 2)
   {

      if (!m_auxData.empty())
      {
         //it is advised to check macau and macauone priors implementation
         //as well as code in PriorFactory that creates macau priors

         //this check does not directly check that input data is Tensor (it only checks number of dimensions)
         //however TensorDataFactory will do an additional check throwing an exception
         THROWERROR("Aux data is not supported for TensorData");
      }
   }

   for (auto p : m_sideInfoConfigs)
   {
      int mode = p.first;
      auto &configItems = p.second;
      for (auto configItem : configItems)
      {
          const auto& sideInfo = configItem->getSideInfo();
          THROWERROR_ASSERT(sideInfo);

          if (sideInfo->getDims()[0] != m_train->getDims()[mode])
          {
              std::stringstream ss;
              ss << "Side info should have the same number of rows as size of dimension " << mode << " in train data";
              THROWERROR(ss.str());
          }
      }
   }

   for(auto& ad1 : getData())
   {
      if (!ad1->hasPos())
      {
         std::stringstream ss;
         ss << "Data \"" << ad1->info() << "\" is missing position info";
         THROWERROR(ss.str());
      }

      const auto& dim1 = ad1->getDims();
      const auto& pos1 = ad1->getPos();

      for(auto& ad2 : getData())
      {
         if (ad1 == ad2)
            continue;

         if (!ad2->hasPos())
         {
            std::stringstream ss;
            ss << "Data \"" << ad2->info() << "\" is missing position info";
            THROWERROR(ss.str());
         }

         const auto& dim2 = ad2->getDims();
         const auto& pos2 = ad2->getPos();

         if (pos1 == pos2)
         {
            std::stringstream ss;
            ss << "Data \"" << ad1->info() <<  "\" and \"" << ad2->info() << "\" at same position";
            THROWERROR(ss.str());
         }

         // if two data blocks are aligned in a certain dimension
         // this dimension should be equal size
         for (std::size_t i = 0; i < pos1.size(); ++i)
         {
            if (pos1.at(i) == pos2.at(i) && (dim1.at(i) != dim2.at(i)))
            {
               std::stringstream ss;
               ss << "Data \"" << ad1->info() << "\" and \"" << ad2->info() << "\" different in size in dimension " << i;
               THROWERROR(ss.str());
            }
         }
      }
   }

   for(std::size_t i = 0; i < m_prior_types.size(); i++)
   {
      PriorTypes pt = m_prior_types[i];
      switch (pt)
      {
         case PriorTypes::normal:
         case PriorTypes::normalone:
         case PriorTypes::spikeandslab:
         case PriorTypes::default_prior:
            THROWERROR_ASSERT_MSG(!hasSideInfo(i), priorTypeToString(pt) + " prior in dimension " + std::to_string(i) + " cannot have side info");
            break;
         case PriorTypes::macau:
         case PriorTypes::macauone:
            THROWERROR_ASSERT_MSG(hasSideInfo(i), priorTypeToString(pt) + " prior in dimension " + std::to_string(i) + " needs side info");
            break;
         default:
            THROWERROR("Unknown prior");
            break;
      }
   }

   std::set<std::string> save_extensions = { ".csv", ".ddm" };

   if (save_extensions.find(m_save_extension) == save_extensions.end())
   {
      THROWERROR("Unknown output extension: " + m_save_extension + " (expected \".csv\" or \".ddm\")");
   }

   m_train->getNoiseConfig().validate();

   return true;
}

void Config::save(std::string fname) const
{
   INIFile ini;
   ini.create(fname);

   //write header with time and version
   ini.appendComment("SMURFF config .ini file");
   ini.appendComment("generated file - copy before editing!");

   auto t = std::time(nullptr);
   auto tm = *std::localtime(&t);
   char time_str[1024];
   strftime (time_str, 1023, "%Y-%m-%d %H:%M:%S", &tm);

   ini.appendComment("generated on " + std::string(time_str));
   ini.appendComment("generated by smurff " + std::string(SMURFF_VERSION));

   //write global options section
   ini.startSection(GLOBAL_SECTION_TAG);

   //count data
   ini.appendComment("count");
   ini.appendItem(GLOBAL_SECTION_TAG, NUM_PRIORS_TAG, std::to_string(m_prior_types.size()));
   ini.appendItem(GLOBAL_SECTION_TAG, NUM_AUX_DATA_TAG, std::to_string(m_auxData.size()));

   //priors data
   ini.appendComment("priors");
   for (std::size_t pIndex = 0; pIndex < m_prior_types.size(); pIndex++)
      ini.appendItem(GLOBAL_SECTION_TAG, std::string(PRIOR_PREFIX) + "_" + std::to_string(pIndex), priorTypeToString(m_prior_types.at(pIndex)));

   //save data
   ini.appendComment("save");
   ini.appendItem(GLOBAL_SECTION_TAG, SAVE_PREFIX_TAG, m_save_prefix);
   ini.appendItem(GLOBAL_SECTION_TAG, SAVE_EXTENSION_TAG, m_save_extension);
   ini.appendItem(GLOBAL_SECTION_TAG, SAVE_FREQ_TAG, std::to_string(m_save_freq));
   ini.appendItem(GLOBAL_SECTION_TAG, CHECKPOINT_FREQ_TAG, std::to_string(m_checkpoint_freq));

   //general data
   ini.appendComment("general");
   ini.appendItem(GLOBAL_SECTION_TAG, VERBOSE_TAG, std::to_string(m_verbose));
   ini.appendItem(GLOBAL_SECTION_TAG, BURNING_TAG, std::to_string(m_burnin));
   ini.appendItem(GLOBAL_SECTION_TAG, NSAMPLES_TAG, std::to_string(m_nsamples));
   ini.appendItem(GLOBAL_SECTION_TAG, NUM_LATENT_TAG, std::to_string(m_num_latent));
   ini.appendItem(GLOBAL_SECTION_TAG, NUM_THREADS_TAG, std::to_string(m_num_threads));
   ini.appendItem(GLOBAL_SECTION_TAG, RANDOM_SEED_SET_TAG, std::to_string(m_random_seed_set));
   ini.appendItem(GLOBAL_SECTION_TAG, RANDOM_SEED_TAG, std::to_string(m_random_seed));
   ini.appendItem(GLOBAL_SECTION_TAG, CSV_STATUS_TAG, m_csv_status);
   ini.appendItem(GLOBAL_SECTION_TAG, INIT_MODEL_TAG, modelInitTypeToString(m_model_init_type));

   //probit prior data
   ini.appendComment("binary classification");
   ini.appendItem(GLOBAL_SECTION_TAG, CLASSIFY_TAG, std::to_string(m_classify));
   ini.appendItem(GLOBAL_SECTION_TAG, THRESHOLD_TAG, std::to_string(m_threshold));

   ini.endSection();

   //write train data section
   TensorConfig::save_tensor_config(ini, TRAIN_SECTION_TAG, -1, m_train);

   //write test data section
   TensorConfig::save_tensor_config(ini, TEST_SECTION_TAG, -1, m_test);

   //write macau prior configs section
   for (auto p : m_sideInfoConfigs)
   {
       int mode = p.first;
       auto &configItems = p.second;
       for (std::size_t mIndex = 0; mIndex < configItems.size(); mIndex++)
       {
           configItems.at(mIndex)->save(ini, mode, mIndex);
       }
   }

   //write aux data section
   for (std::size_t sIndex = 0; sIndex < m_auxData.size(); sIndex++)
   {
      TensorConfig::save_tensor_config(ini, AUX_DATA_PREFIX, sIndex, m_auxData.at(sIndex));
   }
}

bool Config::restore(std::string fname)
{
   THROWERROR_FILE_NOT_EXIST(fname);

   INIFile reader;
   reader.open(fname);

   if (reader.getParseError() < 0)
   {
      std::cout << "Can't load '" << fname << "'\n";
      return false;
   }

   auto add_index = [](const std::string name, int idx = -1) -> std::string
   {
      if (idx >= 0)
         return name + "_" + std::to_string(idx);
      return name;
   };

   //restore train data
   setTest(TensorConfig::restore_tensor_config(reader, TEST_SECTION_TAG));

   //restore test data
   setTrain(TensorConfig::restore_tensor_config(reader, TRAIN_SECTION_TAG));

   //restore global data

   //restore priors
   std::size_t num_priors = reader.getInteger(GLOBAL_SECTION_TAG, NUM_PRIORS_TAG, 0);
   std::vector<std::string> pNames;
   for(std::size_t pIndex = 0; pIndex < num_priors; pIndex++)
   {
      pNames.push_back(reader.get(GLOBAL_SECTION_TAG, add_index(PRIOR_PREFIX, pIndex),  PRIOR_NAME_DEFAULT));
   }
   setPriorTypes(pNames);

   //restore macau prior configs section
   for (std::size_t mPriorIndex = 0; mPriorIndex < num_priors; mPriorIndex++)
   {
      int mSideInfoIndex = 0;
      while (true) {
          auto sideInfoConfig = std::make_shared<SideInfoConfig>();

          if (sideInfoConfig->restore(reader, mPriorIndex, mSideInfoIndex))
             m_sideInfoConfigs[mPriorIndex].push_back(sideInfoConfig);
          else
             break;

          mSideInfoIndex++;
      }
   }

   //restore aux data
   std::size_t num_aux_data = reader.getInteger(GLOBAL_SECTION_TAG, NUM_AUX_DATA_TAG, 0);
   for(std::size_t pIndex = 0; pIndex < num_aux_data; pIndex++)
   {
      m_auxData.push_back(TensorConfig::restore_tensor_config(reader, add_index(AUX_DATA_PREFIX, pIndex)));
   }

   //restore save data
   m_save_prefix = reader.get(GLOBAL_SECTION_TAG, SAVE_PREFIX_TAG, Config::SAVE_PREFIX_DEFAULT_VALUE);
   m_save_extension = reader.get(GLOBAL_SECTION_TAG, SAVE_EXTENSION_TAG, Config::SAVE_EXTENSION_DEFAULT_VALUE);
   m_save_freq = reader.getInteger(GLOBAL_SECTION_TAG, SAVE_FREQ_TAG, Config::SAVE_FREQ_DEFAULT_VALUE);
   m_checkpoint_freq = reader.getInteger(GLOBAL_SECTION_TAG, CHECKPOINT_FREQ_TAG, Config::CHECKPOINT_FREQ_DEFAULT_VALUE);

   //restore general data
   m_verbose = reader.getInteger(GLOBAL_SECTION_TAG, VERBOSE_TAG, Config::VERBOSE_DEFAULT_VALUE);
   m_burnin = reader.getInteger(GLOBAL_SECTION_TAG, BURNING_TAG, Config::BURNIN_DEFAULT_VALUE);
   m_nsamples = reader.getInteger(GLOBAL_SECTION_TAG, NSAMPLES_TAG, Config::NSAMPLES_DEFAULT_VALUE);
   m_num_latent = reader.getInteger(GLOBAL_SECTION_TAG, NUM_LATENT_TAG, Config::NUM_LATENT_DEFAULT_VALUE);
   m_num_threads = reader.getInteger(GLOBAL_SECTION_TAG, NUM_THREADS_TAG, Config::NUM_THREADS_DEFAULT_VALUE);
   m_random_seed_set = reader.getBoolean(GLOBAL_SECTION_TAG, RANDOM_SEED_SET_TAG,  false);
   m_random_seed = reader.getInteger(GLOBAL_SECTION_TAG, RANDOM_SEED_TAG, Config::RANDOM_SEED_DEFAULT_VALUE);
   m_csv_status = reader.get(GLOBAL_SECTION_TAG, CSV_STATUS_TAG, Config::STATUS_DEFAULT_VALUE);
   m_model_init_type = stringToModelInitType(reader.get(GLOBAL_SECTION_TAG, INIT_MODEL_TAG, modelInitTypeToString(Config::INIT_MODEL_DEFAULT_VALUE)));

   //restore probit prior data
   m_classify = reader.getBoolean(GLOBAL_SECTION_TAG, CLASSIFY_TAG,  false);
   m_threshold = reader.getReal(GLOBAL_SECTION_TAG, THRESHOLD_TAG, Config::THRESHOLD_DEFAULT_VALUE);

   return true;
}

bool Config::restoreSaveInfo(std::string fname, std::string& save_prefix, std::string& save_extension)
{
   THROWERROR_FILE_NOT_EXIST(fname);

   INIFile reader;
   reader.open(fname);

   if (reader.getParseError() < 0)
   {
      std::cout << "Can't load '" << fname << "'\n";
      return false;
   }

   save_prefix = reader.get(GLOBAL_SECTION_TAG, SAVE_PREFIX_TAG, Config::SAVE_PREFIX_DEFAULT_VALUE);
   save_extension = reader.get(GLOBAL_SECTION_TAG, SAVE_EXTENSION_TAG, Config::SAVE_EXTENSION_DEFAULT_VALUE);

   return true;
}
