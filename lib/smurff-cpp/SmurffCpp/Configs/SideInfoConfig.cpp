#include <iostream>
#include <fstream>
#include <memory>

#include "SideInfoConfig.h"

#include "TensorConfig.h"

#define MACAU_PRIOR_CONFIG_PREFIX_TAG "macau_prior_config"
#define MACAU_PRIOR_CONFIG_ITEM_PREFIX_TAG "macau_prior_config_item"

#define NUM_SIDE_INFO_TAG "num_side_info"

#define TOL_TAG "tol"
#define MAX_ITER_TAG "max_iter"
#define DIRECT_TAG "direct"
#define THROW_ON_CHOLESKY_ERROR_TAG "throw_on_cholesky_error"
#define SIDE_INFO_PREFIX "side_info"

using namespace smurff;

double SideInfoConfig::BETA_PRECISION_DEFAULT_VALUE = 10.0;
double SideInfoConfig::TOL_DEFAULT_VALUE = 1e-6;
int SideInfoConfig::MAX_ITER_DEFAULT_VALUE = 10;

SideInfoConfig::SideInfoConfig()
{
   m_tol = SideInfoConfig::TOL_DEFAULT_VALUE;
   m_max_iter = SideInfoConfig::MAX_ITER_DEFAULT_VALUE;
   m_direct = false;
   m_throw_on_cholesky_error = false;
}

void SideInfoConfig::save(INIFile& writer, std::size_t prior_index, std::size_t config_item_index) const
{
   std::string sectionName = std::string(MACAU_PRIOR_CONFIG_ITEM_PREFIX_TAG) + "_" + std::to_string(prior_index) + "_" + std::to_string(config_item_index);

   //macau prior config item section
   writer.startSection(sectionName);

   //config item data
   writer.appendItem(sectionName, TOL_TAG, std::to_string(m_tol));
   writer.appendItem(sectionName, MAX_ITER_TAG, std::to_string(m_max_iter));
   writer.appendItem(sectionName, DIRECT_TAG, std::to_string(m_direct));
   writer.appendItem(sectionName, THROW_ON_CHOLESKY_ERROR_TAG, std::to_string(m_throw_on_cholesky_error));

   writer.endSection();

   std::string sideInfoName = std::string(SIDE_INFO_PREFIX) + "_" + std::to_string(prior_index);
   
   //config item side info
   TensorConfig::save_tensor_config(writer, sideInfoName, config_item_index, m_sideInfo);
}

bool SideInfoConfig::restore(const INIFile& reader, std::size_t prior_index, std::size_t config_item_index)
{
   auto add_index = [](const std::string name, int idx = -1) -> std::string
   {
      if (idx >= 0)
         return name + "_" + std::to_string(idx);
      return name;
   };

   std::stringstream section;
   section << MACAU_PRIOR_CONFIG_ITEM_PREFIX_TAG << "_" << prior_index << "_" << config_item_index;

   if (!reader.hasSection(section.str()))
   {
       return false;
   }

   //restore side info properties
   m_tol = reader.getReal(section.str(), TOL_TAG, SideInfoConfig::TOL_DEFAULT_VALUE);
   m_max_iter = reader.getReal(section.str(), MAX_ITER_TAG, SideInfoConfig::MAX_ITER_DEFAULT_VALUE);
   m_direct = reader.getBoolean(section.str(), DIRECT_TAG, false);
   m_throw_on_cholesky_error = reader.getBoolean(section.str(), THROW_ON_CHOLESKY_ERROR_TAG, false);

   std::stringstream ss;
   ss << SIDE_INFO_PREFIX << "_" << prior_index;

   auto tensor_cfg = TensorConfig::restore_tensor_config(reader, add_index(ss.str(), config_item_index));
   m_sideInfo = std::dynamic_pointer_cast<MatrixConfig>(tensor_cfg);

   return true;
}
