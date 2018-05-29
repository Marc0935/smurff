#SETUP PROJECT
set (PROJECT smurff-cpp)
message("Configuring " ${PROJECT} "...")
project (${PROJECT})

FILE (GLOB UTIL_FILES   "../Utils/TruncNorm.h"
                        "../Utils/InvNormCdf.h"
                        "../Utils/Distribution.h"
                        "../Utils/MatrixUtils.h"
                        "../Utils/TensorUtils.h"
                        "../Utils/PVec.hpp"
                        "../Utils/counters.h"
                        "../Utils/linop.h"
                        "../Utils/omp_util.h"
                        "../Utils/Error.h"
                        "../Utils/ThreadVector.hpp"
                        "../Utils/RootFile.h"
                        "../Utils/StepFile.h"
                        "../Utils/StringUtils.h"

                        "../Utils/TruncNorm.cpp"
                        "../Utils/InvNormCdf.cpp"
                        "../Utils/Distribution.cpp"
                        "../Utils/MatrixUtils.cpp"
                        "../Utils/TensorUtils.cpp"
                        "../Utils/counters.cpp"
                        "../Utils/linop.cpp"
                        "../Utils/omp_util.cpp"
                        "../Utils/RootFile.cpp"
                        "../Utils/StepFile.cpp"
                        "../Utils/StringUtils.cpp"
                        )

source_group ("Utils" FILES ${UTIL_FILES})

FILE (GLOB IO_FILES     "../IO/ini.h"
                        "../IO/INIFile.h"
                        "../IO/GenericIO.h"
                        "../IO/MatrixIO.h"
                        "../IO/TensorIO.h"
                        "../IO/IDataWriter.h"
                        "../IO/DataWriter.h"

                        "../IO/ini.c"
                        "../IO/INIFile.cpp"
                        "../IO/GenericIO.cpp"
                        "../IO/MatrixIO.cpp"
                        "../IO/TensorIO.cpp"
                        "../IO/DataWriter.cpp"
                        )

source_group ("IO" FILES ${IO_FILES})

FILE (GLOB CONFIG_FILES "../Configs/Config.h"
                        "../Configs/MatrixConfig.h"
                        "../Configs/TensorConfig.h"
                        "../Configs/NoiseConfig.h"
                        "../Configs/SideInfoConfig.h"

                        "../Configs/Config.cpp"
                        "../Configs/MatrixConfig.cpp"
                        "../Configs/TensorConfig.cpp"
                        "../Configs/NoiseConfig.cpp"
                        "../Configs/SideInfoConfig.cpp"
                        )

source_group ("Configs" FILES ${CONFIG_FILES})

FILE (GLOB NOISE_FILES  "../Noises/GaussianNoise.h"
                        "../Noises/AdaptiveGaussianNoise.h"
                        "../Noises/FixedGaussianNoise.h"
                        "../Noises/ProbitNoise.h"
                        "../Noises/UnusedNoise.h"
                        "../Noises/INoiseModel.h"
                        "../Noises/NoiseFactory.h"

                        "../Noises/INoiseModel.cpp"
                        "../Noises/GaussianNoise.cpp"
                        "../Noises/AdaptiveGaussianNoise.cpp"
                        "../Noises/FixedGaussianNoise.cpp"
                        "../Noises/ProbitNoise.cpp"
                        "../Noises/UnusedNoise.cpp"
                        "../Noises/NoiseFactory.cpp"
                        )

source_group ("Noises" FILES ${NOISE_FILES})

FILE (GLOB PRIOR_FILES  "../Priors/ILatentPrior.h"
                        "../Priors/NormalPrior.h"
                        "../Priors/NormalOnePrior.h"
                        "../Priors/SpikeAndSlabPrior.h"
                        "../Priors/MacauPrior.h"
                        "../Priors/MacauOnePrior.h"
                        "../Priors/IPriorFactory.h"
                        "../Priors/PriorFactory.h"

                        "../Priors/ILatentPrior.cpp"
                        "../Priors/NormalPrior.cpp"
                        "../Priors/NormalOnePrior.cpp"
                        "../Priors/SpikeAndSlabPrior.cpp"
                        "../Priors/MacauPrior.cpp"
                        "../Priors/MacauOnePrior.cpp"
                        "../Priors/PriorFactory.cpp"
                        )

source_group ("Priors" FILES ${PRIOR_FILES})

FILE (GLOB MATRIX_FILES "../DataMatrices/Data.h"
                        "../DataMatrices/DenseMatrixData.h"
                        "../DataMatrices/FullMatrixData.hpp"
                        "../DataMatrices/MatricesData.h"
                        "../DataMatrices/MatrixData.h"
                        "../DataMatrices/MatrixDataTempl.hpp"
                        "../DataMatrices/ScarceMatrixData.h"
                        "../DataMatrices/SparseMatrixData.h"
                        "../DataMatrices/IDataCreator.h"
                        "../DataMatrices/DataCreator.h"
                        "../DataMatrices/DataCreatorBase.h"

                        "../DataMatrices/Data.cpp"
                        "../DataMatrices/DenseMatrixData.cpp"
                        "../DataMatrices/MatricesData.cpp"
                        "../DataMatrices/MatrixData.cpp"
                        "../DataMatrices/ScarceMatrixData.cpp"
                        "../DataMatrices/SparseMatrixData.cpp"
                        "../DataMatrices/DataCreator.cpp"
                        "../DataMatrices/DataCreatorBase.cpp"
                        )

source_group ("DataMatrices" FILES ${MATRIX_FILES})

FILE (GLOB TENSOR_FILES "../DataTensors/TensorData.h"
                        "../DataTensors/SparseMode.h"
                        "../DataTensors/TensorData.cpp"
                        "../DataTensors/SparseMode.cpp"
                        )

source_group ("DataTensors" FILES ${TENSOR_FILES})

FILE (GLOB SESSION_FILES "../Sessions/ISession.h"
                         "../Sessions/BaseSession.h"
                         "../Sessions/Session.h"
                         "../Sessions/PythonSession.h"
                         "../Sessions/SessionFactory.h"

                         "../Sessions/BaseSession.cpp"
                         "../Sessions/Session.cpp"
                         "../Sessions/PythonSession.cpp"
                         "../Sessions/SessionFactory.cpp")

source_group ("Sessions" FILES ${SESSION_FILES})

FILE (GLOB HEADER_FILES "../Version.h"
                        "../Model.h"
                        "../result.h"
                        "../VMatrixIterator.hpp"
                        "../ConstVMatrixIterator.hpp"
                        "../VMatrixExprIterator.hpp"
                        "../ConstVMatrixExprIterator.hpp"
                        )

source_group ("Header Files" FILES ${HEADER_FILES})

FILE (GLOB SOURCE_FILES "../Version.cpp"
                        "../Model.cpp"
                        "../result.cpp"
                        )
source_group ("Source Files" FILES ${SOURCE_FILES})

FILE (GLOB SIDE_INFO_FILES "../SideInfo/ISideInfo.h"
                           "../SideInfo/SparseFeat.h"
                           "../SideInfo/SparseDoubleFeat.h"
                           "../SideInfo/DenseDoubleFeat.h"
                           "../SideInfo/LibFastSparseDependency.h"
                           "../SideInfo/DenseDoubleSideInfo.h"
                           "../SideInfo/DenseDoubleSideInfo.cpp"
                           "../SideInfo/SparseDoubleSideInfo.h"
                           "../SideInfo/SparseDoubleSideInfo.cpp"
                           "../SideInfo/SparseBinarySideinfo.h"
                           "../SideInfo/SparseBinarySideinfo.cpp"
                        )
source_group ("Side Info" FILES ${SIDE_INFO_FILES})

#SETUP OUTPUT
add_library (${PROJECT} STATIC ${HEADER_FILES}
                               ${SOURCE_FILES}
                               ${UTIL_FILES}
                               ${IO_FILES}
                               ${CONFIG_FILES}
                               ${NOISE_FILES}
                               ${PRIOR_FILES}
                               ${MATRIX_FILES}
                               ${TENSOR_FILES}
                               ${SESSION_FILES}
                               ${SIDE_INFO_FILES}
                               )
set_property(TARGET ${PROJECT} PROPERTY FOLDER "Libraries")


#SETUP INCLUDES
include_directories(../)
include_directories(../..)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${BOOST_RANDOM_INCLUDE_DIRS})

#INSTALL
install (TARGETS ${PROJECT} ARCHIVE DESTINATION lib)
install (FILES ../ResultItem.h DESTINATION include/SmurffCpp)
install (FILES ../Utils/Error.h DESTINATION include/SmurffCpp/Utils)
install (FILES ../Utils/PVec.hpp DESTINATION include/SmurffCpp/Utils)
install (FILES ../Configs/Config.h DESTINATION include/SmurffCpp/Configs)
install (FILES ../Configs/MatrixConfig.h DESTINATION include/SmurffCpp/Configs)
install (FILES ../Configs/TensorConfig.h DESTINATION include/SmurffCpp/Configs)
install (FILES ../Configs/NoiseConfig.h DESTINATION include/SmurffCpp/Configs)
install (FILES ../Configs/SideInfoConfig.h DESTINATION include/SmurffCpp/Configs)
install (FILES ../Sessions/SessionFactory.h DESTINATION include/SmurffCpp/Sessions)
install (FILES ../Sessions/ISession.h DESTINATION include/SmurffCpp/Sessions)
install (FILES ../IO/INIFile.h DESTINATION include/SmurffCpp/IO)
