from libcpp cimport bool
from libcpp.memory cimport shared_ptr

from MatrixConfig cimport MatrixConfig

cdef extern from "<SmurffCpp/Configs/SideInfoConfig.h>" namespace "smurff":
    cdef cppclass SideInfoConfig:
        MacauPriorConfigItem() except +
        void setSideInfo(shared_ptr[MatrixConfig] value)
        void setTol(double value)
        void setMaxIter(int value)
        void setDirect(bool value)
