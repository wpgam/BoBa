# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Python bindings. Only processed when BOBA_BUILD_PYTHON is ON, so a normal C++ build
# never needs Python headers or pybind11.

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(boba_python_sources
    bindings/module.cpp
    bindings/tensor_train.cpp
    bindings/construct.cpp
    bindings/algebra.cpp
    bindings/compress.cpp
    bindings/cross.cpp
    bindings/detail/runtime_tensor_train.cpp
    bindings/detail/runtime_rounding.cpp
    bindings/detail/runtime_algebra.cpp
    bindings/detail/runtime_cross.cpp)

pybind11_add_module(_pyboba MODULE ${boba_python_sources})

target_include_directories(_pyboba PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bindings)

# Inherit BoBa's include paths, defines and TPL link line rather than restating them.
target_link_libraries(_pyboba PRIVATE BOBA)

target_compile_features(_pyboba PRIVATE cxx_std_20)
set_target_properties(_pyboba PROPERTIES CXX_EXTENSIONS OFF)

if(SKBUILD)
  # Wheel layout: the compiled extension lands next to the pure-Python package that
  # scikit-build-core copies from python/pyboba.
  install(TARGETS _pyboba LIBRARY DESTINATION pyboba RUNTIME DESTINATION pyboba)
endif()
