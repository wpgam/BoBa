// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "py_tensor_train.hpp"

#include "detail/errors.hpp"

#include "BOBA/boba.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * \file
 * \brief Entry point of the private `pyboba._pyboba` extension module.
 *
 * The public Python surface lives in `pyboba/__init__.py`; nothing here is meant to be
 * imported directly by users.
 */

PYBIND11_MODULE(_pyboba, module)
{
  module.doc() =
    "Private compiled core of the pyboba package. Import pyboba instead.";

  // Backend handles and allocators. Round one is host-only, but init()/finalize() are
  // the same calls a C++ BoBa program makes, so enabling a device backend later needs no
  // change here.
  ::boba::init();
  py::module_::import("atexit").attr("register")(py::cpp_function([]()
  {
    ::boba::finalize();
  }));

  py::register_exception_translator([](std::exception_ptr error)
  {
    try
    {
      if (error)
      {
        std::rethrow_exception(error);
      }
    }
    catch (boba_python::MemoryErrorException const& exception)
    {
      py::set_error(PyExc_MemoryError, exception.what());
    }
  });

  auto tensor_train_class = boba_python::register_tensor_train(module);
  boba_python::register_construct(module);
  boba_python::register_algebra(tensor_train_class, module);
  boba_python::register_structure(module);
  boba_python::register_compress(module);
  boba_python::register_cross(module);
}
