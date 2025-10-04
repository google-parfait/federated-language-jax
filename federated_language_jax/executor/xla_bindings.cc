/* Copyright 2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file contains the pybind11 definitions for exposing the XLA C++ Executor
// interface in Python.

#include "federated_language/proto/computation.pb.h"
#include "federated_language_executor/executor.pb.h"
#include "federated_language_jax/executor/xla_executor.h"
#include "include/pybind11/detail/common.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"
#include "include/pybind11/stl.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/status_casters.h"
#include "pybind11_protobuf/native_proto_caster.h"

namespace federated_language_jax {

namespace py = ::pybind11;

namespace {

////////////////////////////////////////////////////////////////////////////////
// The Python module definition `xla_bindings`.
//
// This will be used with `import xla_bindings` on the Python side. This
// module should _not_ be directly imported into the public pip API. The methods
// here will raise `NotOkStatus` errors from absl, which are not user friendly.
////////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(xla_bindings, m) {
  py::google::ImportStatusModule();

  // IMPORTANT: The binding defined in this module are dependent on the binding
  // defined in the `executor_bindings` module.
  py::module::import("federated_language_executor.executor_bindings_cc");

  m.doc() = "Bindings for the C++ ";

  m.def("create_xla_executor", &CreateXLAExecutor,
        py::arg("platform_name") = "Host", "Creates an XlaExecutor.");
}

}  // namespace
}  // namespace federated_language_jax
