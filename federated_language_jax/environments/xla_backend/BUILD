# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        "//federated_language_jax:__pkg__",
        "//federated_language_jax/backends/xla:__pkg__",
        "//federated_language_jax/environments/jax_frontend:__pkg__",
    ],
)

licenses(["notice"])

py_library(
    name = "xla_backend",
    srcs = ["__init__.py"],
)

py_library(
    name = "xla_executor_bindings",
    srcs = ["xla_executor_bindings.py"],
    deps = ["@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:xla_bindings"],
)

py_test(
    name = "xla_executor_bindings_test",
    srcs = ["xla_executor_bindings_test.py"],
    deps = [
        ":xla_executor_bindings",
        "@federated_language_pypi//absl_py",
    ],
)

py_library(
    name = "xla_serialization",
    srcs = ["xla_serialization.py"],
    deps = [
        "@federated_language//federated_language/proto:computation_py_pb2",
        "@federated_language_jax_pypi//federated_language",
        "@federated_language_jax_pypi//jax",
        "@federated_language_jax_pypi//protobuf",
        "@org_tensorflow_federated//tensorflow_federated",
    ],
)

py_test(
    name = "xla_serialization_test",
    srcs = ["xla_serialization_test.py"],
    deps = [
        ":xla_serialization",
        "@federated_language//federated_language/proto:computation_py_pb2",
        "@federated_language_jax_pypi//federated_language",
        "@federated_language_jax_pypi//jax",
        "@federated_language_jax_pypi//numpy",
        "@federated_language_jax_pypi//protobuf",
        "@federated_language_pypi//absl_py",
    ],
)
