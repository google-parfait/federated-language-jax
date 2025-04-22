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
    ],
)

licenses(["notice"])

py_library(
    name = "xla",
    srcs = ["__init__.py"],
)

py_library(
    name = "cpp_execution_contexts",
    srcs = ["cpp_execution_contexts.py"],
    deps = [
        "//federated_language_jax/environments/jax_frontend:jax_computation",
        "//federated_language_jax/environments/xla_backend:xla_executor_bindings",
        "@federated_language_jax_pypi//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
        "@org_tensorflow_federated//tensorflow_federated/python/core/impl/executor_stacks:cpp_executor_factory",
        "@org_tensorflow_federated//tensorflow_federated/python/core/impl/executors:executor_bindings",
    ],
)

py_test(
    name = "cpp_execution_contexts_test",
    srcs = ["cpp_execution_contexts_test.py"],
    deps = [
        ":cpp_execution_contexts",
        "//federated_language_jax/environments/jax_frontend:jax_computation",
        "@federated_language_jax_pypi//federated_language",
        "@federated_language_jax_pypi//numpy",
        "@federated_language_pypi//absl_py",
    ],
)
