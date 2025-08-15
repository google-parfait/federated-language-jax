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
    name = "execution_contexts",
    srcs = ["execution_contexts.py"],
    data = [
        "//federated_language_jax/executor:xla_bindings",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:executor_bindings",
    ],
    deps = [
        "//federated_language_jax/computation:jax_computation",
        "//federated_language_jax/executor:executor_factory",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "execution_contexts_test",
    srcs = ["execution_contexts_test.py"],
    deps = [
        ":execution_contexts",
        "//federated_language_jax/computation:jax_computation",
        "@federated_language//federated_language",
        "@federated_language_jax_pypi//absl_py",
        "@federated_language_jax_pypi//numpy",
    ],
)
