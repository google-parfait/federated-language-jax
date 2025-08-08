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
        "//federated_language_jax/backend:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "computation",
    srcs = ["__init__.py"],
)

py_library(
    name = "jax_computation",
    srcs = ["jax_computation.py"],
    deps = [
        ":jax_serialization",
        "@federated_language_jax_pypi//dm_tree",
        "@federated_language_jax_pypi//federated_language",
        "@federated_language_jax_pypi//jax",
        "@federated_language_jax_pypi//numpy",
    ],
)

py_test(
    name = "jax_computation_test",
    srcs = ["jax_computation_test.py"],
    deps = [
        ":jax_computation",
        "@federated_language_jax_pypi//absl_py",
        "@federated_language_jax_pypi//federated_language",
        "@federated_language_jax_pypi//jax",
        "@federated_language_jax_pypi//ml_dtypes",
        "@federated_language_jax_pypi//numpy",
    ],
)

py_library(
    name = "jax_computation_context",
    srcs = ["jax_computation_context.py"],
    deps = ["@federated_language_jax_pypi//federated_language"],
)

py_test(
    name = "jax_computation_context_test",
    srcs = ["jax_computation_context_test.py"],
    deps = [
        ":jax_computation",
        ":jax_computation_context",
        "@federated_language_jax_pypi//absl_py",
        "@federated_language_jax_pypi//numpy",
    ],
)

py_library(
    name = "jax_serialization",
    srcs = ["jax_serialization.py"],
    deps = [
        ":jax_computation_context",
        ":xla_serialization",
        "@federated_language//federated_language/common_libs:structure",
        "@federated_language//federated_language/proto:computation_py_pb2",
        "@federated_language_jax_pypi//federated_language",
        "@federated_language_jax_pypi//jax",
        "@federated_language_jax_pypi//numpy",
    ],
)

py_test(
    name = "jax_serialization_test",
    srcs = ["jax_serialization_test.py"],
    deps = [
        ":jax_serialization",
        "@federated_language//federated_language/common_libs:structure",
        "@federated_language//federated_language/proto:computation_py_pb2",
        "@federated_language_jax_pypi//absl_py",
        "@federated_language_jax_pypi//federated_language",
        "@federated_language_jax_pypi//jax",
        "@federated_language_jax_pypi//numpy",
    ],
)

py_library(
    name = "xla_serialization",
    srcs = ["xla_serialization.py"],
    deps = [
        "@federated_language//federated_language/proto:computation_py_pb2",
        "@federated_language_jax_pypi//federated_language",
        "@federated_language_jax_pypi//protobuf",
        "@xla//xla/python:xla_client",
    ],
)

py_test(
    name = "xla_serialization_test",
    srcs = ["xla_serialization_test.py"],
    deps = [
        ":xla_serialization",
        "@federated_language//federated_language/proto:computation_py_pb2",
        "@federated_language_jax_pypi//absl_py",
        "@federated_language_jax_pypi//federated_language",
        "@federated_language_jax_pypi//jax",
        "@federated_language_jax_pypi//numpy",
        "@federated_language_jax_pypi//protobuf",
    ],
)
