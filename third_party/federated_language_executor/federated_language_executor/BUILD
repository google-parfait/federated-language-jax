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

load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_library(
    name = "federated_language_executor",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cpp_to_python_executor",
        ":data_conversions",
        ":executor_bindings",
        ":executor_errors",
        ":value_serialization",
        ":version",
    ],
)

py_library(
    name = "cpp_to_python_executor",
    srcs = ["cpp_to_python_executor.py"],
    deps = [
        ":executor_bindings",
        ":executor_errors",
        ":value_serialization",
        "@federated_language_executor_pypi//federated_language",
    ],
)

py_test(
    name = "cpp_to_python_executor_test",
    srcs = ["cpp_to_python_executor_test.py"],
    deps = [
        ":cpp_to_python_executor",
        ":executor_bindings",
        ":value_serialization",
        "@federated_language_executor_pypi//absl_py",
        "@federated_language_executor_pypi//federated_language",
        "@federated_language_executor_pypi//numpy",
    ],
)

py_library(
    name = "data_conversions",
    srcs = ["data_conversions.py"],
    deps = ["@federated_language_executor_pypi//federated_language"],
)

py_test(
    name = "data_conversions_test",
    srcs = ["data_conversions_test.py"],
    deps = [
        ":data_conversions",
        "@federated_language_executor_pypi//absl_py",
        "@federated_language_executor_pypi//federated_language",
    ],
)

proto_library(
    name = "executor_proto",
    srcs = ["executor.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@federated_language//federated_language/proto:array_proto",
        "@federated_language//federated_language/proto:computation_proto",
    ],
)

py_proto_library(
    name = "executor_py_pb2",
    visibility = ["//visibility:public"],
    deps = [":executor_proto"],
)

py_grpc_library(
    name = "executor_py_pb2_grpc",
    srcs = [":executor_proto"],
    visibility = ["//visibility:public"],
    deps = [":executor_py_pb2"],
)

cc_proto_library(
    name = "executor_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":executor_proto"],
)

cc_grpc_library(
    name = "executor_cc_grpc_proto",
    srcs = [":executor_proto"],
    grpc_only = True,
    visibility = ["//visibility:public"],
    deps = [":executor_cc_proto"],
)

py_library(
    name = "executor_bindings",
    srcs = ["executor_bindings.py"],
    data = [":executor_bindings_cc"],
    deps = [
        ":data_conversions",
        "@federated_language_executor_pypi//federated_language",
    ],
)

py_test(
    name = "executor_bindings_test",
    srcs = ["executor_bindings_test.py"],
    data = [":executor_test_utils_bindings"],
    deps = [
        ":executor_bindings",
        "@federated_language_executor_pypi//absl_py",
        "@federated_language_executor_pypi//federated_language",
        "@federated_language_executor_pypi//portpicker",
    ],
)

pybind_extension(
    name = "executor_bindings_cc",
    srcs = ["executor_bindings.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":executor_cc_proto",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@com_github_grpc_grpc//:grpc++",
        "@federated_language//federated_language/proto:computation_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:composing_executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:federating_executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:reference_resolving_executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:remote_executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:sequence_executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:streaming_remote_executor",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

py_library(
    name = "executor_errors",
    srcs = ["executor_errors.py"],
    deps = [
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@federated_language_executor_pypi//federated_language",
        "@federated_language_executor_pypi//typing_extensions",
    ],
)

pybind_extension(
    name = "executor_test_utils_bindings",
    testonly = True,
    srcs = ["executor_test_utils_bindings.cc"],
    visibility = ["//visibility:public"],
    deps = ["@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:mock_executor"],
)

py_library(
    name = "executor_utils",
    srcs = ["executor_utils.py"],
    deps = ["@federated_language_executor_pypi//federated_language"],
)

py_test(
    name = "executor_utils_test",
    srcs = ["executor_utils_test.py"],
    deps = [
        ":executor_utils",
        "@federated_language_executor_pypi//absl_py",
        "@federated_language_executor_pypi//federated_language",
        "@federated_language_executor_pypi//numpy",
    ],
)

py_library(
    name = "value_serialization",
    srcs = ["value_serialization.py"],
    deps = [
        ":executor_py_pb2",
        ":executor_utils",
        "@federated_language//federated_language/common_libs:structure",
        "@federated_language//federated_language/proto:array_py_pb2",
        "@federated_language//federated_language/proto:computation_py_pb2",
        "@federated_language_executor_pypi//dm_tree",
        "@federated_language_executor_pypi//federated_language",
        "@federated_language_executor_pypi//numpy",
    ],
)

py_test(
    name = "value_serialization_test",
    srcs = ["value_serialization_test.py"],
    deps = [
        ":executor_py_pb2",
        ":value_serialization",
        "@federated_language//federated_language/common_libs:structure",
        "@federated_language//federated_language/proto:computation_py_pb2",
        "@federated_language_executor_pypi//absl_py",
        "@federated_language_executor_pypi//federated_language",
        "@federated_language_executor_pypi//numpy",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
)
