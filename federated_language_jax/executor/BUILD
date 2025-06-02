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

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        "//federated_language_jax:__pkg__",
        "//federated_language_jax/backend:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "executor",
    srcs = ["__init__.py"],
)

py_library(
    name = "executor_factory",
    srcs = ["executor_factory.py"],
    data = ["@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:executor_bindings.so"],
    deps = [
        "@federated_language_jax_pypi//absl_py",
        "@federated_language_jax_pypi//cachetools",
        "@federated_language_jax_pypi//federated_language",
        "@org_tensorflow_federated//tensorflow_federated/python/core/impl/executors:cpp_to_python_executor",
    ],
)

pybind_extension(
    name = "xla_bindings",
    srcs = ["xla_bindings.cc"],
    deps = [
        ":xla_executor",
        "@federated_language//federated_language/proto:computation_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

cc_library(
    name = "xla_executor",
    srcs = ["xla_executor.cc"],
    hdrs = ["xla_executor.h"],
    copts = ["-Wthread-safety-analysis"],
    deps = [
        ":xla_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@federated_language//federated_language/proto:computation_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:status_macros",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:threading",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla:xla_proto_cc",
        "@xla//xla/client",
        "@xla//xla/client:client_library",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/service",
        "@xla//xla/service:cpu_plugin",  # buildcleaner: keep # TODO: b/419576470 - Move to test/binary targets.
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/stream_executor:platform",
        "@xla//xla/stream_executor:platform_manager",
    ],
)

cc_test(
    name = "xla_executor_test",
    srcs = ["xla_executor_test.cc"],
    args = ["--tff_xla_executor_test_platform=Host"],
    copts = ["-Wthread-safety-analysis"],
    deps = [
        ":xla_executor",
        ":xla_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@federated_language//federated_language/proto:array_cc_proto",
        "@federated_language//federated_language/proto:computation_cc_proto",
        "@federated_language//federated_language/proto:data_type_cc_proto",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:array_shape_test_utils",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:array_test_utils",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:status_macros",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:type_test_utils",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:value_test_utils",
        "@org_tensorflow_federated//tensorflow_federated/cc/testing:protobuf_matchers",
        "@org_tensorflow_federated//tensorflow_federated/cc/testing:status_matchers",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/builder:xla_builder",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/service:platform_util",
        "@xla//xla/stream_executor:platform",
    ],
)

cc_library(
    name = "xla_utils",
    srcs = ["xla_utils.cc"],
    hdrs = ["xla_utils.h"],
    copts = ["-Wthread-safety-analysis"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@federated_language//federated_language/proto:array_cc_proto",
        "@federated_language//federated_language/proto:computation_cc_proto",
        "@federated_language//federated_language/proto:data_type_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:status_macros",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla:types",
        "@xla//xla:xla_data_proto_cc",
    ],
)

cc_test(
    name = "xla_utils_test",
    srcs = ["xla_utils_test.cc"],
    copts = ["-Wthread-safety-analysis"],
    deps = [
        ":xla_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@eigen",
        "@federated_language//federated_language/proto:array_cc_proto",
        "@federated_language//federated_language/proto:computation_cc_proto",
        "@federated_language//federated_language/proto:data_type_cc_proto",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:array_shape_test_utils",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:array_test_utils",
        "@org_tensorflow_federated//tensorflow_federated/cc/testing:protobuf_matchers",
        "@org_tensorflow_federated//tensorflow_federated/cc/testing:status_matchers",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla:types",
        "@xla//xla:xla_data_proto_cc",
    ],
)
