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

load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

sh_binary(
    name = "build_package",
    srcs = ["build_package.sh"],
    data = [
        "//:LICENSE",
        "//:README.md",
        "//:pyproject.toml",
        "//federated_language_jax",
        "//federated_language_jax/backend",
        "//federated_language_jax/compiler",
        "//federated_language_jax/computation",
        "//federated_language_jax/executor",
    ],
)
