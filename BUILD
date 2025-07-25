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

load("@python//:defs.bzl", "compile_pip_requirements")
load("@rules_license//rules:license.bzl", "license")

# TODO: b/419584204 - Enable all supported versions of Python.
# load("@python//3.10:defs.bzl", compile_pip_requirements_3_10 = "compile_pip_requirements")
# load("@python//3.11:defs.bzl", compile_pip_requirements_3_11 = "compile_pip_requirements")
# load("@python//3.12:defs.bzl", compile_pip_requirements_3_12 = "compile_pip_requirements")
# load("@python//3.13:defs.bzl", compile_pip_requirements_3_13 = "compile_pip_requirements")
# load("@python//3.9:defs.bzl", compile_pip_requirements_3_9 = "compile_pip_requirements")

package(
    default_applicable_licenses = [":package_license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "package_license",
    package_name = "federated_language_jax",
    license_kinds = ["@rules_license//licenses/spdx:Apache-2.0"],
)

licenses(["notice"])

exports_files([
    "LICENSE",
    "pyproject.toml",
    "requirements_lock_3_10.txt",
    "requirements_lock_3_11.txt",
    "requirements_lock_3_12.txt",
    "requirements_lock_3_13.txt",
    "requirements_lock_3_9.txt",
    "requirements.in",
])

# TODO: b/419584204 - Enable all supported versions of Python.
# compile_pip_requirements_3_9(
#     name = "requirements_3_9",
#     src = "//:requirements.in",
#     extra_args = [
#         "--strip-extras",
#     ],
#     requirements_txt = "//:requirements_lock_3_9.txt",
# )

# TODO: b/419584204 - Enable all supported versions of Python.
# compile_pip_requirements_3_10(
#     name = "requirements_3_10",
#     src = "//:requirements.in",
#     extra_args = [
#         "--strip-extras",
#     ],
#     requirements_txt = "//:requirements_lock_3_10.txt",
# )

# TODO: b/419584204 - Enable all supported versions of Python.
# compile_pip_requirements_3_11(
#     name = "requirements_3_11",
#     src = "//:requirements.in",
#     extra_args = [
#         "--strip-extras",
#     ],
#     requirements_txt = "//:requirements_lock_3_11.txt",
# )

compile_pip_requirements(
    name = "requirements_3_12",
    src = "//:requirements.in",
    extra_args = [
        "--strip-extras",
    ],
    requirements_txt = "//:requirements_lock_3_12.txt",
)

# TODO: b/419584204 - Enable all supported versions of Python.
# compile_pip_requirements_3_13(
#     name = "requirements_3_13",
#     src = "//:requirements.in",
#     extra_args = [
#         "--strip-extras",
#     ],
#     requirements_txt = "//:requirements_lock_3_13.txt",
# )
