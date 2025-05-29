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

workspace(name = "federated_language_jax")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Direct Dependencies

http_archive(
    name = "federated_language",
    patches = [
        "//third_party/federated_language:proto_library_loads.patch",
        "//third_party/federated_language:structure_visibility.patch",
    ],
    repo_mapping = {
        "@federated_language_pypi": "@federated_language_jax_pypi",
        "@protobuf": "@com_google_protobuf",
    },
    sha256 = "927a692e698068a44df691713f6dd74a54eddc536e90c6b1c4e557d3ca7cb9e7",
    strip_prefix = "federated-language-bfc8b09943883cc425d82fd331888e8244a8f4c0",
    url = "https://github.com/google-parfait/federated-language/archive/bfc8b09943883cc425d82fd331888e8244a8f4c0.tar.gz",
)

# TODO: b/417987844 - Federated Language JAX should not depend on TFF.
http_archive(
    name = "org_tensorflow_federated",
    patches = [
        "//third_party/tensorflow_federated:bazel_deps.patch",
        "//third_party/tensorflow_federated:cpp_to_python_executor_visibility.patch",
        "//third_party/tensorflow_federated:executors_errors_deps.patch",
    ],
    sha256 = "fc887d436c2857acafcf5e44c136c041d99ad874b48e90aaac7e69a117e09bfc",
    strip_prefix = "tensorflow-federated-23792d84ad9ab6b1bfe28aed893be85d9c4374c3",
    url = "https://github.com/google-parfait/tensorflow-federated/archive/23792d84ad9ab6b1bfe28aed893be85d9c4374c3.tar.gz",
)

http_archive(
    name = "pybind11_abseil",
    sha256 = "1496b112e86416e2dcf288569a3e7b64f3537f0b18132224f492266e9ff76c44",
    strip_prefix = "pybind11_abseil-202402.0",
    url = "https://github.com/pybind/pybind11_abseil/archive/v202402.0.tar.gz",
)

http_archive(
    name = "pybind11_bazel",
    sha256 = "cae680670bfa6e82703c03f2a3c995408cdcbf43616d7bdd198ef45d3c327731",
    strip_prefix = "pybind11_bazel-2.13.6",
    url = "https://github.com/pybind/pybind11_bazel/archive/refs/tags/v2.13.6.tar.gz",
)

http_archive(
    name = "rules_cc",
    sha256 = "b26168b9a13f094794982b832975eaf53cefc5dced5b3be7df6b8b794dc2744b",
    strip_prefix = "rules_cc-0.0.12",
    url = "https://github.com/bazelbuild/rules_cc/archive/refs/tags/0.0.12.tar.gz",
)

http_archive(
    name = "rules_license",
    sha256 = "75759939aef3aeb726e801417a883deefadadb7fea49946a1f5bb74a5162e81e",
    strip_prefix = "rules_license-1.0.0",
    url = "https://github.com/bazelbuild/rules_license/archive/refs/tags/1.0.0.tar.gz",
)

http_archive(
    name = "xla",
    sha256 = "4d935ee2dac97cf55da02b2305decce0fde6a1f5c9f78b0db821104d6440b67f",
    strip_prefix = "xla-661559150498be4c186e74af3a0c60b1aae0c991",
    url = "https://github.com/openxla/xla/archive/661559150498be4c186e74af3a0c60b1aae0c991.zip",
)

# Transitive Dependencies, inlined

# Required by `pybind11_bazel`
http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11-BUILD.bazel",
    sha256 = "d0a116e91f64a4a2d8fb7590c34242df92258a61ec644b79127951e821b47be6",
    strip_prefix = "pybind11-2.13.6",
    url = "https://github.com/pybind/pybind11/archive/v2.13.6.zip",
)

# Required by `xla`, the version of `rules_proto` that XLA depends on is incompatible with the
# version of `rules_python` that XLA depends on.
http_archive(
    name = "rules_proto",
    sha256 = "6fb6767d1bef535310547e03247f7518b03487740c11b6c6adb7952033fe1295",
    strip_prefix = "rules_proto-6.0.2",
    url = "https://github.com/bazelbuild/rules_proto/archive/refs/tags/6.0.2.tar.gz",
)

# Transitive Dependencies, required by `xla`

load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    requirements = {
        "3.11": "//:requirements_lock_3_11.txt",
    },
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@xla//:workspace4.bzl", "xla_workspace4")

xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")

xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")

xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")

xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")

xla_workspace0()

load(
    "@xla//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@xla//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@xla//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@xla//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@xla//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

# Python Dependencies

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "federated_language_jax_pypi",
    python_interpreter_target = "@python_host//:python",
    requirements_lock = "//:requirements_lock_3_11.txt",
)

load("@federated_language_jax_pypi//:requirements.bzl", "install_deps")

install_deps()
