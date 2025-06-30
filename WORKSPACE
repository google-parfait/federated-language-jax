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

# Commit determined by:
# https://github.com/openxla/xla/blob/661559150498be4c186e74af3a0c60b1aae0c991/third_party/absl/workspace.bzl#L10
http_archive(
    name = "abseil-cpp",
    patch_args = ["-p1"],
    patches = [
        "@xla//third_party/absl:nullability_macros.patch",
    ],
    sha256 = "0320586856674d16b0b7a4d4afb22151bdc798490bb7f295eddd8f6a62b46fea",
    strip_prefix = "abseil-cpp-fb3621f4f897824c0dbe0615fa94543df6192f30",
    url = "https://github.com/abseil/abseil-cpp/archive/fb3621f4f897824c0dbe0615fa94543df6192f30.tar.gz",
)

# Commit determined by:
# https://github.com/openxla/xla/blob/661559150498be4c186e74af3a0c60b1aae0c991/third_party/eigen3/workspace.bzl#L10
http_archive(
    name = "eigen",
    build_file = "//third_party:eigen.BUILD",
    sha256 = "1a432ccbd597ea7b9faa1557b1752328d6adc1a3db8969f6fe793ff704be3bf0",
    strip_prefix = "eigen-4c38131a16803130b66266a912029504f2cf23cd",
    url = "https://gitlab.com/libeigen/eigen/-/archive/4c38131a16803130b66266a912029504f2cf23cd/eigen-4c38131a16803130b66266a912029504f2cf23cd.tar.gz",
)

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

http_archive(
    name = "googletest",
    sha256 = "65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c",
    strip_prefix = "googletest-1.17.0",
    url = "https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz",
)

# TODO: b/417987844 - Federated Language JAX should not depend on TFF.
http_archive(
    name = "org_tensorflow_federated",
    patches = [
        "//third_party/tensorflow_federated:bazel_deps.patch",
        "//third_party/tensorflow_federated:cpp_to_python_executor_visibility.patch",
        "//third_party/tensorflow_federated:executors_errors_deps.patch",
    ],
    repo_mapping = {
        "@com_google_absl": "@abseil-cpp",
        "@com_google_googletest": "@googletest",
    },
    sha256 = "21ef65738f3ed1e3a0b9261f9374bd9d8f9219b9e8b41e1770955ec174f9f048",
    strip_prefix = "tensorflow-federated-fc32ea5c0c22dff1445b311b85c5a6d3b4eacd8c",
    url = "https://github.com/google-parfait/tensorflow-federated/archive/fc32ea5c0c22dff1445b311b85c5a6d3b4eacd8c.tar.gz",
)

http_archive(
    name = "pybind11_abseil",
    repo_mapping = {
        "@com_google_absl": "@abseil-cpp",
    },
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

# Commit determined by
# https://github.com/openxla/xla/blob/661559150498be4c186e74af3a0c60b1aae0c991/workspace2.bzl#L154
http_archive(
    name = "pybind11_protobuf",
    repo_mapping = {
        "@com_google_absl": "@abseil-cpp",
    },
    sha256 = "c7ab64b1ccf9a678694a89035a8c865a693e4e872803778f91f0965c2f281d78",
    strip_prefix = "pybind11_protobuf-80f3440cd8fee124e077e2e47a8a17b78b451363",
    url = "https://github.com/pybind/pybind11_protobuf/archive/80f3440cd8fee124e077e2e47a8a17b78b451363.zip",
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
    patches = [
        "//third_party/ducc:bazel_deps.patch",
        "//third_party/ml_dtypes:bazel_deps.patch",
        "//third_party/tsl:bazel_deps.patch",
        "//third_party/xla:bazel_deps.patch",
    ],
    repo_mapping = {
        "@com_google_absl": "@abseil-cpp",
        "@com_googlesource_code_re2": "@re2",
    },
    sha256 = "4d935ee2dac97cf55da02b2305decce0fde6a1f5c9f78b0db821104d6440b67f",
    strip_prefix = "xla-661559150498be4c186e74af3a0c60b1aae0c991",
    url = "https://github.com/openxla/xla/archive/661559150498be4c186e74af3a0c60b1aae0c991.zip",
)

#
# Inlined Transitive Dependencies
#

# Required by `org_tensorflow_federated` and `xla`, commit determined by
# https://github.com/google-parfait/tensorflow-federated/blob/fc32ea5c0c22dff1445b311b85c5a6d3b4eacd8c/WORKSPACE#L26
http_archive(
    name = "com_github_grpc_grpc",
    repo_mapping = {
        "@com_googlesource_code_re2": "@re2",
    },
    sha256 = "76900ab068da86378395a8e125b5cc43dfae671e09ff6462ddfef18676e2165a",
    strip_prefix = "grpc-1.50.0",
    url = "https://github.com/grpc/grpc/archive/refs/tags/v1.50.0.tar.gz",
)

# Required by `pybind11_bazel`.
http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11-BUILD.bazel",
    sha256 = "d0a116e91f64a4a2d8fb7590c34242df92258a61ec644b79127951e821b47be6",
    strip_prefix = "pybind11-2.13.6",
    url = "https://github.com/pybind/pybind11/archive/v2.13.6.zip",
)

# Required by `googletest`.
http_archive(
    name = "re2",
    sha256 = "eb2df807c781601c14a260a507a5bb4509be1ee626024cb45acbd57cb9d4032b",
    strip_prefix = "re2-2024-07-02",
    url = "https://github.com/google/re2/archive/refs/tags/2024-07-02.tar.gz",
)

# Required by `xla`, the version of `rules_proto` that XLA depends on is incompatible with the
# version of `rules_python` that XLA depends on.
http_archive(
    name = "rules_proto",
    sha256 = "6fb6767d1bef535310547e03247f7518b03487740c11b6c6adb7952033fe1295",
    strip_prefix = "rules_proto-6.0.2",
    url = "https://github.com/bazelbuild/rules_proto/archive/refs/tags/6.0.2.tar.gz",
)

# Required by com_github_grpc_grpc. This commit is determined by
# https://github.com/grpc/grpc/blob/v1.50.0/bazel/grpc_deps.bzl#L344.
http_archive(
    name = "upb",
    sha256 = "017a7e8e4e842d01dba5dc8aa316323eee080cd1b75986a7d1f94d87220e6502",
    strip_prefix = "upb-e4635f223e7d36dfbea3b722a4ca4807a7e882e2",
    url = "https://github.com/protocolbuffers/upb/archive/e4635f223e7d36dfbea3b722a4ca4807a7e882e2.tar.gz",
)

#
# Transitive Dependencies
#

# Required by `xla`.
load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    requirements = {
        "3.12": "//:requirements_lock_3_12.txt",
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
    requirements_lock = "//:requirements_lock_3_12.txt",
)

load("@federated_language_jax_pypi//:requirements.bzl", "install_deps")

install_deps()
