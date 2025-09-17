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

#
# Direct Dependencies
#

# Commit determined by:
# https://github.com/openxla/xla/blob/381088235cac2e4d438d87ada69cf92b13d798a2/third_party/absl/workspace.bzl#L10
http_archive(
    name = "abseil-cpp",
    patch_args = ["-p1"],
    patches = [
        "@xla//third_party/absl:nullability_macros.patch",
    ],
    sha256 = "a862ce94f77979ce36d2ca21ad3ca36b60838083392247b301b085a06d9f2b1a",
    strip_prefix = "abseil-cpp-54fac219c4ef0bc379dfffb0b8098725d77ac81b",
    url = "https://github.com/abseil/abseil-cpp/archive/54fac219c4ef0bc379dfffb0b8098725d77ac81b.tar.gz",
)

# Commit determined by:
# https://github.com/openxla/xla/blob/381088235cac2e4d438d87ada69cf92b13d798a2/tsl_workspace2.bzl#L283
http_archive(
    name = "com_google_protobuf",
    repo_mapping = {
        "@com_google_absl": "@abseil-cpp",
    },
    sha256 = "7fce939b9b7181bd0bd157360e0cc88a8cabf01ac4efe4662494f56dd955d4c1",
    strip_prefix = "protobuf-5.28.3",
    url = "https://github.com/protocolbuffers/protobuf/archive/refs/tags/v5.28.3.tar.gz",
)

# Commit determined by:
# https://github.com/openxla/xla/blob/381088235cac2e4d438d87ada69cf92b13d798a2/third_party/eigen3/workspace.bzl#L10
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
        "//third_party/federated_language:python_toolchain.patch",
        "//third_party/federated_language:structure_visibility.patch",
    ],
    repo_mapping = {
        "@federated_language_pypi": "@federated_language_jax_pypi",
        "@protobuf": "@com_google_protobuf",
    },
    sha256 = "dd767b67d054d8ab14e62a16edad376e25610f2edc5b78ccf41b473ded5662d4",
    strip_prefix = "federated-language-0.5.2",
    url = "https://github.com/google-parfait/federated-language/archive/refs/tags/v0.5.2.tar.gz",
)

# local_repository(
#     name = "federated_language_executor",
#     path = "third_party/federated_language_executor",
# )

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
        "//third_party/tensorflow_federated:protobuf_matchers.patch",
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
# https://github.com/openxla/xla/blob/381088235cac2e4d438d87ada69cf92b13d798a2/workspace2.bzl#L152
http_archive(
    name = "pybind11_protobuf",
    repo_mapping = {
        "@com_google_absl": "@abseil-cpp",
    },
    sha256 = "6c712ca5fc1e15df2ed1f55bd974c32a3065e658483616f3fda7607546db33ab",
    strip_prefix = "pybind11_protobuf-f02a2b7653bc50eb5119d125842a3870db95d251",
    url = "https://github.com/pybind/pybind11_protobuf/archive/f02a2b7653bc50eb5119d125842a3870db95d251.tar.gz",
)

http_archive(
    name = "rules_cc",
    sha256 = "d62624b45e0912713dcd3b8e30ba6ae55418ed6bf99e6d135cd61b8addae312b",
    strip_prefix = "rules_cc-0.1.2",
    url = "https://github.com/bazelbuild/rules_cc/archive/refs/tags/0.1.2.tar.gz",
)

http_archive(
    name = "rules_license",
    sha256 = "75759939aef3aeb726e801417a883deefadadb7fea49946a1f5bb74a5162e81e",
    strip_prefix = "rules_license-1.0.0",
    url = "https://github.com/bazelbuild/rules_license/archive/refs/tags/1.0.0.tar.gz",
)

http_archive(
    name = "rules_shell",
    sha256 = "99bfc7aaefd1ed69613bbd25e24bf7871d68aeafca3a6b79f5f85c0996a41355",
    strip_prefix = "rules_shell-0.5.1",
    url = "https://github.com/bazelbuild/rules_shell/releases/download/v0.5.1/rules_shell-v0.5.1.tar.gz",
)

http_archive(
    name = "xla",
    patches = [
        "//third_party/tsl:bazel_deps.patch",
        "//third_party/xla:bazel_deps.patch",
        "//third_party/xla:zlib.patch",
    ],
    repo_mapping = {
        "@com_google_absl": "@abseil-cpp",
        "@com_googlesource_code_re2": "@re2",
    },
    sha256 = "83547a447618229991e1dff77a10c0770e9b3370b5ac3669c6823e90f22c6814",
    strip_prefix = "xla-381088235cac2e4d438d87ada69cf92b13d798a2",
    url = "https://github.com/openxla/xla/archive/381088235cac2e4d438d87ada69cf92b13d798a2.tar.gz",
)

#
# Inlined Transitive Dependencies
#

# Required by `org_tensorflow_federated` and `xla`; commit determined by
# https://github.com/openxla/xla/blob/381088235cac2e4d438d87ada69cf92b13d798a2/tsl_workspace2.bzl#L339
http_archive(
    name = "com_github_grpc_grpc",
    patch_args = ["-p1"],
    patches = [
        "@xla//third_party/grpc:grpc.patch",
    ],
    repo_mapping = {
        "@com_google_absl": "@abseil-cpp",
        "@com_googlesource_code_re2": "@re2",
    },
    sha256 = "afbc5d78d6ba6d509cc6e264de0d49dcd7304db435cbf2d630385bacf49e066c",
    strip_prefix = "grpc-1.68.2",
    url = "https://github.com/grpc/grpc/archive/refs/tags/v1.68.2.tar.gz",
)

# Required by `pybind11_bazel`.
http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11-BUILD.bazel",
    sha256 = "e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20",
    strip_prefix = "pybind11-2.13.6",
    url = "https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz",
)

# Required by `xla`, the version of `rules_proto` that XLA depends on is incompatible with the
# version of `rules_python` that XLA depends on.
http_archive(
    name = "rules_proto",
    sha256 = "6fb6767d1bef535310547e03247f7518b03487740c11b6c6adb7952033fe1295",
    strip_prefix = "rules_proto-6.0.2",
    url = "https://github.com/bazelbuild/rules_proto/archive/refs/tags/6.0.2.tar.gz",
)

#
# Transitive Dependencies
#

# Required by `googletest`.
load("@googletest//:googletest_deps.bzl", "googletest_deps")

googletest_deps()

# Required by `rules_shell`.
load("@rules_shell//shell:repositories.bzl", "rules_shell_dependencies")

rules_shell_dependencies()

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
    "@rules_ml_toolchain//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
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
    "@rules_ml_toolchain//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@rules_ml_toolchain//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@rules_ml_toolchain//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

#
# CC Toolchains
#

load("@rules_ml_toolchain//cc_toolchain/deps:cc_toolchain_deps.bzl", "cc_toolchain_deps")

cc_toolchain_deps()

register_toolchains("@rules_ml_toolchain//cc_toolchain:lx64_lx64")

register_toolchains("@rules_ml_toolchain//cc_toolchain:lx64_lx64_cuda")

#
# Python Dependencies
#

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "federated_language_jax_pypi",
    python_interpreter_target = "@python_host//:python",
    requirements_lock = "//:requirements_lock_3_12.txt",
)

load("@federated_language_jax_pypi//:requirements.bzl", "install_deps")

install_deps()
