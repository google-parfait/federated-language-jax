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
# https://github.com/openxla/xla/blob/f73bbc1a0dd4bcbdacf6420bd0f517eb2d3fd177/third_party/absl/workspace.bzl#L25
http_archive(
    name = "abseil-cpp",
    patch_args = ["-p1"],
    patches = [
        "@xla//third_party/absl:btree.patch",
        "@xla//third_party/absl:build_dll.patch",
        "@xla//third_party/absl:endian.patch",
        "@xla//third_party/absl:raw_hash_set.patch",
    ],
    sha256 = "6e1aee535473414164bf83e4ebc40240dec71a4701f8a642d906e95bea1aea0c",
    strip_prefix = "abseil-cpp-20260526.0",
    url = "https://github.com/abseil/abseil-cpp/archive/20260526.0.tar.gz",
)

# Commit determined by:
# https://github.com/openxla/xla/blob/381088235cac2e4d438d87ada69cf92b13d798a2/third_party/eigen3/workspace.bzl#L10
http_archive(
    name = "eigen",
    build_file = "//third_party:eigen.BUILD",
    sha256 = "1a432ccbd597ea7b9faa1557b1752328d6adc1a3db8969f6fe793ff704be3bf0",
    strip_prefix = "eigen-4c38131a16803130b66266a912029504f2cf23cd",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/4c38131a16803130b66266a912029504f2cf23cd/eigen-4c38131a16803130b66266a912029504f2cf23cd.tar.gz",
        "https://gitlab.com/libeigen/eigen/-/archive/4c38131a16803130b66266a912029504f2cf23cd/eigen-4c38131a16803130b66266a912029504f2cf23cd.tar.gz",
    ],
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
    sha256 = "d99f3a4cc88f816391777fc53cdc094d9abd72bdd7a6fe2bdccfa43235f351db",
    strip_prefix = "federated-language-0.5.4",
    url = "https://github.com/google-parfait/federated-language/archive/refs/tags/v0.5.4.tar.gz",
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
    ],
    repo_mapping = {
        "@com_google_absl": "@abseil-cpp",
        "@com_google_googletest": "@googletest",
        "@eigen": "@eigen",
    },
    sha256 = "58a0c08290c3f701bdcb30581c95049f50fec26008686162370b558f76d31027",
    strip_prefix = "tensorflow-federated-3f3e1dca6cb2256b2e20db90cf0298b1fbe1d6f8",
    url = "https://github.com/google-parfait/tensorflow-federated/archive/3f3e1dca6cb2256b2e20db90cf0298b1fbe1d6f8.tar.gz",
)

# Commit determined by
# https://github.com/openxla/xla/blob/f73bbc1a0dd4bcbdacf6420bd0f517eb2d3fd177/third_party/pybind11_abseil/workspace.bzl#L29
http_archive(
    name = "pybind11_abseil",
    repo_mapping = {
        "@com_google_absl": "@abseil-cpp",
    },
    sha256 = "c6d0c6784e4d5681919731f1fa86e0b7cd010e770115bdb3a0285b3939ef2394",
    strip_prefix = "pybind11_abseil-13d4f99d5309df3d5afa80fe2ae332d7a2a64c6b",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pybind/pybind11_abseil/archive/13d4f99d5309df3d5afa80fe2ae332d7a2a64c6b.tar.gz",
        "https://github.com/pybind/pybind11_abseil/archive/13d4f99d5309df3d5afa80fe2ae332d7a2a64c6b.tar.gz",
    ],
)

http_archive(
    name = "pybind11_bazel",
    sha256 = "cae680670bfa6e82703c03f2a3c995408cdcbf43616d7bdd198ef45d3c327731",
    strip_prefix = "pybind11_bazel-2.13.6",
    url = "https://github.com/pybind/pybind11_bazel/archive/refs/tags/v2.13.6.tar.gz",
)

# Commit determined by
# https://github.com/openxla/xla/blob/f73bbc1a0dd4bcbdacf6420bd0f517eb2d3fd177/workspace2.bzl#L198
http_archive(
    name = "pybind11_protobuf",
    repo_mapping = {
        "@com_google_absl": "@abseil-cpp",
    },
    sha256 = "6c712ca5fc1e15df2ed1f55bd974c32a3065e658483616f3fda7607546db33ab",
    strip_prefix = "pybind11_protobuf-f02a2b7653bc50eb5119d125842a3870db95d251",
    url = "https://github.com/pybind/pybind11_protobuf/archive/f02a2b7653bc50eb5119d125842a3870db95d251.tar.gz",
)

# Commit determined by:
# https://github.com/openxla/xla/blob/f73bbc1a0dd4bcbdacf6420bd0f517eb2d3fd177/third_party/riegeli/workspace.bzl#L22
http_archive(
    name = "riegeli",
    patch_args = ["-p1"],
    patches = [
        "@xla//third_party/riegeli:layering_check.patch",
    ],
    repo_mapping = {
        "@com_google_absl": "@abseil-cpp",
    },
    sha256 = "f63337f63f794ba9dc7dd281b20af3d036dfe0c1a5a4b7b8dc20b39f7e323b97",
    strip_prefix = "riegeli-9f2744dc23e81d84c02f6f51244e9e9bb9802d57",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/riegeli/archive/9f2744dc23e81d84c02f6f51244e9e9bb9802d57.tar.gz",
        "https://github.com/google/riegeli/archive/9f2744dc23e81d84c02f6f51244e9e9bb9802d57.tar.gz",
    ],
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
    url = "https://github.com/bazelbuild/rules_shell/archive/refs/tags/v0.5.1.tar.gz",
)

http_archive(
    name = "remote_coverage_tools",
    sha256 = "7006375f6756819b7013ca875eab70a541cf7d89142d9c511ed78ea4fefa38af",
    urls = [
        "https://storage.googleapis.com/bazel-mirror/bazel_coverage_output_generator/releases/coverage_output_generator-v2.6.zip",
        "https://mirror.bazel.build/bazel_coverage_output_generator/releases/coverage_output_generator-v2.6.zip",
    ],
)

http_archive(
    name = "xla",
    patch_args = ["-p1"],
    patches = [
        "//third_party/xla:workspace4.patch",
    ],
    repo_mapping = {
        "@com_google_absl": "@abseil-cpp",
        "@com_googlesource_code_re2": "@re2",
        "@eigen_archive": "@eigen",
    },
    sha256 = "77f34f7b925bdfe72d40f0f6ede96c55e3cf7e2599cd3e0da2f3046886b87557",
    strip_prefix = "xla-f73bbc1a0dd4bcbdacf6420bd0f517eb2d3fd177",
    url = "https://github.com/openxla/xla/archive/f73bbc1a0dd4bcbdacf6420bd0f517eb2d3fd177.tar.gz",
)

load("@xla//third_party:repo.bzl", "tf_http_archive")

# Commit determined by:
# https://github.com/openxla/xla/blob/f73bbc1a0dd4bcbdacf6420bd0f517eb2d3fd177/third_party/py/ml_dtypes/workspace.bzl#L25
tf_http_archive(
    name = "ml_dtypes_py",
    build_file = "@xla//third_party/py/ml_dtypes:ml_dtypes_py.BUILD",
    link_files = {
        "@xla//third_party/py/ml_dtypes:ml_dtypes.BUILD": "ml_dtypes/BUILD.bazel",
    },
    repo_mapping = {
        "@eigen_archive": "@eigen",
    },
    sha256 = "f6e5880666661351e6cd084ac4178ddc4dabcde7e9a73722981c0d1500cf5937",
    strip_prefix = "ml_dtypes-00d98cd92ade342fef589c0470379abb27baebe9",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/jax-ml/ml_dtypes/archive/00d98cd92ade342fef589c0470379abb27baebe9/ml_dtypes-00d98cd92ade342fef589c0470379abb27baebe9.tar.gz",
        "https://github.com/jax-ml/ml_dtypes/archive/00d98cd92ade342fef589c0470379abb27baebe9/ml_dtypes-00d98cd92ade342fef589c0470379abb27baebe9.tar.gz",
    ],
)

# Commit determined by:
# https://github.com/openxla/xla/blob/f73bbc1a0dd4bcbdacf6420bd0f517eb2d3fd177/third_party/ducc/workspace.bzl#L21
tf_http_archive(
    name = "ducc",
    build_file = "@xla//third_party/ducc:ducc.BUILD",
    link_files = {
        "@xla//third_party/ducc:ducc0_custom_lowlevel_threading.h": "google/ducc0_custom_lowlevel_threading.h",
        "@xla//third_party/ducc:fft.h": "google/fft.h",
        "@xla//third_party/ducc:fft.cc": "google/fft.cc",
        "@xla//third_party/ducc:threading.cc": "google/threading.cc",
        "@xla//third_party/ducc:threading.h": "google/threading.h",
    },
    repo_mapping = {
        "@eigen_archive": "@eigen",
    },
    sha256 = "077cf4bd0bd7eddaa6649a024285fff96e2662c5e6f2fb6ed5c5771f9de093f3",
    strip_prefix = "ducc-aa46a4c21e440b3d416c16eca3c96df19c74f316",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.mpcdf.mpg.de/mtr/ducc/-/archive/aa46a4c21e440b3d416c16eca3c96df19c74f316/ducc-aa46a4c21e440b3d416c16eca3c96df19c74f316.tar.gz",
        "https://gitlab.mpcdf.mpg.de/mtr/ducc/-/archive/aa46a4c21e440b3d416c16eca3c96df19c74f316/ducc-aa46a4c21e440b3d416c16eca3c96df19c74f316.tar.gz",
    ],
)

#
# Inlined Transitive Dependencies
#

# Required by `org_tensorflow_federated` and `xla`; commit determined by
# https://github.com/openxla/xla/blob/f73bbc1a0dd4bcbdacf6420bd0f517eb2d3fd177/workspace2.bzl#L684
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
    sha256 = "41b695614b26652ff9e97ce50cfd4a6c7a3d45a9fe598d1454407746499bbf2c",
    strip_prefix = "grpc-1.81.0",
    url = "https://github.com/grpc/grpc/archive/refs/tags/v1.81.0.tar.gz",
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
load("@xla//:workspace4.bzl", "xla_workspace4")

xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")

xla_workspace3()

load("@bazel_features//:deps.bzl", "bazel_features_deps")

bazel_features_deps()

# Initialize hermetic C++
load("@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl", "cc_toolchain_deps")

cc_toolchain_deps()

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64")

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64_cuda")

# Commit determined by:
# https://github.com/openxla/xla/blob/f73bbc1a0dd4bcbdacf6420bd0f517eb2d3fd177/third_party/py/python_init_rules.bzl#L14
http_archive(
    name = "com_google_protobuf",
    patch_args = ["-p1"],
    patches = [
        "@xla//third_party/protobuf:protobuf.patch",
        "@xla//third_party/protobuf:protobuf_arena.patch",
    ],
    repo_mapping = {
        "@com_google_absl": "@abseil-cpp",
        "@protobuf_pip_deps": "@pypi",
    },
    sha256 = "6e09bbc950ba60c3a7b30280210cd285af8d7d8ed5e0a6ed101c72aff22e8d88",
    strip_prefix = "protobuf-6.31.1",
    urls = [
        "https://github.com/protocolbuffers/protobuf/archive/refs/tags/v6.31.1.zip",
    ],
)

# Initialize hermetic Python
load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@rules_ml_toolchain//py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    requirements = {
        "3.10": "//:requirements_lock_3_10.txt",
        "3.11": "//:requirements_lock_3_11.txt",
        "3.12": "//:requirements_lock_3_12.txt",
        "3.13": "//:requirements_lock_3_13.txt",
    },
)

load("@rules_ml_toolchain//py:python_register_toolchain.bzl", "python_register_toolchain")

python_register_toolchain()

load("@rules_ml_toolchain//py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

load("@xla//:workspace2.bzl", "xla_workspace2")

xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")

xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")

xla_workspace0()

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_versions.bzl",
    "REDIST_VERSIONS_TO_BUILD_TEMPLATES",
)
load("@xla//third_party/cccl:workspace.bzl", "CCCL_3_2_0_DIST_DICT", "CCCL_GITHUB_VERSIONS_TO_BUILD_TEMPLATES")

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS | CCCL_3_2_0_DIST_DICT,
    redist_versions_to_build_templates = REDIST_VERSIONS_TO_BUILD_TEMPLATES | CCCL_GITHUB_VERSIONS_TO_BUILD_TEMPLATES,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

load(
    "@rules_ml_toolchain//gpu/nvshmem:nvshmem_json_init_repository.bzl",
    "nvshmem_json_init_repository",
)

nvshmem_json_init_repository()

load(
    "@nvshmem_redist_json//:distributions.bzl",
    "NVSHMEM_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/nvshmem:nvshmem_redist_init_repository.bzl",
    "nvshmem_redist_init_repository",
)

nvshmem_redist_init_repository(
    nvshmem_redistributions = NVSHMEM_REDISTRIBUTIONS,
)

#
# Python Dependencies
#

load("@rules_ml_toolchain//py:python_register_toolchain.bzl", "get_toolchain_name_per_python_version")
load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "federated_language_jax_pypi",
    python_interpreter_target = "@{}_host//:python".format(
        get_toolchain_name_per_python_version("python"),
    ),
    requirements_lock = "//:requirements_lock_3_12.txt",
)

load("@federated_language_jax_pypi//:requirements.bzl", install_flang_deps = "install_deps")

install_flang_deps()
