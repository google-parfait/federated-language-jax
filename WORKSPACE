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
    strip_prefix = "tensorflow-federated-23792d84ad9ab6b1bfe28aed893be85d9c4374c3",
    url = "https://github.com/google-parfait/tensorflow-federated/archive/23792d84ad9ab6b1bfe28aed893be85d9c4374c3.tar.gz",
)

http_archive(
    name = "xla",
    strip_prefix = "xla-661559150498be4c186e74af3a0c60b1aae0c991",
    url = "https://github.com/openxla/xla/archive/661559150498be4c186e74af3a0c60b1aae0c991.zip",
)

# Transitive Dependencies, inlined

# The version of `rules_proto` that XLA depends on is incompatible with the version of
# `rules_python` that XLA depends on.
http_archive(
    name = "rules_proto",
    strip_prefix = "rules_proto-6.0.2",
    url = "https://github.com/bazelbuild/rules_proto/archive/refs/tags/6.0.2.tar.gz",
)

# Transitive Dependencies, required by xla

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
