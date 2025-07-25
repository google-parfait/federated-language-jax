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
"""JAX computation, compiler, and executor for the Federated Language library."""

# pylint: disable=g-importing-member
from federated_language_jax.backend.execution_contexts import create_async_local_execution_context
from federated_language_jax.backend.execution_contexts import create_sync_local_execution_context
from federated_language_jax.computation.jax_computation import jax_computation as computation
# pylint: enable=g-importing-member
