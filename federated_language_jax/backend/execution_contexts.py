# Copyright 2022 Google LLC
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
"""Execution contexts for the XLA backend."""

import federated_language
from federated_language_jax.computation import jax_computation
from federated_language_jax.executor import executor_factory
from federated_language_jax.executor import xla_bindings
from tensorflow_federated.cc.core.impl.executors import executor_bindings


def _create_xla_backend_execution_stack(max_concurrent_computation_calls):
  del max_concurrent_computation_calls  # Unused.
  xla_executor = xla_bindings.create_xla_executor()
  reference_resolving_executor = (
      executor_bindings.create_reference_resolving_executor(xla_executor)
  )
  return executor_bindings.create_sequence_executor(
      reference_resolving_executor
  )


def create_async_local_execution_context(
    default_num_clients: int = 0, max_concurrent_computation_calls: int = -1
):
  """Creates a local execution context backed by TFF-C++ runtime.

  Args:
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the CPP runtime. If nonpositive, there is no
      limit.

  Returns:
    An instance of `tff.framework.AyncContext` representing the TFF-C++ runtime.
  """
  factory = executor_factory.local_executor_factory(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      leaf_executor_fn=_create_xla_backend_execution_stack,
  )

  return federated_language.framework.AsyncExecutionContext(
      executor_fn=factory,
      transform_args=jax_computation.transform_args,
      transform_result=jax_computation.transform_result,
  )


def create_sync_local_execution_context(
    default_num_clients: int = 0, max_concurrent_computation_calls: int = -1
):
  """Creates a local execution context backed by TFF-C++ runtime.

  Args:
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the CPP runtime. If nonpositive, there is no
      limit.

  Returns:
    An instance of `federated_language.framework.SyncContext` representing the
    TFF-C++ runtime.
  """
  factory = executor_factory.local_executor_factory(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      leaf_executor_fn=_create_xla_backend_execution_stack,
  )

  # TODO: b/255978089 - implement lowering to federated_aggregate to create JAX
  # computations instead of TensorFlow, similar to "desugar intrinsics" in the
  # native backend.
  return federated_language.framework.SyncExecutionContext(
      executor_fn=factory,
      transform_args=jax_computation.transform_args,
      transform_result=jax_computation.transform_result,
  )
