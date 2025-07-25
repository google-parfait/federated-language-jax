# Copyright 2019 Google LLC
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
"""A collection of constructors for basic types of executor stacks."""

from collections.abc import Callable
import concurrent
import math
from typing import Optional

from absl import logging
import cachetools
import federated_language

from tensorflow_federated.cc.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.executors import cpp_to_python_executor


# Users likely do not intend to run 4 or more TensorFlow functions sequentially;
# we special-case to warn users explicitly in this case, in addition to
# logging in the case of any implied sequential execution.
_CONCURRENCY_LEVEL_TO_WARN = 4


def _get_hashable_key(
    cardinalities: federated_language.framework.CardinalitiesType,
):
  return tuple(sorted((str(k), v) for k, v in cardinalities.items()))


class CPPExecutorFactory(federated_language.framework.ExecutorFactory):
  """An ExecutorFactory which wraps a simple executor_fn."""

  def __init__(
      self,
      executor_fn: Callable[
          [federated_language.framework.CardinalitiesType],
          executor_bindings.Executor,
      ],
      executor_cache_size: int = 5,
  ):
    self._executor_fn = executor_fn
    self._cache_size = executor_cache_size
    self._executors = cachetools.LRUCache(self._cache_size)

  def create_executor(
      self, cardinalities: federated_language.framework.CardinalitiesType
  ) -> federated_language.framework.Executor:
    cardinalities_key = _get_hashable_key(cardinalities)
    if cardinalities_key not in self._executors:
      cpp_executor = self._executor_fn(cardinalities)
      futures_executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
      executor = cpp_to_python_executor.CppToPythonExecutorBridge(
          cpp_executor, futures_executor
      )
      self._executors[cardinalities_key] = executor
    return self._executors[cardinalities_key]

  def clean_up_executor(
      self, cardinalities: federated_language.framework.CardinalitiesType
  ):
    cardinalities_key = _get_hashable_key(cardinalities)
    ex = self._executors.get(cardinalities_key)
    if ex is None:
      return
    del self._executors[cardinalities_key]


def _log_and_warn_on_sequential_execution(
    max_concurrent_computation_calls: int,
    num_clients: int,
    expected_concurrency_factor: int,
):
  """Logs warnings that users may be using the runtime in an unexpected way."""

  if expected_concurrency_factor >= _CONCURRENCY_LEVEL_TO_WARN:
    logging.warning(
        'Running %s clients with max concurrency %s will result in significant '
        'serialization of execution; running %s TensorFlow functions '
        'sequentially. This invocation could benefit significantly from more '
        "resources (e.g. more GPUs), or moving to TFF's distributed runtime."
    )
  else:
    logging.info(
        (
            'TFF-C++ local executor configured to maximally run %s '
            'calls into TensorFlow in  parallel; asked to run %s '
            'clients. This will result in %s invocations running '
            'sequentially, indicating that this invocation will run '
            'faster when equipped with increased resources or invoked '
            'against the distributed TFF runtime.'
        ),
        max_concurrent_computation_calls,
        num_clients,
        expected_concurrency_factor,
    )


def _check_num_clients_is_valid(default_num_clients: int):
  if default_num_clients < 0:
    raise ValueError('Default number of clients must be nonnegative.')


def local_executor_factory(
    *,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    leaf_executor_fn: Optional[Callable[[int], executor_bindings.Executor]],
    client_leaf_executor_fn: Optional[
        Callable[[int], executor_bindings.Executor]
    ] = None,
) -> federated_language.framework.ExecutorFactory:
  """Local ExecutorFactory backed by C++ Executor bindings."""
  _check_num_clients_is_valid(default_num_clients)

  def _executor_fn(
      cardinalities: federated_language.framework.CardinalitiesType,
  ) -> executor_bindings.Executor:
    if cardinalities.get(federated_language.CLIENTS) is None:
      cardinalities[federated_language.CLIENTS] = default_num_clients
    num_clients = cardinalities[federated_language.CLIENTS]
    if (
        max_concurrent_computation_calls > 0
        and num_clients > max_concurrent_computation_calls
    ):
      expected_concurrency_factor = math.ceil(
          num_clients / max_concurrent_computation_calls
      )
      _log_and_warn_on_sequential_execution(
          max_concurrent_computation_calls,
          num_clients,
          expected_concurrency_factor,
      )

    server_leaf_executor = leaf_executor_fn(max_concurrent_computation_calls)
    sub_federating_reference_resolving_server_executor = (
        executor_bindings.create_reference_resolving_executor(
            server_leaf_executor
        )
    )
    if client_leaf_executor_fn is None:
      sub_federating_reference_resolving_client_executor = (
          sub_federating_reference_resolving_server_executor
      )
    else:
      client_leaf_executor = client_leaf_executor_fn(
          max_concurrent_computation_calls
      )

      sub_federating_reference_resolving_client_executor = (
          executor_bindings.create_reference_resolving_executor(
              client_leaf_executor
          )
      )

    cardinalities = {k.uri: v for k, v in cardinalities.items()}
    federating_ex = executor_bindings.create_federating_executor(
        sub_federating_reference_resolving_server_executor,
        sub_federating_reference_resolving_client_executor,
        cardinalities,
    )
    top_level_reference_resolving_ex = (
        executor_bindings.create_reference_resolving_executor(federating_ex)
    )
    return top_level_reference_resolving_ex

  return CPPExecutorFactory(_executor_fn)
