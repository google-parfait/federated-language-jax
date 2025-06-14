# Copyright 2020 Google LLC
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
"""The implementation of an experimental JAX computation context."""

import federated_language


class JaxComputationContext(federated_language.framework.SyncContext):
  """An experimental context for building JAX computations."""

  def __init__(self):
    pass

  def ingest(self, val, type_spec):
    raise NotImplementedError(
        'Invoking computations from JAX code is not currently supported.'
    )

  def invoke(self, comp, arg):
    raise NotImplementedError(
        'Invoking computations from JAX code is not currently supported.'
    )
