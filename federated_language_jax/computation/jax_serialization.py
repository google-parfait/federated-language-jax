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
"""Utilities for serializing JAX computations."""

from collections.abc import Callable, Mapping, Sequence
import typing
from typing import Protocol, Union

import federated_language
from federated_language.proto import computation_pb2 as pb
from federated_language_jax.computation import jax_computation_context
from federated_language_jax.computation import xla_serialization
import jax
import numpy as np


class _XlaSerializerTensorArg(
    jax.ShapeDtypeStruct, federated_language.TypedObject
):
  """Represents tensor type info understood by both TFF and JAX serializer."""

  def __init__(
      self, tensor_type: federated_language.TensorType, tensor_index: int
  ):
    jax.ShapeDtypeStruct.__init__(self, tensor_type.shape, tensor_type.dtype)
    self._type_signature = tensor_type
    self._tensor_index = tensor_index

  @property
  def type_signature(self) -> federated_language.TensorType:
    return self._type_signature

  @property
  def tensor_index(self) -> int:
    return self._tensor_index


@typing.runtime_checkable
class _NamedTuple(Protocol):

  def _fields(self) -> tuple[str, ...]:
    ...

  def _asdict(self) -> dict[str, object]:
    ...

  def __hash__(self) -> int:
    ...


def _tff_type_to_xla_serializer_arg(
    type_spec: federated_language.Type,
) -> Union[Mapping[str, object], Sequence[object], _XlaSerializerTensorArg]:
  """Converts TFF type into an argument for the JAX-to-XLA serializer.

  Args:
    type_spec: An instance of `federated_language.Type` containing only
      structure and tensor elements.

  Returns:
    An object that carries both TFF and JAX type info, to be fed into the JAX
    serializer.
  """

  def _undefined_shape_predicate(type_element: federated_language.Type) -> bool:
    if not isinstance(type_element, federated_language.TensorType):
      return False
    return not federated_language.array_shape_is_fully_defined(
        type_element.shape
    )

  has_undefined_shapes = federated_language.framework.type_contains(
      type_spec, _undefined_shape_predicate
  )
  if has_undefined_shapes:
    raise TypeError(
        'Can only serialize XLA computations whose parameters '
        'contain fully-defined TensorShapes at the leaves; '
        'encountered undefined tensor shapes (or unknown rank '
        'tensors) in the signature:\n'
        f'{type_spec.formatted_representation()}'
    )

  def _make(
      type_spec: federated_language.Type, next_unused_tensor_index: int
  ) -> tuple[
      Union[Mapping[str, object], Sequence[object], _XlaSerializerTensorArg],
      int,
  ]:
    if isinstance(type_spec, federated_language.TensorType):
      obj = _XlaSerializerTensorArg(type_spec, next_unused_tensor_index)
      next_unused_tensor_index = next_unused_tensor_index + 1
      return obj, next_unused_tensor_index
    elif isinstance(type_spec, federated_language.StructType):
      elements = []
      for k, v in type_spec.items():
        obj, next_unused_tensor_index = _make(v, next_unused_tensor_index)
        elements.append((k, obj))

      def _get_container_cls(
          type_spec: federated_language.StructType,
      ) -> type[object]:
        container_cls = type_spec.python_container
        if container_cls is None:
          has_names = [name is not None for name, _ in type_spec.items()]
          if any(has_names):
            if not all(has_names):
              raise ValueError(
                  'Expected `type_spec` to have either all named or unnamed'
                  f' elements, found {type_spec}.'
              )
            container_cls = dict
          else:
            container_cls = list
        return container_cls

      container_cls = _get_container_cls(type_spec)

      if issubclass(container_cls, _NamedTuple):
        result = container_cls(**dict(elements))  # pylint: disable=too-many-function-args
      elif issubclass(container_cls, Mapping):
        result = container_cls(elements)  # pylint: disable=too-many-function-args
      elif issubclass(container_cls, Sequence):
        result = container_cls([x for _, x in elements])  # pylint: disable=too-many-function-args
      else:
        raise NotImplementedError(
            f'Unexpected container_cls found: {type(container_cls)}.'
        )

      return result, next_unused_tensor_index
    else:
      raise TypeError(
          'Can only construct an XLA serializer for TFF types '
          'which contain exclusively structure and tensor types; '
          f'found type:\n {type_spec.formatted_representation()}'
      )

  obj, _ = _make(type_spec, 0)
  return obj


def _jax_shape_dtype_struct_to_tff_tensor(
    val: jax.ShapeDtypeStruct,
) -> federated_language.TensorType:
  """Converts `jax.ShapeDtypeStruct` to `federated_language.TensorType`.

  Args:
    val: An instance of `jax.ShapeDtypeStruct`.

  Returns:
    A corresponding instance of `federated_language.TensorType`.

  Raises:
    TypeError: if arg type mismatches.
  """
  return federated_language.TensorType(val.dtype, val.shape)


def serialize_jax_computation(
    fn: Callable[..., object],
    parameter_type: Union[
        federated_language.StructType, federated_language.TensorType
    ],
    context_stack: federated_language.framework.ContextStack,
) -> tuple[pb.Computation, federated_language.FunctionType]:
  """Serializes a Python function containing JAX code as a TFF computation.

  Args:
    fn: The Python function containing JAX code to be traced by JAX and
      serialized as a TFF computation containing XLA code.
    parameter_type: An instance of `federated_language.Type` that represents the
      TFF type of the computation parameter, or `None` if the function does not
      take any parameters.
    context_stack: The context stack to use during serialization.

  Returns:
    A 2-tuple of `pb.Computation` with the constructed computation and a
    `federated_language.FunctionType` containing the full type including
    Python container annotations.

  Raises:
    TypeError: if the arguments are of the wrong types.
  """

  if parameter_type is not None:
    parameter_type = federated_language.to_type(parameter_type)
    packed_arg = _tff_type_to_xla_serializer_arg(parameter_type)
  else:
    packed_arg = None

  args, kwargs = federated_language.framework.unpack_arg(
      fn, parameter_type, packed_arg
  )

  # While the fake parameters are fed via args/kwargs during serialization,
  # it is possible for them to get reordered in the actual generated XLA code.
  # We use here the same flattening function as that one, which is used by
  # the JAX serializer to determine the ordering and allow it to be captured
  # in the parameter binding. We do not need to do anything special for the
  # results, since the results, if multiple, are always returned as a tuple.
  flattened_obj, _ = jax.tree_util.tree_flatten((args, kwargs))
  tensor_indexes = list(np.argsort([x.tensor_index for x in flattened_obj]))

  context = jax_computation_context.JaxComputationContext()
  with context_stack.install(context):
    wrapped = jax.jit(fn, keep_unused=True)
    lowered = wrapped.lower(*args, **kwargs)
    compiled_xla = lowered.compiler_ir('hlo')

  # Test if the output is a tuple, or a single array and construct the
  # return spec accordingly.
  if isinstance(lowered.out_info, jax.ShapeDtypeStruct):
    returned_type_spec = _jax_shape_dtype_struct_to_tff_tensor(
        jax.ShapeDtypeStruct(
            shape=lowered.out_info.shape, dtype=lowered.out_info.dtype
        )
    )
  else:
    returned_type_spec = federated_language.to_type(
        jax.tree_util.tree_map(
            _jax_shape_dtype_struct_to_tff_tensor, lowered.out_info
        )
    )

  computation_type = federated_language.FunctionType(
      parameter_type, returned_type_spec
  )
  return (
      xla_serialization.create_xla_tff_computation(
          compiled_xla, tensor_indexes, computation_type
      ),
      computation_type,
  )
