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

import collections

from absl.testing import absltest
import federated_language
from federated_language.common_libs import structure
from federated_language.proto import computation_pb2 as pb
from federated_language_jax.computation import jax_serialization
import jax
import numpy as np


class JaxSerializationTest(absltest.TestCase):

  def test_serialize_jax_computation_raises_type_error_with_unknown_rank(
      self,
  ):
    def fn(x):
      return x + 10

    parameter_type = federated_language.TensorType(dtype=np.int32, shape=None)
    with self.assertRaisesRegex(TypeError, 'fully-defined TensorShapes'):
      jax_serialization.serialize_jax_computation(
          fn, parameter_type, federated_language.framework.get_context_stack()
      )

  def test_serialize_jax_computation_raises_type_error_with_unknown_dimension(
      self,
  ):
    def fn(x):
      return x + 10

    parameter_type = federated_language.TensorType(dtype=np.int32, shape=[None])
    with self.assertRaisesRegex(TypeError, 'fully-defined TensorShapes'):
      jax_serialization.serialize_jax_computation(
          fn, parameter_type, federated_language.framework.get_context_stack()
      )

  def test_serialize_jax_with_noarg_to_int32(self):
    def traced_fn():
      return 10

    parameter_type = None
    comp_pb, annotated_type = jax_serialization.serialize_jax_computation(
        traced_fn,
        parameter_type,
        federated_language.framework.get_context_stack(),
    )
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = federated_language.FunctionType.from_proto(comp_pb.type)
    self.assertTrue(
        type_spec.is_equivalent_to(
            federated_language.FunctionType(
                parameter=parameter_type, result=np.int32
            )
        )
    )
    self.assertEqual(
        annotated_type,
        federated_language.FunctionType(
            parameter=parameter_type, result=np.int32
        ),
    )
    self.assertEqual(str(comp_pb.xla.parameter), '')
    self.assertEqual(str(comp_pb.xla.result), 'tensor {\n  index: 0\n}\n')

  def test_serialize_jax_with_unused_arg_to_int32(self):
    def traced_fn(x):
      del x
      return 10

    parameter_type = federated_language.to_type(np.int32)
    comp_pb, annotated_type = jax_serialization.serialize_jax_computation(
        traced_fn,
        parameter_type,
        federated_language.framework.get_context_stack(),
    )
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = federated_language.FunctionType.from_proto(comp_pb.type)
    self.assertTrue(
        type_spec.is_equivalent_to(
            federated_language.FunctionType(
                parameter=parameter_type, result=np.int32
            )
        )
    )
    self.assertEqual(
        annotated_type,
        federated_language.FunctionType(
            parameter=parameter_type, result=np.int32
        ),
    )
    self.assertEqual(str(comp_pb.xla.parameter), 'tensor {\n  index: 0\n}\n')
    self.assertEqual(str(comp_pb.xla.result), 'tensor {\n  index: 0\n}\n')

  def test_serialize_jax_with_int32_to_int32(self):
    def traced_fn(x):
      return x + 10

    parameter_type = federated_language.to_type(np.int32)
    comp_pb, annotated_type = jax_serialization.serialize_jax_computation(
        traced_fn,
        parameter_type,
        federated_language.framework.get_context_stack(),
    )
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = federated_language.FunctionType.from_proto(comp_pb.type)
    self.assertTrue(
        type_spec.is_equivalent_to(
            federated_language.FunctionType(
                parameter=parameter_type, result=np.int32
            )
        )
    )
    self.assertEqual(
        annotated_type,
        federated_language.FunctionType(
            parameter=parameter_type, result=np.int32
        ),
    )
    self.assertEqual(str(comp_pb.xla.result), str(comp_pb.xla.parameter))
    self.assertEqual(str(comp_pb.xla.result), 'tensor {\n  index: 0\n}\n')

  def test_serialize_jax_with_2xint32_to_2xint32(self):
    def traced_fn(x):
      return collections.OrderedDict(
          sum=x['foo'] + x['bar'], difference=x['bar'] - x['foo']
      )

    parameter_type = federated_language.to_type(
        collections.OrderedDict(foo=np.int32, bar=np.int32)
    )
    comp_pb, annotated_type = jax_serialization.serialize_jax_computation(
        traced_fn,
        parameter_type,
        federated_language.framework.get_context_stack(),
    )
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = federated_language.FunctionType.from_proto(comp_pb.type)
    self.assertTrue(
        type_spec.is_equivalent_to(
            federated_language.FunctionType(
                parameter=parameter_type,
                result=federated_language.StructType(
                    [('sum', np.int32), ('difference', np.int32)]
                ),
            )
        )
    )
    self.assertEqual(
        annotated_type,
        federated_language.FunctionType(
            parameter=parameter_type,
            result=federated_language.StructWithPythonType(
                [('sum', np.int32), ('difference', np.int32)],
                container_type=collections.OrderedDict,
            ),
        ),
    )
    self.assertEqual(str(comp_pb.xla.result), str(comp_pb.xla.parameter))
    self.assertEqual(
        str(comp_pb.xla.parameter),
        (
            'struct {\n'
            '  element {\n'
            '    tensor {\n'
            '      index: 0\n'
            '    }\n'
            '  }\n'
            '  element {\n'
            '    tensor {\n'
            '      index: 1\n'
            '    }\n'
            '  }\n'
            '}\n'
        ),
    )

  def test_serialize_jax_with_two_args(self):
    def traced_fn(x, y):
      return x + y

    parameter_type = federated_language.to_type(
        collections.OrderedDict(x=np.int32, y=np.int32)
    )
    comp_pb, annotated_type = jax_serialization.serialize_jax_computation(
        traced_fn,
        parameter_type,
        federated_language.framework.get_context_stack(),
    )
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = federated_language.FunctionType.from_proto(comp_pb.type)
    self.assertTrue(
        type_spec.is_equivalent_to(
            federated_language.FunctionType(
                parameter=parameter_type, result=np.int32
            )
        )
    )
    self.assertEqual(
        annotated_type,
        federated_language.FunctionType(
            parameter=parameter_type, result=np.int32
        ),
    )
    self.assertEqual(
        str(comp_pb.xla.parameter),
        (
            'struct {\n'
            '  element {\n'
            '    tensor {\n'
            '      index: 0\n'
            '    }\n'
            '  }\n'
            '  element {\n'
            '    tensor {\n'
            '      index: 1\n'
            '    }\n'
            '  }\n'
            '}\n'
        ),
    )
    self.assertEqual(str(comp_pb.xla.result), 'tensor {\n  index: 0\n}\n')

  def test_serialize_jax_with_nested_struct_arg(self):
    def traced_fn(x, y):
      return x[0] + y

    parameter_type = federated_language.StructType([
        (None, federated_language.StructType([(None, np.int32)])),
        (None, np.int32),
    ])
    comp_pb, annotated_type = jax_serialization.serialize_jax_computation(
        traced_fn,
        parameter_type,
        federated_language.framework.get_context_stack(),
    )
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = federated_language.FunctionType.from_proto(comp_pb.type)
    self.assertTrue(
        type_spec.is_equivalent_to(
            federated_language.FunctionType(
                parameter=parameter_type, result=np.int32
            )
        )
    )
    self.assertEqual(
        annotated_type,
        federated_language.FunctionType(
            parameter=parameter_type, result=np.int32
        ),
    )

  def test_nested_structure_type_signature_roundtrip(self):
    def traced_fn(x):
      return x[0][0]

    parameter_type = federated_language.to_type([(np.int32,)])
    comp_pb, annotated_type = jax_serialization.serialize_jax_computation(
        traced_fn,
        parameter_type,
        federated_language.framework.get_context_stack(),
    )
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = federated_language.FunctionType.from_proto(comp_pb.type)
    self.assertTrue(
        type_spec.is_equivalent_to(
            federated_language.FunctionType(
                parameter=parameter_type, result=np.int32
            )
        )
    )
    self.assertEqual(
        annotated_type,
        federated_language.FunctionType(
            parameter=parameter_type, result=np.int32
        ),
    )

  def test_arg_ordering(self):
    parameter_type = federated_language.to_type((
        federated_language.TensorType(np.int32, (10,)),
        federated_language.TensorType(np.int32),
    ))

    def traced_fn(b, a):
      return jax.numpy.add(a, jax.numpy.sum(b))

    comp_pb, annotated_type = jax_serialization.serialize_jax_computation(
        traced_fn,
        parameter_type,
        federated_language.framework.get_context_stack(),
    )
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = federated_language.FunctionType.from_proto(comp_pb.type)
    self.assertTrue(
        type_spec.is_equivalent_to(
            federated_language.FunctionType(
                parameter=parameter_type, result=np.int32
            )
        )
    )
    self.assertEqual(
        annotated_type,
        federated_language.FunctionType(
            parameter=parameter_type, result=np.int32
        ),
    )

  def test_tracing_with_float64_input(self):
    self.skipTest('b/237566862')
    parameter_type = federated_language.TensorType(np.float64)
    identity_fn = lambda x: x
    comp_pb, annotated_type = jax_serialization.serialize_jax_computation(
        identity_fn,
        parameter_type,
        federated_language.framework.get_context_stack(),
    )
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = federated_language.FunctionType.from_proto(comp_pb.type)
    self.assertTrue(
        type_spec.is_equivalent_to(
            federated_language.FunctionType(
                parameter=np.float64, result=np.float64
            )
        )
    )
    self.assertEqual(
        annotated_type,
        federated_language.FunctionType(
            parameter=np.float64, result=np.float64
        ),
    )


class StructPytreeTest(absltest.TestCase):

  def test_named_struct(self):
    struct = structure.Struct.named(a=1, b=2, c=3)
    children, aux_data = jax.tree_util.tree_flatten(struct)
    self.assertEqual(children, [1, 2, 3])
    new_struct = jax.tree_util.tree_unflatten(aux_data, children)
    self.assertEqual(struct, new_struct)

  def test_unnamed_struct(self):
    struct = structure.Struct.unnamed(1, 'a', 2)
    children, aux_data = jax.tree_util.tree_flatten(struct)
    self.assertEqual(children, [1, 'a', 2])
    new_struct = jax.tree_util.tree_unflatten(aux_data, children)
    self.assertEqual(struct, new_struct)

  def test_mixed_named_struct(self):
    struct = structure.Struct([('a', 1), (None, 2), ('b', 3), (None, 4)])
    children, aux_data = jax.tree_util.tree_flatten(struct)
    self.assertEqual(children, [1, 2, 3, 4])
    new_struct = jax.tree_util.tree_unflatten(aux_data, children)
    self.assertEqual(struct, new_struct)

  def test_nested_structs(self):
    struct = structure.Struct.named(
        a=1,
        b=structure.Struct.named(c=4, d=structure.Struct.unnamed(5, 6), e=7),
    )
    children, aux_data = jax.tree_util.tree_flatten(struct)
    self.assertEqual(children, [1, 4, 5, 6, 7])
    new_struct = jax.tree_util.tree_unflatten(aux_data, children)
    self.assertEqual(struct, new_struct)

  def test_mixed_nested_structs_and_python_containers(self):
    struct = structure.Struct.named(
        a=1,
        b=[
            2,
            structure.Struct.named(c=4, d=(5, 6), e=7),
            3,
        ],
    )
    children, aux_data = jax.tree_util.tree_flatten(struct)
    self.assertEqual(children, [1, 2, 4, 5, 6, 7, 3])
    new_struct = jax.tree_util.tree_unflatten(aux_data, children)
    self.assertEqual(struct, new_struct)


class XlaSerializerStructArgPytreeTest(absltest.TestCase):

  def test_named_struct(self):
    struct_arg = jax_serialization._XlaSerializerStructArg(
        federated_language.StructType(
            [('a', np.float32), ('b', np.int64), ('c', np.int32)]
        ),
        [('a', 1.0), ('b', 2), ('c', 3)],
    )
    children, aux_data = jax.tree_util.tree_flatten(struct_arg)
    self.assertEqual(children, [1.0, 2, 3])
    new_struct_arg = jax.tree_util.tree_unflatten(aux_data, children)
    self.assertEqual(struct_arg, new_struct_arg)

  def test_unnamed_struct(self):
    struct_arg = jax_serialization._XlaSerializerStructArg(
        federated_language.StructType([
            (None, np.float32),
            (None, np.int64),
            (None, np.int32),
        ]),
        elements=[(None, 1.0), (None, 2), (None, 3)],
    )
    children, aux_data = jax.tree_util.tree_flatten(struct_arg)
    self.assertEqual(children, [1.0, 2, 3])
    new_struct_arg = jax.tree_util.tree_unflatten(aux_data, children)
    self.assertEqual(struct_arg, new_struct_arg)

  def test_mixed_named_struct(self):
    struct_arg = jax_serialization._XlaSerializerStructArg(
        federated_language.StructType([
            ('a', np.int32),
            (None, np.int32),
            ('b', np.int64),
            (None, np.int64),
        ]),
        elements=[('a', 1), (None, 2), ('b', 3), (None, 4)],
    )
    children, aux_data = jax.tree_util.tree_flatten(struct_arg)
    self.assertEqual(children, [1, 2, 3, 4])
    new_struct_arg = jax.tree_util.tree_unflatten(aux_data, children)
    self.assertEqual(struct_arg, new_struct_arg)

  def test_nested_structs(self):
    struct_arg = jax_serialization._XlaSerializerStructArg(
        federated_language.StructType([
            ('a', np.int32),
            (
                'b',
                federated_language.StructType([
                    ('c', np.int32),
                    (
                        'd',
                        federated_language.StructType(
                            [(None, np.int32), (None, np.int32)]
                        ),
                    ),
                    ('e', np.int32),
                ]),
            ),
        ]),
        elements=[
            ('a', 1),
            (
                'b',
                jax_serialization._XlaSerializerStructArg(
                    federated_language.StructType([
                        ('c', np.int32),
                        (
                            'd',
                            federated_language.StructType(
                                [(None, np.int32), (None, np.int32)]
                            ),
                        ),
                        ('e', np.int32),
                    ]),
                    elements=[
                        ('c', 4),
                        (
                            'd',
                            jax_serialization._XlaSerializerStructArg(
                                federated_language.StructType(
                                    [(None, np.int32), (None, np.int32)]
                                ),
                                elements=[(None, 5), (None, 6)],
                            ),
                        ),
                        ('e', 7),
                    ],
                ),
            ),
        ],
    )
    children, aux_data = jax.tree_util.tree_flatten(struct_arg)
    self.assertEqual(children, [1, 4, 5, 6, 7])
    new_struct_arg = jax.tree_util.tree_unflatten(aux_data, children)
    self.assertEqual(struct_arg, new_struct_arg)

  def test_mixed_nested_structs_and_python_containers(self):
    struct_arg = jax_serialization._XlaSerializerStructArg(
        federated_language.StructType([
            ('a', np.int32),
            federated_language.StructType([(
                (None, np.int32),
                (
                    None,
                    federated_language.StructType(
                        [(None, np.int32), (None, np.int32)]
                    ),
                ),
                (None, np.int32),
            )]),
        ]),
        elements=[
            ('a', 1),
            (
                None,
                [
                    4,
                    jax_serialization._XlaSerializerStructArg(
                        federated_language.StructType(
                            [(None, np.int32), (None, np.int32)]
                        ),
                        elements=[(None, 5), (None, 6)],
                    ),
                    7,
                ],
            ),
        ],
    )
    children, aux_data = jax.tree_util.tree_flatten(struct_arg)
    self.assertEqual(children, [1, 4, 5, 6, 7])
    new_struct_arg = jax.tree_util.tree_unflatten(aux_data, children)
    self.assertEqual(struct_arg, new_struct_arg)


if __name__ == '__main__':
  absltest.main()
