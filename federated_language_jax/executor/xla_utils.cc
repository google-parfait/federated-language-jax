/* Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "federated_language_jax/executor/xla_utils.h"

#include <algorithm>
#include <complex>
#include <cstdint>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "google/protobuf/repeated_field.h"
#include "federated_language/proto/array.pb.h"
#include "federated_language/proto/computation.pb.h"
#include "federated_language/proto/data_type.pb.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"

namespace tensorflow_federated {

absl::StatusOr<federated_language::DataType> DataTypeFromPrimitiveType(
    xla::PrimitiveType primative_type) {
  switch (primative_type) {
    case xla::PRED:
      return federated_language::DataType::DT_BOOL;
    case xla::S8:
      return federated_language::DataType::DT_INT8;
    case xla::S16:
      return federated_language::DataType::DT_INT16;
    case xla::S32:
      return federated_language::DataType::DT_INT32;
    case xla::S64:
      return federated_language::DataType::DT_INT64;
    case xla::U8:
      return federated_language::DataType::DT_UINT8;
    case xla::U16:
      return federated_language::DataType::DT_UINT16;
    case xla::U32:
      return federated_language::DataType::DT_UINT32;
    case xla::U64:
      return federated_language::DataType::DT_UINT64;
    case xla::F16:
      return federated_language::DataType::DT_HALF;
    case xla::F32:
      return federated_language::DataType::DT_FLOAT;
    case xla::F64:
      return federated_language::DataType::DT_DOUBLE;
    case xla::C64:
      return federated_language::DataType::DT_COMPLEX64;
    case xla::C128:
      return federated_language::DataType::DT_COMPLEX128;
    case xla::BF16:
      return federated_language::DataType::DT_BFLOAT16;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected PrimitiveType found:", primative_type));
  }
}

absl::StatusOr<xla::PrimitiveType> PrimitiveTypeFromDataType(
    federated_language::DataType data_type) {
  switch (data_type) {
    case federated_language::DataType::DT_BOOL:
      return xla::PRED;
    case federated_language::DataType::DT_INT8:
      return xla::S8;
    case federated_language::DataType::DT_INT16:
      return xla::S16;
    case federated_language::DataType::DT_INT32:
      return xla::S32;
    case federated_language::DataType::DT_INT64:
      return xla::S64;
    case federated_language::DataType::DT_UINT8:
      return xla::U8;
    case federated_language::DataType::DT_UINT16:
      return xla::U16;
    case federated_language::DataType::DT_UINT32:
      return xla::U32;
    case federated_language::DataType::DT_UINT64:
      return xla::U64;
    case federated_language::DataType::DT_HALF:
      return xla::F16;
    case federated_language::DataType::DT_FLOAT:
      return xla::F32;
    case federated_language::DataType::DT_DOUBLE:
      return xla::F64;
    case federated_language::DataType::DT_COMPLEX64:
      return xla::C64;
    case federated_language::DataType::DT_COMPLEX128:
      return xla::C128;
    case federated_language::DataType::DT_BFLOAT16:
      return xla::BF16;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found:", data_type));
  }
}

federated_language::ArrayShape ArrayShapeFromShape(const xla::Shape& shape) {
  federated_language::ArrayShape shape_pb;
  for (int i = 0; i < shape.dimensions().size(); i++) {
    shape_pb.mutable_dim()->Add(shape.dimensions(i));
  }
  return shape_pb;
}

absl::StatusOr<xla::Shape> ShapeFromTensorType(
    const federated_language::TensorType& tensor_type_pb) {
  if (tensor_type_pb.unknown_rank()) {
    return absl::InvalidArgumentError(
        "Shapes of unknown rank are not supported in the XLA executor.");
  }
  return xla::ShapeUtil::MakeValidatedShape(
      TFF_TRY(PrimitiveTypeFromDataType(tensor_type_pb.dtype())),
      tensor_type_pb.dims());
}

absl::StatusOr<xla::Shape> ShapeFromArrayShape(
    federated_language::DataType data_type,
    const federated_language::ArrayShape& shape_pb) {
  if (shape_pb.unknown_rank()) {
    return absl::InvalidArgumentError(
        "Shapes of unknown rank are not supported in the XLA executor.");
  }
  return xla::ShapeUtil::MakeValidatedShape(
      TFF_TRY(PrimitiveTypeFromDataType(data_type)), shape_pb.dim());
}

template <typename T>
static void CopyToRepeatedField(const absl::Span<const T> src,
                                google::protobuf::RepeatedField<T>* dest) {
  std::copy(src.begin(), src.end(), google::protobuf::RepeatedFieldBackInserter(dest));
}

// Overload for different SrcType and DestType.
template <typename SrcType, typename DestType>
static void CopyToRepeatedField(const absl::Span<const SrcType> src,
                                google::protobuf::RepeatedField<DestType>* dest) {
  std::transform(
      src.begin(), src.end(), google::protobuf::RepeatedFieldBackInserter(dest),
      [](const SrcType& x) -> DestType { return static_cast<DestType>(x); });
}

// Overload for Eigen::half.
static void CopyToRepeatedField(const absl::Span<const Eigen::half> src,
                                google::protobuf::RepeatedField<int32_t>* dest) {
  // Values of dtype Eigen::half are packed to and unpacked from a protobuf
  // field of type int32 using the following logic in order to maintain
  // compatibility with how other external libraries (e.g. TensorFlow, Jax)
  // represent values of Eigen::half.
  std::transform(
      src.begin(), src.end(), google::protobuf::RepeatedFieldBackInserter(dest),
      [](Eigen::half x) -> int32_t {
        return static_cast<int32_t>(Eigen::numext::bit_cast<uint16_t>(x));
      });
}

// Overload for complex.
template <typename T>
static void CopyToRepeatedField(const absl::Span<const std::complex<T>> src,
                                google::protobuf::RepeatedField<T>* dest) {
  for (std::complex<T> value : src) {
    dest->Add(value.real());
    dest->Add(value.imag());
  }
}

// Overload for xla::bfloat16.
static void CopyToRepeatedField(const absl::Span<const xla::bfloat16> src,
                                google::protobuf::RepeatedField<int32_t>* dest) {
  // Values of dtype Eigen::bfloat16 are packed to and unpacked from a
  // protobuf field of type int32 using the following logic in order to maintain
  // compatibility with how other external libraries (e.g. TensorFlow, Jax)
  // represent values of Eigen::bfloat16.
  std::transform(
      src.begin(), src.end(), google::protobuf::RepeatedFieldBackInserter(dest),
      [](xla::bfloat16 x) -> int32_t {
        return static_cast<int32_t>(Eigen::numext::bit_cast<uint16_t>(x));
      });
}

absl::StatusOr<federated_language::Array> ArrayFromLiteral(
    const xla::Literal& literal) {
  federated_language::Array array_pb;
  federated_language::DataType data_type =
      TFF_TRY(DataTypeFromPrimitiveType(literal.shape().element_type()));
  array_pb.set_dtype(data_type);
  federated_language::ArrayShape shape_pb =
      ArrayShapeFromShape(literal.shape());
  array_pb.mutable_shape()->Swap(&shape_pb);

  switch (literal.shape().element_type()) {
    case xla::PRED:
      CopyToRepeatedField(literal.data<bool>(),
                          array_pb.mutable_bool_list()->mutable_value());
      break;
    case xla::S8:
      CopyToRepeatedField(literal.data<int8_t>(),
                          array_pb.mutable_int8_list()->mutable_value());
      break;
    case xla::S16:
      CopyToRepeatedField(literal.data<int16_t>(),
                          array_pb.mutable_int16_list()->mutable_value());
      break;
    case xla::S32:
      CopyToRepeatedField(literal.data<int32_t>(),
                          array_pb.mutable_int32_list()->mutable_value());
      break;
    case xla::S64:
      CopyToRepeatedField(literal.data<int64_t>(),
                          array_pb.mutable_int64_list()->mutable_value());
      break;
    case xla::U8:
      CopyToRepeatedField(literal.data<uint8_t>(),
                          array_pb.mutable_uint8_list()->mutable_value());
      break;
    case xla::U16:
      CopyToRepeatedField(literal.data<uint16_t>(),
                          array_pb.mutable_uint16_list()->mutable_value());
      break;
    case xla::U32:
      CopyToRepeatedField(literal.data<uint32_t>(),
                          array_pb.mutable_uint32_list()->mutable_value());
      break;
    case xla::U64:
      CopyToRepeatedField(literal.data<uint64_t>(),
                          array_pb.mutable_uint64_list()->mutable_value());
      break;
    case xla::F16:
      CopyToRepeatedField(literal.data<xla::half>(),
                          array_pb.mutable_float16_list()->mutable_value());
      break;
    case xla::F32:
      CopyToRepeatedField(literal.data<float>(),
                          array_pb.mutable_float32_list()->mutable_value());
      break;
    case xla::F64:
      CopyToRepeatedField(literal.data<double>(),
                          array_pb.mutable_float64_list()->mutable_value());
      break;
    case xla::C64:
      CopyToRepeatedField(literal.data<xla::complex64>(),
                          array_pb.mutable_complex64_list()->mutable_value());
      break;
    case xla::C128:
      CopyToRepeatedField(literal.data<xla::complex128>(),
                          array_pb.mutable_complex128_list()->mutable_value());
      break;
    case xla::BF16:
      CopyToRepeatedField(literal.data<xla::bfloat16>(),
                          array_pb.mutable_bfloat16_list()->mutable_value());
      break;
    default:
      return absl::UnimplementedError(absl::StrCat(
          "Unexpected DataType found:", literal.shape().element_type()));
  }

  return array_pb;
}

template <typename T>
static void CopyFromRepeatedField(const google::protobuf::RepeatedField<T>& src,
                                  T* dest) {
  std::copy(src.begin(), src.end(), dest);
}

// Overload for different SrcType and DestType.
template <typename SrcType, typename DestType>
static void CopyFromRepeatedField(const google::protobuf::RepeatedField<SrcType>& src,
                                  DestType* dest) {
  std::transform(
      src.begin(), src.end(), dest,
      [](const SrcType& x) -> DestType { return static_cast<DestType>(x); });
}

// Overload for Eigen::half.
static void CopyFromRepeatedField(const google::protobuf::RepeatedField<int32_t>& src,
                                  Eigen::half* dest) {
  // Values of dtype Eigen::half are packed to and unpacked from a protobuf
  // field of type int32 using the following logic in order to maintain
  // compatibility with how other external libraries (e.g. TensorFlow, Jax)
  // represent values of Eigen::half.
  std::transform(src.begin(), src.end(), dest, [](int32_t x) -> Eigen::half {
    return Eigen::numext::bit_cast<Eigen::half>(static_cast<uint16_t>(x));
  });
}

// Overload for complex.
template <typename T>
static void CopyFromRepeatedField(const google::protobuf::RepeatedField<T>& src,
                                  std::complex<T>* dest) {
  std::copy(src.begin(), src.end(), reinterpret_cast<T*>(dest));
}

// Overload for Eigen::bfloat16.
static void CopyFromRepeatedField(const google::protobuf::RepeatedField<int32_t>& src,
                                  Eigen::bfloat16* dest) {
  // Values of dtype Eigen::bfloat16 are packed to and unpacked from a
  // protobuf field of type int32 using the following logic in order to maintain
  // compatibility with how other external libraries (e.g. TensorFlow, Jax)
  // represent values of Eigen::bfloat16.
  std::transform(src.begin(), src.end(), dest,
                 [](int32_t x) -> Eigen::bfloat16 {
                   return Eigen::numext::bit_cast<Eigen::bfloat16>(
                       static_cast<uint16_t>(x));
                 });
}

absl::StatusOr<xla::Literal> LiteralFromArray(
    const federated_language::Array& array_pb) {
  switch (array_pb.kind_case()) {
    case federated_language::Array::kBoolList: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          federated_language::DataType::DT_BOOL, array_pb.shape())));
      CopyFromRepeatedField(array_pb.bool_list().value(),
                            literal.data<bool>().begin());
      return literal;
    }
    case federated_language::Array::kInt8List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          federated_language::DataType::DT_INT8, array_pb.shape())));
      CopyFromRepeatedField(array_pb.int8_list().value(),
                            literal.data<int8_t>().begin());
      return literal;
    }
    case federated_language::Array::kInt16List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          federated_language::DataType::DT_INT16, array_pb.shape())));
      CopyFromRepeatedField(array_pb.int16_list().value(),
                            literal.data<int16_t>().begin());
      return literal;
    }
    case federated_language::Array::kInt32List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          federated_language::DataType::DT_INT32, array_pb.shape())));
      CopyFromRepeatedField(array_pb.int32_list().value(),
                            literal.data<int32_t>().begin());
      return literal;
    }
    case federated_language::Array::kInt64List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          federated_language::DataType::DT_INT64, array_pb.shape())));
      CopyFromRepeatedField(array_pb.int64_list().value(),
                            literal.data<int64_t>().begin());
      return literal;
    }
    case federated_language::Array::kUint8List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          federated_language::DataType::DT_UINT8, array_pb.shape())));
      CopyFromRepeatedField(array_pb.uint8_list().value(),
                            literal.data<uint8_t>().begin());
      return literal;
    }
    case federated_language::Array::kUint16List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          federated_language::DataType::DT_UINT16, array_pb.shape())));
      CopyFromRepeatedField(array_pb.uint16_list().value(),
                            literal.data<uint16_t>().begin());
      return literal;
    }
    case federated_language::Array::kUint32List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          federated_language::DataType::DT_UINT32, array_pb.shape())));
      CopyFromRepeatedField(array_pb.uint32_list().value(),
                            literal.data<uint32_t>().begin());
      return literal;
    }
    case federated_language::Array::kUint64List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          federated_language::DataType::DT_UINT64, array_pb.shape())));
      CopyFromRepeatedField(array_pb.uint64_list().value(),
                            literal.data<uint64_t>().begin());
      return literal;
    }
    case federated_language::Array::kFloat16List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          federated_language::DataType::DT_HALF, array_pb.shape())));
      CopyFromRepeatedField(array_pb.float16_list().value(),
                            literal.data<xla::half>().begin());
      return literal;
    }
    case federated_language::Array::kFloat32List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          federated_language::DataType::DT_FLOAT, array_pb.shape())));
      CopyFromRepeatedField(array_pb.float32_list().value(),
                            literal.data<float>().begin());
      return literal;
    }
    case federated_language::Array::kFloat64List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          federated_language::DataType::DT_DOUBLE, array_pb.shape())));
      CopyFromRepeatedField(array_pb.float64_list().value(),
                            literal.data<double>().begin());
      return literal;
    }
    case federated_language::Array::kComplex64List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          federated_language::DataType::DT_COMPLEX64, array_pb.shape())));
      CopyFromRepeatedField(array_pb.complex64_list().value(),
                            literal.data<xla::complex64>().begin());
      return literal;
    }
    case federated_language::Array::kComplex128List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          federated_language::DataType::DT_COMPLEX128, array_pb.shape())));
      CopyFromRepeatedField(array_pb.complex128_list().value(),
                            literal.data<xla::complex128>().begin());
      return literal;
    }
    case federated_language::Array::kBfloat16List: {
      xla::Literal literal(TFF_TRY(ShapeFromArrayShape(
          federated_language::DataType::DT_BFLOAT16, array_pb.shape())));
      CopyFromRepeatedField(array_pb.bfloat16_list().value(),
                            literal.data<xla::bfloat16>().begin());
      return literal;
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found:", array_pb.kind_case()));
  }
}

}  // namespace tensorflow_federated
