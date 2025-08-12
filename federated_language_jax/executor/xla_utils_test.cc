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

#include <complex>
#include <cstdint>
#include <initializer_list>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "Eigen/Core"
#include "federated_language/proto/array.pb.h"
#include "federated_language/proto/computation.pb.h"
#include "federated_language/proto/data_type.pb.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/array_shape_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/array_test_utils.h"
#include "tensorflow_federated/cc/testing/protobuf_matchers.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace federated_language_jax {
namespace {

TEST(ShapeFromTensorTypeTest, TestReturnsShape_fully_defined) {
  std::initializer_list<int64_t> dims = {2, 3};
  federated_language::TensorType type_pb;
  type_pb.set_dtype(federated_language::DataType::DT_INT32);
  type_pb.mutable_dims()->Assign(dims.begin(), dims.end());
  const xla::Shape& expected_shape = xla::ShapeUtil::MakeShape(
      xla::primitive_util::NativeToPrimitiveType<int32_t>(), {2, 3});

  const xla::Shape& actual_shape = TFF_ASSERT_OK(ShapeFromTensorType(type_pb));

  EXPECT_TRUE(xla::Shape::Equal()(actual_shape, expected_shape));
}

TEST(ShapeFromTensorTypeTest, TestReturnsShape_scalar) {
  std::initializer_list<int64_t> dims = {};
  federated_language::TensorType type_pb;
  type_pb.set_dtype(federated_language::DataType::DT_INT32);
  type_pb.mutable_dims()->Assign(dims.begin(), dims.end());
  const xla::Shape& expected_shape = xla::ShapeUtil::MakeShape(
      xla::primitive_util::NativeToPrimitiveType<int32_t>(), {});

  const xla::Shape& actual_shape = TFF_ASSERT_OK(ShapeFromTensorType(type_pb));

  EXPECT_TRUE(xla::Shape::Equal()(actual_shape, expected_shape));
}

TEST(ShapeFromTensorTypeTest, TestFails_partially_defined) {
  std::initializer_list<int64_t> dims = {2, -1};
  federated_language::TensorType type_pb;
  type_pb.set_dtype(federated_language::DataType::DT_INT32);
  type_pb.mutable_dims()->Assign(dims.begin(), dims.end());

  const absl::StatusOr<xla::Shape>& result = ShapeFromTensorType(type_pb);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ShapeFromTensorTypeTest, TestFails_unknown) {
  std::initializer_list<int64_t> dims = {};
  federated_language::TensorType type_pb;
  type_pb.set_dtype(federated_language::DataType::DT_INT32);
  type_pb.mutable_dims()->Assign(dims.begin(), dims.end());
  type_pb.set_unknown_rank(true);

  const absl::StatusOr<xla::Shape>& result = ShapeFromTensorType(type_pb);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ShapeFromArrayShapeTest, TestReturnsShape_fully_defined) {
  const federated_language::ArrayShape& shape_pb =
      tensorflow_federated::testing::CreateArrayShape({2, 3});

  const xla::Shape& actual_shape = TFF_ASSERT_OK(
      ShapeFromArrayShape(federated_language::DataType::DT_INT32, shape_pb));

  const xla::Shape& expected_shape = xla::ShapeUtil::MakeShape(
      xla::primitive_util::NativeToPrimitiveType<int32_t>(), {2, 3});
  EXPECT_TRUE(xla::Shape::Equal()(actual_shape, expected_shape));
}

TEST(ShapeFromArrayShapeTest, TestReturnsShape_scalar) {
  const federated_language::ArrayShape& shape_pb =
      tensorflow_federated::testing::CreateArrayShape({});

  const xla::Shape& actual_shape = TFF_ASSERT_OK(
      ShapeFromArrayShape(federated_language::DataType::DT_INT32, shape_pb));

  const xla::Shape& expected_shape = xla::ShapeUtil::MakeShape(
      xla::primitive_util::NativeToPrimitiveType<int32_t>(), {});
  EXPECT_TRUE(xla::Shape::Equal()(actual_shape, expected_shape));
}

TEST(ShapeFromArrayShapeTest, TestFails_partially_defined) {
  const federated_language::ArrayShape& shape_pb =
      tensorflow_federated::testing::CreateArrayShape({2, -1});

  const absl::StatusOr<xla::Shape>& result =
      ShapeFromArrayShape(federated_language::DataType::DT_INT32, shape_pb);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ShapeFromArrayShapeTest, TestFails_unknown) {
  const federated_language::ArrayShape& shape_pb =
      tensorflow_federated::testing::CreateArrayShape({}, true);

  const absl::StatusOr<xla::Shape>& result =
      ShapeFromArrayShape(federated_language::DataType::DT_INT32, shape_pb);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ArrayFromLiteralTest, TestReturnsArray_bool) {
  const xla::Literal literal = xla::LiteralUtil::CreateR0(true);

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromLiteral(literal));

  const federated_language::Array& expected_array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_BOOL,
          tensorflow_federated::testing::CreateArrayShape({}), {true}));
  EXPECT_THAT(actual_array_pb,
              tensorflow_federated::testing::EqualsProto(expected_array_pb));
}

TEST(ArrayFromLiteralTest, TestReturnsArray_int8) {
  const xla::Literal literal = xla::LiteralUtil::CreateR0<int8_t>(1);

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromLiteral(literal));

  const federated_language::Array& expected_array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_INT8,
          tensorflow_federated::testing::CreateArrayShape({}), {1}));
  EXPECT_THAT(actual_array_pb,
              tensorflow_federated::testing::EqualsProto(expected_array_pb));
}

TEST(ArrayFromLiteralTest, TestReturnsArray_int16) {
  const xla::Literal literal = xla::LiteralUtil::CreateR0<int16_t>(1);

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromLiteral(literal));

  const federated_language::Array& expected_array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_INT16,
          tensorflow_federated::testing::CreateArrayShape({}), {1}));
  EXPECT_THAT(actual_array_pb,
              tensorflow_federated::testing::EqualsProto(expected_array_pb));
}

TEST(ArrayFromLiteralTest, TestReturnsArray_int32) {
  const xla::Literal literal = xla::LiteralUtil::CreateR0<int32_t>(1);

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromLiteral(literal));

  const federated_language::Array& expected_array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_INT32,
          tensorflow_federated::testing::CreateArrayShape({}), {1}));
  EXPECT_THAT(actual_array_pb,
              tensorflow_federated::testing::EqualsProto(expected_array_pb));
}

TEST(ArrayFromLiteralTest, TestReturnsArray_int64) {
  const xla::Literal literal = xla::LiteralUtil::CreateR0<int64_t>(1);

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromLiteral(literal));

  const federated_language::Array& expected_array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_INT64,
          tensorflow_federated::testing::CreateArrayShape({}), {1}));
  EXPECT_THAT(actual_array_pb,
              tensorflow_federated::testing::EqualsProto(expected_array_pb));
}

TEST(ArrayFromLiteralTest, TestReturnsArray_uint8) {
  const xla::Literal literal = xla::LiteralUtil::CreateR0<uint8_t>(1);

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromLiteral(literal));

  const federated_language::Array& expected_array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_UINT8,
          tensorflow_federated::testing::CreateArrayShape({}), {1}));
  EXPECT_THAT(actual_array_pb,
              tensorflow_federated::testing::EqualsProto(expected_array_pb));
}

TEST(ArrayFromLiteralTest, TestReturnsArray_uint16) {
  const xla::Literal literal = xla::LiteralUtil::CreateR0<uint16_t>(1);

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromLiteral(literal));

  const federated_language::Array& expected_array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_UINT16,
          tensorflow_federated::testing::CreateArrayShape({}), {1}));
  EXPECT_THAT(actual_array_pb,
              tensorflow_federated::testing::EqualsProto(expected_array_pb));
}

TEST(ArrayFromLiteralTest, TestReturnsArray_uint32) {
  const xla::Literal literal = xla::LiteralUtil::CreateR0<uint32_t>(1);

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromLiteral(literal));

  const federated_language::Array& expected_array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_UINT32,
          tensorflow_federated::testing::CreateArrayShape({}), {1}));
  EXPECT_THAT(actual_array_pb,
              tensorflow_federated::testing::EqualsProto(expected_array_pb));
}

TEST(ArrayFromLiteralTest, TestReturnsArray_uint64) {
  const xla::Literal literal = xla::LiteralUtil::CreateR0<uint64_t>(1);

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromLiteral(literal));

  const federated_language::Array& expected_array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_UINT64,
          tensorflow_federated::testing::CreateArrayShape({}), {1}));
  EXPECT_THAT(actual_array_pb,
              tensorflow_federated::testing::EqualsProto(expected_array_pb));
}

TEST(ArrayFromLiteralTest, TestReturnsArray_float16) {
  const xla::Literal literal = xla::LiteralUtil::CreateR0(Eigen::half{1.0});

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromLiteral(literal));

  const federated_language::Array& expected_array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_HALF,
          tensorflow_federated::testing::CreateArrayShape({}),
          {Eigen::half{1.0}}));
  EXPECT_THAT(actual_array_pb,
              tensorflow_federated::testing::EqualsProto(expected_array_pb));
}

TEST(ArrayFromLiteralTest, TestReturnsArray_float32) {
  const xla::Literal literal = xla::LiteralUtil::CreateR0<float>(1.0);

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromLiteral(literal));

  const federated_language::Array& expected_array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_FLOAT,
          tensorflow_federated::testing::CreateArrayShape({}), {1.0}));
  EXPECT_THAT(actual_array_pb,
              tensorflow_federated::testing::EqualsProto(expected_array_pb));
}

TEST(ArrayFromLiteralTest, TestReturnsArray_float64) {
  const xla::Literal literal = xla::LiteralUtil::CreateR0<double>(1.0);

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromLiteral(literal));

  const federated_language::Array& expected_array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_DOUBLE,
          tensorflow_federated::testing::CreateArrayShape({}), {1.0}));
  EXPECT_THAT(actual_array_pb,
              tensorflow_federated::testing::EqualsProto(expected_array_pb));
}

TEST(ArrayFromLiteralTest, TestReturnsArray_complex64) {
  const xla::Literal literal =
      xla::LiteralUtil::CreateR0(xla::complex64{1.0, 1.0});

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromLiteral(literal));

  const federated_language::Array& expected_array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_COMPLEX64,
          tensorflow_federated::testing::CreateArrayShape({}),
          {std::complex<float>{1.0, 1.0}}));
  EXPECT_THAT(actual_array_pb,
              tensorflow_federated::testing::EqualsProto(expected_array_pb));
}

TEST(ArrayFromLiteralTest, TestReturnsArray_complex128) {
  const xla::Literal literal =
      xla::LiteralUtil::CreateR0(xla::complex128{1.0, 1.0});

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromLiteral(literal));

  const federated_language::Array& expected_array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_COMPLEX128,
          tensorflow_federated::testing::CreateArrayShape({}),
          {std::complex<double>{1.0, 1.0}}));
  EXPECT_THAT(actual_array_pb,
              tensorflow_federated::testing::EqualsProto(expected_array_pb));
}

TEST(ArrayFromLiteralTest, TestReturnsArray_bfloat16) {
  const xla::Literal literal = xla::LiteralUtil::CreateR0(Eigen::bfloat16{1.0});

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromLiteral(literal));

  const federated_language::Array& expected_array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_BFLOAT16,
          tensorflow_federated::testing::CreateArrayShape({}),
          {Eigen::bfloat16{1.0}}));
  EXPECT_THAT(actual_array_pb,
              tensorflow_federated::testing::EqualsProto(expected_array_pb));
}

TEST(ArrayFromLiteralTest, TestReturnsArray_array) {
  const xla::Literal literal =
      xla::LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}});

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromLiteral(literal));

  const federated_language::Array& expected_array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_INT32,
          tensorflow_federated::testing::CreateArrayShape({2, 3}),
          {1, 2, 3, 4, 5, 6}));
  EXPECT_THAT(actual_array_pb,
              tensorflow_federated::testing::EqualsProto(expected_array_pb));
}

TEST(LiteralFromArrayTest, TestReturnsLiteral_bool) {
  const federated_language::Array& array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_BOOL,
          tensorflow_federated::testing::CreateArrayShape({}), {true}));

  const xla::Literal& actual_literal =
      TFF_ASSERT_OK(LiteralFromArray(array_pb));

  const xla::Literal expected_literal = xla::LiteralUtil::CreateR0(true);
  EXPECT_EQ(actual_literal, expected_literal);
}

TEST(LiteralFromArrayTest, TestReturnsLiteral_int8) {
  const federated_language::Array& array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_INT8,
          tensorflow_federated::testing::CreateArrayShape({}), {1}));

  const xla::Literal& actual_literal =
      TFF_ASSERT_OK(LiteralFromArray(array_pb));

  const xla::Literal expected_literal = xla::LiteralUtil::CreateR0<int8_t>(1);
  EXPECT_EQ(actual_literal, expected_literal);
}

TEST(LiteralFromArrayTest, TestReturnsLiteral_int16) {
  const federated_language::Array& array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_INT16,
          tensorflow_federated::testing::CreateArrayShape({}), {1}));

  const xla::Literal& actual_literal =
      TFF_ASSERT_OK(LiteralFromArray(array_pb));

  const xla::Literal expected_literal = xla::LiteralUtil::CreateR0<int16_t>(1);
  EXPECT_EQ(actual_literal, expected_literal);
}

TEST(LiteralFromArrayTest, TestReturnsLiteral_int32) {
  const federated_language::Array& array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_INT32,
          tensorflow_federated::testing::CreateArrayShape({}), {1}));

  const xla::Literal& actual_literal =
      TFF_ASSERT_OK(LiteralFromArray(array_pb));

  const xla::Literal expected_literal = xla::LiteralUtil::CreateR0<int32_t>(1);
  EXPECT_EQ(actual_literal, expected_literal);
}

TEST(LiteralFromArrayTest, TestReturnsLiteral_int64) {
  const federated_language::Array& array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_INT64,
          tensorflow_federated::testing::CreateArrayShape({}), {1}));

  const xla::Literal& actual_literal =
      TFF_ASSERT_OK(LiteralFromArray(array_pb));

  const xla::Literal expected_literal = xla::LiteralUtil::CreateR0<int64_t>(1);
  EXPECT_EQ(actual_literal, expected_literal);
}

TEST(LiteralFromArrayTest, TestReturnsLiteral_uint8) {
  const federated_language::Array& array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_UINT8,
          tensorflow_federated::testing::CreateArrayShape({}), {1}));

  const xla::Literal& actual_literal =
      TFF_ASSERT_OK(LiteralFromArray(array_pb));

  const xla::Literal expected_literal = xla::LiteralUtil::CreateR0<uint8_t>(1);
  EXPECT_EQ(actual_literal, expected_literal);
}

TEST(LiteralFromArrayTest, TestReturnsLiteral_uint16) {
  const federated_language::Array& array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_UINT16,
          tensorflow_federated::testing::CreateArrayShape({}), {1}));

  const xla::Literal& actual_literal =
      TFF_ASSERT_OK(LiteralFromArray(array_pb));

  const xla::Literal expected_literal = xla::LiteralUtil::CreateR0<uint16_t>(1);
  EXPECT_EQ(actual_literal, expected_literal);
}

TEST(LiteralFromArrayTest, TestReturnsLiteral_uint32) {
  const federated_language::Array& array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_UINT32,
          tensorflow_federated::testing::CreateArrayShape({}), {1}));

  const xla::Literal& actual_literal =
      TFF_ASSERT_OK(LiteralFromArray(array_pb));

  const xla::Literal expected_literal = xla::LiteralUtil::CreateR0<uint32_t>(1);
  EXPECT_EQ(actual_literal, expected_literal);
}

TEST(LiteralFromArrayTest, TestReturnsLiteral_uint64) {
  const federated_language::Array& array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_UINT64,
          tensorflow_federated::testing::CreateArrayShape({}), {1}));

  const xla::Literal& actual_literal =
      TFF_ASSERT_OK(LiteralFromArray(array_pb));

  const xla::Literal expected_literal = xla::LiteralUtil::CreateR0<uint64_t>(1);
  EXPECT_EQ(actual_literal, expected_literal);
}

TEST(LiteralFromArrayTest, TestReturnsLiteral_float16) {
  const federated_language::Array& array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_HALF,
          tensorflow_federated::testing::CreateArrayShape({}),
          {Eigen::half{1.0}}));

  const xla::Literal& actual_literal =
      TFF_ASSERT_OK(LiteralFromArray(array_pb));

  const xla::Literal expected_literal =
      xla::LiteralUtil::CreateR0(Eigen::half{1.0});
  EXPECT_EQ(actual_literal, expected_literal);
}

TEST(LiteralFromArrayTest, TestReturnsLiteral_float32) {
  const federated_language::Array& array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_FLOAT,
          tensorflow_federated::testing::CreateArrayShape({}), {1.0}));

  const xla::Literal& actual_literal =
      TFF_ASSERT_OK(LiteralFromArray(array_pb));

  const xla::Literal expected_literal = xla::LiteralUtil::CreateR0<float>(1.0);
  EXPECT_EQ(actual_literal, expected_literal);
}

TEST(LiteralFromArrayTest, TestReturnsLiteral_float64) {
  const federated_language::Array& array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_DOUBLE,
          tensorflow_federated::testing::CreateArrayShape({}), {1.0}));

  const xla::Literal& actual_literal =
      TFF_ASSERT_OK(LiteralFromArray(array_pb));

  const xla::Literal expected_literal = xla::LiteralUtil::CreateR0<double>(1.0);
  EXPECT_EQ(actual_literal, expected_literal);
}

TEST(LiteralFromArrayTest, TestReturnsLiteral_complex64) {
  const federated_language::Array& array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_COMPLEX64,
          tensorflow_federated::testing::CreateArrayShape({}),
          {std::complex<float>{1.0, 1.0}}));

  const xla::Literal& actual_literal =
      TFF_ASSERT_OK(LiteralFromArray(array_pb));

  const xla::Literal expected_literal =
      xla::LiteralUtil::CreateR0(xla::complex64{1.0, 1.0});
  EXPECT_EQ(actual_literal, expected_literal);
}

TEST(LiteralFromArrayTest, TestReturnsLiteral_complex128) {
  const federated_language::Array& array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_COMPLEX128,
          tensorflow_federated::testing::CreateArrayShape({}),
          {std::complex<double>{1.0, 1.0}}));

  const xla::Literal& actual_literal =
      TFF_ASSERT_OK(LiteralFromArray(array_pb));

  const xla::Literal expected_literal =
      xla::LiteralUtil::CreateR0(xla::complex128{1.0, 1.0});
  EXPECT_EQ(actual_literal, expected_literal);
}

TEST(LiteralFromArrayTest, TestReturnsLiteral_bfloat16) {
  const federated_language::Array& array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_BFLOAT16,
          tensorflow_federated::testing::CreateArrayShape({}),
          {Eigen::bfloat16{1.0}}));

  const xla::Literal& actual_literal =
      TFF_ASSERT_OK(LiteralFromArray(array_pb));

  const xla::Literal expected_literal =
      xla::LiteralUtil::CreateR0(Eigen::bfloat16{1.0});
  EXPECT_EQ(actual_literal, expected_literal);
}

TEST(LiteralFromArrayTest, TestReturnsLiteral_array) {
  const federated_language::Array& array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_INT32,
          tensorflow_federated::testing::CreateArrayShape({2, 3}),
          {1, 2, 3, 4, 5, 6}));

  const xla::Literal& actual_literal =
      TFF_ASSERT_OK(LiteralFromArray(array_pb));

  const xla::Literal expected_literal =
      xla::LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}});
  EXPECT_EQ(actual_literal, expected_literal);
}

TEST(LiteralFromArrayTest, TestFails_string) {
  const federated_language::Array& array_pb =
      TFF_ASSERT_OK(tensorflow_federated::testing::CreateArray(
          federated_language::DataType::DT_STRING,
          tensorflow_federated::testing::CreateArrayShape({}), {"a"}));

  const absl::StatusOr<xla::Literal>& result = LiteralFromArray(array_pb);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace federated_language_jax
