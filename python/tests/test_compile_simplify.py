# Copyright © 2026 Apple Inc.

import os
import tempfile

import mlx.core as mx
import mlx_tests


class TestCompileSimplify(mlx_tests.MLXTestCase):
    def test_inverse_transposes(self):
        for shapeless in (False, True):
            traces = []

            def fun(x):
                traces.append(1)
                inner = x.transpose(1, -1, 0)
                outer = inner.transpose(2, 0, 1)
                return inner, outer, outer, mx.sin(outer)

            compiled = mx.compile(fun, shapeless=shapeless)
            inputs = (
                mx.arange(24, dtype=mx.float32).reshape(2, 3, 4),
                mx.arange(48, dtype=mx.float32).reshape(4, 3, 4)[::2],
                mx.arange(24, dtype=mx.float32).reshape(2, 3, 4).transpose(1, 0, 2),
                mx.zeros((0, 2, 4)),
            )
            for x in inputs:
                outputs = compiled(x)
                expected = (x.transpose(1, 2, 0), x, x, mx.sin(x))
                for actual, reference in zip(outputs, expected):
                    self.assertEqual(actual.shape, reference.shape)
                    self.assertTrue(mx.array_equal(actual, reference))
            self.assertEqual(len(traces), 1 if shapeless else 3)

        def identity(x):
            return x.transpose(1, 2, 0).transpose(2, 0, 1)

        x = mx.arange(24, dtype=mx.float32).reshape(2, 3, 4)

        def loss(x):
            return mx.sum(mx.square(identity(x)))

        for grad in (mx.grad(mx.compile(loss)), mx.compile(mx.grad(loss))):
            self.assertTrue(mx.array_equal(grad(x), 2 * x))
        batch = mx.stack((x, x + 1))
        for mapped in (mx.vmap(mx.compile(identity)), mx.compile(mx.vmap(identity))):
            self.assertTrue(mx.array_equal(mapped(batch), batch))

    def test_inverse_transpose_logs(self):
        stderr = self.run_compile_log_test(
            """
x = mx.arange(9).reshape(3, 3)
mx.compile(lambda x: x.T.T, shapeless=True)(x)
mx.compile(lambda x: x.T.transpose(0, 1))(x)
s = mx.new_stream(mx.default_device())
mx.compile(lambda x: mx.transpose(x.T, stream=s))(x)
""",
            "passes,guards",
        )
        self.assertIn("event=rewrite rule=inverse_transposes", stderr)
        self.assertIn("check=inverse_permutation", stderr)
        self.assertIn("reason=index_map_mismatch", stderr)
        self.assertIn("reason=stream_mismatch", stderr)

    def test_roundtrip_views(self):
        def fun(x):
            inner = x.reshape(-1)
            outer = inner.reshape(x.shape)
            return inner, outer, outer, mx.sin(outer)

        compiled = mx.compile(fun)
        for x in (
            mx.arange(12, dtype=mx.float32).reshape(4, 3),
            mx.arange(24, dtype=mx.float32).reshape(4, 6)[:, ::2],
            mx.arange(12, dtype=mx.float32).reshape(3, 4).T,
            mx.zeros((0, 3)),
        ):
            for actual, expected in zip(compiled(x), fun(x)):
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(mx.array_equal(actual, expected))

        def identity(x):
            return x.reshape(3, 2).reshape(2, 3)

        def loss(x):
            return mx.sum(mx.square(identity(x)))

        x = mx.arange(6, dtype=mx.float32).reshape(2, 3)
        for grad in (mx.grad(mx.compile(loss)), mx.compile(mx.grad(loss))):
            self.assertTrue(mx.array_equal(grad(x), 2 * x))
        batch = mx.stack((x, x + 1))
        for mapped in (mx.vmap(mx.compile(identity)), mx.compile(mx.vmap(identity))):
            self.assertTrue(mx.array_equal(mapped(batch), batch))

    def test_roundtrip_views_shapeless_guards(self):
        def fun(x):
            return x.reshape(2, -1).reshape(-1)

        compiled = mx.compile(fun, shapeless=True)
        for n in (6, 8):
            x = mx.arange(n)
            self.assertTrue(mx.array_equal(compiled(x), x))
        with self.assertRaises(ValueError):
            compiled(mx.arange(7))

        stderr = self.run_compile_log_test(
            """
x = mx.arange(6)
fun = lambda x: x.reshape(2, -1).reshape(-1)
mx.compile(fun)(x)
mx.compile(fun, shapeless=True)(x)
""",
            "passes,guards",
        )
        self.assertIn("event=rewrite rule=roundtrip_views", stderr)
        self.assertIn("reason=shape_not_proven", stderr)

    def test_split_concat(self):
        for indices in ([3, 7], [0, 3, 12], [3, 3, 7], list(range(1, 12))):

            def fun(x):
                parts = mx.split(x, indices, axis=1)
                output = mx.concatenate(parts, axis=1)
                return parts[0], output, output

            compiled = mx.compile(fun)
            for x in (
                mx.arange(24).reshape(2, 12),
                mx.arange(48).reshape(2, 24)[:, ::2],
                mx.zeros((0, 12)),
                mx.zeros((2, 0)),
            ):
                for actual, expected in zip(compiled(x), fun(x)):
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertTrue(mx.array_equal(actual, expected))

        def identity(x):
            return mx.concatenate(mx.split(x, [1, 2], axis=1), axis=1)

        def loss(x):
            return mx.sum(mx.square(identity(x)))

        x = mx.arange(6, dtype=mx.float32).reshape(2, 3)
        for grad in (mx.grad(mx.compile(loss)), mx.compile(mx.grad(loss))):
            self.assertTrue(mx.array_equal(grad(x), 2 * x))
        batch = mx.stack((x, x + 1))
        for mapped in (mx.vmap(mx.compile(identity)), mx.compile(mx.vmap(identity))):
            self.assertTrue(mx.array_equal(mapped(batch), batch))

    def test_split_concat_shapeless_guards(self):
        for indices in ([2, 4], [0, 2, 6]):

            def fun(x):
                return mx.concatenate(mx.split(x, indices, axis=1), axis=1)

            with self.assertRaisesRegex(ValueError, "cannot infer output shapes"):
                mx.compile(fun, shapeless=True)(mx.arange(12).reshape(2, 6))

        stderr = self.run_compile_log_test(
            """
x = mx.arange(12).reshape(2, 6)
fun = lambda x: mx.concatenate(mx.split(x, [2, 4], axis=1), axis=1)
mx.compile(fun)(x)
mx.compile(lambda x: mx.concatenate(mx.split(x, [2, 4], axis=1)[::-1], axis=1))(x)
try:
    mx.compile(fun, shapeless=True)(x)
except ValueError as e:
    assert "cannot infer output shapes" in str(e)
else:
    raise AssertionError("Unproven shapeless split was removed")
""",
            "passes,guards",
        )
        self.assertIn("event=rewrite rule=split_concat", stderr)
        self.assertIn("reason=index_map_mismatch", stderr)
        self.assertIn("reason=shape_not_proven", stderr)

    def assert_outputs(self, actual, expected):
        actual = (actual,) if isinstance(actual, mx.array) else actual
        expected = (expected,) if isinstance(expected, mx.array) else expected
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(mx.array_equal(a, b, equal_nan=True))

    def test_concat_split(self):
        for widths in ((2, 3), (0, 2, 0, 3, 0), (1,) * 12):
            boundaries = []
            for width in widths[:-1]:
                boundaries.append((boundaries[-1] if boundaries else 0) + width)

            def fun(*xs):
                joined = mx.concatenate(xs, axis=1)
                parts = mx.split(joined, boundaries, axis=1)
                return (*parts[::-1], parts[0], joined)

            for dtype in (mx.float32, mx.float16, mx.bfloat16, mx.int32):
                inputs = [
                    mx.arange(2 * w).reshape(2, w).astype(dtype)[:, ::-1]
                    for w in widths
                ]
                compiled = mx.compile(fun)
                self.assert_outputs(compiled(*inputs), fun(*inputs))
                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, "concat_split.mlxfn")
                    mx.export_function(path, fun, *inputs)
                    self.assert_outputs(mx.import_function(path)(*inputs), fun(*inputs))

    def test_concat_split_promotion_and_transforms(self):
        def fun(a, b):
            first, second = mx.split(mx.concatenate([a, b], axis=1), [2], axis=1)
            return first, second, first

        a = mx.arange(4).reshape(2, 2).astype(mx.int16)
        b = mx.arange(6, dtype=mx.float32).reshape(2, 3)
        self.assert_outputs(mx.compile(fun)(a, b), fun(a, b))
        a = a.astype(mx.float32)

        def loss(a, b):
            return sum(mx.sum(x) for x in fun(a, b))

        self.assert_outputs(
            mx.grad(mx.compile(loss), argnums=(0, 1))(a, b),
            mx.grad(loss, argnums=(0, 1))(a, b),
        )
        batch_a, batch_b = mx.stack([a, a + 1]), mx.stack([b, b + 1])
        self.assert_outputs(
            mx.vmap(mx.compile(fun))(batch_a, batch_b), mx.vmap(fun)(batch_a, batch_b)
        )

    def test_concat_split_shapeless_preserves_validation(self):
        for boundary in (0, 2):

            def fun(a, b):
                return mx.split(mx.concatenate([a, b]), [boundary])

            compiled = mx.compile(fun, shapeless=True)
            with self.assertRaisesRegex(ValueError, "infer output shapes"):
                compiled(mx.ones((2,)), mx.ones((3,)))

    def test_integer_cast_chains(self):
        cases = (
            (mx.int8, mx.int64, [-128, -1, 0, 127]),
            (mx.uint8, mx.int16, [0, 127, 128, 255]),
            (mx.int16, mx.int32, [-32768, -1, 0, 32767]),
            (mx.uint32, mx.int64, [0, 2**31, 2**32 - 1]),
            (mx.bool_, mx.uint64, [False, True]),
        )
        for dtype, via, values in cases:

            def fun(x):
                inner = x.astype(via)
                return inner.astype(dtype), inner, inner.astype(mx.uint8)

            x = mx.array(values, dtype=dtype)
            for shapeless in (False, True):
                compiled = mx.compile(fun, shapeless=shapeless)
                for value in (x, x[::-1], x[:0], mx.concatenate([x, x])):
                    self.assert_outputs(compiled(value), fun(value))
                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, "casts.mlxfn")
                    mx.export_function(path, fun, x, shapeless=shapeless)
                    self.assert_outputs(mx.import_function(path)(x), fun(x))

    def test_cast_rounding_and_signedness(self):
        cases = (
            (mx.int16, mx.int8, [-257, -128, 127, 256]),
            (mx.int32, mx.uint16, [-65536, -1, 0, 65536]),
            (mx.uint16, mx.int16, [0, 32767, 32768, 65535]),
            (mx.int32, mx.bool_, [-1, 0, 1, 2]),
        )
        for dtype, via, values in cases:
            x = mx.array(values, dtype=dtype)

            def fun(x):
                return x.astype(via).astype(dtype)

            self.assert_outputs(mx.compile(fun)(x), fun(x))

        x = mx.array([0.0, -0.0, 1.0003, 0.000001, float("inf"), float("nan")])
        for via in (mx.float16, mx.bfloat16):

            def fun(x):
                return x.astype(via).astype(mx.float32)

            actual, expected = mx.compile(fun)(x), fun(x)
            self.assert_outputs(actual, expected)
            self.assertTrue(
                mx.array_equal(actual.view(mx.uint32)[:2], expected.view(mx.uint32)[:2])
            )

    def test_cast_chain_logs(self):
        stderr = self.run_compile_log_test(
            """
x = mx.array([-128, 127], dtype=mx.int8)
mx.compile(lambda x: x.astype(mx.int64).astype(mx.int8))(x)
mx.compile(lambda x: x.astype(mx.uint16).astype(mx.int8))(x)
""",
            "passes,guards",
        )
        self.assertIn("event=rewrite rule=redundant_cast", stderr)
        self.assertIn("reason=lossy_or_unsupported_conversion", stderr)

    def test_uniform_values(self):
        def fun(s):
            filled = mx.full((2, 3), s)
            value = mx.square(filled.T).astype(mx.float32) + mx.ones((3, 2))
            return value, filled, value

        for dtype in (mx.float32, mx.float16, mx.bfloat16, mx.int32):
            compiled = mx.compile(fun)
            for value in (2, -3, 0):
                s = mx.array(value, dtype=dtype)
                self.assert_outputs(compiled(s), fun(s))
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "uniform.mlxfn")
                mx.export_function(path, fun, s)
                s = mx.array(7, dtype=dtype)
                self.assert_outputs(mx.import_function(path)(s), fun(s))

        def loss(s):
            return mx.sum(fun(s)[0])

        s = mx.array(2.0)
        for grad in (mx.grad(mx.compile(loss)), mx.compile(mx.grad(loss))):
            self.assert_outputs(grad(s), mx.grad(loss)(s))
        batch = mx.array([2.0, -3.0, 0.0])
        for mapped in (mx.vmap(mx.compile(fun)), mx.compile(mx.vmap(fun))):
            self.assert_outputs(mapped(batch), mx.vmap(fun)(batch))

    def test_uniform_boolean_conditions(self):
        x = mx.arange(6, dtype=mx.int16)
        y = mx.array(2.5)

        def fun(x, y):
            condition = mx.logical_and(
                mx.full((6,), True), mx.logical_not(mx.full((6,), False))
            )
            return mx.where(condition, x, y), mx.where(
                mx.not_equal(condition, True), x, y
            )

        self.assert_outputs(mx.compile(fun)(x, y), fun(x, y))

        def runtime_condition(s, x, y):
            return mx.where(mx.full((6,), s), x, y)

        compiled = mx.compile(runtime_condition)
        for condition in (True, False):
            s = mx.array(condition)
            self.assert_outputs(compiled(s, x, y), runtime_condition(s, x, y))

        def rounded_condition(x, y):
            condition = mx.full((6,), 1e-20).astype(mx.float16)
            return mx.where(condition, x, y)

        self.assert_outputs(
            mx.compile(rounded_condition)(x, y), rounded_condition(x, y)
        )

    def test_uniform_numerics_and_shapeless_validation(self):
        for dtype in (mx.float32, mx.float16, mx.bfloat16):

            def fun(s):
                return mx.negative(mx.full((8,), s).astype(dtype)).astype(mx.float32)

            compiled = mx.compile(fun)
            for value in (0.0, -0.0, 1.0003, float("inf"), float("nan")):
                s = mx.array(value)
                actual, expected = compiled(s), fun(s)
                self.assert_outputs(actual, expected)
                if value == 0:
                    self.assert_outputs(
                        actual.view(mx.uint32), expected.view(mx.uint32)
                    )

        def empty(s):
            return mx.square(mx.full((0, 3), s))

        self.assert_outputs(mx.compile(empty)(mx.array(2.0)), mx.zeros((0, 3)))

        def guarded(x, y):
            return mx.where(mx.full((4,), False), x.reshape(2, -1).reshape(-1), y)

        compiled = mx.compile(guarded, shapeless=True)
        self.assert_outputs(compiled(mx.arange(4), mx.ones(4)), mx.ones(4))
        with self.assertRaises(ValueError):
            compiled(mx.arange(5), mx.ones(4))

    def test_uniform_value_logs(self):
        stderr = self.run_compile_log_test(
            """
x = mx.arange(4)
mx.compile(lambda x: mx.where(mx.full((4,), True), x, -x))(x)
fun = lambda s: mx.square(mx.full((4,), s))
mx.compile(fun)(mx.array(2.0))
mx.compile(fun, shapeless=True)(mx.array(2.0))
""",
            "passes,guards",
        )
        self.assertIn("event=rewrite rule=uniform_value", stderr)
        self.assertIn("event=rewrite rule=uniform_select", stderr)
        self.assertIn("reason=nonempty_shape_not_proven", stderr)

    def test_combine_concatenate_slices(self):
        for start, stop in ((0, 2), (1, 5), (3, 7), (0, 7), (2, 2)):

            def fun(a, b):
                joined = mx.concatenate([a, b], axis=1)
                part = joined[:, start:stop]
                output = mx.concatenate([part, joined, joined, part], axis=1)
                return output, output

            compiled = mx.compile(fun)
            for dtype in (mx.int32, mx.float32, mx.float16, mx.bfloat16):
                a = mx.arange(12).reshape(2, 6)[:, ::2].astype(dtype)
                b = mx.arange(8).reshape(2, 4)[:, ::-1].astype(dtype)
                self.assert_outputs(compiled(a, b), fun(a, b))
            a = mx.array([[16777217, -16777217, 3]], dtype=mx.int32)
            b = mx.array([[0.5, 1.5, 2.5, 3.5]])
            self.assert_outputs(compiled(a, b), fun(a, b))

    def test_combine_concatenate_slices_transforms_and_export(self):
        def fun(a, b):
            joined = mx.concatenate([a, b], axis=-1)
            return mx.concatenate([joined, joined[..., 1:5]], axis=-1)

        a = mx.arange(6, dtype=mx.float32).reshape(2, 3)
        b = mx.arange(8, dtype=mx.float32).reshape(2, 4)

        def loss(a, b):
            return mx.sum(mx.square(fun(a, b)))

        for grad in (
            mx.grad(mx.compile(loss), argnums=(0, 1)),
            mx.compile(mx.grad(loss, argnums=(0, 1))),
        ):
            self.assert_outputs(grad(a, b), mx.grad(loss, argnums=(0, 1))(a, b))
        batch_a, batch_b = mx.stack([a, a + 1]), mx.stack([b, b + 1])
        for mapped in (mx.vmap(mx.compile(fun)), mx.compile(mx.vmap(fun))):
            self.assert_outputs(
                mapped(batch_a, batch_b), mx.vmap(fun)(batch_a, batch_b)
            )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "concat_slices.mlxfn")
            mx.export_function(path, fun, a, b)
            self.assert_outputs(mx.import_function(path)(a, b), fun(a, b))

    def test_combine_concatenate_slices_logs(self):
        stderr = self.run_compile_log_test(
            """
a, b = mx.arange(3), mx.arange(4)
def fun(a, b):
    joined = mx.concatenate([a, b])
    return mx.concatenate([joined, joined[1:5]])
mx.compile(fun)(a, b)
def shared(a, b):
    joined = mx.concatenate([a, b])
    return mx.concatenate([joined, joined[1:5]]), joined
mx.compile(shared)(a, b)
""",
            "passes,guards",
        )
        self.assertIn("event=rewrite rule=concatenate_slices", stderr)
        self.assertIn("reason=concat_not_removable", stderr)

    def test_integer_algebraic_identities(self):
        for dtype in (
            mx.int8,
            mx.uint8,
            mx.int16,
            mx.uint16,
            mx.int32,
            mx.uint32,
            mx.int64,
            mx.uint64,
        ):

            def fun(x):
                zero = mx.zeros(x.shape, dtype=dtype)
                one = mx.ones(x.shape, dtype=dtype)
                return (x + zero) * one - zero, zero + x, one * x

            x = mx.array([-1, 0, 1, 127]).astype(dtype)
            compiled = mx.compile(fun)
            for value in (x, x[::-1], x[:0]):
                self.assert_outputs(compiled(value), fun(value))
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "identities.mlxfn")
                mx.export_function(path, fun, x)
                self.assert_outputs(mx.import_function(path)(x), fun(x))

        def promoted(x):
            return x + mx.zeros((2, 3), dtype=mx.int32)

        x = mx.array([[1], [2]], dtype=mx.int16)
        self.assert_outputs(mx.compile(promoted)(x), promoted(x))

    def test_boolean_algebraic_identities(self):
        def fun(x):
            return (
                mx.logical_and(x, mx.ones((2, 3), dtype=mx.bool_)),
                mx.logical_or(mx.zeros((2, 3), dtype=mx.bool_), x),
            )

        x = mx.array([[True], [False]])
        self.assert_outputs(mx.compile(fun)(x), fun(x))
        batch = mx.stack([x, mx.logical_not(x)])
        for mapped in (mx.vmap(mx.compile(fun)), mx.compile(mx.vmap(fun))):
            self.assert_outputs(mapped(batch), mx.vmap(fun)(batch))

    def test_algebraic_identity_runtime_values_and_floats(self):
        traces = []

        def fun(x, s):
            traces.append(1)
            return x + s, x * s, s - x

        x = mx.arange(4)
        compiled = mx.compile(fun)
        for scalar in (0, 1, 2, -1):
            s = mx.array(scalar)
            self.assert_outputs(compiled(x, s), (x + s, x * s, s - x))
        self.assertEqual(len(traces), 1)

        def floating(x):
            return x + 0.0, x * 1.0, x * 0.0

        x = mx.array([-0.0, 0.0, float("inf"), float("nan")])
        for actual, expected in zip(mx.compile(floating)(x), floating(x)):
            self.assert_outputs(actual, expected)
            self.assert_outputs(
                actual[:2].view(mx.uint32), expected[:2].view(mx.uint32)
            )

    def test_algebraic_identity_logs(self):
        stderr = self.run_compile_log_test(
            """
x = mx.arange(4)
mx.compile(lambda x: (x + 0) * 1)(x)
mx.compile(lambda x: x + 0, shapeless=True)(x)
""",
            "passes,guards",
        )
        self.assertIn("event=rewrite rule=algebraic_identity", stderr)
        self.assertIn("reason=constant_value_not_proven", stderr)

    def test_uniform_cumsum(self):
        for dtype in (
            mx.bool_,
            mx.int8,
            mx.uint8,
            mx.int16,
            mx.uint16,
            mx.int32,
            mx.uint32,
            mx.int64,
            mx.uint64,
        ):
            traces = []

            def fun(c):
                traces.append(1)
                x = mx.full((2, 260), c)
                return tuple(
                    mx.cumsum(x, axis=axis, reverse=reverse, inclusive=inclusive)
                    for axis in (0, 1)
                    for reverse in (False, True)
                    for inclusive in (False, True)
                )

            compiled = mx.compile(fun)
            for value in (0, 1, -1, 127):
                scalar = mx.array(value).astype(dtype)
                actual = compiled(scalar)
                self.assertEqual(len(traces), 1)
                x = mx.full((2, 260), scalar)
                expected = tuple(
                    mx.cumsum(x, axis=axis, reverse=reverse, inclusive=inclusive)
                    for axis in (0, 1)
                    for reverse in (False, True)
                    for inclusive in (False, True)
                )
                self.assert_outputs(actual, expected)

    def test_uniform_cumsum_shapes_and_export(self):
        def fun(c, x):
            uniform = mx.broadcast_arrays(c, x)[0]
            return mx.cumsum(uniform, axis=-1), mx.cumsum(uniform, axis=0)

        for shapeless in (False, True):
            compiled = mx.compile(fun, shapeless=shapeless)
            for shape in ((2, 3), (3, 7), (0, 3), (2, 0)):
                x = mx.zeros(shape, mx.int32).T
                for c in (mx.array(3), mx.array(-7)):
                    self.assert_outputs(compiled(c, x), fun(c, x))

        def scan(c):
            return mx.cumsum(mx.full((4, 7), c * c + c), axis=1)

        c = mx.array(3)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "cumsum.mlxfn")
            mx.export_function(path, scan, c)
            imported = mx.import_function(path)
            for value in (3, -4):
                scalar = mx.array(value)
                self.assert_outputs(imported(scalar), scan(scalar))
        batch = mx.array([2, 3, -4])
        for mapped in (mx.vmap(mx.compile(scan)), mx.compile(mx.vmap(scan))):
            self.assert_outputs(mapped(batch), mx.vmap(scan)(batch))

    def test_uniform_cumsum_logs(self):
        stderr = self.run_compile_log_test(
            """
mx.compile(lambda c: mx.cumsum(mx.full((8,), c)))(mx.array(3))
mx.compile(lambda c: mx.cumsum(mx.full((8,), c)))(mx.array(0.1))
""",
            "passes,guards",
        )
        self.assertIn("event=rewrite rule=uniform_cumsum", stderr)
        self.assertIn("reason=scan_dtype_not_supported", stderr)

    def test_reuse_minmax_reductions(self):
        for dtype in (
            mx.bool_,
            mx.int8,
            mx.uint8,
            mx.int16,
            mx.uint16,
            mx.int32,
            mx.uint32,
            mx.int64,
            mx.uint64,
        ):
            for reduce in (mx.min, mx.max):
                for keepdims in (False, True):

                    def fun(x):
                        local = reduce(x, axis=1, keepdims=keepdims)
                        middle = reduce(x, axis=(0, 1), keepdims=keepdims)
                        total = reduce(x, keepdims=keepdims)
                        return total, local, middle, total

                    compiled = mx.compile(fun)
                    x = (mx.arange(48).reshape(2, 3, 8) - 24).astype(dtype)
                    for value in (x, x[:, ::-1, ::2], x.transpose(2, 1, 0)):
                        self.assert_outputs(compiled(value), fun(value))

    def test_reuse_minmax_replay_and_export(self):
        def fun(x):
            return mx.max(x, axis=1), mx.max(x), mx.min(x, axis=0), mx.min(x)

        for shapeless in (False, True):
            compiled = mx.compile(fun, shapeless=shapeless)
            shapes = ((2, 3), (3, 7)) if shapeless else ((2, 3), (3, 7), (1, 5), (4, 1))
            for shape in shapes:
                x = mx.arange(shape[0] * shape[1]).reshape(shape)
                self.assert_outputs(compiled(x), fun(x))

        x = mx.arange(24).reshape(4, 6)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "minmax.mlxfn")
            mx.export_function(path, fun, x)
            imported = mx.import_function(path)
            self.assert_outputs(imported(-x), fun(-x))
        batch = mx.stack([x, -x])
        for mapped in (mx.vmap(mx.compile(fun)), mx.compile(mx.vmap(fun))):
            self.assert_outputs(mapped(batch), mx.vmap(fun)(batch))

        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            value = mx.array([[0.0, -0.0], [float("nan"), float("inf")]], dtype)
            self.assert_outputs(mx.compile(fun)(value), fun(value))

    def test_reuse_minmax_logs(self):
        stderr = self.run_compile_log_test(
            """
fun = lambda x: (mx.max(x, axis=1), mx.max(x))
x = mx.arange(12).reshape(3, 4)
mx.compile(fun)(x)
mx.compile(fun, shapeless=True)(x)
mx.compile(fun)(x.astype(mx.float32))
""",
            "passes,guards",
        )
        self.assertIn("event=rewrite rule=reuse_minmax", stderr)
        self.assertIn("reason=nonempty_shape_not_proven", stderr)
        self.assertIn("reason=reduction_dtype_not_supported", stderr)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
