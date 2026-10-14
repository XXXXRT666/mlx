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


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
