# Copyright © 2026 Apple Inc.

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


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
