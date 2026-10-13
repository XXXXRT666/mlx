# Copyright © 2026 Apple Inc.

import os
import tempfile

import mlx.core as mx
import mlx_tests


class TestCompileReorder(mlx_tests.MLXTestCase):
    def assert_outputs(self, actual, expected):
        actual = (actual,) if isinstance(actual, mx.array) else actual
        expected = (expected,) if isinstance(expected, mx.array) else expected
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(mx.allclose(a, b, atol=1e-5, rtol=1e-5, equal_nan=True))

    def test_views(self):
        views = (
            lambda x: x.reshape(3, 2),
            lambda x: x.T,
            lambda x: x.flatten(),
            lambda x: mx.expand_dims(x, 0),
            lambda x: mx.squeeze(x, 1),
            lambda x: mx.unflatten(x, 2, (1, 3)),
            lambda x: mx.broadcast_to(x, (4, 2, 1, 3)),
            lambda x: mx.repeat(x, 3, axis=1),
            lambda x: mx.tile(x, (2, 1, 2)),
        )
        for dtype in (mx.float32, mx.float16, mx.bfloat16, mx.int32):
            base = mx.arange(6).reshape(2, 1, 3).astype(dtype) - 3
            for x in (base, base[:, :, ::-1], base.transpose(2, 1, 0).reshape(2, 1, 3)):
                y = -x
                mask = x > 0
                for view in views:
                    for fun in (
                        lambda x, y, m: -view(mx.abs(x)),
                        lambda x, y, m: view(x) + view(y),
                        lambda x, y, m: mx.where(view(m), view(x), view(y)),
                        lambda x, y, m: view(x) + 2,
                    ):
                        self.assert_outputs(
                            mx.compile(fun)(x, y, mask), fun(x, y, mask)
                        )

    def test_selections(self):
        selections = (
            lambda x, i: x[::2, 1::2],
            lambda x, i: x[::-1, ::-2],
            lambda x, i: mx.take(x, i, axis=0),
            lambda x, i: mx.take_along_axis(
                x, mx.broadcast_to(i[:, None], (2, 4)), axis=0
            ),
        )
        for dtype in (mx.float32, mx.float16, mx.bfloat16, mx.int32):
            x = mx.arange(12).reshape(3, 4).astype(dtype) - 5
            for selection in selections:
                for fun in (
                    lambda x, y, i: selection(-mx.abs(x), i),
                    lambda x, y, i: selection(x + y, i),
                    lambda x, y, i: selection(mx.where(x > 0, x, y), i),
                ):
                    compiled = mx.compile(fun)
                    for indices in (mx.array([2, 0]), mx.array([-1, -1])):
                        self.assert_outputs(
                            compiled(x, -x, indices), fun(x, -x, indices)
                        )

    def test_concat_and_shared_outputs(self):
        x = mx.arange(6, dtype=mx.float32).reshape(2, 3)
        y = mx.arange(9, dtype=mx.float32).reshape(3, 3)
        for op in (lambda a: -a, lambda a: a + 2, lambda a: mx.where(True, a, 2)):

            def fun(x, y):
                out = mx.concatenate([op(x), op(y)])
                return out, out, x

            self.assert_outputs(mx.compile(fun)(x, y), fun(x, y))

        def shared(x):
            view = x.T
            pointwise = mx.abs(x)
            return (
                -view,
                view,
                pointwise[::2],
                pointwise,
                mx.concatenate([pointwise, pointwise]),
            )

        self.assert_outputs(mx.compile(shared)(x), shared(x))

    def test_shape_is_not_an_index_mapping(self):
        def fun(x, y):
            return x.T + y.reshape(3, 3), mx.where(x > 0, x.T, y)

        x = mx.arange(9, dtype=mx.float32).reshape(3, 3) - 4
        y = x[::-1]
        self.assert_outputs(mx.compile(fun)(x, y), fun(x, y))

    def test_shapeless_replay_and_shape_checks(self):
        for view in (
            lambda x: x.reshape(3, -1),
            lambda x: x.T,
            lambda x: x.flatten(),
            lambda x: mx.expand_dims(x, 0),
            lambda x: mx.squeeze(x, 1),
            lambda x: mx.unflatten(x, 2, (1, 3)),
        ):
            traces = []

            def fun(x):
                traces.append(1)
                return -view(mx.abs(x))

            compiled = mx.compile(fun, shapeless=True)
            for n in (2, 5, 0):
                x = mx.arange(n * 3, dtype=mx.float32).reshape(n, 1, 3)
                actual = compiled(x)
                self.assertEqual(len(traces), 1)
                self.assert_outputs(actual, -view(mx.abs(x)))

        for fun, valid, invalid in (
            (lambda x: -mx.abs(x).reshape(2, -1), mx.arange(6), mx.arange(7)),
            (
                lambda x: -mx.broadcast_to(mx.abs(x), (2, 3)),
                mx.ones((1, 3)),
                mx.ones((2, 2)),
            ),
        ):
            compiled = mx.compile(fun, shapeless=True)
            self.assert_outputs(compiled(valid), fun(valid))
            with self.assertRaises(ValueError):
                compiled(invalid)

        def binary(x, y):
            return x.reshape(2, -1) + y.reshape(2, -1)

        compiled = mx.compile(binary, shapeless=True)
        for n, m in ((6, 6), (2, 8), (8, 2)):
            x, y = mx.arange(n), mx.arange(m)
            self.assert_outputs(compiled(x, y), binary(x, y))
        with self.assertRaises(ValueError):
            compiled(mx.arange(6), mx.arange(7))

        def concat(x, y):
            return mx.concatenate([-mx.abs(x), -mx.abs(y)])

        compiled = mx.compile(concat, shapeless=True)
        for n, m in ((2, 3), (0, 5), (4, 1)):
            x, y = mx.arange(n), mx.arange(m)
            self.assert_outputs(compiled(x, y), concat(x, y))

    def test_rounding_special_values_and_casts(self):
        x = mx.array([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1.0007])
        for dtype in (mx.float16, mx.bfloat16):

            def fun(x):
                y = x.astype(dtype).reshape(2, 3)
                return y.astype(mx.float32), mx.concatenate(
                    [x.astype(dtype), x.astype(dtype)]
                )

            self.assert_outputs(mx.compile(fun)(x), fun(x))
        x = mx.array([1 + 2j, -3 + 4j, 0 + 1j, -1 - 2j])
        self.assert_outputs(
            mx.compile(lambda x: mx.real(x.reshape(2, 2)))(x), mx.real(x.reshape(2, 2))
        )

    def test_grad_vmap_and_export(self):
        functions = (
            lambda x: mx.square(mx.sin(x).reshape(3, -1)),
            lambda x: mx.square(mx.sin(x))[::2],
            lambda x: mx.concatenate([mx.sin(x), mx.sin(x + 1)]),
        )
        x = mx.arange(12, dtype=mx.float32).reshape(4, 3) * 0.1
        with tempfile.TemporaryDirectory() as directory:
            for i, fun in enumerate(functions):
                compiled = mx.compile(fun)
                self.assert_outputs(
                    mx.grad(lambda x: mx.sum(compiled(x)))(x),
                    mx.grad(lambda x: mx.sum(fun(x)))(x),
                )
                batch = mx.stack([x, x + 1])
                self.assert_outputs(mx.vmap(compiled)(batch), mx.vmap(fun)(batch))
                path = os.path.join(directory, f"reorder_{i}.mlxfn")
                mx.export_function(path, fun, x)
                self.assert_outputs(mx.import_function(path)(x), fun(x))

            def fun(x):
                return -mx.abs(x).reshape(3, -1)

            path = os.path.join(directory, "shapeless.mlxfn")
            mx.export_function(path, fun, mx.arange(6), shapeless=True)
            imported = mx.import_function(path)
            self.assert_outputs(imported(mx.arange(9)), fun(mx.arange(9)))
            with self.assertRaises(ValueError):
                imported(mx.arange(7))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
