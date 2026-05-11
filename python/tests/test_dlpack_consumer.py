# Copyright © 2026 Apple Inc.
"""Tests for the DLPack consumer path in ``mx.array``.

These tests cover scenarios that don't depend on a third-party Metal producer:

1. Round trip through NumPy's DLPack exporter (``kDLCPU`` path).
2. Self round trip via ``mx.array.__dlpack__`` (``kDLMetal`` on Metal hosts,
   ``kDLCPU`` otherwise).
3. Negative paths: used capsule, unsupported strided view, etc.
"""

from __future__ import annotations

import unittest
import ctypes

import numpy as np

try:
    import mlx.core as mx
except ImportError as exc:  # pragma: no cover - import error is environment specific
    raise unittest.SkipTest(f"mlx.core unavailable: {exc}")


class TestArrayDLPackBasic(unittest.TestCase):
    @staticmethod
    def _capsule_device(capsule):
        class DLDevice(ctypes.Structure):
            _fields_ = [
                ("device_type", ctypes.c_int32),
                ("device_id", ctypes.c_int32),
            ]

        class DLDataType(ctypes.Structure):
            _fields_ = [
                ("code", ctypes.c_uint8),
                ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16),
            ]

        class DLTensor(ctypes.Structure):
            _fields_ = [
                ("data", ctypes.c_void_p),
                ("device", DLDevice),
                ("ndim", ctypes.c_int32),
                ("dtype", DLDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64),
            ]

        class DLManagedTensor(ctypes.Structure):
            _fields_ = [
                ("dl_tensor", DLTensor),
                ("manager_ctx", ctypes.c_void_p),
                ("deleter", ctypes.c_void_p),
            ]

        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        get_pointer.restype = ctypes.c_void_p
        ptr = get_pointer(capsule, b"dltensor")
        tensor = ctypes.cast(ptr, ctypes.POINTER(DLManagedTensor)).contents
        device = tensor.dl_tensor.device
        return device.device_type, device.device_id

    def test_mx_array_dlpack_device_matches_capsule(self):
        x = mx.arange(8, dtype=mx.float32)
        capsule = x.__dlpack__()
        self.assertEqual(self._capsule_device(capsule), x.__dlpack_device__())

    def test_mx_array_dlpack_explicit_cpu_device(self):
        x = mx.arange(8, dtype=mx.float32)
        capsule = x.__dlpack__(dl_device=(1, 0))
        self.assertEqual(self._capsule_device(capsule), (1, 0))

    def test_mx_array_accepts_dlpack_capsule(self):
        # Pass a raw PyCapsule rather than the producer object.
        arr_np = np.arange(8, dtype=np.int32).reshape(2, 4)
        capsule = arr_np.__dlpack__()
        arr_mx = mx.array(capsule)
        self.assertEqual(tuple(arr_mx.shape), (2, 4))
        self.assertEqual(arr_mx.dtype, mx.int32)
        self.assertTrue(np.array_equal(np.asarray(arr_mx), arr_np))

    def test_mx_array_accepts_dlpack_producer(self):
        class DLPackProducer:
            def __init__(self, array):
                self.array = array

            def __dlpack__(self):
                return self.array.__dlpack__()

            def __dlpack_device__(self):
                return self.array.__dlpack_device__()

        arr_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        arr_mx = mx.array(DLPackProducer(arr_np))
        self.assertEqual(tuple(arr_mx.shape), (3, 4))
        self.assertEqual(arr_mx.dtype, mx.float32)
        self.assertTrue(np.allclose(np.asarray(arr_mx), arr_np))

    def test_mx_array_accepts_mlx_dlpack_producer(self):
        class DLPackProducer:
            def __init__(self, array):
                self.array = array

            def __dlpack__(self):
                return self.array.__dlpack__()

            def __dlpack_device__(self):
                return self.array.__dlpack_device__()

        x = mx.arange(20, dtype=mx.float32).reshape(4, 5)
        y = mx.array(DLPackProducer(x))
        self.assertTrue(mx.array_equal(x, y).item())

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_mx_array_exports_lazy_metal_array(self):
        x = mx.arange(20, dtype=mx.float32).reshape(4, 5)
        y = x + 1
        z = mx.array(y.__dlpack__())
        self.assertTrue(mx.array_equal(y, z).item())

    def test_mx_array_dlpack_export_inside_custom_vjp_transform(self):
        @mx.custom_function
        def roundtrip_double(x):
            y = x * 2
            return mx.array(y.__dlpack__())

        @roundtrip_double.vjp
        def roundtrip_double_vjp(primals, cotangent, _outputs):
            return cotangent * 2

        def loss(x):
            return mx.sum(roundtrip_double(x))

        x = mx.arange(8, dtype=mx.float32)
        value, grad = mx.value_and_grad(loss)(x)
        mx.eval(value, grad)

        self.assertEqual(value.item(), 56.0)
        self.assertTrue(mx.array_equal(grad, mx.ones_like(x) * 2).item())

    def test_mx_array_dlpack_dtype_override_rejected(self):
        arr_np = np.arange(6, dtype=np.int32).reshape(2, 3)
        with self.assertRaises(Exception):
            mx.array(arr_np.__dlpack__(), dtype=mx.float32)

    def test_mx_array_prefers_mlx_array_protocol_over_dlpack(self):
        class BothProtocols:
            def __mlx_array__(self):
                return mx.array([1, 2, 3], dtype=mx.int32)

            def __dlpack__(self):
                raise AssertionError("__dlpack__ should not be called")

        arr_mx = mx.array(BothProtocols())
        self.assertEqual(arr_mx.dtype, mx.int32)
        self.assertTrue(np.array_equal(np.asarray(arr_mx), np.array([1, 2, 3])))

    def test_dtypes(self):
        cases = [
            (np.bool_, mx.bool_),
            (np.int8, mx.int8),
            (np.int16, mx.int16),
            (np.int32, mx.int32),
            (np.int64, mx.int64),
            (np.uint8, mx.uint8),
            (np.uint16, mx.uint16),
            (np.uint32, mx.uint32),
            (np.uint64, mx.uint64),
            (np.float16, mx.float16),
            (np.float32, mx.float32),
            (np.float64, mx.float64),
            (np.complex64, mx.complex64),
        ]
        for np_dtype, mx_dtype in cases:
            with self.subTest(np_dtype=np_dtype):
                arr = np.zeros((2, 3), dtype=np_dtype)
                if np_dtype is np.bool_:
                    arr[0, 0] = True
                else:
                    arr[0, 0] = 1
                converted = mx.array(arr.__dlpack__())
                self.assertEqual(converted.dtype, mx_dtype)
                self.assertEqual(tuple(converted.shape), (2, 3))


class TestArrayDLPackErrors(unittest.TestCase):
    def test_rejects_used_capsule(self):
        arr_np = np.arange(4, dtype=np.float32)
        capsule = arr_np.__dlpack__()
        # First call consumes; second must fail because the capsule was
        # renamed to "used_dltensor".
        _ = mx.array(capsule)
        with self.assertRaises(Exception):
            mx.array(capsule)


class TestArrayDLPackNonContiguous(unittest.TestCase):
    def test_strided_view_rejected(self):
        # MLX's first-cut consumer does not support arbitrary DLPack strides.
        # NumPy emits __dlpack__ with explicit strides for slices; producers
        # may or may not encode strides depending on contiguity. We assert
        # that a non-row-contiguous slice is rejected with a clear error
        # rather than silently misinterpreting the layout.
        big = np.arange(16, dtype=np.float32).reshape(4, 4)
        view = big[::2, :]
        try:
            capsule = view.__dlpack__()
        except (TypeError, BufferError):
            self.skipTest(
                "NumPy refused to export a non-contiguous DLPack capsule"
            )
        with self.assertRaises(Exception):
            mx.array(capsule)

    def test_rejected_capsule_is_not_marked_used(self):
        big = np.arange(16, dtype=np.float32).reshape(4, 4)
        view = big[::2, :]
        try:
            capsule = view.__dlpack__()
        except (TypeError, BufferError):
            self.skipTest(
                "NumPy refused to export a non-contiguous DLPack capsule"
            )

        with self.assertRaises(Exception):
            mx.array(capsule)

        get_name = ctypes.pythonapi.PyCapsule_GetName
        get_name.argtypes = [ctypes.py_object]
        get_name.restype = ctypes.c_char_p
        self.assertEqual(get_name(capsule), b"dltensor")


if __name__ == "__main__":
    unittest.main()
