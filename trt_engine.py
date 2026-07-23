"""TensorRT 10.x 공통 엔진 실행 (이름 기반 API). Jetson 전용."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np


class TensorRTEngineError(RuntimeError):
    pass


class _CudaBackend:
    """pycuda 또는 cuda-python 중 사용 가능한 쪽."""

    def __init__(self):
        self.kind = None
        self.stream = None
        self._pycuda = None
        self._cudart = None

        try:
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401

            self._pycuda = cuda
            self.stream = cuda.Stream()
            self.kind = "pycuda"
            return
        except Exception:
            pass

        try:
            from cuda import cudart

            self._cudart = cudart
            err, self.stream = cudart.cudaStreamCreate()
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaStreamCreate: {err}")
            self.kind = "cuda-python"
            return
        except Exception as e:
            raise TensorRTEngineError(
                "CUDA 바인딩 없음. Jetson에서 `pycuda` 또는 `cuda-python`이 필요합니다."
            ) from e

    def alloc(self, nbytes: int):
        if self.kind == "pycuda":
            return self._pycuda.mem_alloc(nbytes)
        err, ptr = self._cudart.cudaMalloc(nbytes)
        if err != self._cudart.cudaError_t.cudaSuccess:
            raise TensorRTEngineError(f"CUDA 메모리 할당 실패: {err}")
        return ptr

    def free(self, ptr) -> None:
        if ptr is None:
            return
        if self.kind == "pycuda":
            ptr.free()
        else:
            self._cudart.cudaFree(ptr)

    def htod(self, device_ptr, host_arr: np.ndarray) -> None:
        host_arr = np.ascontiguousarray(host_arr)
        nbytes = host_arr.nbytes
        if self.kind == "pycuda":
            self._pycuda.memcpy_htod_async(device_ptr, host_arr, self.stream)
        else:
            err, = self._cudart.cudaMemcpyAsync(
                device_ptr,
                host_arr.ctypes.data,
                nbytes,
                self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            )
            if err != self._cudart.cudaError_t.cudaSuccess:
                raise TensorRTEngineError(f"H2D 실패: {err}")

    def dtoh(self, host_arr: np.ndarray, device_ptr) -> None:
        host_arr = np.ascontiguousarray(host_arr)
        nbytes = host_arr.nbytes
        if self.kind == "pycuda":
            self._pycuda.memcpy_dtoh_async(host_arr, device_ptr, self.stream)
        else:
            err, = self._cudart.cudaMemcpyAsync(
                host_arr.ctypes.data,
                device_ptr,
                nbytes,
                self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )
            if err != self._cudart.cudaError_t.cudaSuccess:
                raise TensorRTEngineError(f"D2H 실패: {err}")

    def sync(self) -> None:
        if self.kind == "pycuda":
            self.stream.synchronize()
        else:
            err, = self._cudart.cudaStreamSynchronize(self.stream)
            if err != self._cudart.cudaError_t.cudaSuccess:
                raise TensorRTEngineError(f"stream sync 실패: {err}")

    def stream_handle(self):
        if self.kind == "pycuda":
            return self.stream.handle
        return self.stream

    def device_addr(self, ptr) -> int:
        return int(ptr)


class TensorRTEngine:
    """
    TensorRT 10.3:
      set_tensor_address / execute_async_v3
    금지: get_binding_* / execute_async_v2
    """

    def __init__(self, engine_path: str | Path):
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"engine 파일 없음: {self.engine_path}")

        try:
            import tensorrt as trt
        except ImportError as e:
            raise TensorRTEngineError(
                "tensorrt import 실패. Jetson 시스템 TensorRT를 확인하세요."
            ) from e

        self.trt = trt
        self.cuda = _CudaBackend()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        blob = self.engine_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise TensorRTEngineError(
                f"엔진 역직렬화 실패 (TensorRT 버전 불일치 가능): {self.engine_path}"
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise TensorRTEngineError("execution context 생성 실패")

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.bindings: dict[str, dict[str, Any]] = {}
        self.last_infer_ms = 0.0

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            is_input = mode == trt.TensorIOMode.INPUT
            np_dtype = self._trt_to_np(dtype)
            entry = {
                "name": name,
                "is_input": is_input,
                "dtype": np_dtype,
                "shape": shape,
                "dynamic": any(d < 0 for d in shape),
                "host": None,
                "device": None,
                "nbytes": 0,
            }
            self.bindings[name] = entry
            (self.input_names if is_input else self.output_names).append(name)

        self._alloc_static()
        self.log_io()

    def _trt_to_np(self, trt_dtype) -> np.dtype:
        trt = self.trt
        mapping = {
            trt.float32: np.float32,
            trt.float16: np.float16,
            trt.int32: np.int32,
            trt.int8: np.int8,
            trt.bool: np.bool_,
        }
        if trt_dtype not in mapping:
            raise TensorRTEngineError(f"지원하지 않는 TRT dtype: {trt_dtype}")
        return np.dtype(mapping[trt_dtype])

    @staticmethod
    def _nbytes(shape, dtype) -> int:
        if any(d < 0 for d in shape):
            return 0
        return int(np.prod(shape)) * np.dtype(dtype).itemsize

    def _alloc_static(self) -> None:
        for b in self.bindings.values():
            if b["dynamic"]:
                continue
            self._ensure_alloc(b["name"], b["shape"])

    def _ensure_alloc(self, name: str, shape: tuple[int, ...]) -> None:
        b = self.bindings[name]
        nbytes = self._nbytes(shape, b["dtype"])
        need = int(np.prod(shape))
        if b["device"] is not None and b["nbytes"] >= nbytes and b["host"] is not None and b["host"].size == need:
            b["shape"] = shape
            return
        if b["device"] is not None:
            self.cuda.free(b["device"])
        b["shape"] = shape
        b["nbytes"] = nbytes
        b["host"] = np.empty(need, dtype=b["dtype"])
        try:
            b["device"] = self.cuda.alloc(nbytes)
        except Exception as e:
            raise TensorRTEngineError(f"CUDA 메모리 할당 실패 ({name}, {nbytes} bytes)") from e

    def log_io(self) -> None:
        print(f"[TRT] engine: {self.engine_path} (cuda={self.cuda.kind})")
        for name in self.input_names + self.output_names:
            b = self.bindings[name]
            kind = "Input" if b["is_input"] else "Output"
            print(f"  {kind}: {name}, shape={b['shape']}, dtype={b['dtype']}, dynamic={b['dynamic']}")

    def infer(self, inputs: dict[str, np.ndarray] | np.ndarray) -> dict[str, np.ndarray]:
        if isinstance(inputs, np.ndarray):
            if len(self.input_names) != 1:
                raise TensorRTEngineError("입력이 여러 개입니다. dict로 이름을 지정하세요.")
            inputs = {self.input_names[0]: inputs}

        t0 = time.perf_counter()

        for name, arr in inputs.items():
            if name not in self.bindings or not self.bindings[name]["is_input"]:
                raise TensorRTEngineError(f"알 수 없는 입력 텐서: {name}")
            arr = np.ascontiguousarray(arr)
            expect = self.bindings[name]["dtype"]
            if arr.dtype != expect:
                arr = arr.astype(expect, copy=False)

            shape = tuple(arr.shape)
            if self.bindings[name]["dynamic"] or any(d < 0 for d in self.engine.get_tensor_shape(name)):
                ok = self.context.set_input_shape(name, shape)
                if ok is False:
                    raise TensorRTEngineError(f"동적 shape 설정 실패: {name} {shape}")
            else:
                fixed = tuple(self.bindings[name]["shape"])
                if shape != fixed:
                    raise TensorRTEngineError(f"입력 shape 불일치: {name} got {shape} expect {fixed}")

            self._ensure_alloc(name, shape)
            np.copyto(self.bindings[name]["host"], arr.ravel())
            self.cuda.htod(self.bindings[name]["device"], self.bindings[name]["host"])

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise TensorRTEngineError(f"출력 shape 미확정: {name} {shape}")
            self._ensure_alloc(name, shape)

        for name, b in self.bindings.items():
            self.context.set_tensor_address(name, self.cuda.device_addr(b["device"]))

        ok = self.context.execute_async_v3(self.cuda.stream_handle())
        if not ok:
            raise TensorRTEngineError("execute_async_v3 실패")

        for name in self.output_names:
            b = self.bindings[name]
            self.cuda.dtoh(b["host"], b["device"])
        self.cuda.sync()

        outputs = {
            name: np.array(self.bindings[name]["host"]).reshape(self.bindings[name]["shape"]).copy()
            for name in self.output_names
        }
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return outputs

    def close(self) -> None:
        for b in self.bindings.values():
            self.cuda.free(b.get("device"))
            b["device"] = None
        self.bindings.clear()
