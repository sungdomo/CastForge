"""Zero-dependency Python ctypes bindings for libcastforge (`bindings/python/castforge.py`)."""

from __future__ import annotations
import ctypes
import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _default_lib_name() -> str:
    sys_name = platform.system().lower()
    if sys_name == "darwin":
        return "libcastforge.dylib"
    if sys_name == "windows":
        return "castforge.dll"
    return "libcastforge.so"


class CastForgeError(RuntimeError):
    """Raised when a CastForge compilation or optimization call fails."""


class CastForge:
    """High-level Python SDK wrapper around the CastForge C ABI (`libcastforge`)."""

    def __init__(self, lib_path: str | os.PathLike[str] | None = None) -> None:
        if lib_path is None:
            repo_root = Path(__file__).resolve().parents[2]
            rel_candidate = repo_root / "target" / "release" / _default_lib_name()
            dbg_candidate = repo_root / "target" / "debug" / _default_lib_name()
            lib_path = rel_candidate if rel_candidate.exists() else dbg_candidate
        self._lib = ctypes.CDLL(str(lib_path))

        self._lib.castforge_version.argtypes = []
        self._lib.castforge_version.restype = ctypes.c_char_p

        self._lib.castforge_free_string.argtypes = [ctypes.c_void_p]
        self._lib.castforge_free_string.restype = None

        self._lib.castforge_optimize_mlir.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.castforge_optimize_mlir.restype = ctypes.c_int32

        self._lib.castforge_compile_mlir.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.castforge_compile_mlir.restype = ctypes.c_int32

    def version(self) -> str:
        raw = self._lib.castforge_version()
        return raw.decode("utf-8") if raw else "unknown"

    def _take_string(self, ptr: ctypes.c_void_p) -> str:
        if not ptr.value:
            return ""
        try:
            raw = ctypes.cast(ptr, ctypes.c_char_p).value
            return raw.decode("utf-8") if raw else ""
        finally:
            self._lib.castforge_free_string(ptr)

    def optimize(
        self, mlir_source: str, pipeline: str = "O2"
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Optimize `mlir_source` with `pipeline` and return `(optimized_mlir, telemetry)`."""
        out_mlir = ctypes.c_void_p()
        out_tel = ctypes.c_void_p()
        out_err = ctypes.c_void_p()
        rc = self._lib.castforge_optimize_mlir(
            mlir_source.encode("utf-8"),
            pipeline.encode("utf-8"),
            ctypes.byref(out_mlir),
            ctypes.byref(out_tel),
            ctypes.byref(out_err),
        )
        if rc != 0:
            err_msg = self._take_string(out_err)
            raise CastForgeError(err_msg or f"castforge_optimize_mlir failed (rc={rc})")
        opt_text = self._take_string(out_mlir)
        tel_text = self._take_string(out_tel)
        return opt_text, json.loads(tel_text) if tel_text else []

    def compile(
        self, mlir_source: str, target: str = "rust", pipeline: str = "O2"
    ) -> str:
        """Optimize `mlir_source` and generate code for `target` (rust/c/wgsl/cuda/metal/ptx)."""
        out_code = ctypes.c_void_p()
        out_err = ctypes.c_void_p()
        rc = self._lib.castforge_compile_mlir(
            mlir_source.encode("utf-8"),
            pipeline.encode("utf-8"),
            target.encode("utf-8"),
            ctypes.byref(out_code),
            ctypes.byref(out_err),
        )
        if rc != 0:
            err_msg = self._take_string(out_err)
            raise CastForgeError(err_msg or f"castforge_compile_mlir failed (rc={rc})")
        return self._take_string(out_code)
