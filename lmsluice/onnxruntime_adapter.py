"""Optional ONNX Runtime CPU consumer boundary.

The import is lazy so importing lmsluice, using its plain bundle provider and
running the standard-library probe do not require ONNX Runtime or ONNX's
Python package.  This adapter accepts an already verified/materialized bundle;
it does not download models, execute model supplied code or choose a runtime
policy for the caller.
"""

from __future__ import annotations

import importlib
import os
import re
import time

from .bundle import BundleCancelled, BundleDescriptor, BundleError, BundleResult


class OnnxRuntimeUnavailable(BundleError):
    def __init__(self, message="ONNX Runtime CPU provider is unavailable", **details):
        super().__init__(message, code="provider_unavailable", provider="onnxruntime",
                         **details)


def capability() -> dict:
    """Report the configured ONNX Runtime CPU capability without installing it."""
    try:
        ort = importlib.import_module("onnxruntime")
    except Exception as exc:
        return {
            "provider": "onnxruntime",
            "available": False,
            "cpu": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    try:
        providers = list(ort.get_available_providers())
    except Exception as exc:
        providers = []
        reason = f"provider query failed: {type(exc).__name__}: {exc}"
    else:
        reason = None if "CPUExecutionProvider" in providers else \
            "CPUExecutionProvider is not advertised"
    return {
        "provider": "onnxruntime",
        "available": "CPUExecutionProvider" in providers,
        "cpu": "CPUExecutionProvider" in providers,
        "version": getattr(ort, "__version__", None),
        "providers": providers,
        "reason": reason,
    }


class OnnxCPUConsumer:
    """One explicitly owned ONNX Runtime CPU session.

    ``validity_hook`` is required by :meth:`run`.  A successful session
    initialization or a returned Python object is not a consumer result until
    the caller accepts the actual output.  ``release`` is idempotent and is the
    only place this object closes its session.
    """

    def __init__(self, bundle, *, graph_path=None, observer=None,
                 validity_hook=None, cancellation=None):
        self.bundle = bundle
        self.graph_path = graph_path
        self.observer = observer
        self.validity_hook = validity_hook
        self.cancellation = cancellation
        self.session = None
        self.ort = None
        self.closed = False
        self.initialized_at = None
        self.first_inference_at = None
        self.first_valid_output_at = None
        self.release_at = None
        self._constraints = self._bundle_constraints()

    def _check_cancel(self, boundary):
        token = self.cancellation
        if token is None:
            return
        try:
            if callable(token):
                cancelled = bool(token())
            elif hasattr(token, "is_set"):
                cancelled = bool(token.is_set())
            elif hasattr(token, "cancelled"):
                cancelled = bool(token.cancelled)
            else:
                cancelled = bool(token)
        except Exception as exc:
            raise BundleCancelled("consumer cancellation probe failed",
                                  boundary=boundary,
                                  exception=type(exc).__name__) from exc
        if cancelled:
            raise BundleCancelled(boundary=boundary)

    def _bundle_directory(self) -> str:
        if isinstance(self.bundle, BundleResult):
            return self.bundle.destination
        if isinstance(self.bundle, BundleDescriptor):
            if self.bundle.source_root is None:
                raise BundleError("bundle descriptor has no materialized destination",
                                  code="consumer_input")
            return self.bundle.source_root
        path = os.fspath(self.bundle)
        if not os.path.isdir(path):
            raise BundleError("consumer bundle destination is not a directory",
                              code="consumer_input", path=path)
        return os.path.abspath(path)

    def _bundle_constraints(self) -> dict:
        if isinstance(self.bundle, BundleDescriptor):
            return dict(self.bundle.consumer or {})
        if isinstance(self.bundle, BundleResult):
            return dict(self.bundle.consumer or {})
        return {}

    def _graph(self) -> str:
        root = self._bundle_directory()
        if self.graph_path is not None:
            candidate = os.path.abspath(os.fspath(self.graph_path))
            if not candidate.startswith(root + os.sep):
                raise BundleError("graph path escapes materialized bundle",
                                  code="unsafe_path", path=candidate)
        else:
            entry_point = None
            if isinstance(self.bundle, (BundleDescriptor, BundleResult)):
                entry_point = self.bundle.entry_point
            if entry_point:
                candidate = os.path.join(root, entry_point)
            else:
                graphs = []
                for directory, _names, files in os.walk(root):
                    for name in files:
                        if name.endswith(".onnx"):
                            graphs.append(os.path.join(directory, name))
                if not graphs:
                    raise BundleError("materialized bundle has no ONNX graph",
                                      code="missing_entry")
                candidate = sorted(graphs)[0]
        candidate = os.path.abspath(candidate)
        if not os.path.isfile(candidate) or os.path.islink(candidate):
            raise BundleError("materialized ONNX graph is unavailable",
                              code="missing_entry", path=candidate)
        return candidate

    def _check_constraints(self, ort):
        constraints = self._constraints
        engine = constraints.get("engine") or constraints.get("runtime")
        if engine and str(engine).lower() not in ("onnxruntime", "onnx-runtime"):
            raise BundleError("bundle requires an incompatible consumer engine",
                              code="provider_incompatible", expected=engine)
        backend = constraints.get("backend")
        if backend and backend != "CPUExecutionProvider":
            raise BundleError("bundle requires an incompatible consumer backend",
                              code="provider_incompatible", expected=backend,
                              actual="CPUExecutionProvider")
        extensions = constraints.get("extensions") or constraints.get("custom_ops")
        if extensions:
            raise BundleError("custom ONNX extensions are disabled",
                              code="unsupported_extension", extensions=extensions)
        version_range = constraints.get("version_range") or constraints.get("version")
        if version_range and not _version_allowed(getattr(ort, "__version__", ""),
                                                  str(version_range)):
            raise BundleError("ONNX Runtime version is outside bundle constraints",
                              code="provider_incompatible", expected=version_range,
                              actual=getattr(ort, "__version__", None))

    def initialize(self):
        if self.session is not None:
            return self
        if self.closed:
            raise BundleError("consumer has been released", code="consumer_released")
        started = time.perf_counter()
        try:
            self._check_cancel("before_consumer_init")
            self.ort = importlib.import_module("onnxruntime")
            providers = list(self.ort.get_available_providers())
            if "CPUExecutionProvider" not in providers:
                raise OnnxRuntimeUnavailable(providers=providers)
            self._check_constraints(self.ort)
            graph = self._graph()
            # Deliberately do not register custom operator libraries or execute
            # model-supplied code.  The provider list pins this boundary to CPU.
            self.session = self.ort.InferenceSession(
                graph, providers=["CPUExecutionProvider"])
            self.initialized_at = time.perf_counter()
            _mark(self.observer, "consumer_initialized", provider="onnxruntime",
                  backend="CPUExecutionProvider", graph=graph,
                  duration_s=self.initialized_at - started)
            _set_execution(self.observer, backend_memory={
                "status": "UNMEASURED",
                "reason": "ONNX Runtime session allocator is not exposed by this adapter",
                "measurement_method": "no backend allocator hook",
            }, measurement_method={
                "session_memory": "UNMEASURED",
                "rss": "ReadinessRecord current-process sampler when enabled",
            })
            self._check_cancel("after_consumer_init")
            return self
        except BundleError as exc:
            if isinstance(exc, BundleCancelled):
                _mark(self.observer, "cancelled",
                      boundary=exc.details.get("boundary"))
                _set_execution(self.observer,
                               cancellation_boundary=exc.details.get("boundary"))
            _failure(self.observer, exc, phase="consumer_init")
            if isinstance(exc, BundleCancelled):
                self.release()
            raise
        except Exception as exc:
            error = BundleError("ONNX Runtime session initialization failed",
                                code="consumer_init_failed",
                                exception=type(exc).__name__, message_detail=str(exc))
            _failure(self.observer, error, phase="consumer_init")
            raise error from exc

    def run(self, inputs: dict, *, validity_hook=None):
        if self.session is None:
            self.initialize()
        hook = validity_hook if validity_hook is not None else self.validity_hook
        if not callable(hook):
            error = BundleError("caller validity hook is required",
                                code="validation_required")
            _failure(self.observer, error, phase="consumer_output")
            raise error
        started = time.perf_counter()
        try:
            self._check_cancel("before_inference")
            if self.first_inference_at is None:
                self.first_inference_at = time.perf_counter()
                _mark(self.observer, "use_started", provider="onnxruntime",
                      backend="CPUExecutionProvider")
            outputs = self.session.run(None, inputs)
            accepted = bool(hook(outputs))
        except BundleCancelled as exc:
            _mark(self.observer, "cancelled",
                  boundary=exc.details.get("boundary"))
            _set_execution(self.observer,
                           cancellation_boundary=exc.details.get("boundary"))
            _failure(self.observer, exc, phase="consumer_inference")
            self.release()
            raise
        except BundleError:
            raise
        except Exception as exc:
            error = BundleError("ONNX Runtime inference failed", code="consumer_run_failed",
                                exception=type(exc).__name__, message_detail=str(exc))
            _failure(self.observer, error, phase="consumer_inference")
            raise error from exc
        if not accepted:
            error = BundleError("caller rejected ONNX Runtime output",
                                code="output_rejected")
            _failure(self.observer, error, phase="consumer_output")
            raise error
        if self.first_valid_output_at is None:
            self.first_valid_output_at = time.perf_counter()
            _mark(self.observer, "consumer_first_valid_output",
                  provider="onnxruntime", backend="CPUExecutionProvider",
                  duration_s=self.first_valid_output_at - started)
            _mark(self.observer, "consumer_first_useful",
                  provider="onnxruntime", validated=True)
            _mark(self.observer, "consumer_ready",
                  provider="onnxruntime", validated=True)
        return outputs

    def release(self):
        if self.release_at is not None:
            return
        self.session = None
        self.release_at = time.perf_counter()
        self.closed = True
        _set_execution(self.observer,
                       cleanup={"consumer_session": "released"})
        _mark(self.observer, "release", owner="consumer", cleanup="complete",
              provider="onnxruntime")

    close = release

    def __enter__(self):
        return self.initialize()

    def __exit__(self, exc_type, exc, _tb):
        self.release()
        return False

    def to_dict(self) -> dict:
        return {
            "provider": "onnxruntime",
            "backend": "CPUExecutionProvider",
            "initialized": self.initialized_at is not None,
            "first_inference": self.first_inference_at is not None,
            "first_valid_output": self.first_valid_output_at is not None,
            "released": self.release_at is not None,
            "io_binding": {
                "status": "UNMEASURED",
                "meaning": "I/O binding concerns graph inputs/outputs and does not prove zero-copy compressed initializer loading",
            },
        }


def _version_allowed(actual: str, expression: str) -> bool:
    if not actual:
        return False
    def parse(value):
        match = re.match(r"\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
        return tuple(int(part or 0) for part in match.groups()) if match else None
    got = parse(actual)
    if got is None:
        return False
    terms = [term.strip() for term in expression.split(",") if term.strip()]
    for term in terms:
        match = re.match(r"(<=|>=|==|<|>|=)?\s*([0-9][0-9.]*)", term)
        if not match:
            return False
        op, value = match.group(1) or "==", match.group(2)
        want = parse(value)
        if want is None:
            return False
        if op in ("=", "==") and got != want:
            return False
        if op == "<" and not got < want:
            return False
        if op == "<=" and not got <= want:
            return False
        if op == ">" and not got > want:
            return False
        if op == ">=" and not got >= want:
            return False
    return True


def _mark(observer, event, **details):
    if observer is None:
        return
    try:
        callback = getattr(observer, "mark", None)
        if callback:
            callback(event, **details)
    except Exception:
        return


def _failure(observer, exc, *, phase):
    if observer is None:
        return
    try:
        callback = getattr(observer, "failure_event", None)
        if callback:
            callback(exc, phase=phase)
    except Exception:
        return


def _set_execution(observer, **values):
    if observer is None:
        return
    try:
        callback = getattr(observer, "set_execution", None)
        if callback:
            callback(**values)
    except Exception:
        return


__all__ = ["OnnxRuntimeUnavailable", "OnnxCPUConsumer", "capability"]
