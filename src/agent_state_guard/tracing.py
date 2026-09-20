"""tracing.py - OpenTelemetry instrumentation for agent-state-guard.

Wraps DeterministicGraph.run() so every node execution, retry attempt,
fallback invocation, and schema-validation failure becomes a named span in
your OTLP-compatible backend (Jaeger, Grafana Tempo, Langfuse, ...).

Quick start
-----------
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

from agent_state_guard.tracing import init_telemetry, traced_graph

init_telemetry(
    service_name="my-agent",
    otlp_endpoint="http://localhost:4318",  # Jaeger / Tempo / Langfuse
)

graph = DeterministicGraph()
graph.add_node("summarise", summarise_fn)

result = traced_graph(graph).run(AgentState(task="..."))
# -> spans appear in your backend under the service name "my-agent"

Environment variables (alternative to keyword args)
------------------------------------------------------
OTEL_SERVICE_NAME               default: agent-state-guard
OTEL_EXPORTER_OTLP_ENDPOINT     default: http://localhost:4318
OTEL_EXPORTER_OTLP_HEADERS      JSON string, e.g. for Langfuse auth
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from .graph import DeterministicGraph, ExecutionResult
from .schemas import AgentState, NodeStatus


# ---------------------------------------------------------------------------
# SDK bootstrap - import guards make the SDK optional at runtime
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace import StatusCode

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OTEL_AVAILABLE = False


def init_telemetry(
    service_name: Optional[str] = None,
    service_version: str = "1.0.0",
    otlp_endpoint: Optional[str] = None,
    otlp_headers: Optional[dict] = None,
) -> None:
    """Bootstrap the OpenTelemetry SDK.

    Safe to call multiple times (idempotent after the first call).
    If opentelemetry-sdk is not installed, this is a no-op.
    """
    if not _OTEL_AVAILABLE:
        return

    svc = service_name or os.getenv("OTEL_SERVICE_NAME", "agent-state-guard")
    endpoint = otlp_endpoint or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"
    )

    raw_headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
    headers = otlp_headers or (json.loads(raw_headers) if raw_headers else {})

    resource = Resource.create(
        {"service.name": svc, "service.version": service_version}
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=f"{endpoint.rstrip('/')}/v1/traces",
        headers=headers,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def _get_tracer():
    if not _OTEL_AVAILABLE:
        return None
    return trace.get_tracer("agent-state-guard")


# ---------------------------------------------------------------------------
# Traced wrapper
# ---------------------------------------------------------------------------
class TracedGraph:
    """Thin wrapper around DeterministicGraph that emits OTLP spans.

    One root span is created per run() call.  Each node execution
    produces a child span with attributes for the node name, attempt number,
    input/output hashes, and duration.  Schema violations and retries each
    get their own error / retry event on the relevant span.

    If OpenTelemetry is not installed, TracedGraph delegates to the
    underlying graph with zero overhead.
    """

    def __init__(self, graph: DeterministicGraph) -> None:
        self._graph = graph
        self._tracer = _get_tracer()

    def run(self, initial_state: AgentState) -> ExecutionResult:
        if self._tracer is None:
            return self._graph.run(initial_state)

        with self._tracer.start_as_current_span(
            "agent_graph.run",
            attributes={
                "run_id": str(initial_state.run_id),
                "task": str(initial_state.task)[:256],
                "node_count": len(self._graph.node_names),
            },
        ) as root_span:
            try:
                result = self._graph.run(initial_state)
                root_span.set_attribute("transitions_count", len(result.transitions))
                root_span.set_attribute("final_step", result.final_state.step)
                root_span.set_status(StatusCode.OK)

                # Child spans, one per transition record
                for record in result.transitions:
                    with self._tracer.start_as_current_span(
                        f"node.{record.node_name}",
                        attributes={
                            "node_name": record.node_name,
                            "attempt": record.attempt,
                            "status": record.status.value,
                            "input_hash": record.input_hash or "",
                            "output_hash": record.output_hash or "",
                            "duration_ms": record.duration_ms or 0.0,
                        },
                    ) as node_span:
                        if record.status == NodeStatus.FAILED:
                            node_span.set_status(StatusCode.ERROR, record.error or "")
                        elif record.status == NodeStatus.FALLBACK:
                            node_span.add_event("fallback_invoked")
                            node_span.set_status(StatusCode.OK)
                        else:
                            node_span.set_status(StatusCode.OK)

                return result

            except Exception as exc:
                root_span.record_exception(exc)
                root_span.set_status(StatusCode.ERROR, str(exc))
                raise


def traced_graph(graph: DeterministicGraph) -> TracedGraph:
    """Wrap graph in a TracedGraph.  Call init_telemetry() first."""
    return TracedGraph(graph)
