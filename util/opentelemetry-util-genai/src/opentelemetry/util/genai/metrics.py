# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for emitting GenAI metrics from LLM invocations."""

from __future__ import annotations

import timeit
from typing import Optional

from opentelemetry.metrics import Histogram, Meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.instruments import (
    create_duration_histogram,
    create_time_per_chunk_histogram,
    create_token_histogram,
    create_ttfc_histogram,
)

from ._inference_invocation import InferenceInvocation
from ._invocation import GenAIInvocation


class InvocationMetricsRecorder:
    """Records duration and token usage histograms for GenAI invocations."""

    def __init__(self, meter: Meter):
        self._duration_histogram: Histogram = create_duration_histogram(meter)
        self._token_histogram: Histogram = create_token_histogram(meter)
        self._ttfc_histogram: Histogram = create_ttfc_histogram(meter)
        self._time_per_chunk_histogram: Histogram = (
            create_time_per_chunk_histogram(meter)
        )

    def record(self, invocation: GenAIInvocation) -> None:
        """Record duration and token metrics for an invocation if possible."""
        attributes = invocation._get_metric_attributes()
        token_counts = invocation._get_metric_token_counts()

        duration_seconds: Optional[float] = None
        if invocation.monotonic_start_s is not None:
            duration_seconds = max(
                timeit.default_timer() - invocation.monotonic_start_s,
                0.0,
            )

        if duration_seconds is not None:
            self._duration_histogram.record(
                duration_seconds,
                attributes=attributes,
                context=invocation._span_context,
            )

        for token_type, token_count in token_counts.items():
            self._token_histogram.record(
                token_count,
                attributes=attributes | {GenAI.GEN_AI_TOKEN_TYPE: token_type},
                context=invocation._span_context,
            )

        # Streaming timing metrics
        if isinstance(invocation, InferenceInvocation):
            ttfc, gaps = invocation._consume_streaming_timing()
            if ttfc is not None:
                self._ttfc_histogram.record(
                    ttfc,
                    attributes=attributes,
                    context=invocation._span_context,
                )
            for gap in gaps:
                self._time_per_chunk_histogram.record(
                    gap,
                    attributes=attributes,
                    context=invocation._span_context,
                )


__all__ = ["InvocationMetricsRecorder"]
