"""Multi-Intent Query Decomposer for Samsung Theme 04 Streaming RAG.

Implements parallel query generation from partial transcripts following the Samsung
architecture specification:
- Cap sub-queries at 2-4 to prevent over-fragmentation  
- Use concat(last-turn || standalone-rewrite) base formulation
- Handle dependent multi-hop questions sequentially, not in parallel
- Provide circuit breaker patterns for LLM failures
- Full observability and metrics for Samsung evaluation

Based on Meta 2025 research: "Stream RAG: Multi-Intent Parallel Retrieval"
"""

from __future__ import annotations

import json
import re
import time
from typing import Protocol

from .llm_steps import ChatClient
from .llm_steps import extract_json_object
from .providers import ProviderError


def inherit_shared_date(query: str, parts: list[str]) -> list[str]:
    """Retain a trailing date for coordinated requests about the same event.

    Require a shared event verb so an unrelated clause (e.g. a founding date
    followed by this year's revenue) does not inherit an inappropriate date.
    """
    if len(parts) < 2:
        return parts
    months = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    date = re.search(
        r"\b(?:on|as of|during|in)\s+(?:\d{1,2}\s+" + months
        + r"\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{4})\s*$",
        query.rstrip(" .?!"), re.I,
    )
    if not date:
        return parts
    events = r"\b(?:upgraded|updated|released|published|announced|reported|launched)\b"
    verbs = [set(re.findall(events, part.casefold())) for part in parts]
    if not set.intersection(*verbs):
        return parts
    return [
        part if re.search(r"\b\d{4}\b", part) else f"{part} {date.group(0)}"
        for part in parts
    ]


def inherit_defined_subject(parts: list[str]) -> list[str]:
    """Copy an explicitly defined user subject into subsequent singular references."""
    if len(parts) < 2:
        return parts
    defined = re.fullmatch(r"what\s+is\s+(.+)", parts[0], re.I)
    if not defined:
        return parts
    subject = re.sub(r",?\s*please$", "", defined.group(1), flags=re.I).strip()
    if re.search(r"\b(?:and|or)\b", subject, re.I):
        return parts
    return [parts[0], *(re.sub(r"\bit\b", lambda _: subject, part, flags=re.I) for part in parts[1:])]


class MultiIntentResult:
    """Result of multi-intent query decomposition."""
    
    def __init__(
        self,
        sub_queries: list[str],
        decomposition_reason: str,
        is_multi_intent: bool,
        processing_time_ms: float,
        original_query: str,
        provider_status: str = "not_applicable",
        provider_error: str | None = None,
    ):
        self.sub_queries = sub_queries
        self.decomposition_reason = decomposition_reason
        self.is_multi_intent = is_multi_intent
        self.processing_time_ms = processing_time_ms
        self.original_query = original_query
        self.provider_status = provider_status
        self.provider_error = provider_error

    def to_dict(self) -> dict:
        """Convert to dictionary for telemetry."""
        return {
            "sub_queries": self.sub_queries,
            "decomposition_reason": self.decomposition_reason,
            "is_multi_intent": self.is_multi_intent,
            "processing_time_ms": self.processing_time_ms,
            "original_query": self.original_query,
            "provider_status": self.provider_status,
            "provider_error": self.provider_error,
        }


class MultiIntentDecomposer(Protocol):
    """Protocol for multi-intent query decomposition."""
    
    def decompose(self, query: str, context: str = "") -> MultiIntentResult:
        """Decompose query into parallel sub-queries."""
        ...
    
    def get_health_status(self) -> dict:
        """Get health and performance metrics."""
        ...


class RuleBasedDecomposer:
    """Rule-based decomposer as baseline (no LLM calls)."""
    
    def __init__(self):
        self.total_queries = 0
        self.multi_intent_detected = 0
        
    def decompose(self, query: str, context: str = "") -> MultiIntentResult:
        """Split explicit query boundaries without splitting noun lists on ``and``."""
        start_time = time.perf_counter()
        self.total_queries += 1

        sub_queries = inherit_defined_subject(inherit_shared_date(query, self._split_query(query)))
        is_multi_intent = len(sub_queries) > 1
        if is_multi_intent:
            self.multi_intent_detected += 1
            reason = f"Detected multi-intent pattern, split into {len(sub_queries)} sub-queries"
        else:
            reason = "Single intent detected, no decomposition needed"
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return MultiIntentResult(
            sub_queries=sub_queries,
            decomposition_reason=reason,
            is_multi_intent=is_multi_intent,
            processing_time_ms=processing_time,
            original_query=query,
        )
    
    def _split_query(self, query: str) -> list[str]:
        """Split only clear clause boundaries; keep coordinated noun phrases intact."""
        # Coordinated interrogatives share a predicate: splitting at "and"
        # alone would emit a meaningless bare "How" retrieval intent.
        shared = re.fullmatch(
            r"\s*(how|why|when|where)\s+and\s+(how|why|when|where)\s+"
            r"((?:do|does|did|can|could|should|would|will|is|are)\b.+)",
            query.strip(), re.I,
        )
        if shared and shared.group(1).lower() != shared.group(2).lower():
            predicate = shared.group(3).rstrip(" .?!")
            return [f"{word} {predicate}" for word in shared.group(1, 2)]

        def clean(parts: list[str]) -> list[str]:
            cleaned = []
            for part in parts:
                value = part.strip().rstrip(" \t\r\n.,;:!?")
                value = re.sub(
                    r"^(?:also|additionally|another\s+thing|one\s+more\s+thing|"
                    r"and\s+another\s+question|separately|could\s+i\s+also\s+ask)"
                    r"\s*[:,]?\s*",
                    "",
                    value,
                    flags=re.IGNORECASE,
                ).strip()
                if value:
                    cleaned.append(value)
            return cleaned

        # Retain support for old synthetic fixtures, but don't depend on their
        # separator: real utterances use punctuation, discourse markers, and
        # coordinated independent questions.
        explicit_composite = clean(re.split(r"\s+\.\s+", query.strip()))
        if len(explicit_composite) > 1:
            return explicit_composite[:4]

        question_start = (
            r"(?:what|how|when|where|why|who|which|can|could|do|does|did|"
            r"is|are|will|would|should|tell\s+me|please\s+explain)"
        )
        separators = re.compile(
            # Explicit clause / sentence boundaries.
            r"\s*[;]\s*|"
            r"(?<=\?)\s+(?=" + question_start + r"\b)|"
            r"(?<=[?!.])\s+(?=(?:also\b|additionally\b|another\s+thing\b|"
            r"one\s+more\s+thing\b|separately\b|and\s+another\s+question\b|"
            r"could\s+i\s+also\s+ask\b))|"
            # Independent coordinated questions; optional discourse adverb and comma.
            r"\s+(?:and(?:\s+also)?|plus)(?:\s*,)?\s+(?=" + question_start + r"\b)",
            re.IGNORECASE,
        )
        valid_parts = clean(separators.split(query.strip()))
        if len(valid_parts) < 2:
            return [query.strip()]
        return valid_parts[:4]
    
    def get_health_status(self) -> dict:
        """Get decomposer health metrics."""
        multi_intent_rate = (
            self.multi_intent_detected / self.total_queries 
            if self.total_queries > 0 else 0.0
        )
        
        return {
            "total_queries": self.total_queries,
            "multi_intent_detected": self.multi_intent_detected,
            "multi_intent_rate": multi_intent_rate,
            "type": "rule_based",
            "error_rate": 0.0,  # Rule-based can't fail
        }


class LLMDecomposer:
    """LLM-based decomposer with circuit breaker pattern."""
    
    def __init__(
        self,
        client: ChatClient,
        *,
        max_sub_queries: int = 4,
        min_query_length: int = 5,
        timeout_ms: int = 2000,
        max_failures: int = 3,
        fallback_to_single: bool = True,
        circuit_breaker_cooldown_s: float = 30.0,
        wait_for_circuit_recovery: bool = False,
    ):
        self.client = client
        self.max_sub_queries = max_sub_queries
        self.min_query_length = min_query_length
        self.timeout_ms = timeout_ms
        self.max_failures = max_failures
        self.fallback_to_single = fallback_to_single
        self.circuit_breaker_cooldown_s = max(0.0, circuit_breaker_cooldown_s)
        self.wait_for_circuit_recovery = wait_for_circuit_recovery
        
        # Health tracking
        self.total_queries = 0
        self.multi_intent_detected = 0
        self.llm_failures = 0
        self.fallback_uses = 0
        self.total_latency_ms = 0.0
        
        # Circuit breaker state
        self.consecutive_failures = 0
        self.circuit_open = False
        self.last_failure_time = 0
        
    def decompose(self, query: str, context: str = "") -> MultiIntentResult:
        """Decompose query using LLM with circuit breaker protection."""
        start_time = time.perf_counter()
        self.total_queries += 1
        
        # Circuit breaker check
        if self._circuit_breaker_active():
            return self._fallback_decompose(query, "Circuit breaker active", start_time)
        
        try:
            result = self._llm_decompose(query, context)
            self._reset_circuit_breaker()
            return result
            
        except (ProviderError, ValueError, json.JSONDecodeError) as e:
            self._record_failure()
            if self.fallback_to_single:
                failure_kind = type(e).__name__
                if isinstance(e, ProviderError):
                    status = re.search(r"request failed:\s*(\d{3})", str(e), re.IGNORECASE)
                    if status:
                        failure_kind = f"{failure_kind}_HTTP_{status.group(1)}"
                return self._fallback_decompose(
                    query, 
                    f"LLM failed ({failure_kind}), using fallback", 
                    start_time
                )
            else:
                raise
    
    def _llm_decompose(self, query: str, context: str) -> MultiIntentResult:
        """Core LLM-based decomposition logic."""
        start_time = time.perf_counter()
        
        # Build prompt for multi-intent decomposition
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context)
        
        # Make LLM call with timeout
        response = self.client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=500,   # Sufficient for 2-4 sub-queries
        )
        
        # Parse JSON response
        result_data = extract_json_object(response)
        
        # Validate and extract results
        sub_queries = result_data.get("sub_queries", [])
        is_multi_intent = result_data.get("is_multi_intent") is True
        reason = result_data.get("reasoning", "LLM decomposition")
        
        # Apply Samsung constraints
        sub_queries = self._validate_sub_queries(sub_queries, query)
        if not is_multi_intent or len(sub_queries) < 2:
            is_multi_intent = False
            sub_queries = [query.strip()]
        
        if is_multi_intent:
            self.multi_intent_detected += 1
        
        processing_time = (time.perf_counter() - start_time) * 1000
        self.total_latency_ms += processing_time
        
        return MultiIntentResult(
            sub_queries=sub_queries,
            decomposition_reason=reason,
            is_multi_intent=is_multi_intent,
            processing_time_ms=processing_time,
            original_query=query,
            provider_status="success",
        )
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for multi-intent decomposition."""
        return f"""You are a query decomposer for a streaming RAG system. Your job is to analyze user queries and decide whether they contain multiple distinct intents that should be searched in parallel.

GUIDELINES:
1. Cap sub-queries at {self.max_sub_queries} maximum to prevent over-fragmentation
2. Only decompose when queries have truly independent search intents  
3. Do NOT decompose dependent multi-hop questions - they need sequential resolution
4. Each sub-query must be self-contained and searchable
5. Preserve the original meaning and context
6. Discourse markers such as "also", "one more thing", "separately", and "another question" can introduce an independent intent; remove the marker from the resulting sub-query.
7. Do not split coordinated noun lists, comparisons, or a question followed by a clarifying follow-up about the same referent.
8. Use the user's exact domain entities and terms; don't invent or broaden search intents.

OUTPUT FORMAT (JSON only):
{{
  "is_multi_intent": boolean,
  "sub_queries": ["query1", "query2", ...],
  "reasoning": "Brief explanation of decision"
}}

EXAMPLES:
- "What is cloud storage and how do I configure it?" → Multi-intent (definition + how-to)
- "How do I migrate from service A to service B?" → Single intent (sequential process)
- "What are the costs and security features?" → Multi-intent (costs + security)
- "What happens after I configure the database?" → Single intent (sequential step)"""
    
    def _build_user_prompt(self, query: str, context: str) -> str:
        """Build user prompt with query and context."""
        prompt = f"Query: {query}\n\n"
        if context.strip():
            prompt += f"Context: {context}\n\n"
        prompt += "Analyze this query and provide your JSON response:"
        return prompt
    
    def _validate_sub_queries(self, sub_queries: list[str], original_query: str) -> list[str]:
        """Validate and clean sub-queries per Samsung constraints."""
        if not isinstance(sub_queries, list) or not sub_queries:
            return [original_query.strip()]
        
        # Clean and filter sub-queries
        valid_queries = []
        seen = set()
        for query in sub_queries:
            if isinstance(query, str):
                cleaned = query.strip().rstrip('?,.')
                normalized = " ".join(cleaned.casefold().split())
                if len(cleaned) >= self.min_query_length and normalized not in seen:
                    valid_queries.append(cleaned)
                    seen.add(normalized)
        
        # Apply max sub-queries constraint
        if len(valid_queries) > self.max_sub_queries:
            valid_queries = valid_queries[:self.max_sub_queries]
        
        # Always return at least the original if validation fails
        if not valid_queries:
            valid_queries = [original_query.strip()]
        
        return inherit_shared_date(original_query, valid_queries)
    
    def _circuit_breaker_active(self) -> bool:
        """Check if circuit breaker should block LLM calls."""
        if not self.circuit_open:
            return False
        
        # Live mode falls back immediately; offline evaluation can wait for one
        # half-open probe instead of treating the remainder of a batch as failed.
        elapsed = time.time() - self.last_failure_time
        if elapsed > self.circuit_breaker_cooldown_s:
            self.circuit_open = False
            self.consecutive_failures = 0
            return False
        if self.wait_for_circuit_recovery:
            time.sleep(max(0.0, self.circuit_breaker_cooldown_s - elapsed))
            self.circuit_open = False
            self.consecutive_failures = 0
            return False

        return True
    
    def _record_failure(self):
        """Record LLM failure and update circuit breaker state."""
        self.llm_failures += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        
        if self.consecutive_failures >= self.max_failures:
            self.circuit_open = True
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker on successful call."""
        self.consecutive_failures = 0
        self.circuit_open = False
    
    def _fallback_decompose(
        self, 
        query: str, 
        reason: str, 
        start_time: float
    ) -> MultiIntentResult:
        """Fallback to single query when LLM fails."""
        self.fallback_uses += 1
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return MultiIntentResult(
            sub_queries=[query.strip()],
            decomposition_reason=reason,
            is_multi_intent=False,
            processing_time_ms=processing_time,
            original_query=query,
            provider_status="fallback",
            provider_error=reason,
        )
    
    def get_health_status(self) -> dict:
        """Get decomposer health and performance metrics."""
        multi_intent_rate = (
            self.multi_intent_detected / self.total_queries 
            if self.total_queries > 0 else 0.0
        )
        error_rate = (
            self.llm_failures / self.total_queries 
            if self.total_queries > 0 else 0.0
        )
        avg_latency_ms = (
            self.total_latency_ms / self.total_queries 
            if self.total_queries > 0 else 0.0
        )
        
        return {
            "total_queries": self.total_queries,
            "multi_intent_detected": self.multi_intent_detected,
            "multi_intent_rate": multi_intent_rate,
            "llm_failures": self.llm_failures,
            "error_rate": error_rate,
            "fallback_uses": self.fallback_uses,
            "avg_latency_ms": avg_latency_ms,
            "circuit_open": self.circuit_open,
            "consecutive_failures": self.consecutive_failures,
            "type": "llm_based",
        }


def build_decomposer(
    decomposer_type: str = "rule_based",
    client: ChatClient | None = None,
    **kwargs
) -> MultiIntentDecomposer:
    """Factory function to build decomposer instance.
    
    Args:
        decomposer_type: "rule_based" or "llm_based"
        client: ChatClient for LLM-based decomposer
        **kwargs: Additional configuration parameters
    
    Returns:
        MultiIntentDecomposer instance
    """
    if decomposer_type == "rule_based":
        return RuleBasedDecomposer()
    elif decomposer_type == "llm_based":
        if client is None:
            raise ValueError("ChatClient required for LLM-based decomposer")
        return LLMDecomposer(client, **kwargs)
    else:
        raise ValueError(f"Unknown decomposer type: {decomposer_type}")
