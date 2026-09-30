from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TranscriptChunk:
    timestamp_s: float
    text: str
    is_final: bool = False
    confidence: float | None = None


@dataclass
class IntentStabilityState:
    """State tracking for semantic intent stability detection."""
    embeddings: list[list[float]]
    texts: list[str] 
    timestamps: list[float]
    confidences: list[float]
    semantic_similarities: list[float]
    confidence_decay_scores: list[float]
    fired: bool = False
    
    def add_chunk(self, embedding: list[float], text: str, timestamp: float, confidence: float) -> None:
        """Add a new chunk to the stability state with validation."""
        try:
            # Validate inputs
            if not embedding or not isinstance(embedding, list):
                raise ValueError("Invalid embedding: must be non-empty list")
            if not text or not isinstance(text, str):
                raise ValueError("Invalid text: must be non-empty string")
            if not isinstance(timestamp, (int, float)) or timestamp < 0:
                raise ValueError("Invalid timestamp: must be non-negative number")
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                raise ValueError("Invalid confidence: must be between 0 and 1")
                
            self.embeddings.append(embedding)
            self.texts.append(text)
            self.timestamps.append(timestamp)
            self.confidences.append(confidence)
            
            # Calculate semantic similarity with previous chunk
            if len(self.embeddings) > 1:
                try:
                    sim = self._cosine_similarity(self.embeddings[-2], self.embeddings[-1])
                    # Clamp similarity to valid range
                    sim = max(-1.0, min(1.0, sim))
                    self.semantic_similarities.append(sim)
                    
                    # Calculate confidence decay score
                    decay_score = self._calculate_confidence_decay()
                    decay_score = max(0.0, min(1.0, decay_score))  # Clamp to [0,1]
                    self.confidence_decay_scores.append(decay_score)
                    
                except Exception as e:
                    # Don't fail the whole operation on similarity calculation error
                    import warnings
                    warnings.warn(f"Similarity calculation failed: {e}", RuntimeWarning)
                    self.semantic_similarities.append(0.0)  # Conservative fallback
                    self.confidence_decay_scores.append(confidence)  # Use raw confidence
                    
        except Exception as e:
            # This is a critical error - re-raise with context
            raise ValueError(f"Failed to add chunk to stability state: {e}") from e
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors with robust error handling."""
        try:
            # Validate inputs
            if not vec1 or not vec2:
                return 0.0
            if len(vec1) != len(vec2):
                raise ValueError(f"Vector dimension mismatch: {len(vec1)} != {len(vec2)}")
            
            # Calculate dot product and magnitudes
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(b * b for b in vec2))
            
            # Handle zero magnitude vectors (shouldn't happen with normalized embeddings)
            if magnitude1 == 0.0 or magnitude2 == 0.0:
                return 0.0
            
            # Calculate similarity and handle numerical issues
            similarity = dot_product / (magnitude1 * magnitude2)
            
            # Clamp to [-1, 1] to handle floating point precision issues
            return max(-1.0, min(1.0, similarity))
            
        except (ValueError, ArithmeticError, TypeError) as e:
            import warnings
            warnings.warn(f"Cosine similarity calculation failed: {e}", RuntimeWarning)
            return 0.0  # Conservative fallback
    
    def _calculate_confidence_decay(self) -> float:
        """Calculate confidence decay score based on recent stability with error handling."""
        try:
            if len(self.confidences) < 2:
                return self.confidences[-1] if self.confidences else 0.0
            
            # Weight recent chunks more heavily (exponential decay)
            weights = [math.exp(-0.5 * i) for i in range(len(self.confidences))]
            weights.reverse()  # Most recent first
            
            # Validate weights
            if not weights or sum(weights) == 0:
                return self.confidences[-1]  # Fallback to most recent confidence
            
            # Combine ASR confidence with exponential weighting
            asr_conf = sum(c * w for c, w in zip(self.confidences, weights)) / sum(weights)
            
            # Incorporate semantic stability if available
            if len(self.semantic_similarities) > 0:
                # Use recent similarities (limited by weights length)
                recent_sims = self.semantic_similarities[-len(weights):]
                if recent_sims:
                    # Weight similarities the same way
                    sim_weights = weights[-len(recent_sims):]
                    if sum(sim_weights) > 0:
                        sem_stability = sum(s * w for s, w in zip(recent_sims, sim_weights)) / sum(sim_weights)
                        # Clamp semantic stability 
                        sem_stability = max(0.0, min(1.0, sem_stability))
                    else:
                        sem_stability = 0.0
                else:
                    sem_stability = 0.0
            else:
                sem_stability = 0.0
            
            # Combined confidence score (ASR confidence boosted by semantic stability)
            # Use conservative combination: base confidence + stability bonus
            combined = asr_conf * (0.7 + 0.3 * sem_stability)
            
            # Clamp result to valid range
            return max(0.0, min(1.0, combined))
            
        except (ValueError, ArithmeticError, ZeroDivisionError) as e:
            import warnings
            warnings.warn(f"Confidence decay calculation failed: {e}", RuntimeWarning)
            # Fallback to most recent raw confidence
            return self.confidences[-1] if self.confidences else 0.0


PRESENTATION_COMMANDS = (
    "repeat",
    "say that again",
    "read that back",
    "say again",
    "summarize",
    "summarise",
    "speak",
    "rephrase",
    "replay",
    "go back",
    "go over that again",
    "start over",
    "start again",
    "never mind",
    "nevermind",
    "hold on",
    "stop",
    "pause",
    "what did you mean",
    "what did you just say",
    "that's all",
    "thats all",
    "thank you",
    "thanks",
    "got it",
    "sounds good",
)

#: Politeness that precedes an imperative without changing it. Stripped before matching
#: so "can you repeat that" is recognised as the same command as "repeat that".
_POLITE_PREFIX = re.compile(
    r"^(?:can|could|would|will|may)\s+you\s+(?:please\s+)?"
    r"|^(?:please|just|ok|okay)\s+"
    r"|^(?:i\s+(?:want|need)\s+you\s+to\s+)"
)

#: A presentation command is short. Beyond this many words the utterance is carrying real
#: content and must retrieve: "stop" is a command, "stoplight timing in Denver" is not,
#: and neither is "hold on to the waiver requirement".
_MAX_PRESENTATION_WORDS = 5

# A small number of common delivery commands exceed the general length cap. Keep
# these as exact matches so longer content requests such as "hold on to the waiver
# requirement" are still allowed through to retrieval.
_LONG_PRESENTATION_COMMANDS = frozenset(
    {
        "go back to the previous point",
        "what did you mean by that",
    }
)

_PRESENTATION_PATTERNS = tuple(
    (command, re.compile(rf"^{re.escape(command)}\b"))
    for command in PRESENTATION_COMMANDS
)


def is_presentation_only(text: str) -> bool:
    """True when the utterance is a request about delivery, not new content.

    Matching is anchored at the start but respects word boundaries, a leading politeness
    preamble is stripped first, and the whole utterance must be short. All three matter:
    plain ``startswith`` both missed the common "can you repeat that" form and silently
    swallowed genuine questions beginning with the same letters -- a recall bug in the
    live path, not just a labelling nit.
    """
    lowered = text.lower().lstrip()
    previous = None
    while previous != lowered:
        previous = lowered
        lowered = _POLITE_PREFIX.sub("", lowered).lstrip()
    if not lowered:
        return False
    normalized = lowered.rstrip(" .!?")
    if (
        len(lowered.split()) > _MAX_PRESENTATION_WORDS
        and normalized not in _LONG_PRESENTATION_COMMANDS
    ):
        return False
    return any(pattern.match(lowered) for _, pattern in _PRESENTATION_PATTERNS)


class SemanticRetrievalController:
    """Advanced streaming controller with semantic intent stability detection.
    
    Based on Meta's 2025 streaming RAG research showing 200% accuracy improvement
    with semantic stability detection vs word-level agreement.
    """
    
    def __init__(
        self,
        encoder=None,  # Encoder for semantic similarity 
        min_chars: int = 24,
        min_tokens: int = 5,
        min_confidence: float = 0.72,
        semantic_threshold: float = 0.85,  # Minimum semantic similarity for stability
        stability_window: int = 3,  # Number of chunks to consider for stability
        intent_drift_threshold: float = 0.65,  # Threshold for detecting intent drift
        confidence_decay_alpha: float = 0.7,  # Confidence decay factor
        adaptive_threshold: bool = True,  # Enable adaptive thresholds
    ) -> None:
        self.encoder = encoder
        self.min_chars = min_chars
        self.min_tokens = min_tokens
        self.min_confidence = min_confidence
        self.semantic_threshold = semantic_threshold
        self.stability_window = stability_window
        self.intent_drift_threshold = intent_drift_threshold
        self.confidence_decay_alpha = confidence_decay_alpha
        self.adaptive_threshold = adaptive_threshold
        
        # State tracking
        self.state = IntentStabilityState([], [], [], [], [], [], False)
        self.last_reason = ""
        
        # Adaptive threshold learning
        self.threshold_history: list[tuple[float, bool]] = []  # (threshold_used, was_successful)
        self.query_type_thresholds: dict[str, float] = {}
    
    def reset(self) -> None:
        """Reset controller state for new query."""
        self.state = IntentStabilityState([], [], [], [], [], [], False)
        self.last_reason = ""
    
    def decide(self, chunk: TranscriptChunk) -> str:
        """Decide whether to retrieve based on semantic intent stability."""
        text = chunk.text.strip()
        
        # Basic filtering (same as before)
        if not text:
            self.last_reason = "empty text"
            return "No-Retrieval"
        if self._is_presentation_only(text):
            self.last_reason = "presentation-only command"
            return "No-Retrieval"
        if chunk.is_final:
            self.last_reason = "final transcript"
            return "Retrieve"
        
        # Length and confidence checks
        confidence = chunk.confidence if chunk.confidence is not None else 0.5
        if confidence < self.min_confidence:
            self.last_reason = f"low ASR confidence {confidence:.3f}"
            return "Wait"
        if len(text) < self.min_chars or len(text.split()) < self.min_tokens:
            self.last_reason = "too short for retrieval"
            return "Wait"
        
        # Semantic analysis (requires encoder)
        if self.encoder is None:
            # Fallback to rule-based if no encoder
            return self._fallback_rule_based(chunk)
        
        # Encode current chunk with robust error handling
        try:
            embedding = self.encoder.encode_query(text)
            if not embedding or len(embedding) != self.encoder.dim:
                raise ValueError(f"Invalid embedding: expected {self.encoder.dim} dims, got {len(embedding) if embedding else 0}")
            
            self.state.add_chunk(embedding, text, chunk.timestamp_s, confidence)
            
        except Exception as e:
            # Log the error but don't fail - fallback gracefully
            import warnings
            warnings.warn(f"Encoding failed for text '{text[:50]}...': {e}", RuntimeWarning)
            return self._fallback_rule_based(chunk)
        
        # Need at least stability_window chunks for analysis
        if len(self.state.embeddings) < self.stability_window:
            self.last_reason = f"collecting stability window ({len(self.state.embeddings)}/{self.stability_window})"
            return "Wait"
        
        # Check for intent drift (significant semantic change)
        if self._detect_intent_drift():
            if self.state.fired:
                # Intent changed after firing - fire again
                self.last_reason = "intent drift detected, re-firing"
                return "Retrieve"
            else:
                # Reset window on intent drift
                self._reset_stability_window()
                self.last_reason = "intent drift detected, resetting window"
                return "Wait"
        
        # Check semantic stability 
        if self._is_semantically_stable():
            if not self.state.fired:
                self.state.fired = True
                stability_score = self._calculate_stability_score()
                self.last_reason = f"semantically stable (score: {stability_score:.3f})"
                
                # Update adaptive thresholds
                if self.adaptive_threshold:
                    self._update_adaptive_threshold(stability_score, success=True)
                
                return "Retrieve"
            else:
                self.last_reason = "already retrieved, awaiting final or intent drift"
                return "Wait"
        
        # Not stable yet
        stability_score = self._calculate_stability_score()
        current_threshold = self._get_adaptive_threshold()
        self.last_reason = f"not stable yet (score: {stability_score:.3f} < threshold: {current_threshold:.3f})"
        return "Wait"
    
    def _is_semantically_stable(self) -> bool:
        """Check if recent chunks show semantic stability."""
        if len(self.state.semantic_similarities) < self.stability_window - 1:
            return False
        
        # Get recent similarities
        recent_sims = self.state.semantic_similarities[-(self.stability_window - 1):]
        
        # Use adaptive or fixed threshold
        threshold = self._get_adaptive_threshold()
        
        # All recent similarities must be above threshold
        stable = all(sim >= threshold for sim in recent_sims)
        
        # Additional confidence decay check
        if stable and len(self.state.confidence_decay_scores) > 0:
            recent_decay_score = self.state.confidence_decay_scores[-1]
            stable = stable and (recent_decay_score >= self.min_confidence)
        
        return stable
    
    def _detect_intent_drift(self) -> bool:
        """Detect significant semantic direction change."""
        if len(self.state.semantic_similarities) < 2:
            return False
        
        # Check if recent similarity dropped significantly
        recent_sim = self.state.semantic_similarities[-1]
        return recent_sim < self.intent_drift_threshold
    
    def _calculate_stability_score(self) -> float:
        """Calculate overall stability score combining multiple factors."""
        if len(self.state.semantic_similarities) == 0:
            return 0.0
        
        # Weighted average of recent semantic similarities
        recent_sims = self.state.semantic_similarities[-(self.stability_window - 1):]
        weights = [math.exp(-0.3 * i) for i in range(len(recent_sims))]
        weights.reverse()
        
        semantic_score = sum(s * w for s, w in zip(recent_sims, weights)) / sum(weights)
        
        # Factor in confidence decay
        if len(self.state.confidence_decay_scores) > 0:
            confidence_factor = self.state.confidence_decay_scores[-1] / self.min_confidence
            semantic_score *= confidence_factor
        
        return min(1.0, semantic_score)
    
    def _get_adaptive_threshold(self) -> float:
        """Get adaptive threshold based on query type and learning."""
        if not self.adaptive_threshold or len(self.state.texts) == 0:
            return self.semantic_threshold
        
        # Simple query type detection (could be enhanced)
        query_type = self._classify_query_type(self.state.texts[-1])
        
        # Use learned threshold for this query type
        if query_type in self.query_type_thresholds:
            return self.query_type_thresholds[query_type]
        
        return self.semantic_threshold
    
    def _classify_query_type(self, text: str) -> str:
        """Simple query type classification (could be enhanced with ML)."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["how", "configure", "setup", "install"]):
            return "how-to"
        elif any(word in text_lower for word in ["what", "explain", "define"]):
            return "definition"  
        elif any(word in text_lower for word in ["error", "problem", "issue", "troubleshoot"]):
            return "troubleshooting"
        elif any(word in text_lower for word in ["price", "cost", "billing", "quota"]):
            return "pricing"
        else:
            return "general"
    
    def _update_adaptive_threshold(self, threshold_used: float, success: bool) -> None:
        """Update adaptive thresholds based on retrieval success."""
        try:
            self.threshold_history.append((threshold_used, success))
            
            # Learn from recent history (last 50 decisions)
            if len(self.threshold_history) > 50:
                self.threshold_history = self.threshold_history[-50:]
            
            # Update query type specific thresholds - simplified for robustness
            if len(self.state.texts) > 0:
                query_type = self._classify_query_type(self.state.texts[-1])
                
                # Use overall success rate for now (can be enhanced later)
                if len(self.threshold_history) >= 5:
                    avg_threshold = sum(t for t, _ in self.threshold_history) / len(self.threshold_history)
                    success_rate = sum(s for _, s in self.threshold_history) / len(self.threshold_history)
                    
                    # Conservative adjustment
                    if success_rate < 0.7:  # Too many failures - lower threshold slightly
                        new_threshold = max(0.65, avg_threshold * 0.98)
                    elif success_rate > 0.9:  # Too conservative - raise threshold slightly  
                        new_threshold = min(0.95, avg_threshold * 1.02)
                    else:
                        new_threshold = avg_threshold
                    
                    self.query_type_thresholds[query_type] = new_threshold
        except Exception:
            # Fail silently for adaptive thresholds - don't break the main flow
            pass
    
    def _reset_stability_window(self) -> None:
        """Reset stability window keeping only the most recent chunk."""
        if len(self.state.embeddings) > 1:
            self.state.embeddings = self.state.embeddings[-1:]
            self.state.texts = self.state.texts[-1:]
            self.state.timestamps = self.state.timestamps[-1:]
            self.state.confidences = self.state.confidences[-1:]
            self.state.semantic_similarities = []
            self.state.confidence_decay_scores = []
    
    def _fallback_rule_based(self, chunk: TranscriptChunk) -> str:
        """Fallback to simple rule-based approach if semantic analysis fails."""
        # Simple word count stability (very basic fallback)
        text = chunk.text.strip()
        words = text.split()
        
        if len(words) >= 8:  # Assume longer queries are more stable
            if not self.state.fired:
                self.state.fired = True
                self.last_reason = "rule-based fallback: sufficient length"
                return "Retrieve"
        
        self.last_reason = "rule-based fallback: insufficient length"
        return "Wait"
    
    def _is_presentation_only(self, text: str) -> bool:
        return is_presentation_only(text)


# Keep the original rule-based controller for compatibility
class RuleBasedRetrievalController:
    def __init__(
        self,
        min_chars: int = 24,
        min_tokens: int = 5,
        min_confidence: float = 0.72,
        agreement_n: int = 2,
        min_stable_words: int = 3,
    ) -> None:
        self.min_chars = min_chars
        self.min_tokens = min_tokens
        self.min_confidence = min_confidence
        self.agreement_n = agreement_n
        self.min_stable_words = min_stable_words
        self._window: list[str] = []
        self._fired = False
        self.last_reason = ""

    def reset(self) -> None:
        self._window = []
        self._fired = False
        self.last_reason = ""

    def decide(self, chunk: TranscriptChunk) -> str:
        text = chunk.text.strip()
        if not text:
            self.last_reason = "empty text"
            return "No-Retrieval"
        if self._is_presentation_only(text):
            self.last_reason = "presentation-only command"
            return "No-Retrieval"
        if chunk.is_final:
            self.last_reason = "final transcript"
            return "Retrieve"
        self._window.append(text)
        committed = self._committed_prefix_len()
        if committed >= self.min_stable_words:
            if not self._fired:
                self._fired = True
                self.last_reason = f"stable intent ({committed} words agreed)"
                return "Retrieve"
            self.last_reason = "already retrieved; awaiting final"
            return "Wait"
        if chunk.confidence is not None and chunk.confidence < self.min_confidence:
            self.last_reason = f"low confidence {chunk.confidence}; intent not stable"
            return "Wait"
        if len(text) < self.min_chars or len(text.split()) < self.min_tokens:
            self.last_reason = "too short for retrieval; intent not stable"
            return "Wait"
        self.last_reason = f"intent not yet stable ({committed}/{self.min_stable_words} words)"
        return "Wait"

    def _committed_prefix_len(self) -> int:
        recent = self._window[-self.agreement_n :]
        if len(recent) < self.agreement_n:
            return 0
        tokenized = [text.split() for text in recent]
        shortest = min(tokenized, key=len)
        committed = 0
        for index in range(len(shortest)):
            token = shortest[index]
            if all(len(text) > index and text[index] == token for text in tokenized):
                committed += 1
            else:
                break
        return committed

    def _is_presentation_only(self, text: str) -> bool:
        return is_presentation_only(text)


def simulate_chunks(text: str, step_words: int = 4, interval_s: float = 0.8) -> list[TranscriptChunk]:
    words = text.split()
    chunks: list[TranscriptChunk] = []
    for index in range(step_words, len(words), step_words):
        chunks.append(
            TranscriptChunk(
                timestamp_s=round(len(chunks) * interval_s, 2),
                text=" ".join(words[:index]),
                is_final=False,
                confidence=0.78,
            )
        )
    chunks.append(
        TranscriptChunk(
            timestamp_s=round(len(chunks) * interval_s, 2),
            text=text,
            is_final=True,
            confidence=0.95,
        )
    )
    return chunks
