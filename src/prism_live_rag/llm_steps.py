from __future__ import annotations

import json
import re
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Protocol

from .models import RetrievedPassage
from .providers import ProviderError
from .synthesis import SENTENCE_RE, content_terms, is_answer_span, select_answer_evidence, supports_definition, synthesize_answer


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        ...


@dataclass(frozen=True)
class EvidenceSpan:
    passage_id: str
    sentence: str


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    elif not stripped.startswith("{"):
        stripped = _first_balanced_json_object(stripped)
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ProviderError("Provider JSON response is not an object")
    return payload


def _first_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ProviderError("Provider did not return a JSON object")
    depth = 0
    in_string = False
    escaped = False
    for offset, char in enumerate(text[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = in_string
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : offset + 1]
    raise ProviderError("Provider did not return a balanced JSON object")


def rewrite_query(client: ChatClient, transcript_text: str) -> str:
    payload = extract_json_object(
        client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the latest user utterance into a standalone retrieval query. "
                        "Resolve references only from the supplied conversation context, keeping "
                        "the current question's entities, negations, and constraints. Do not answer "
                        "the question or add facts. Ignore unrelated earlier discussion. "
                        "If the utterance is incomplete, copy it into rewritten_version. "
                        "Return only JSON with keys class and rewritten_version."
                    ),
                },
                {"role": "user", "content": transcript_text},
            ],
            temperature=0.0,
            max_tokens=160,
        )
    )
    value = payload.get("rewritten_version", "")
    rewritten = value.strip() if isinstance(value, str) else ""
    if rewritten.lower() in {"null", "none", "n/a"}:
        rewritten = ""
    return rewritten or transcript_text


def extract_spans_with_provider(
    client: ChatClient,
    query: str,
    passages: list[RetrievedPassage],
    max_spans: int = 4,
    *,
    defer_semantic_checks: bool = False,
) -> list[EvidenceSpan]:
    if not passages:
        return []
    passages = [item for item in passages if _jurisdiction_matches_context(query, item.passage.url)]
    if not passages:
        return []
    try:
        return _extract_spans_once(client, query, passages, max_spans, defer_semantic_checks)
    except (ProviderError, json.JSONDecodeError) as exc:
        if not _is_provider_parse_error(exc):
            raise
        return _extract_spans_once(client, query, passages, max_spans, defer_semantic_checks)


def _is_provider_parse_error(exc: Exception) -> bool:
    if isinstance(exc, json.JSONDecodeError):
        return True
    message = str(exc)
    return (
        "JSON" in message
        or "balanced JSON" in message
        or "extracted_spans" in message
        or "did not return" in message
    )


def _extract_spans_once(
    client: ChatClient,
    query: str,
    passages: list[RetrievedPassage],
    max_spans: int,
    defer_semantic_checks: bool = False,
) -> list[EvidenceSpan]:
    indexed_passages = []
    id_by_index: dict[str, str] = {}
    text_by_id: dict[str, str] = {}
    valid_ids = set()
    sentence_by_id: dict[str, tuple[str, str]] = {}
    for index, item in enumerate(passages, start=1):
        pid = item.passage.id
        id_by_index[str(index)] = pid
        text_by_id[pid] = " ".join(item.passage.text.split())
        valid_ids.add(pid)
        catalog = []
        for offset, sentence in enumerate(
            part.strip() for block in item.passage.text.splitlines()
            for part in SENTENCE_RE.split(block) if part.strip()
        ):
            sid = f"{index}.{offset + 1}"
            sentence_by_id[sid] = (pid, " ".join(sentence.split()))
            catalog.append(f"[{sid}] {sentence}")
        indexed_passages.append(f"Passage {index}\nchunk_id: {pid}\ntitle: {item.passage.title}\nurl: {item.passage.url}\n" + "\n".join(catalog))
    payload = extract_json_object(
        client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Extract exact evidence spans that directly answer the user's query, using only the "
                        "provided passages. Do not return merely topically related facts. If the passages "
                        "do not support an answer to the query, return an empty extracted_spans array. "
                        "Each sentence value MUST be a contiguous verbatim substring of one passage. "
                        "Do not summarize, paraphrase, merge distant sentences, invent separators, or "
                        "shorten numbered instructions. For a procedure, copy its actual instruction "
                        "sentences or contiguous numbered steps, including qualifications. "
                        "A question may contain a false premise: extract text that explicitly "
                        "corrects it rather than requiring the premise to be true. Conversation "
                        "context resolves the subject only; it is not factual evidence. "
                        "Select the bracketed sentence IDs instead of copying or rewriting text. "
                        "Do not select navigation menus, headings alone, or a fact about a different "
                        "subject or attribute. Select sufficient sentences to answer the actual "
                        "requested relation, including nearby qualifications. For yes/no questions, "
                        "a topical mention is not proof. If an enumeration is requested, select its "
                        "items and identifying sentence. Return only JSON: {\"span_ids\":[\"1.3\",\"1.4\"]}."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\n" + "\n\n".join(indexed_passages),
                },
            ],
            temperature=0.0,
            max_tokens=700,
        )
    )
    if "span_ids" not in payload and "extracted_spans" not in payload:
        raise ProviderError("Evidence response missing extracted_spans or span_ids list")
    raw_spans = payload.get("extracted_spans", [])
    if "span_ids" in payload:
        selected_ids = payload["span_ids"]
        if not isinstance(selected_ids, list):
            raise ProviderError("Evidence JSON response missing span_ids list")
        raw_spans = [
            {"passage_id": sentence_by_id[sid][0], "sentence": sentence_by_id[sid][1]}
            for sid in selected_ids if isinstance(sid, str) and sid in sentence_by_id
        ]
    if not isinstance(raw_spans, list):
        raise ProviderError("Evidence response missing extracted_spans list")
    # Retrieved/gold passages are not automatically evidence for this question.
    # Reject topical but non-responsive exact sentences with a cheap overlap gate.
    answer_query = _current_question(query)
    query_terms = content_terms(answer_query)
    min_overlap = max(1, int(len(query_terms) * 0.30 + 0.999))
    spans: list[EvidenceSpan] = []
    for row in raw_spans:
        if not isinstance(row, dict):
            continue
        raw_id = str(row.get("passage_id", "")).strip()
        passage_id = id_by_index.get(raw_id, raw_id)
        sentence = " ".join(str(row.get("sentence", "")).split())
        expanded_measurement = False
        if defer_semantic_checks and passage_id in valid_ids:
            source = next(item.passage.text for item in passages if item.passage.id == passage_id)
            sentence, expanded_measurement = _expand_selected_structure(sentence, source, answer_query)
        sentence_terms = content_terms(sentence)
        overlap = sum(
            1 for term in query_terms
            if any(
                term == evidence_term
                or (len(term) >= 5 and (term.startswith(evidence_term) or evidence_term.startswith(term)))
                for evidence_term in sentence_terms
            )
        )
        if (
            passage_id in valid_ids
            and sentence
            and (is_answer_span(sentence) or expanded_measurement)
            and _matches_required_answer_shape(answer_query, sentence)
            and sentence in text_by_id[passage_id]
            and (
                defer_semantic_checks
                or (
                    (not query_terms or overlap >= min_overlap or _supports_procedural_span(
                        query, sentence, text_by_id[passage_id]
                    ))
                    and _matches_question_relation(answer_query, sentence)
                )
            )
        ):
            spans.append(EvidenceSpan(passage_id=passage_id, sentence=sentence))
        if len(spans) >= max_spans:
            break
    observer = getattr(client, "observe_evidence_selection", None)
    if observer is not None:
        observer({
            "query": answer_query,
            "requested_span_ids": payload.get("span_ids"),
            "selected_count": len(raw_spans),
            "retained_count": len(spans),
            "deferred_semantic_checks": defer_semantic_checks,
        })
    return spans


def _expand_selected_structure(sentence: str, source: str, question: str) -> tuple[str, bool]:
    """Keep an explicit list/measurement with its selected heading or label.

    All expansions are contiguous source substrings. Never assign an unlabeled
    value to an entity or join cells across intervening labels. Semantic review
    is mandatory for this path.
    """
    if "\n" not in source or not sentence:
        return sentence, False
    heading = content_terms(sentence)
    if 2 <= len(heading) and len(sentence.split()) <= 5 and heading.issubset(content_terms(question)):
        block = re.search(r"(?m)^" + re.escape(sentence) + r"\s*\n\s*\n([^\n]+(?:\n[^\n]+){1,11})(?:\n\s*\n|$)", source)
        if block and len(block.group(0)) <= 700:
            return " ".join(block.group(0).split()), False
    number_unit = r"\d[\d,.]*\s+(?:hours?|minutes?|seconds?|days?|kilometers?|meters?|kilograms?|GB|MB)"
    label = r"[A-Za-z][A-Za-z -]{0,60}"
    row = re.search(r"(?m)^" + re.escape(sentence) + r"\n(?:[ \t]*\n)*[ \t]*" + number_unit + r"[ \t]*$", source) if re.fullmatch(label, sentence) else None
    if row is None and re.fullmatch(number_unit, sentence, re.I):
        row = re.search(r"(?m)^" + label + r"\n(?:[ \t]*\n)*[ \t]*" + re.escape(sentence) + r"[ \t]*$", source)
    if row and re.fullmatch(label, row.group(0).splitlines()[0]):
        return " ".join(row.group(0).split()), True
    return sentence, False


def _supports_procedural_span(query: str, sentence: str, source: str) -> bool:
    """Procedure steps often omit the subject already stated in their heading.

    Require an explicit procedural question, an imperative operation, and topic
    anchors in the same source passage. This does not admit generic topic facts.
    """
    question = _current_question(query)
    if not re.search(r"\bhow\b.*\b(?:configure|install|setup|set up|apply|enable)\b", question, re.I):
        return False
    if not re.match(r"\s*(?:(?:step\s+)?\d+[.:)]\s*)?(?:edit|append|close|save|restart|configure|select|choose|open|run|use|click|complete|create|submit)\b", sentence, re.I):
        return False
    actions = {"configure", "install", "setup", "apply", "enable"}
    topic = content_terms(question) - actions
    if len(topic) < 2 and " Current question:" in query:
        topic = content_terms(query.rsplit(" Current question:", 1)[0]) - actions
    return len(topic & content_terms(source)) >= 2


def _current_question(query: str) -> str:
    """Score against the latest utterance, not the whole dialogue wrapper."""
    marker = " Current question:"
    return query.rsplit(marker, 1)[-1].strip() if marker in query else query.strip()


def _matches_required_answer_shape(question: str, sentence: str) -> bool:
    """Hard attribute guards remain mandatory even under semantic review."""
    dates = re.findall(r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b", question, re.I)
    if any(date.casefold() not in sentence.casefold() for date in dates):
        # An undated body fragment from a multi-date changelog cannot support
        # the user's requested date, even when the model selects it.
        return False
    if re.search(r"\b(?:phone|telephone|tele|tel)\s*(?:number|no\b)|\bphone\s+number\b", question, re.I):
        # North American corpus numbers include optional country code and
        # conventional spaces, dots, parentheses and hyphens. An address or ZIP
        # code cannot satisfy a request for a telephone number.
        return bool(re.search(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)", sentence))
    return True


_STATE_NAMES = dict(pair.split(":", 1) for pair in (
    "al:Alabama|ak:Alaska|az:Arizona|ar:Arkansas|ca:California|co:Colorado|ct:Connecticut|"
    "de:Delaware|fl:Florida|ga:Georgia|hi:Hawaii|id:Idaho|il:Illinois|in:Indiana|ia:Iowa|"
    "ks:Kansas|ky:Kentucky|la:Louisiana|me:Maine|md:Maryland|ma:Massachusetts|mi:Michigan|"
    "mn:Minnesota|ms:Mississippi|mo:Missouri|mt:Montana|ne:Nebraska|nv:Nevada|nh:New Hampshire|"
    "nj:New Jersey|nm:New Mexico|ny:New York|nc:North Carolina|nd:North Dakota|oh:Ohio|"
    "ok:Oklahoma|or:Oregon|pa:Pennsylvania|ri:Rhode Island|sc:South Carolina|sd:South Dakota|"
    "tn:Tennessee|tx:Texas|ut:Utah|vt:Vermont|va:Virginia|wa:Washington|wv:West Virginia|"
    "wi:Wisconsin|wy:Wyoming"
).split("|"))


def _jurisdiction_matches_context(query: str, source_url: str) -> bool:
    """Reject an explicitly different US state authority, not federal sources.

    This is a conservative scope guard, not a claim that an unknown host has
    matching jurisdiction. Multiple-state requests remain with semantic review.
    """
    host = (urlparse(source_url).hostname or "").lower()
    # va.gov is the federal Department of Veterans Affairs, not Virginia.
    if host == "va.gov" or host.endswith(".va.gov"):
        return True
    official = re.search(r"(?:^|\.)([a-z]{2})\.gov$", host)
    state = _STATE_NAMES.get(official.group(1)) if official else None
    if state is None:
        return True
    current = _current_question(query)
    states = lambda text: {name for name in _STATE_NAMES.values() if re.search(r"\b" + re.escape(name) + r"\b", text, re.I)}
    requested = states(current) or states(query)
    return len(requested) != 1 or state in requested


def _matches_question_relation(question: str, sentence: str) -> bool:
    """Require explicit answer-type evidence for common high-risk questions."""
    q = question.lower()
    s = sentence.lower()
    if not supports_definition(question, sentence):
        return False
    # A flattened multi-column table row does not identify which value applies.
    # Do not present several unlabeled counts as one grounded count answer.
    if re.search(r"\bhow many\b", q) and len(re.findall(r"\b\d+(?:\.\d+)?\b", s)) > 1:
        if len(s.split()) <= 10 and not re.search(r"\b(?:is|are|includes|allows|per|each|for)\b", s):
            return False
    if re.search(r"\b(?:why|reason|purpose|cause)\b", q):
        return bool(re.search(
            r"\b(?:because|due to|reason|purpose|so that|in order to|"
            r"protect|prevent|ensure|allow|enable|designed to|helps?|gain insight)\b", s
        ))
    if re.search(r"\b(?:same or different|same\s*/\s*different|"
                 r"the same|different from|similar to|compared with)\b", q):
        relation_present = bool(re.search(
            r"\b(?:same|different|similar|differ|unlike|whereas|versus|"
            r"compared|equivalent|not identical)\b", s
        ))
        # Generic overlap (e.g. "IBM Cloud Object Storage" and "same") can
        # describe a different comparison. Require the evidence sentence to
        # mention each distinctive entity from the question as well.
        entity_question = q.rsplit(" if ", 1)[-1] if " if " in q else q
        query_terms = content_terms(entity_question)
        sentence_terms = content_terms(sentence)
        generic = {
            "object", "storage", "cloud", "service", "platform", "product",
            "same", "different", "similar", "differ", "compared",
        }
        entities = query_terms - generic
        entities_present = len(entities) >= 2 and all(
            any(term == candidate or (
                len(term) >= 5 and (term.startswith(candidate) or candidate.startswith(term))
            ) for candidate in sentence_terms)
            for term in entities
        )
        return relation_present and entities_present
    if re.search(r"\b(?:who do i call|whom should i call|how do i contact|"
                 r"who should i contact)\b", q):
        return bool(re.search(r"\b(?:call|contact|phone|email|hotline|number)\b", s))
    return True


def generate_grounded_answer(client: ChatClient, query: str, spans: list[EvidenceSpan]) -> str:
    facts = "\n".join(f"- [{span.passage_id}] {span.sentence}" for span in spans)
    answer = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied facts. Do not add outside facts. "
                    "If the facts are insufficient, say so briefly."
                ),
            },
            {"role": "user", "content": f"Question: {query}\n\nFacts:\n{facts}"},
        ],
        temperature=0.0,
        max_tokens=260,
    )
    return " ".join(answer.split())


def provider_synthesize_answer(
    evidence_client: ChatClient | None,
    generation_client: ChatClient | None,
    query: str,
    passages: list[RetrievedPassage],
    *,
    verify_relevance: bool = False,
) -> tuple[str, list[str], str | None]:
    if evidence_client is None or generation_client is None:
        answer, citations, uncertainty = synthesize_answer(query, passages)
        return answer, citations, uncertainty or "Provider clients are not configured; used deterministic fallback."
    try:
        # In verified mode, semantic review replaces the lexical relevance gate,
        # not the mandatory exact-substring, ID allowlist or navigation checks.
        spans = extract_spans_with_provider(
            evidence_client, query, passages, max_spans=8 if verify_relevance else 4,
            defer_semantic_checks=verify_relevance,
        )
        missing_information = None
        if spans and verify_relevance:
            # Selection checks exact provenance; this separate, bounded check
            # checks responsiveness. It cannot add or rewrite any evidence.
            by_id = {item.passage.id: item.passage for item in passages}
            catalog = "\n".join(
                f"[{i}] Source: {by_id[span.passage_id].title} {by_id[span.passage_id].url}\n"
                f"Source opening: {by_id[span.passage_id].text[:240]}\n"
                f"Selected evidence: {span.sentence}"
                for i, span in enumerate(spans, 1)
            )
            observer = getattr(evidence_client, "observe_relevance_check", None)
            if observer is not None:
                observer(query)
            review_messages = [
                {"role": "system", "content": (
                    "Check whether each cited excerpt answers the CURRENT QUESTION about the correct "
                    "subject, location, time, and requested attribute. Dialogue resolves references "
                    "only; it is not evidence. Reject merely topical facts, navigation/link stubs, "
                    "and answers for another jurisdiction or product. Do not infer missing facts. "
                    "Keep explicit corrections of false premises. A partial answer may retain only "
                    "supported aspects; identify missing requested aspects. Return only JSON: "
                    '{"supported_span_ids":[1],"missing_information":null}. '
                    "If no excerpt answers the question, return an empty list and a brief reason."
                )},
                {"role": "user", "content": f"Question and context: {query}\nSelected cited evidence:\n{catalog}"},
            ]
            for review_attempt in range(2):
                try:
                    checked = extract_json_object(evidence_client.chat(
                        review_messages, temperature=0.0, max_tokens=512,
                    ))
                    break
                except (ProviderError, json.JSONDecodeError) as exc:
                    if review_attempt or not _is_provider_parse_error(exc):
                        raise
            accepted = checked.get("supported_span_ids")
            if not isinstance(accepted, list):
                raise ProviderError("Relevance JSON response missing supported_span_ids list")
            spans = [span for i, span in enumerate(spans, 1) if any(type(n) is int and n == i for n in accepted)]
            missing = checked.get("missing_information")
            missing_information = missing.strip() if isinstance(missing, str) and missing.strip() else None
        if not spans:
            return (
                "I do not have enough information in the retrieved corpus to answer that.",
                [],
                missing_information or "No provider evidence spans were found.",
            )
        # Keep the factual surface form extractive. A second generative pass can
        # introduce unsupported modifiers even when its input facts are correct.
        retained = select_answer_evidence(query, [(span.passage_id, span.sentence) for span in spans])
        answer = " ".join(sentence for _, sentence in retained)
        citations = list(dict.fromkeys(pid for pid, _ in retained))
        return answer, citations, missing_information
    except (ProviderError, json.JSONDecodeError) as exc:
        # Do not turn an API/rate-limit failure into a confident-looking answer
        # from a weak lexical fallback. In a provider run, failure is uncertainty.
        return (
            "I could not verify a supported answer from the available evidence.",
            [],
            f"Provider evidence check failed ({type(exc).__name__}); no answer was emitted.",
        )
