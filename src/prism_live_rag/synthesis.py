from __future__ import annotations

import re

from .models import RetrievedPassage


SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+|(?<=[a-z][.!?])(?=[A-Z])"
    r"|(?<=[a-z])(?=(?:From|This|The|Our)\s)"
)
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
CONTENT_STOPWORDS = frozenset(
    "a an and are as at be by can could did do does for from had has have how i if "
    "in is it its may of on or should that the their them then there these they this those "
    "to was were what when where which who why will with would you your "
    "please sorry thank thanks yes okay oops meant want know tell bit confused "
    "think correct sure really also about".split()
)
AMBIGUOUS_REFERENCE_RE = re.compile(
    r"\b(?:(?:this|that|these|those)\s+"
    r"(?P<head>(?:[A-Za-z0-9_-]+\s+){0,2}"
    r"(?:case|court|county|loan|mission|program|sdk|fee|library|facilities|"
    r"clinic|hospital|department|office|service|location|policy|provider|app|types?))"
    r"|the\s+(?P<the_head>case|loan|clinic|hospital|mission|program|"
    r"department|office|location|provider|park))\b",
    re.IGNORECASE,
)
CONTEXT_DEPENDENT_RE = re.compile(
    r"\b(?:it|its|they|them|their|there|this|that|these|those|one|ones)\b"
    r"|\bwhat\s+about\b|\bi\s+meant\b|\bwhich\s+one\b"
    r"|\b(?:then|other|next steps|more concrete)\b",
    re.IGNORECASE,
)
PROVIDER_COMPLEXITY_RE = re.compile(
    r"\b(?:why|reason|compare|compared|same or different|difference between|"
    r"what if|steps involved|how does .* work together|i(?:'m| am) not sure|"
    r"i do not understand|i don't understand|confused about)\b",
    re.IGNORECASE,
)
UNRESOLVED_PRONOUN_RE = re.compile(
    r"\b(?:it|its|they|them|their|this|that|these|those|one|ones)\b",
    re.IGNORECASE,
)


def content_terms(text: str) -> set[str]:
    return {
        term.lower()
        for term in TOKEN_RE.findall(text)
        if len(term) > 2 and term.lower() not in CONTENT_STOPWORDS
    }


def deduplicate_evidence(spans: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Keep exact evidence once, including overlap across passage windows.

    Only whitespace is normalized. Containment removes a redundant span, not
    words inside an otherwise retained claim; numbers and qualifications survive.
    """
    selected: list[tuple[str, str]] = []
    for passage_id, text in spans:
        text = " ".join(text.split())
        if not text or any(text == old for _, old in selected):
            continue
        if len(text) >= 40 and any(text in old for _, old in selected):
            continue
        selected = [
            (pid, old) for pid, old in selected
            if not (len(old) >= 40 and old in text)
        ]
        selected.append((passage_id, text))
    return selected


def refine_answer(
    previous_answer: str,
    previous_citations: list[str],
    delta_answer: str,
    delta_citations: list[str],
    delta_query: str,
) -> tuple[str, list[str], str | None]:
    """Patch the affected sentence(s), retaining unrelated grounded statements.

    Without claim-level provenance for an older answer, a restrictive delta whose
    topic cannot be located must replace the old answer. Keeping an unverified old
    claim would make the refined answer contradictory or misleading.
    """
    old_sentences = [s.strip() for s in SENTENCE_RE.split(previous_answer.strip()) if s.strip()]
    new_terms = content_terms(delta_answer)
    changed = [s for s in old_sentences if content_terms(s) & new_terms]
    restrictive = bool(re.search(r"\b(?:only|instead|actually|except|rather|not)\b", delta_query, re.I))
    if not restrictive and re.match(r"\s*(?:what\s+about|also\b|and\b)", delta_query, re.I):
        # An additive follow-up is not a correction. Shared subject words in
        # its evidence must not erase unrelated established attributes.
        return previous_answer.rstrip() + "\n\n" + delta_answer.strip(), list(dict.fromkeys(
            [*previous_citations, *delta_citations]
        )), None
    if restrictive and not changed:
        return delta_answer, list(delta_citations), (
            "Earlier claims were omitted because their applicability to the new constraint could not be verified."
        )
    retained = [s for s in old_sentences if s not in changed]
    answer = " ".join([*retained, delta_answer]).strip()
    citations = list(dict.fromkeys([*(previous_citations if retained else []), *delta_citations]))
    return answer, citations, None


def present_previous_answer(answer: str, request: str) -> str:
    """Handle the common presentation-only follow-up without another corpus search."""
    if re.search(r"\b(?:two|2)\s+bullets?\b", request, re.I):
        sentences = [s.strip() for s in SENTENCE_RE.split(answer.strip()) if s.strip()]
        if len(sentences) >= 2:
            return f"- {sentences[0]}\n- {' '.join(sentences[1:])}"
        return f"- {answer.strip()}"
    return answer


def focus_refinement_query(delta_query: str, previous_answer: str) -> tuple[str, set[str]]:
    """Use terms newly introduced by the constraint when selecting delta evidence."""
    old_terms = content_terms(previous_answer)
    operators = {"only", "instead", "actually", "rather", "except"}
    novel = [
        token for token in TOKEN_RE.findall(delta_query)
        if token.lower() not in old_terms
        and token.lower() not in CONTENT_STOPWORDS
        and token.lower() not in operators
    ]
    anchors = {
        token.lower() for token in novel
        if len(token) > 2 and sum(char.isupper() for char in token) >= 2
    }
    return (" ".join(novel) or delta_query), anchors


def needs_clarification(
    query: str, context_turns: tuple[tuple[str, str], ...] = ()
) -> bool:
    """Detect unresolved references cheaply, before attempting an answer.

    This intentionally targets high-precision conversational ambiguity. It does not
    try to classify general answerability, which remains an evidence decision.
    """
    context = " ".join(text for _, text in context_turns[-2:])
    match = AMBIGUOUS_REFERENCE_RE.search(query)
    if match:
        if not context.strip():
            return True
        context_terms = content_terms(context)
        head_terms = content_terms(match.group("head") or match.group("the_head") or "")
        singular = lambda term: term[:-3] + "y" if term.endswith("ies") else term.rstrip("s")
        normalized_context = {singular(term) for term in context_terms}
        normalized_head = {singular(term) for term in head_terms}
        # A generic facility noun need not be repeated verbatim when its
        # identifying modifier is present (camp facilities -> camping).
        if "camp" in normalized_head and {"camping", "campground"} & normalized_context:
            normalized_context.add("camp")
        if "facility" in normalized_head and len(normalized_head) > 1:
            normalized_head.remove("facility")
        # A generic overlap (e.g. "court") must not resolve a referent qualified by
        # a different name (e.g. "Superior Court"). Require all identifying terms.
        return not normalized_head.issubset(normalized_context)

    # A bare pronoun with no conversational antecedent is not a grounded query.
    # With context, require a topical bridge; unrelated dialogue is not enough.
    pronoun = UNRESOLVED_PRONOUN_RE.search(query)
    if pronoun:
        # Coordinated questions often introduce an antecedent earlier in the
        # same utterance ("...cloud storage and how do I configure it?").
        # Two content terms before the pronoun are a conservative local bridge.
        local_antecedent = len(content_terms(query[:pronoun.start()])) >= 2
        if not local_antecedent:
            query_terms = content_terms(query)
            context_terms = content_terms(context)
            if not context_terms or not (query_terms & context_terms):
                # Generic follow-ups may share no words with their antecedent.
                # Resolve only if the immediate user/agent pair agrees on at
                # least two topical terms, and the new words name no new entity.
                last = context_turns[-2:]
                generic_actions = {"configure", "configured", "work", "works", "provide", "online", "services", "service", "use", "used", "apply", "application", "how", "setting", "settings", "policies", "policy"}
                bridge = (
                    len(last) == 2
                    and last[0][0] == "user" and last[1][0] in {"agent", "assistant"}
                    and (
                        len(content_terms(last[0][1]) & content_terms(last[1][1])) >= 2
                        or (0 < len(content_terms(last[0][1])) <= 2
                            and content_terms(last[0][1]).issubset(content_terms(last[1][1])))
                        or any(
                            token in TOKEN_RE.findall(last[1][1])
                            for token in TOKEN_RE.findall(last[0][1])
                            if len(token) >= 3 and token.isupper()
                        )
                    )
                    and query_terms.issubset(generic_actions)
                )
                if not bridge:
                    return True

    # "SDK" without a product/vendor/context is materially ambiguous; retrieval
    # otherwise tends to pick a plausible but arbitrary vendor's documentation.
    if re.search(r"\bsdk\b", query, re.IGNORECASE):
        identifying_terms = content_terms(query) - {"sdk", "best", "practices"}
        if not identifying_terms and not context.strip():
            return True

    if not match:
        return False
    return False


def contextualize_query(
    query: str, context_turns: tuple[tuple[str, str], ...] = ()
) -> str:
    """Resolve explicit follow-up references with the last two dialogue turns."""
    if not context_turns or not CONTEXT_DEPENDENT_RE.search(query):
        return query
    # Keep only the immediate conversational neighborhood. Older turns add
    # prompt noise and can overwhelm the active question during retrieval.
    context = [
        f"{speaker}: {text.strip()}"
        for speaker, text in context_turns[-2:]
        if text.strip()
    ]
    if not context:
        return query
    # Never slice the serialized dialogue: that silently removes its speaker or
    # entity anchors and can even truncate the active ASR question. Keep complete
    # turns here; the provider/index tokenizer owns its explicit input budget.
    return "Conversation context: " + " ".join(context) + " Current question: " + query


def extract_evidence(
    query: str,
    passages: list[RetrievedPassage],
    max_spans: int = 4,
    required_terms: set[str] | None = None,
) -> list[tuple[str, str]]:
    query_terms = content_terms(query)
    spans: list[tuple[float, int, str, str]] = []
    for passage_rank, item in enumerate(passages):
        for sentence in SENTENCE_RE.split(item.passage.text.strip()):
            clean = " ".join(sentence.split())
            if not clean:
                continue
            if required_terms and clean.endswith("?"):
                continue
            sentence_terms = content_terms(clean)
            if required_terms and not (required_terms & sentence_terms):
                continue
            overlap = len(query_terms & sentence_terms)
            coverage = overlap / max(len(query_terms), 1)
            if overlap >= 2 or (overlap == 1 and coverage >= 0.34):
                spans.append(
                    (
                        coverage + min(item.score, 1.0) * 0.05,
                        passage_rank,
                        item.passage.id,
                        clean,
                    )
                )
    spans.sort(key=lambda row: row[0], reverse=True)

    # Preserve retrieval diversity: one supported sentence from each of the top
    # passages is safer than four near-duplicate sentences from a single document.
    selected: list[tuple[float, int, str, str]] = []
    selected_texts: set[str] = set()
    for passage_rank in range(min(3, len(passages))):
        best = next((row for row in spans if row[1] == passage_rank and row[3] not in selected_texts), None)
        if best is not None:
            selected.append(best)
            selected_texts.add(best[3])
    for row in spans:
        if len(selected) >= max_spans:
            break
        if row[3] not in selected_texts:
            selected.append(row)
            selected_texts.add(row[3])
    return [(pid, sentence) for _, _, pid, sentence in selected[:max_spans]]


def route_to_provider(
    query: str,
    passages: list[RetrievedPassage],
    *,
    multi_intent: bool = False,
) -> tuple[bool, str]:
    """Route only queries that benefit from model-based evidence selection.

    Direct, well-supported fact lookups stay extractive and incur no provider
    request. Compound/contextual/relational or locally unsupported questions can
    use the provider as an evidence checker. No retrieved evidence means no
    provider call: the answer path should abstain locally.
    """
    if not passages:
        return False, "No passages retrieved; abstain locally without an LLM call."
    if multi_intent and re.search(r"\band\s+(?:what|how|why|who|when|where)\b", query, re.I):
        return True, "Compound request; check each intent against cited evidence."
    if PROVIDER_COMPLEXITY_RE.search(query):
        return True, "Reasoning, comparison, or explanation wording needs evidence selection."
    if CONTEXT_DEPENDENT_RE.search(query):
        return True, "Context-dependent wording needs antecedent-aware evidence selection."
    if not direct_evidence(query, passages):
        return True, "Retrieved passages lack a direct local evidence match; check before abstaining."
    return False, "Direct factual query has extractive support; no LLM call is needed."


def supports_definition(query: str, sentence: str) -> bool:
    """A topic mention alone cannot answer an explicit definition request."""
    match = re.match(r"\s*(?:what\s+(?:is|are)|define)\s+(.+?)[?.]*$", query, re.I)
    if match is None:
        match = re.search(r"\b(?:understand|explain)\s+what\s+(.+?)\s+(?:is|are)[?.]*$", query, re.I)
    if match is None:
        return True
    # Enumerating named types/concepts is not asking for a copular definition
    # of the entire question ("the other three ... concepts").
    if re.match(r"(?:the\s+)?(?:other\s+)?(?:two|three|four|\d+)\b", match.group(1), re.I):
        return True
    # "What is the release date/cost/name ..." asks for an attribute,
    # not a definition. Do not demand a copular definition for these lookups.
    if re.match(
        r"(?:the\s+)?(?:name|date|release\s+date|cost|price|default|number|"
        r"difference|purpose|reason|process|procedure|status|version|timeout|"
        r"address|phone|limit|maximum|minimum|recommended)\b",
        match.group(1), re.I,
    ):
        return True
    topic_text = re.sub(r"\s+in\s+(?:an?\s+)?(?:api\s+request|request|response)\s*$", "", match.group(1), flags=re.I)
    topic = content_terms(topic_text)
    named = re.search(r"\b(?:called|known as|termed)\s+(.+?)(?:[.!?]|$)", sentence, re.I)
    topic_name = " ".join(t.lower() for t in TOKEN_RE.findall(topic_text) if t.lower() in topic)
    if topic and named and re.search(r"(?:^|\s)" + re.escape(topic_name) + r"$", named.group(1).strip(), re.I):
        return True
    copula = re.search(r"\b(?:is|are|means|refers to|defined as|specifies|defines|represents|consists of)\b", sentence, re.I)
    if topic and topic.issubset(content_terms(sentence)) and re.search(
        r"\b(?:vehicles|robots|devices|tools|systems|programs)\b.{0,60}\bcalled\b", sentence, re.I
    ):
        return True
    if not topic or copula is None:
        return False
    subject = sentence[:copula.start()]
    return len(subject.split()) <= 12 and topic.issubset(content_terms(subject))


def direct_evidence(query: str, passages: list[RetrievedPassage]) -> list[tuple[str, str]]:
    """Conservative local support for automatic routing, excluding question headings."""
    steps = procedural_evidence(query, passages)
    if steps:
        return steps
    terms = content_terms(query)
    name_lookup = bool(re.fullmatch(
        r"what\s+is\s+(?:the\s+)?name\s+of\s+[^?]+[?.]*", query.strip(), re.I,
    )) and not re.search(r"\b(?:and|or)\b", query, re.I)
    if name_lookup:
        terms -= {"name", "our", "my"}
    supported = []
    for passage_id, sentence in extract_evidence(query, passages, max_spans=12):
        if sentence.endswith("?") or not is_answer_span(sentence):
            continue
        if len(terms & content_terms(sentence)) / max(len(terms), 1) < 0.8:
            continue
        if not supports_definition(query, sentence):
            continue
        if name_lookup and not re.search(r"\b(?:called|named|name\s+is)\b", sentence, re.I):
            continue
        question = query.rsplit(" Current question:", 1)[-1].strip()
        if re.search(r"\b(?:pay|paid|payment|payments)\b", question, re.I) and not re.search(
            r"\b(?:pay|paid|payment|payments|cash|credit|debit|card|fare|fares|ticket|tickets)\b", sentence, re.I
        ):
            continue
        dates = re.findall(r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b", question, re.I)
        if any(date.casefold() not in sentence.casefold() for date in dates):
            continue
        supported.append((passage_id, sentence))
    return supported


def is_answer_span(sentence: str) -> bool:
    """Reject navigation and link stubs even if they mention the query topic.

    This leaves the corpus unchanged. Long menu-contaminated text is routed to
    evidence selection instead of being emitted as a supposedly direct answer.
    """
    markers = ("contact us", "site map", "skip to content", "skip to main content", "advanced search")
    lowered = sentence.casefold()
    return (
        len(sentence.split()) >= 4
        and not sentence.rstrip().endswith(("?", ":"))
        and sum(marker in lowered for marker in markers) < 2
        and not re.search(r"\b(?:for more information|for .+ record),?\s*see\s*:?$", sentence, re.I)
    )


def procedural_evidence(query: str, passages: list[RetrievedPassage]) -> list[tuple[str, str]]:
    """Exact numbered headings for a specifically named configuration procedure.

    Only one source is used, with all query topic terms present and at least two
    sequential steps. Generic follow-ups continue through provider validation.
    """
    question = query.rsplit(" Current question:", 1)[-1].strip()
    match = re.fullmatch(r"how\s+(?:do|can|should)\s+(?:i|we|you)\s+(?:configure|install|enable)\s+(.+?)[?.]*", question, re.I)
    if not match:
        return []
    topic = content_terms(match.group(1))
    if len(topic) < 2:
        return []
    for item in passages:
        if not topic.issubset(content_terms(item.passage.text)):
            continue
        steps = re.findall(r"(?m)^\s*(Step\s+(\d+)\.\s+[^\n]+)", item.passage.text)
        if len(steps) < 2 or [int(number) for _, number in steps] != list(range(1, len(steps) + 1)):
            continue
        return [(item.passage.id, " ".join(text.split())) for text, _ in steps]
    return []


def select_answer_evidence(query: str, spans: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Prefer an explicit naming assertion for a narrow name lookup.

    Do not truncate procedures, comparisons, or compound questions. Preserve an
    entire verbatim sentence, including its qualifications, and its source ID.
    """
    spans = deduplicate_evidence(spans)
    question = query.rsplit(" Current question:", 1)[-1].strip()
    match = re.fullmatch(r"what\s+is\s+(?:the\s+)?name\s+of\s+(.+?)[?.]*", question, re.I)
    if not match or re.search(r"\b(?:and|or)\b", match.group(1), re.I):
        return spans
    topic = content_terms(match.group(1)) - {"our", "my"}
    candidates = [
        (pid, text) for pid, text in spans
        if topic and topic.issubset(content_terms(text))
        and re.search(r"\b(?:called|named|name\s+is)\b", text, re.I)
        and not text.endswith("?")
    ]
    if not candidates:
        return spans
    # A disagreement must remain visible rather than silently selecting a name.
    names = {
        re.split(r"\b(?:called|named|name\s+is)\b", text, flags=re.I)[-1].strip().lower()
        for _, text in candidates
    }
    return [min(candidates, key=lambda span: len(span[1]))] if len(names) == 1 else candidates


def synthesize_answer(
    query: str,
    passages: list[RetrievedPassage],
    *,
    min_relevance_score: float | None = None,
    required_terms: set[str] | None = None,
    require_direct_support: bool = False,
) -> tuple[str, list[str], str | None]:
    if (
        min_relevance_score is not None
        and (not passages or passages[0].score < min_relevance_score)
    ):
        return (
            "I do not have enough information in the retrieved corpus to answer that.",
            [],
            "The strongest retrieved passage was below the relevance threshold.",
        )
    evidence = (
        direct_evidence(query, passages) if require_direct_support else
        extract_evidence(query, passages, required_terms=required_terms)
    )
    if not evidence:
        return (
            "I do not have enough information in the retrieved corpus to answer that.",
            [],
            "No supporting evidence spans were found.",
        )
    citations = []
    answer_parts = []
    for passage_id, sentence in select_answer_evidence(query, evidence):
        citations.append(passage_id)
        answer_parts.append(sentence)
    return " ".join(answer_parts), list(dict.fromkeys(citations)), None
