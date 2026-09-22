from __future__ import annotations

from .controller import RuleBasedRetrievalController, TranscriptChunk
from .llm_steps import ChatClient, provider_synthesize_answer, rewrite_query
from .models import RagResponse, RetrievalEvent
from .providers import ProviderError
from .retrieval import HybridRetriever
from .synthesis import synthesize_answer


class StreamingRagPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        controller: RuleBasedRetrievalController | None = None,
        *,
        synthesis_mode: str = "deterministic",
        evidence_client: ChatClient | None = None,
        generation_client: ChatClient | None = None,
        enable_query_rewrite: bool = False,
        refine_on_final: bool = False,
    ) -> None:
        self.retriever = retriever
        self.controller = controller or RuleBasedRetrievalController()
        self.synthesis_mode = synthesis_mode
        self.evidence_client = evidence_client
        self.generation_client = generation_client
        self.enable_query_rewrite = enable_query_rewrite
        self.refine_on_final = refine_on_final

    def run(self, chunks: list[TranscriptChunk], domain: str = "cloud") -> RagResponse:
        response = RagResponse()
        retrieved_for = None
        passages = []
        active_query = None
        for chunk in chunks:
            decision = self.controller.decide(chunk)
            if decision != "Retrieve" or retrieved_for == chunk.text:
                continue
            retrieved_for = chunk.text
            response.retrieval_events.append(
                RetrievalEvent(timestamp_s=chunk.timestamp_s, query=chunk.text, trigger="provisional" if not chunk.is_final else "final")
            )
            retrieval_query = chunk.text
            if self.enable_query_rewrite and self.evidence_client is not None:
                try:
                    retrieval_query = rewrite_query(self.evidence_client, chunk.text)
                except (ProviderError, ValueError):
                    retrieval_query = chunk.text
            response.sub_queries.append(retrieval_query)
            candidate_passages = self.retriever.search(retrieval_query, domain=domain)
            if candidate_passages or not passages:
                passages = candidate_passages
                active_query = retrieval_query
            if chunk.is_final or (candidate_passages and not self.refine_on_final):
                break
        if not response.sub_queries and chunks:
            response.sub_queries = [chunks[-1].text]
            active_query = chunks[-1].text
        query_for_answer = active_query or response.sub_queries[-1]
        if self.synthesis_mode == "provider":
            answer, citations, uncertainty = provider_synthesize_answer(
                self.evidence_client,
                self.generation_client,
                query_for_answer,
                passages,
            )
        else:
            answer, citations, uncertainty = synthesize_answer(query_for_answer, passages)
        response.answer = answer
        response.citations = citations
        response.uncertainty = uncertainty
        return response
