from __future__ import annotations

from .controller import RuleBasedRetrievalController, TranscriptChunk
from .models import RagResponse, RetrievalEvent
from .retrieval import HybridRetriever
from .synthesis import synthesize_answer


class StreamingRagPipeline:
    def __init__(self, retriever: HybridRetriever, controller: RuleBasedRetrievalController | None = None) -> None:
        self.retriever = retriever
        self.controller = controller or RuleBasedRetrievalController()

    def run(self, chunks: list[TranscriptChunk], domain: str = "cloud") -> RagResponse:
        response = RagResponse()
        retrieved_for = None
        passages = []
        for chunk in chunks:
            decision = self.controller.decide(chunk)
            if decision != "Retrieve" or retrieved_for == chunk.text:
                continue
            retrieved_for = chunk.text
            response.retrieval_events.append(
                RetrievalEvent(timestamp_s=chunk.timestamp_s, query=chunk.text, trigger="provisional" if not chunk.is_final else "final")
            )
            response.sub_queries = [chunk.text]
            passages = self.retriever.search(chunk.text, domain=domain)
            if passages:
                break
        if not response.sub_queries and chunks:
            response.sub_queries = [chunks[-1].text]
        answer, citations, uncertainty = synthesize_answer(response.sub_queries[-1], passages)
        response.answer = answer
        response.citations = citations
        response.uncertainty = uncertainty
        return response

