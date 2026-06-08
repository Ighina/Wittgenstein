"""Tests for boundary-aware chunking and chunk-aggregating verification."""

import pytest

from src.config import PipelineConfig
from src.models import SnippetType, VerificationSnippet, VerificationStatus
from src.utils.chunking import chunk_text
from src.verifiers.text_verifier import TextVerifier


class TestChunkText:
    def test_short_text_single_chunk(self):
        assert chunk_text("hello world", max_chars=200) == ["hello world"]

    def test_empty_returns_one_empty_chunk(self):
        assert chunk_text("", max_chars=200) == [""]

    def test_long_text_splits(self):
        text = "\n\n".join(f"Paragraph number {i} with some content." for i in range(50))
        chunks = chunk_text(text, max_chars=200, overlap=20)
        assert len(chunks) > 1
        # Each chunk is bounded (soft bound: overlap may add a little).
        assert all(len(c) <= 200 + 20 + 40 for c in chunks)

    def test_reassembles_content(self):
        text = "Alpha beta gamma. " * 40
        chunks = chunk_text(text, max_chars=120, overlap=0)
        joined = " ".join(chunks).replace("\n", " ")
        # Every original word survives somewhere.
        for word in ("Alpha", "beta", "gamma"):
            assert word in joined


class TestChunkAggregation:
    """Directly exercise BaseVerifier._analyze_in_chunks aggregation."""

    @pytest.fixture
    def verifier(self):
        cfg = PipelineConfig()
        cfg.llm.provider = "mock"
        cfg.verify_chunk_chars = 300  # force multi-chunk on the content below
        cfg.verify_chunk_overlap = 0
        return TextVerifier(config=cfg)

    @staticmethod
    def _multichunk_content(marker_para: str) -> str:
        clean = "Routine background prose sentence. " * 8
        return clean + "\n\n" + marker_para + "\n\n" + clean

    def test_error_chunk_wins_and_is_prefixed(self, verifier):
        content = self._multichunk_content("MARKER_BAD a load-bearing wrong step.")

        def analyze(chunk):
            if "MARKER_BAD" in chunk:
                return {"error_detected": True, "confidence": 0.9, "reasoning": "bad step"}
            return {"error_detected": False, "confidence": 0.5, "reasoning": "ok"}

        chosen, n_chunks, n_failed = verifier._analyze_in_chunks(content, analyze)
        assert n_chunks > 1
        assert chosen["error_detected"] is True
        assert chosen["confidence"] == 0.9
        assert chosen["reasoning"].startswith("[chunk")

    def test_one_failing_chunk_is_tolerated(self, verifier):
        content = self._multichunk_content("BOOM")

        def analyze(chunk):
            if "BOOM" in chunk:
                raise RuntimeError("empty LLM response")
            return {"error_detected": False, "confidence": 0.7, "reasoning": "ok"}

        chosen, n_chunks, n_failed = verifier._analyze_in_chunks(content, analyze)
        assert n_failed >= 1
        assert chosen is not None  # surviving chunks still produce a verdict

    def test_all_chunks_failing_returns_none(self, verifier):
        def analyze(chunk):
            raise RuntimeError("empty LLM response")

        chosen, n_chunks, n_failed = verifier._analyze_in_chunks("word " * 400, analyze)
        assert chosen is None
        assert n_failed == n_chunks


class TestChunkedTextVerifierEndToEnd:
    @pytest.fixture
    def mock_config(self):
        cfg = PipelineConfig()
        cfg.llm.provider = "mock"
        cfg.llm.num_workers = 1
        cfg.verify_chunk_chars = 300
        return cfg

    def test_long_snippet_runs_through_chunked_path(self, mock_config):
        content = "A routine paragraph of prose. " * 60  # multi-chunk
        snip = VerificationSnippet(
            snippet_id="p_sec_0", snippet_type=SnippetType.SECTION, paper_id="p",
            location="Methods", content=content,
        )
        r = TextVerifier(config=mock_config).verify(snip)
        # Mock default is no-error; the point is the chunked path returns cleanly.
        assert r.status in (VerificationStatus.NO_ERROR, VerificationStatus.ERROR_DETECTED)
        assert r.snippet_id == "p_sec_0"

    def test_short_snippet_not_prefixed(self, mock_config):
        snip = VerificationSnippet(
            snippet_id="p_sec_1", snippet_type=SnippetType.SECTION, paper_id="p",
            location="Intro", content="A short, clean sentence.",
        )
        r = TextVerifier(config=mock_config).verify(snip)
        assert "chunk" not in r.reasoning
