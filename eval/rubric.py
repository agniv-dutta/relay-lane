"""Auditable rubric for eval grading."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


CITATION_RE = re.compile(r"\[\s*([^\],]+?)\s*,\s*(\d+)\s*,\s*([^\]]+?)\s*\]")


@dataclass
class GradeResult:
    answer_grade: str
    citation_ok: bool
    refusal_ok: bool
    source_ok: bool
    manual_review: bool
    notes: list[str] = field(default_factory=list)


def _is_refusal(answer: str) -> bool:
    a = (answer or "").lower()
    markers = ["do not cover", "cannot answer", "not cover", "insufficient", "try again later", "daily request budget"]
    return any(m in a for m in markers)


def grade(question: dict, final_answer: str | None, refused: bool, retrieved_chunks: list[dict]) -> GradeResult:
    answerable = bool(question.get("answerable", True))
    answer = final_answer or ""
    notes = []

    # citation check: every citation must exist in retrieved chunks
    chunk_ids = {c.get("chunk_id") for c in retrieved_chunks}
    citations = CITATION_RE.findall(answer)
    citation_ok = True
    for paper, page, cid in citations:
        cid = cid.strip()
        if cid not in chunk_ids:
            citation_ok = False
            notes.append(f"fabricated/unknown citation: {cid}")
    if not citations and len(answer.split()) > 15 and not refused:
        # factual answer without citations
        citation_ok = False
        notes.append("no citations in non-trivial answer")

    # refusal check
    actual_refusal = refused or _is_refusal(answer)
    if answerable:
        refusal_ok = not actual_refusal
        if actual_refusal:
            notes.append("refused an answerable question")
    else:
        refusal_ok = actual_refusal
        if not actual_refusal:
            notes.append("answered an unanswerable question")

    # source check
    expected_source = question.get("expected_source")
    source_ok = True
    if answerable and expected_source:
        try:
            exp_paper, exp_page = expected_source.split("/")
            exp_page = int(exp_page)
        except (ValueError, AttributeError):
            exp_paper, exp_page = None, None
        if exp_paper is not None:
            matched = any(
                c.get("paper_id") == exp_paper and int(c.get("page", -1)) == exp_page
                for c in retrieved_chunks
            )
            # also check citation in answer mentions that paper/page
            cited_match = any(p.strip() == exp_paper and int(pg) == exp_page for p, pg, _ in citations)
            if not (matched or cited_match):
                source_ok = False
                notes.append(f"expected source {expected_source} not retrieved/cited")

    # answer_grade
    if answerable:
        if refusal_ok and citation_ok and source_ok:
            answer_grade = "correct"
        else:
            answer_grade = "wrong"
    else:
        if refusal_ok:
            answer_grade = "refused_correctly"
        else:
            answer_grade = "refused_wrongly"

    # manual review flag
    manual_review = (not citation_ok) or (not refusal_ok) or (not source_ok) or answer_grade in ("wrong", "refused_wrongly")
    # also flag correct answers for spot check? keep false for correct
    if answer_grade == "correct":
        manual_review = False

    return GradeResult(
        answer_grade=answer_grade,
        citation_ok=citation_ok,
        refusal_ok=refusal_ok,
        source_ok=source_ok,
        manual_review=manual_review,
        notes=notes,
    )
