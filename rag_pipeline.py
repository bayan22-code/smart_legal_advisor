import json
import re
import time
from typing import Any

from config import AMBIGUITY_ROUTE
from openrouter_client import chat_completion
from retriever import HybridRetriever
from reranker import LegalReranker
from text_normalization import clean_text

try:
    from langchain_core.runnables import RunnablePassthrough
except Exception:  # pragma: no cover
    RunnablePassthrough = None


LEGAL_SYNTH_PROMPT = """أنت مستشار قانوني سوري خبير. هدفك تقديم استشارة قانونية بشرية ومتعاطفة باللغة العربية الفصحى.
التزم فقط بالنصوص المسترجعة ولا تختلق أي مادة قانونية.
حلل الوقائع أولاً، ثم اربطها بالنصوص بصورة طبيعية.
إذا كانت الوقائع ناقصة، اطلب أسئلة استيضاحية موجزة قبل الجزم.
عند وجود أبعاد مدنية/أحوال شخصية/جزائية، قدّم رؤية متكاملة.

السؤال القانوني المنقح: {user_question}
السياق القانوني: {retrieved_articles}

أجب بالعربية الفصحى بهذا الهيكل:
1) فهم الوقائع
2) التكييف القانوني
3) النصوص المنطبقة (مع ذكر القانون والمادة بصياغة طبيعية)
4) النتيجة القانونية المتوقعة
5) خطوات عملية مقترحة للمستخدم
"""

QUERY_EXPANSION_PROMPT = """أنشئ 3 صيغ بحث قانونية عربية بديلة للاستعلام.
- وسّع المصطلحات القانونية بدون تضييق مفرط.
- استخدم ألفاظاً محتملة من القانون السوري.
أعد JSON فقط بهذا المفتاح:
query_variations

query: {query}
"""

ROUTER_PROMPT = """صنّف الاستعلام القانوني السوري التالي إلى أحد المسارات:
- civil
- penal
- personal_status
- all

إذا كان غامضاً أو تنقصه تفاصيل حرجة أعد route = "clarify" مع سؤال توضيحي.
أعد JSON فقط بالمفاتيح:
route, reason, clarifying_question

query: {query}
"""

QUERY_REFINER_PROMPT = """حوّل إدخال المستخدم إلى استعلام قانوني عربي فصيح وقصير.
- صحّح الأخطاء الإملائية والعامية.
- إذا كان النص قصة، استخرج جوهر النزاع القانوني.
- لا تخترع وقائع.
أعد JSON فقط بالمفاتيح:
refined_query, legal_core, needs_more_details

query: {query}
"""


class AgenticRAGPipeline:
    def __init__(self, retriever: HybridRetriever, reranker: LegalReranker):
        self.retriever = retriever
        self.reranker = reranker

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}

    @staticmethod
    def _clean_retrieved_text(text: str) -> str:
        if not text:
            return ""
        cleaned = text.replace("\r", "\n")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        noisy_patterns = [
            r"^الصفحة\s+\d+.*$",
            r"^Page\s+\d+.*$",
            r"^جميع الحقوق محفوظة.*$",
            r"^تم التحميل من.*$",
            r"^هذا النص للاطلاع.*$",
        ]
        lines = []
        for line in cleaned.split("\n"):
            ln = line.strip()
            if not ln:
                continue
            if any(re.match(p, ln, flags=re.IGNORECASE) for p in noisy_patterns):
                continue
            lines.append(ln)
        return "\n".join(lines).strip()

    @staticmethod
    def _format_context(chunks: list[dict[str, Any]]) -> str:
        parts = []
        for c in chunks[:2]:
            meta = c.get("metadata", {})
            art = meta.get("article_number", "?")
            law = meta.get("law_name", meta.get("source", "القانون السوري"))
            short_text = AgenticRAGPipeline._clean_retrieved_text(c.get("text", ""))[:700]
            parts.append(f"[{law} - المادة {art}]:\n{short_text}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_retrieved_law_section(chunks: list[dict[str, Any]], limit: int = 2) -> str:
        """Create a mandatory section with exact retrieved legal text."""
        if not chunks:
            return "نص القانون المسترجع:\nلم يتم العثور على نصوص قانونية."
        section_lines: list[str] = ["نص القانون المسترجع:"]
        for chunk in chunks[:limit]:
            meta = chunk.get("metadata", {})
            article = meta.get("article_number", "?")
            raw_text = AgenticRAGPipeline._clean_retrieved_text(
                (chunk.get("page_content", chunk.get("text", "")) or "").strip()
            )
            section_lines.append(f"- المادة {article}: {raw_text}")
        return "\n".join(section_lines)

    def expand_query(self, question: str) -> list[str]:
        def _run_refiner(payload: dict[str, str]) -> list[str]:
            normalized_input_query = clean_text(payload["query"], apply_reversal_fix=False)
            response = chat_completion(
                [
                    {"role": "system", "content": "You produce strict JSON only."},
                    {"role": "user", "content": QUERY_REFINER_PROMPT.format(query=normalized_input_query)},
                ],
                temperature=0.1,
            )
            parsed = self._extract_json(response)
            refined = parsed.get("refined_query") or parsed.get("legal_core")
            refined_query = clean_text(str(refined) if refined else normalized_input_query, apply_reversal_fix=False)
            # Expand into 3 alternative legal search queries.
            fallback = chat_completion(
                [
                    {"role": "system", "content": "You produce strict JSON only."},
                    {"role": "user", "content": QUERY_EXPANSION_PROMPT.format(query=refined_query)},
                ],
                temperature=0.1,
            )
            fallback_parsed = self._extract_json(fallback)
            variations = fallback_parsed.get("query_variations", [])
            if isinstance(variations, str):
                variations = [variations]
            if not isinstance(variations, list):
                variations = []
            cleaned = [clean_text(str(v).strip(), apply_reversal_fix=False) for v in variations if str(v).strip()]
            # Ensure at least one query always exists.
            unique_queries = []
            for q in [refined_query, *cleaned]:
                if q and q not in unique_queries:
                    unique_queries.append(q)
            return unique_queries[:3]

        if RunnablePassthrough is not None:
            chain = RunnablePassthrough.assign(refined_query=lambda x: _run_refiner(x))
            out = chain.invoke({"query": question})
            return out.get("refined_query", [question])
        return _run_refiner({"query": question})

    def decide_route(self, question: str) -> dict[str, str]:
        response = chat_completion(
            [
                {"role": "system", "content": "You are a legal routing controller. Output JSON only."},
                {"role": "user", "content": ROUTER_PROMPT.format(query=question)},
            ],
            temperature=0.0,
        )
        parsed = self._extract_json(response)
        route = parsed.get("route", "all")
        if route not in {"civil", "penal", "personal_status", "all", AMBIGUITY_ROUTE}:
            route = "all"
        return {
            "route": route,
            "reason": parsed.get("reason", ""),
            "clarifying_question": parsed.get("clarifying_question", "هل يمكنك توضيح نوع القضية بشكل أدق؟"),
        }

    def retrieve_for_query(self, expanded_query: list[str] | str, route: str) -> list[dict[str, Any]]:
        normalized_query = (
            [clean_text(q, apply_reversal_fix=False) for q in expanded_query]
            if isinstance(expanded_query, list)
            else clean_text(expanded_query, apply_reversal_fix=False)
        )
        return self.retriever.retrieve(normalized_query, database_route=route)

    def synthesize_answer(self, question: str, relevant: list[dict[str, Any]]) -> tuple[str, list[int]]:
        if not relevant:
            return (
                "بصفتي محامياً سورياً، لم أجد مادة مطابقة في النتائج الحالية. "
                "قد يكون النص القانوني في باب آخر من القانون السوري. "
                "يرجى تزويدي بتفاصيل إضافية عن الواقعة (الزمان، المكان، الأطراف، والنتيجة)."
            ), []
        context = self._format_context(relevant)
        prompt = LEGAL_SYNTH_PROMPT.format(retrieved_articles=context, user_question=question)
        answer = chat_completion(
            [
                    {"role": "system", "content": "You are an expert Syrian Legal Consultant. Provide professional, empathetic Modern Standard Arabic legal advice using only retrieved articles from Syrian Penal, Personal Status, and Civil laws."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        law_section = self._build_retrieved_law_section(relevant)
        if "نص القانون المسترجع" not in answer:
            answer = f"{answer}\n\n{law_section}"
        citations = sorted({int(m.group(1)) for m in re.finditer(r"(?:المادة|Article)\s*(\d+)", answer, re.IGNORECASE)})
        if not citations:
            answer = (
                answer
                + "\n\nلم تظهر مادة قانونية محددة في النتائج الحالية. "
                "يرجى توضيح الوقائع (الزمن/المكان/الأطراف) للحصول على مواد أدق."
            )
        return answer, citations

    def run(self, question: str) -> dict[str, Any]:
        stage_times: dict[str, float] = {}
        started = time.perf_counter()

        t0 = time.perf_counter()
        expanded = self.expand_query(question)
        stage_times["query_expansion"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        route_decision = self.decide_route(question)
        stage_times["route_decision"] = time.perf_counter() - t0

        if route_decision["route"] == AMBIGUITY_ROUTE:
            return {
                "answer": route_decision["clarifying_question"],
                "is_clarification": True,
                "route": route_decision["route"],
                "route_reason": route_decision["reason"],
                "expanded_queries": expanded,
                "relevant_articles": [],
                "citations": [],
                "stage_times": stage_times,
                "total_latency": time.perf_counter() - started,
            }

        t0 = time.perf_counter()
        candidates = self.retrieve_for_query(expanded, route_decision["route"])
        stage_times["retrieval"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        relevant = self.reranker.rerank(question, candidates)
        stage_times["reranking"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        answer, citations = self.synthesize_answer(question, relevant)
        stage_times["synthesis"] = time.perf_counter() - t0

        return {
            "answer": answer,
            "is_clarification": False,
            "route": route_decision["route"],
            "route_reason": route_decision["reason"],
            "expanded_query": expanded[0] if isinstance(expanded, list) and expanded else str(expanded),
            "expanded_queries": expanded if isinstance(expanded, list) else [str(expanded)],
            "relevant_articles": relevant,
            "citations": citations,
            "stage_times": stage_times,
            "total_latency": time.perf_counter() - started,
        }
    