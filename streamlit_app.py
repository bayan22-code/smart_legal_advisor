"""Streamlit interface for the upgraded Agentic Arabic Legal RAG."""

import re
import time
from pathlib import Path

import streamlit as st

from config import TOP_K_RERANK
from evaluation import append_eval_log, evaluate_with_ragas
from ingest import load_bm25_index
from rag_pipeline import AgenticRAGPipeline
from retriever import HybridRetriever
from reranker import LegalReranker
from embeddings import ArabicEmbedder
from bm25_search import BM25Index


def apply_rtl_styles():
    """Apply RTL and Arabic-friendly styles."""
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri&display=swap');
    * { direction: rtl !important; text-align: right !important; font-family: 'Amiri', serif !important; }
    div[data-testid="stExpander"] { direction: rtl !important; text-align: right !important; }
</style>
""", unsafe_allow_html=True)
    st.markdown('<style>div[data-testid="stMarkdownContainer"] {direction: rtl; text-align: right;}</style>', unsafe_allow_html=True)
    st.markdown("""
    <style>
    /* RTL support */
    [data-testid="stAppViewContainer"] {
        direction: rtl;
        text-align: right;
    }
    [data-testid="stChatMessage"] {
        direction: rtl;
        text-align: right;
    }
    .stChatInput {
        direction: rtl;
    }
    .stTextInput input {
        direction: rtl !important;
        text-align: right !important;
    }
    .stChatInput textarea {
        direction: rtl !important;
        text-align: right !important;
    }
    /* Arabic font */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Arial', 'Tahoma', sans-serif;
    }
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .legal-icon {
        font-size: 2rem;
    }
    .rtl-output {
        direction: rtl;
        text-align: right;
        unicode-bidi: bidi-override;
        line-height: 1.8;
    }
    </style>
    """, unsafe_allow_html=True)


def create_pipeline():
    """Create cached agentic RAG pipeline with persisted BM25 index."""
    embedder = ArabicEmbedder()
    bm25 = BM25Index()
    load_bm25_index(bm25)
    retriever = HybridRetriever(embedder, bm25)
    reranker = LegalReranker()
    return AgenticRAGPipeline(retriever, reranker)


@st.cache_resource
def get_pipeline():
    """Cached pipeline instance."""
    return create_pipeline()


def format_citations(citations: list[int]) -> str:
    """Format citations as (المادة N) for display."""
    return "، ".join(f"(المادة {n})" for n in sorted(citations))


def clean_arabic_text(text: str) -> str:
    """Lightweight encoding cleanup while preserving raw Arabic order."""
    if text is None:
        return ""
    return str(text).replace("\ufeff", "").strip()


def clean_retrieved_display_text(text: str) -> str:
    """Remove non-legal boilerplate from retrieved article display."""
    raw = clean_arabic_text(text).replace("\r", "\n")
    noisy_patterns = [
        r"^الصفحة\s+\d+.*$",
        r"^Page\s+\d+.*$",
        r"^جميع الحقوق محفوظة.*$",
        r"^تم التحميل من.*$",
        r"^هذا النص للاطلاع.*$",
    ]
    lines = []
    for line in raw.split("\n"):
        ln = line.strip()
        if not ln:
            continue
        if any(re.match(p, ln, re.IGNORECASE) for p in noisy_patterns):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def display_arabic(text: str) -> str:
    """Keep Arabic in logical order; rely on RTL container for visual direction."""
    if not text:
        return text
    raw = clean_arabic_text(text)
    return raw


def rtl_container(text: str, already_shaped: bool = False) -> str:
    rendered = text if already_shaped else display_arabic(text)
    return f'<div style="text-align: right; direction: rtl;">{rendered}</div>'


def extract_citations_from_text(answer: str) -> list[int]:
    return sorted({int(m.group(1)) for m in re.finditer(r"(?:المادة|Article)\s*(\d+)", answer, re.IGNORECASE)})


def build_context_window(messages: list[dict], latest_question: str, max_messages: int = 3) -> str:
    """Build lightweight memory using only last 3 messages."""
    turns = []
    for msg in messages[-max_messages:]:
        role = "المستخدم" if msg.get("role") == "user" else "المستشار"
        turns.append(f"{role}: {msg.get('content', '')}")
    turns.append(f"المستخدم: {latest_question}")
    return "\n".join(turns)


def highlight_article_mentions(answer: str) -> str:
    """Highlight English article mentions such as 'Article 155'."""
    return re.sub(
        r"(Article\s+\d+)",
        r"<mark>\1</mark>",
        answer,
        flags=re.IGNORECASE,
    )


def log_debug(step: str):
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    st.session_state.debug_logs.append(step)
    st.session_state.debug_logs = st.session_state.debug_logs[-20:]


def render_legal_sources(articles: list[dict], top_k: int = TOP_K_RERANK):
    """
    Render expander with legal sources. Shows article number and law name from metadata.
    Uses icons for each article.
    """
    if not articles:
        return
    display = articles[:top_k]
    with st.expander("المصادر القانونية المستخرجة", expanded=False):
        for i, art in enumerate(display, start=1):
            meta = art.get("metadata", {})
            article_num = meta.get("article_number", "?")
            part_index = int(meta.get("part_index", 0))
            total_parts = int(meta.get("total_parts", 1))
            text = clean_retrieved_display_text(art.get("page_content", art.get("text", "")))
            article_label = f"المادة رقم {article_num}"
            if total_parts > 1:
                article_label += f" (جزء {part_index + 1}/{total_parts})"
            st.markdown(rtl_container(article_label), unsafe_allow_html=True)
            st.markdown(rtl_container(text), unsafe_allow_html=True)
            if i < len(display):
                st.divider()


def main():
    st.set_page_config(
        page_title="المستشار القانوني الذكي",
        page_icon="⚖️",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    apply_rtl_styles()

    st.markdown("""
    <div class="main-header">
        <h1 class="legal-icon">⚖️ المستشار القانوني الذكي</h1>
        <p style="color: #666;">اسأل عن القانون السوري</p>
    </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "eval_logs" not in st.session_state:
        st.session_state.eval_logs = []
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = []
    if "selected_article_text" not in st.session_state:
        st.session_state.selected_article_text = ""
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    if "last_latency" not in st.session_state:
        st.session_state.last_latency = 0.0

    # Sidebar: clear chat
    with st.sidebar:
        st.markdown("### ⚙️ الإعدادات")
        if st.button("🗑️ مسح سجل المحادثة", use_container_width=True):
            st.session_state.messages = []
            st.session_state.eval_logs = []
            st.session_state.eval_results = []
            st.session_state.selected_article_text = ""
            st.session_state.debug_logs = []
            st.session_state.last_latency = 0.0
            st.rerun()

        st.markdown("### 🚦 Status")
        st.metric("Execution Time (s)", f"{st.session_state.last_latency:.2f}")
        st.markdown("**Debug Log**")
        if st.session_state.debug_logs:
            for line in st.session_state.debug_logs[-8:]:
                st.caption(f"- {line}")
        else:
            st.caption("No debug logs yet.")
        try:
            vector_db = get_pipeline().retriever
            st.write(f"Debug: Total chunks in DB: {vector_db.collection.count()}")
        except Exception as e:
            st.caption(f"Debug DB count unavailable: {e}")

        st.markdown("### 📊 Evaluation Dashboard")
        eval_source = st.session_state.eval_results or st.session_state.eval_logs
        if eval_source:
            latest = eval_source[-1]
            st.metric("Faithfulness", f"{latest['faithfulness']:.3f}")
            st.metric("Answer Relevancy", f"{latest['answer_relevancy']:.3f}")
            if not latest.get("used_ragas", True):
                st.caption("RAGAS runtime fallback used.")
                if latest.get("error"):
                    st.caption(f"Error: {latest['error']}")
        else:
            st.caption("No evaluations yet.")

        if st.session_state.selected_article_text:
            st.markdown("### 📘 نص المادة المختارة")
            st.markdown(rtl_container(st.session_state.selected_article_text), unsafe_allow_html=True)

    # Chat history: render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(rtl_container(msg["content"]), unsafe_allow_html=True)
            if msg["role"] == "assistant":
                if msg.get("refined_question") and msg["refined_question"] != msg.get("original_question", ""):
                    st.caption(display_arabic(f"السؤال المُصحَّح: {msg['refined_question']}"))
                if msg.get("relevant_articles"):
                    render_legal_sources(msg["relevant_articles"])
                if msg.get("citations"):
                    st.markdown(f"<div class='rtl-output'><b>الاستشهادات:</b> {display_arabic(format_citations(msg['citations']))}</div>", unsafe_allow_html=True)
                    for c in msg["citations"]:
                        if st.button(f"المادة {c}", key=f"hist-cit-{id(msg)}-{c}"):
                            article = next(
                                (
                                    a for a in msg.get("relevant_articles", [])
                                    if str(a.get("metadata", {}).get("article_number")) == str(c)
                                ),
                                None,
                            )
                            if article:
                                st.session_state.selected_article_text = article.get("page_content", article.get("text", ""))
                                st.rerun()

    bm25_path = Path(__file__).resolve().parent / "bm25_index.pkl"
    if not bm25_path.exists():
        st.error(
            "يرجى تشغيل السكربت ingest.py أولاً لوضع البيانات في قاعدة المعرفة.\n"
            "Run: cd legal_rag && python ingest.py"
        )
        st.stop()

    if prompt := st.chat_input("اطرح سؤالك هنا..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(rtl_container(prompt), unsafe_allow_html=True)

        with st.chat_message("assistant"):
            current_question = build_context_window(st.session_state.chat_history, prompt)
            relevant = []
            citations = []
            answer = ""
            stage_times = {}
            total_latency = 0.0
            route = "all"
            expanded_query = prompt
            eval_result = None
            try:
                pipeline = get_pipeline()

                # Pipeline Trace with st.status
                with st.status("جاري معالجة سؤالك...", expanded=True) as status:
                    # Step 1: Query expansion + route decision
                    step1 = st.container()
                    with step1:
                        st.markdown("**1️⃣ Refining Query (Query Expansion)...**")
                    status.update(label="Refining Query...", state="running")
                    log_debug("Step: Correcting Arabic Encoding...")
                    log_debug("Step: Refining query...")
                    t0 = time.perf_counter()
                    expanded_query = pipeline.expand_query(current_question)
                    stage_times["query_expansion"] = time.perf_counter() - t0
                    with step1:
                        st.success("تم توليد صياغة قانونية احترافية واحدة.")
                        display_query = expanded_query[0] if isinstance(expanded_query, list) and expanded_query else str(expanded_query)
                        st.markdown(rtl_container(f"الاستعلام القانوني المنقح: {display_query}"), unsafe_allow_html=True)
                        if isinstance(expanded_query, list) and len(expanded_query) > 1:
                            st.markdown(
                                rtl_container("بدائل البحث: " + " | ".join(expanded_query)),
                                unsafe_allow_html=True,
                            )

                    step1b = st.container()
                    with step1b:
                        st.markdown("**2️⃣ Agent Deciding Route...**")
                    status.update(label="Agent Deciding Route...", state="running")
                    log_debug("Step: Agent deciding route...")
                    t0 = time.perf_counter()
                    route_decision = pipeline.decide_route(current_question)
                    stage_times["route_decision"] = time.perf_counter() - t0
                    route = route_decision["route"]
                    with step1b:
                        st.info(f"المسار المختار: {route}")

                    if route == "clarify":
                        answer = route_decision["clarifying_question"]
                        status.update(label="يلزم توضيح من المستخدم", state="complete", expanded=False)
                    else:
                        # Step 3: Retrieval
                        step2 = st.container()
                        with step2:
                            st.markdown("**3️⃣ Retrieving...**")
                        status.update(label="Retrieving...", state="running")
                        log_debug("Step: Retrieving Articles...")
                        log_debug("Step: Searching Vector Store...")
                        t0 = time.perf_counter()
                        candidates = pipeline.retrieve_for_query(expanded_query, route)
                        print("[DEBUG] Raw retrieval results:", candidates)
                        stage_times["retrieval"] = time.perf_counter() - t0
                        log_debug(f"Step: Found {len(candidates)} articles.")
                        with step2:
                            st.success(f"تم استرجاع {len(candidates)} مادة مرشحة.")

                        # Step 4: Rerank
                        step3 = st.container()
                        with step3:
                            st.markdown("**4️⃣ Reranking...**")
                        status.update(label="Reranking...", state="running")
                        log_debug("Step: Reranking with BGE...")
                        t0 = time.perf_counter()
                        relevant = pipeline.reranker.rerank(current_question, candidates)
                        stage_times["reranking"] = time.perf_counter() - t0
                        with step3:
                            st.success(f"تم اختيار أفضل {len(relevant)} مواد (Top-{TOP_K_RERANK}).")

                        # Step 5: Synthesis
                        step4 = st.container()
                        with step4:
                            st.markdown("**5️⃣ توليد الإجابة...**")
                        status.update(label="Synthesizing answer...", state="running")
                        log_debug("Step: Synthesizing answer...")
                        t0 = time.perf_counter()
                        answer, citations = pipeline.synthesize_answer(current_question, relevant)
                        stage_times["synthesis"] = time.perf_counter() - t0

                        # Step 6: Evaluate
                        step5 = st.container()
                        with step5:
                            st.markdown("**6️⃣ Evaluating with RAGAS...**")
                        status.update(label="Evaluating with RAGAS...", state="running")
                        log_debug("Step: Calculating RAGAS metrics...")
                        t0 = time.perf_counter()
                        eval_result = evaluate_with_ragas(
                            question=clean_arabic_text(current_question),
                            answer=clean_arabic_text(answer),
                            contexts=[clean_arabic_text(a.get("page_content", a.get("text", ""))) for a in relevant],
                        )
                        stage_times["ragas_evaluation"] = time.perf_counter() - t0
                        append_eval_log(st.session_state, eval_result, current_question)
                        st.session_state.eval_results = st.session_state.eval_logs
                        with step5:
                            st.success(
                                f"Faithfulness={eval_result.faithfulness:.3f} | "
                                f"Relevancy={eval_result.answer_relevancy:.3f}"
                            )
                        status.update(label="اكتملت المعالجة ✓", state="complete", expanded=False)
                        log_debug("Step: Completed.")

                total_latency = sum(stage_times.values())
                st.session_state.last_latency = total_latency

            except Exception as e:
                answer = f"حدث خطأ: {str(e)}"
                relevant = []
                citations = []
                log_debug(f"Error: {str(e)}")

            if not citations and answer:
                citations = extract_citations_from_text(answer)

            # Ensure evaluation is populated in dashboard even if branch flow changed.
            if eval_result is None and answer:
                log_debug("Step: Calculating RAGAS metrics...")
                eval_result = evaluate_with_ragas(
                    question=clean_arabic_text(current_question),
                    answer=clean_arabic_text(answer),
                    contexts=[clean_arabic_text(a.get("page_content", a.get("text", ""))) for a in relevant],
                )
                append_eval_log(st.session_state, eval_result, current_question)
                st.session_state.eval_results = st.session_state.eval_logs

            st.markdown("**الإجابة:**")
            fixed_answer = highlight_article_mentions(clean_arabic_text(answer))
            st.markdown(
                f'<p style="direction: rtl; text-align: right;">{fixed_answer}</p>',
                unsafe_allow_html=True,
            )
            st.caption(f"Route: {route} | Latency: {total_latency:.2f}s")
            if stage_times:
                st.json({k: round(v, 3) for k, v in stage_times.items()})
            if citations:
                st.markdown(
                    f"<div class='rtl-output'><b>الاستشهادات:</b> {display_arabic(format_citations(citations))}</div>",
                    unsafe_allow_html=True,
                )
                for c in citations:
                    if st.button(f"عرض المادة {c}", key=f"live-cit-{c}"):
                        article = next(
                            (
                                a for a in relevant
                                if str(a.get("metadata", {}).get("article_number")) == str(c)
                            ),
                            None,
                        )
                        if article:
                            st.session_state.selected_article_text = article.get("page_content", article.get("text", ""))
                            st.rerun()

            render_legal_sources(relevant)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "original_question": prompt,
                "refined_question": expanded_query[0] if isinstance(expanded_query, list) and expanded_query else str(expanded_query),
                "expanded_query": expanded_query[0] if isinstance(expanded_query, list) and expanded_query else str(expanded_query),
                "expanded_queries": expanded_query if isinstance(expanded_query, list) else [str(expanded_query)],
                "route": route,
                "stage_times": stage_times,
                "relevant_articles": relevant,
                "citations": citations,
            })
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.chat_history = st.session_state.chat_history[-3:]


if __name__ == "__main__":
    main()
