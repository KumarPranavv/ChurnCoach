"""
RAG Engine for Personalized Churn Retention Recommendations

Uses a pre-built FAISS index (faiss_index/) for retrieval and
Google Gemini / Anthropic Claude / OpenAI GPT-4 for generation.
Falls back to rule-based recommendations if LLM is unavailable.

To rebuild the FAISS index, run:  python build_faiss_index.py
"""

import os
import pickle
from typing import Dict, Any, Optional

# --- No heavy dependencies needed at runtime (no sentence-transformers, no faiss) ---

# --- LLM providers ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Load pre-built FAISS index from disk (NO sentence-transformers needed)
# ---------------------------------------------------------------------------

class ChunkStore:
    """
    Loads pre-built chunk texts from chunks.pkl and provides
    keyword-based retrieval (TF-IDF-like scoring) so we do NOT
    need sentence-transformers or any embedding model at runtime.
    """

    def __init__(self, index_dir: str = "faiss_index"):
        chunks_path = os.path.join(index_dir, "chunks.pkl")
        if not os.path.isfile(chunks_path):
            raise FileNotFoundError(
                f"chunks.pkl not found in {index_dir}/. Run: python build_faiss_index.py"
            )
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)  # list of {"text": ..., "metadata": {...}}

    # ---- keyword-based retrieval (no embeddings needed) ---------------------
    def search(self, query: str, k: int = 6) -> list[dict]:
        """
        Score each chunk against the query using simple keyword overlap
        (normalised by chunk length).  Returns top-k results.
        """
        import re
        query_tokens = set(re.findall(r"[a-z]+", query.lower()))
        scored: list[tuple[float, dict]] = []
        for chunk in self.chunks:
            text = chunk.get("text", "")
            chunk_tokens = re.findall(r"[a-z]+", text.lower())
            if not chunk_tokens:
                continue
            chunk_set = set(chunk_tokens)
            overlap = len(query_tokens & chunk_set)
            # normalise: overlap / sqrt(len(query_tokens) * len(chunk_set))
            import math
            denom = math.sqrt(len(query_tokens) * len(chunk_set)) if query_tokens and chunk_set else 1
            score = overlap / denom
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"text": c["text"], "score": s,
                 "metadata": c.get("metadata", {})} for s, c in scored[:k]]


# ---------------------------------------------------------------------------
# LLM initialisation
# ---------------------------------------------------------------------------

def _get_llm(api_key: str, provider: str = "gemini"):
    """Return a LangChain chat model."""
    if provider == "gemini" and GEMINI_AVAILABLE:
        # gemini-2.0-flash: fast, high output quality, no thinking-token overhead
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.3,
            max_output_tokens=8192,
        )
    if provider == "anthropic" and ANTHROPIC_AVAILABLE:
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            anthropic_api_key=api_key,
            temperature=0.3,
            max_tokens=4096,
        )
    if provider == "openai" and OPENAI_AVAILABLE:
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
            temperature=0.3,
            max_tokens=4096,
        )
    return None


# ---------------------------------------------------------------------------
# Build search query from customer profile
# ---------------------------------------------------------------------------

def _build_search_queries(profile: Dict[str, Any], churn_prob: float) -> list[str]:
    """Build multiple targeted search queries from customer data for broader retrieval."""
    queries = []

    tenure = profile.get("tenure", 0)
    contract = profile.get("Contract", "")
    internet = profile.get("InternetService", "")
    monthly = profile.get("MonthlyCharges", 0)
    security = profile.get("OnlineSecurity", "No")
    protection = profile.get("DeviceProtection", "No")
    tech = profile.get("TechSupport", "No")
    payment = profile.get("PaymentMethod", "")
    partner = profile.get("Partner", "No")
    dependents = profile.get("Dependents", "No")
    senior = profile.get("SeniorCitizen", 0)
    streaming_tv = profile.get("StreamingTV", "No")
    streaming_movies = profile.get("StreamingMovies", "No")

    # --- Query 1: Primary risk profile for playbook matching ---
    q1_parts = []
    if churn_prob > 0.7:
        q1_parts.append("HIGH churn risk")
    elif churn_prob > 0.3:
        q1_parts.append("moderate churn risk")
    else:
        q1_parts.append("low churn risk")

    if tenure <= 12:
        q1_parts.append(f"tenure {tenure} months new customer short tenure")
    elif tenure <= 24:
        q1_parts.append(f"tenure {tenure} months")
    elif tenure <= 48:
        q1_parts.append(f"tenure {tenure} months mid-tenure customer")
    else:
        q1_parts.append(f"tenure {tenure} months loyal veteran long-tenure")

    if contract == "Month-to-month":
        q1_parts.append("Contract Month-to-month no commitment")
    else:
        q1_parts.append(f"Contract {contract}")

    q1_parts.append(f"MonthlyCharges ${monthly:.0f}")
    if monthly > 100:
        q1_parts.append("high charges expensive plan premium")
    elif monthly > 70:
        q1_parts.append("above-median charges")

    queries.append(" ".join(q1_parts))

    # --- Query 2: Service-specific interventions ---
    q2_parts = []
    if internet == "Fiber optic":
        q2_parts.append("Fiber optic internet high speed fiber")
    elif internet == "DSL":
        q2_parts.append("DSL internet upgrade fiber")
    else:
        q2_parts.append("no internet service")

    if security == "No" and protection == "No" and tech == "No":
        q2_parts.append("OnlineSecurity No DeviceProtection No TechSupport No no protection services security suite")
    else:
        if security == "Yes":
            q2_parts.append("OnlineSecurity Yes")
        if protection == "Yes":
            q2_parts.append("DeviceProtection Yes")
        if tech == "Yes":
            q2_parts.append("TechSupport Yes")

    if streaming_tv == "Yes" or streaming_movies == "Yes":
        q2_parts.append("streaming user StreamingTV StreamingMovies content entertainment")
    else:
        q2_parts.append("no streaming add streaming bundle upsell")

    queries.append(" ".join(q2_parts))

    # --- Query 3: Demographic and billing ---
    q3_parts = []
    if senior == 1:
        q3_parts.append("SeniorCitizen senior citizen elderly")
        if partner == "No":
            q3_parts.append("senior alone no partner isolation")
    if partner == "No" and dependents == "No":
        q3_parts.append("Partner No Dependents No single no family")
    elif partner == "Yes" or dependents == "Yes":
        q3_parts.append(f"Partner {partner} Dependents {dependents} family household")

    if "Electronic check" in payment:
        q3_parts.append("PaymentMethod Electronic check payment risk auto-pay switch")
    else:
        q3_parts.append(f"PaymentMethod {payment}")

    if contract == "Month-to-month" and monthly > 70:
        q3_parts.append("price sensitive contract upgrade discount pricing")

    queries.append(" ".join(q3_parts))

    # --- Query 4: Pricing and discount rules ---
    q4_parts = ["pricing discount offer savings"]
    if contract == "Month-to-month":
        q4_parts.append("contract upgrade incentive Month-to-month annual")
    if tenure >= 36:
        q4_parts.append("loyalty retention discount veteran")
    if senior == 1:
        q4_parts.append("senior citizen special rate discount")
    if monthly > 90 and internet == "Fiber optic":
        q4_parts.append("fiber optic price protection price lock")
    if "Electronic check" in payment:
        q4_parts.append("payment method migration incentive auto-pay credit")

    queries.append(" ".join(q4_parts))

    return queries


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are ChurnCoach AI, a telecom retention strategist. You MUST:
- Reference the customer's EXACT data values (tenure, MonthlyCharges, services, contract) in every point
- Calculate dollar savings with their real MonthlyCharges (e.g. "25% off $90 = $67.50/mo, saving $270/yr")
- Match them to the specific playbook/segment from the provided knowledge base

Use EXACTLY these headers:
**🎯 Customer Segment Match:** (which segment + why based on their data)
**⚠️ Key Risk Factors:** (3 bullet points citing specific data values)
**🎁 Recommended Retention Offers:** (3 offers with exact $ calculations)
**⚡ Immediate Action Plan:** (step-by-step with timelines)
**📊 Expected Outcome:** (retention % improvement with reasoning)"""


def _build_user_prompt(profile: Dict[str, Any], churn_prob: float, context: str) -> str:
    """Build a detailed user message with full customer data for the LLM."""
    tenure = profile.get("tenure", 0)
    monthly = profile.get("MonthlyCharges", 0)
    total = profile.get("TotalCharges", 0)
    senior = profile.get("SeniorCitizen", 0)
    partner = profile.get("Partner", "N/A")
    dependents = profile.get("Dependents", "N/A")
    contract = profile.get("Contract", "N/A")
    internet = profile.get("InternetService", "N/A")
    security = profile.get("OnlineSecurity", "No")
    backup = profile.get("OnlineBackup", "No")
    protection = profile.get("DeviceProtection", "No")
    tech = profile.get("TechSupport", "No")
    streaming_tv = profile.get("StreamingTV", "No")
    streaming_movies = profile.get("StreamingMovies", "No")
    payment = profile.get("PaymentMethod", "N/A")

    # Count active services
    active = sum(1 for s in [profile.get("PhoneService"), security, backup,
                              protection, tech, streaming_tv, streaming_movies]
                  if s == "Yes")
    has_protection = any(s == "Yes" for s in [security, protection, tech])
    has_streaming = streaming_tv == "Yes" or streaming_movies == "Yes"

    # Pre-compute key data-driven insights for the LLM
    insights = []
    if contract == "Month-to-month":
        insights.append(f"CRITICAL: Month-to-month contract — 42% churn rate for this contract type")
    if tenure <= 12:
        insights.append(f"WARNING: New customer ({tenure} months) — in highest churn window")
    elif tenure >= 48:
        insights.append(f"LOYALTY: Long-tenure customer ({tenure} months) — high lifetime value")
    if not has_protection and internet != "No":
        insights.append("GAP: No protection services (OnlineSecurity=No, DeviceProtection=No, TechSupport=No) — protection customers churn 60-70% less")
    if "Electronic check" in payment:
        insights.append("RISK: Electronic check payment — 45% churn rate vs 15% for auto-pay")
    if monthly > 90 and internet == "Fiber optic":
        insights.append(f"HIGH-VALUE: Fiber optic customer paying ${monthly:.2f}/month — highest ARPU, must retain")
    if senior == 1 and partner == "No":
        insights.append("VULNERABLE: Senior citizen living alone — needs dedicated care approach")
    if not has_streaming and internet != "No":
        insights.append("UPSELL OPPORTUNITY: No streaming services — adding streaming reduces churn by 20%")
    if monthly > 70 and contract == "Month-to-month" and tenure < 24:
        insights.append(f"PRICE-SENSITIVE: Paying ${monthly:.2f}/month on month-to-month with only {tenure} months tenure — high flight risk")

    insights_text = "\n".join(f"  • {i}" for i in insights) if insights else "  • No critical flags"

    return f"""KNOWLEDGE BASE CONTEXT:
{context}

CUSTOMER PROFILE:
- Senior: {'Yes' if senior == 1 else 'No'} | Partner: {partner} | Dependents: {dependents} | Tenure: {tenure} months
- Internet: {internet} | Phone: {profile.get('PhoneService', 'N/A')} | MultipleLines: {profile.get('MultipleLines', 'N/A')}
- OnlineSecurity: {security} | OnlineBackup: {backup} | DeviceProtection: {protection} | TechSupport: {tech}
- StreamingTV: {streaming_tv} | StreamingMovies: {streaming_movies}
- Contract: {contract} | Monthly: ${monthly:.2f} | Total: ${total:.2f} | AvgMonthly: ${total / max(tenure, 1):.2f}
- Payment: {payment} | PaperlessBilling: {profile.get('PaperlessBilling', 'N/A')}
- Active Services: {active} | Has Protection Bundle: {'Yes' if has_protection else 'No'} | Has Streaming: {'Yes' if has_streaming else 'No'}

CHURN RISK: {churn_prob:.1%} ({'HIGH' if churn_prob > 0.7 else 'MODERATE' if churn_prob > 0.3 else 'LOW'})

KEY INSIGHTS:
{insights_text}

Generate a complete retention plan using the knowledge base. Calculate all discounts using MonthlyCharges=${monthly:.2f}."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RAGEngine:
    """Retrieval-Augmented Generation engine for churn recommendations."""

    def __init__(self, index_dir: str = "faiss_index"):
        try:
            self.chunk_store = ChunkStore(index_dir)
            self.ready = True
        except Exception:
            self.chunk_store = None
            self.ready = False
        self.llm = None

    # -- LLM connection (call after __init__) ---------------------------------
    def connect_llm(self, api_key: str, provider: str = "gemini") -> bool:
        """Attempt to connect to an LLM provider. Returns True on success."""
        if not api_key or not api_key.strip():
            return False
        try:
            self.llm = _get_llm(api_key.strip(), provider)
            return self.llm is not None
        except Exception:
            self.llm = None
            return False

    # -- Core recommendation method -------------------------------------------
    def get_recommendation(
        self, profile: Dict[str, Any], churn_prob: float, k: int = 5
    ) -> str:
        """
        Generate a personalized retention recommendation using RAG.

        Uses multi-query retrieval for broader, more relevant context.
        Falls back to rule-based output when LLM/vector store is unavailable.
        """
        # --- Fallback: rule-based ------------------------------------------------
        if not self.ready or self.llm is None:
            return self._fallback_recommendation(profile, churn_prob)

        # --- RAG path with multi-query retrieval ---------------------------------
        import time

        try:
            queries = _build_search_queries(profile, churn_prob)
            # Retrieve top-k results for each query, deduplicate
            seen_texts = set()
            all_results = []
            for query in queries:
                results = self.chunk_store.search(query, k=k)
                for r in results:
                    text_key = r["text"][:100]  # dedup by first 100 chars
                    if text_key not in seen_texts:
                        seen_texts.add(text_key)
                        all_results.append(r)
            # Sort by score and take top results
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            top_results = all_results[:6]  # Compact context to maximize output budget
            context = "\n\n---\n\n".join(r["text"] for r in top_results)
        except Exception as e:
            return self._fallback_recommendation(profile, churn_prob)

        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_build_user_prompt(profile, churn_prob, context)),
        ]

        # Try once, retry once on rate-limit (max ~5s wait)
        for attempt in range(2):
            try:
                response = self.llm.invoke(messages)
                if response and hasattr(response, 'content') and response.content:
                    text = response.content.strip()
                    # Check if response was truncated (missing expected sections)
                    if ("Expected Outcome" in text or "📊" in text) and len(text) > 200:
                        return text  # Complete response
                    elif len(text) > 500:
                        # Partial but substantial — append a note
                        return text + "\n\n---\n*⚠️ Response may have been trimmed. Key actions are listed above.*"
                    elif len(text) > 100:
                        # Got something but too short — use it + supplement with fallback
                        fallback = self._fallback_recommendation(profile, churn_prob)
                        return text + "\n\n---\n**📋 Additional Data-Driven Recommendations:**\n\n" + fallback
                    # Tiny/empty → fall back entirely
                    return self._fallback_recommendation(profile, churn_prob)
                return self._fallback_recommendation(profile, churn_prob)
            except Exception as e:
                err_str = str(e)
                if attempt == 0 and ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str):
                    time.sleep(5)
                    continue
                return self._fallback_recommendation(profile, churn_prob)

        return self._fallback_recommendation(profile, churn_prob)

    # -- Fallback (no LLM) ----------------------------------------------------
    @staticmethod
    def _fallback_recommendation(profile: Dict[str, Any], churn_prob: float) -> str:
        """Data-driven rule-based recommendations when LLM is not available."""
        recs: list[str] = []
        tenure = profile.get("tenure", 0)
        monthly = profile.get("MonthlyCharges", 0)
        total = profile.get("TotalCharges", 0)
        contract = profile.get("Contract", "")
        internet = profile.get("InternetService", "")
        security = profile.get("OnlineSecurity", "No")
        protection = profile.get("DeviceProtection", "No")
        tech = profile.get("TechSupport", "No")
        payment = profile.get("PaymentMethod", "")
        partner = profile.get("Partner", "No")
        dependents = profile.get("Dependents", "No")
        senior = profile.get("SeniorCitizen", 0)
        streaming_tv = profile.get("StreamingTV", "No")
        streaming_movies = profile.get("StreamingMovies", "No")
        has_protection = security == "Yes" or protection == "Yes" or tech == "Yes"
        has_streaming = streaming_tv == "Yes" or streaming_movies == "Yes"
        active_services = sum(1 for s in [
            profile.get("PhoneService"), security,
            profile.get("OnlineBackup", "No"), protection,
            tech, streaming_tv, streaming_movies
        ] if s == "Yes")

        # --- Segment matching ---
        if senior == 1 and partner == "No":
            segment = "Price-Sensitive Senior (SeniorCitizen=Yes, Partner=No)"
        elif tenure <= 12 and contract == "Month-to-month":
            segment = f"Uncommitted New User (tenure={tenure} months, Month-to-month contract)"
        elif monthly > 70 and contract == "Month-to-month" and tenure < 24:
            segment = f"Price-Sensitive Churner (MonthlyCharges=${monthly:.2f}, Month-to-month, tenure={tenure}mo)"
        elif tenure >= 48:
            segment = f"Loyal But Neglected Veteran (tenure={tenure} months)"
        elif not has_protection and internet != "No":
            segment = "No-Protection Customer (missing OnlineSecurity, DeviceProtection, TechSupport)"
        elif internet == "Fiber optic" and monthly > 90:
            segment = f"Fiber Optic High Spender (${monthly:.2f}/month on Fiber)"
        else:
            segment = f"General Customer (tenure={tenure}mo, ${monthly:.2f}/month, {contract})"

        recs.append(f"**🎯 Customer Segment Match:** {segment}\n")

        # --- Risk factors (data-specific) ---
        recs.append("**⚠️ Key Risk Factors:**")
        risk_found = False
        if contract == "Month-to-month":
            recs.append(f"- Month-to-month contract — 42% avg churn rate for this type")
            risk_found = True
        if tenure <= 12:
            recs.append(f"- New customer with only {tenure} months tenure — in highest churn window")
            risk_found = True
        if not has_protection and internet != "No":
            recs.append(f"- No protection services (OnlineSecurity={security}, DeviceProtection={protection}, TechSupport={tech}) — protection users churn 60-70% less")
            risk_found = True
        if monthly > 80:
            recs.append(f"- High monthly charges (${monthly:.2f}/month) — above-median pricing increases flight risk")
            risk_found = True
        if "Electronic check" in payment:
            recs.append(f"- Electronic check payment — 45% churn rate vs 15% for auto-pay")
            risk_found = True
        if senior == 1 and partner == "No":
            recs.append(f"- Senior citizen living alone — needs dedicated care approach")
            risk_found = True
        if partner == "No" and dependents == "No":
            recs.append(f"- Single user (no partner, no dependents) — low switching cost")
            risk_found = True
        if not risk_found:
            recs.append("- No significant risk factors identified in current data")

        # --- Offers (data-specific with dollar calculations) ---
        recs.append("\n**🎁 Recommended Retention Offers:**")
        if contract == "Month-to-month":
            discount_pct = 25 if monthly > 80 else 20
            new_price = monthly * (1 - discount_pct / 100)
            annual_savings = (monthly - new_price) * 12
            recs.append(f"- 📝 Contract upgrade: {discount_pct}% off for 1-year contract → ${new_price:.2f}/month (saves ${annual_savings:.2f}/year)")
        if "Electronic check" in payment:
            credit = 7 if monthly > 80 else 5
            recs.append(f"- � Auto-pay switch: ${credit}/month credit + $25 one-time bonus (${credit * 12 + 25:.0f} first-year savings)")
        if not has_protection and internet != "No":
            recs.append(f"- 🛡️ Free Protection Suite trial (OnlineSecurity + DeviceProtection + TechSupport) for 3 months, then 25% off ongoing")
        if senior == 1:
            senior_savings = monthly * 0.15
            recs.append(f"- � Senior discount: 15% off → saves ${senior_savings:.2f}/month (${senior_savings * 12:.2f}/year)")
            if tech == "No":
                recs.append(f"- 📞 Free TechSupport for 6 months — seniors without tech support churn 40% more")
        if internet == "Fiber optic" and monthly > 90:
            recs.append(f"- 🔒 Fiber price lock: guarantee ${monthly:.2f}/month rate for 24 months + free speed upgrade")
        if not has_streaming and internet != "No":
            recs.append(f"- 📺 Streaming bundle (TV + Movies) at 25% discount — streaming users churn 20% less")
        if tenure >= 36:
            tier = "20%" if tenure > 60 else "15%" if tenure > 48 else "10%"
            recs.append(f"- ⭐ Loyalty discount: {tier} off as a {tenure}-month customer appreciation")
        if internet == "DSL" and has_streaming:
            recs.append(f"- ⚡ Free Fiber optic upgrade for 3 months at DSL price — 10x faster streaming")
        if active_services >= 5 and monthly > 100:
            bundle_savings = monthly * 0.15
            recs.append(f"- 📦 Multi-service bundle: 15% discount on {active_services} services → saves ${bundle_savings:.2f}/month")

        # --- Immediate action ---
        recs.append("\n**⚡ Immediate Action Plan:**")
        if churn_prob > 0.7:
            recs.append(f"1. URGENT: Assign retention specialist — contact within 24 hours")
            recs.append(f"2. Prepare personalized offer package with the discounts above")
            recs.append(f"3. If customer mentions competitor, authorize up to 30% retention rate for 6 months")
        elif churn_prob > 0.3:
            recs.append(f"1. Send personalized email with satisfaction survey within 48 hours")
            recs.append(f"2. Prepare targeted offer based on their specific risk factors")
            recs.append(f"3. Schedule follow-up call within 7 days")
        else:
            recs.append(f"1. Enroll in loyalty rewards program")
            recs.append(f"2. Send appreciation message recognizing {tenure} months of loyalty")
            recs.append(f"3. Explore upsell opportunities at next positive interaction")

        # --- Expected outcome ---
        recs.append("\n**📊 Expected Outcome:**")
        if churn_prob > 0.7:
            recs.append(f"With immediate outreach and targeted offers, expect 30-45% reduction in churn probability. Contract upgrade alone would drop churn from ~42% to ~11%.")
        elif churn_prob > 0.3:
            recs.append(f"Proactive engagement typically yields 20-30% retention improvement. Protection suite addition reduces churn by an additional 15-25%.")
        else:
            recs.append(f"Maintain strong retention. Focus on upselling for 10-15% revenue uplift. Each additional service reduces churn probability by 8-12%.")

        return "\n".join(recs)
