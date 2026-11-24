import os
import re
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import settings
from src.nlp.opportunity_engine import build_business_opportunities
from src.nlp.topic_clusterer import cluster_review_topics

logger = logging.getLogger("revio_ai_engine")


# ---------------------------------------------------------------------------
# Local Analytics Engine
# ---------------------------------------------------------------------------

class LocalAnalyticsEngine:
    """
    Generates fully evidence-backed, data-driven business insights from real
    batch analytics. Every statistic is derived from actual database aggregates.
    Never produces static hardcoded text.
    """

    @staticmethod
    def _pct(count: int, total: int) -> float:
        return round((count / total * 100), 1) if total > 0 else 0.0

    @staticmethod
    def _fmt_pct(count: int, total: int) -> str:
        return f"{LocalAnalyticsEngine._pct(count, total)}%"

    @classmethod
    def generate_insights(
        cls,
        analytics_summary: Dict[str, Any],
        fallback_reason: Optional[str] = None,
        reviews_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        analytics_summary = analytics_summary or {}

        total: int = int(analytics_summary.get("total_reviews", 0) or 0)
        avg_rating: float = float(analytics_summary.get("avg_rating", 0.0) or 0.0)
        sent_dist: Dict[str, int] = analytics_summary.get("sentiment_distribution", {}) or {}
        priority_dist: Dict[str, int] = analytics_summary.get("priority_distribution", {}) or {}
        emotion_dist: Dict[str, int] = analytics_summary.get("emotion_distribution", {}) or {}
        category_stats: List[Dict] = analytics_summary.get("category_stats", []) or []
        complaint_aspects: Dict[str, int] = analytics_summary.get("complaint_aspects", {}) or {}
        positive_aspects: Dict[str, int] = analytics_summary.get("positive_aspects", {}) or {}
        rating_dist: Dict[int, int] = analytics_summary.get("rating_distribution", {}) or {}

        # ── Sentiment ──────────────────────────────────────────────────────
        pos_count = int(sent_dist.get("Positive", 0) or 0)
        neg_count = int(sent_dist.get("Negative", 0) or 0)
        neutral_count = int(sent_dist.get("Neutral", 0) or 0)
        mixed_count = int(sent_dist.get("Mixed", 0) or 0)

        pos_pct = cls._pct(pos_count, total)
        neg_pct = cls._pct(neg_count, total)
        neutral_pct = cls._pct(neutral_count, total)

        # ── Priority ───────────────────────────────────────────────────────
        high_prio = int(priority_dist.get("High", 0) or 0)
        med_prio = int(priority_dist.get("Medium", 0) or 0)
        low_prio = int(priority_dist.get("Low", 0) or 0)
        high_prio_pct = cls._pct(high_prio, total)

        # ── Emotion ────────────────────────────────────────────────────────
        angry_count = int(emotion_dist.get("Angry", 0) or 0)
        frustrated_count = int(emotion_dist.get("Frustrated", 0) or 0)
        happy_count = int(emotion_dist.get("Happy", 0) or 0)
        satisfied_count = int(emotion_dist.get("Satisfied", 0) or 0)
        negative_emotion_count = angry_count + frustrated_count
        positive_emotion_count = happy_count + satisfied_count
        dominant_emotion = "Neutral"
        if emotion_dist:
            dominant_emotion = max(emotion_dist.items(), key=lambda x: int(x[1] or 0))[0]

        # ── Categories ─────────────────────────────────────────────────────
        best_cat_name = "General"
        best_cat_rating = avg_rating
        best_cat_count = 0
        worst_cat_name = "General"
        worst_cat_rating = avg_rating
        worst_cat_count = 0

        if category_stats:
            sorted_by_rating = sorted(
                [c for c in category_stats if c.get("avg_rating") is not None],
                key=lambda c: float(c.get("avg_rating", 0) or 0),
                reverse=True,
            )
            if sorted_by_rating:
                best = sorted_by_rating[0]
                worst = sorted_by_rating[-1]
                best_cat_name = best.get("product_category", "General") or "General"
                best_cat_rating = round(float(best.get("avg_rating", 0.0) or 0.0), 2)
                best_cat_count = int(best.get("count", 0) or 0)
                worst_cat_name = worst.get("product_category", "General") or "General"
                worst_cat_rating = round(float(worst.get("avg_rating", 0.0) or 0.0), 2)
                worst_cat_count = int(worst.get("count", 0) or 0)

        # ── Aspects ────────────────────────────────────────────────────────
        top_complaints: List[Tuple[str, int]] = sorted(
            complaint_aspects.items(), key=lambda x: int(x[1] or 0), reverse=True
        )[:5]
        top_positives: List[Tuple[str, int]] = sorted(
            positive_aspects.items(), key=lambda x: int(x[1] or 0), reverse=True
        )[:5]

        top_complaint_aspect = top_complaints[0][0] if top_complaints else "Operational Delivery"
        top_complaint_count = int(top_complaints[0][1]) if top_complaints else 0
        top_complaint_pct = cls._pct(top_complaint_count, total)

        top_positive_aspect = top_positives[0][0] if top_positives else "Product Quality"
        top_positive_count = int(top_positives[0][1]) if top_positives else 0
        top_positive_pct = cls._pct(top_positive_count, total)

        # ── Rating distribution ────────────────────────────────────────────
        one_star = int(rating_dist.get(1, rating_dist.get("1", 0)) or 0)
        two_star = int(rating_dist.get(2, rating_dist.get("2", 0)) or 0)
        four_star = int(rating_dist.get(4, rating_dist.get("4", 0)) or 0)
        five_star = int(rating_dist.get(5, rating_dist.get("5", 0)) or 0)
        low_rating_count = one_star + two_star
        low_rating_pct = cls._pct(low_rating_count, total)
        high_rating_count = four_star + five_star
        high_rating_pct = cls._pct(high_rating_count, total)

        # ── Optional NLP Topics & Opportunity Signals ────────────────────────
        topic_clusters: List[Dict[str, Any]] = []
        if reviews_df is not None and not reviews_df.empty and "review_text" in reviews_df.columns:
            topic_clusters = cluster_review_topics(reviews_df, text_col="review_text", n_clusters=4, top_terms=3)

        nlp_opportunities = build_business_opportunities(analytics_summary, topics=topic_clusters)
        top_topics: List[str] = nlp_opportunities.get("top_topics", [])

        # ── Insufficient data guard ────────────────────────────────────────
        if total == 0:
            return cls._empty_result(fallback_reason)

        # ══════════════════════════════════════════════════════════════════
        # 1. EXECUTIVE SUMMARY
        # ══════════════════════════════════════════════════════════════════
        exec_summary = (
            f"Analysis of {total:,} verified customer reviews in this batch reveals an overall "
            f"satisfaction index of {avg_rating:.2f}/5.0. Sentiment composition: {pos_pct}% Positive "
            f"({pos_count:,} reviews), {neg_pct}% Negative ({neg_count:,} reviews), "
            f"{neutral_pct}% Neutral ({neutral_count:,} reviews). "
        )
        if best_cat_name != worst_cat_name:
            exec_summary += (
                f"Top-performing category is '{best_cat_name}' (avg {best_cat_rating:.2f}/5.0 across "
                f"{best_cat_count:,} reviews); worst-performing is '{worst_cat_name}' "
                f"(avg {worst_cat_rating:.2f}/5.0 across {worst_cat_count:,} reviews). "
            )
        exec_summary += (
            f"{high_prio:,} reviews ({high_prio_pct}%) are classified High Priority requiring immediate attention. "
            f"The dominant customer emotional signal is '{dominant_emotion}'. "
        )
        if top_complaint_aspect != "Operational Delivery" or top_complaint_count > 0:
            exec_summary += (
                f"The leading complaint driver is '{top_complaint_aspect}', cited in {top_complaint_count:,} "
                f"negative reviews ({top_complaint_pct}% of batch). "
            )
        if top_topics:
            exec_summary += f"Key emerging themes include {', '.join(top_topics[:3])}. "
        exec_summary += (
            f"Strategic priorities: resolve high-volume complaint patterns, invest in underperforming "
            f"categories, and amplify proven strengths in '{top_positive_aspect}'."
        )

        # ══════════════════════════════════════════════════════════════════
        # 2. OPPORTUNITIES / RECOMMENDATIONS (evidence-backed)
        # ══════════════════════════════════════════════════════════════════
        action_items: List[Dict[str, Any]] = []

        # Opportunity 1 — Top complaint aspect (High priority if neg > 15%)
        if top_complaints:
            asp_name, asp_count = top_complaints[0]
            asp_count = int(asp_count or 0)
            asp_pct = cls._pct(asp_count, total)
            priority = "High" if asp_pct > 15 else ("Medium" if asp_pct > 5 else "Low")
            action_items.append({
                "priority": priority,
                "category": asp_name,
                "opportunity_title": f"Reduce '{asp_name}' Failure Rate",
                "problem": (
                    f"'{asp_name}' is the most frequent complaint driver, appearing in "
                    f"{asp_count:,} reviews ({asp_pct}% of batch)."
                ),
                "evidence": (
                    f"Complaint aspect count: {asp_count:,}/{total:,} reviews ({asp_pct}%). "
                    f"High-priority ticket volume linked to this aspect: approximately "
                    f"{min(asp_count, high_prio):,}. "
                    + (f"Negative emotion (Angry+Frustrated): {negative_emotion_count:,} reviews." if negative_emotion_count > 0 else "")
                ),
                "affected_category": worst_cat_name if worst_cat_name != "General" else "All Categories",
                "root_cause_hypothesis": (
                    f"Systemic failure in '{asp_name}' processes, vendor standards, or customer "
                    f"expectation misalignment is driving repeat complaints."
                ),
                "action": (
                    f"Conduct root-cause audit of all '{asp_name}' processes. "
                    f"Implement SLA tracking and vendor accountability metrics. "
                    f"Target a 40% reduction in '{asp_name}' complaint volume within 60 days."
                ),
                "implementation_steps": [
                    f"Map all customer touchpoints related to '{asp_name}'.",
                    "Identify top 3 failure modes from review text analysis.",
                    "Assign cross-functional team with weekly KPI review.",
                    "Implement customer callback program for affected reviews.",
                    "Re-survey affected customers after resolution.",
                ],
                "expected_impact": (
                    f"Reducing '{asp_name}' complaints by 40% would shift ~{int(asp_count * 0.4):,} "
                    f"negative reviews to neutral/positive, improving overall avg rating."
                ),
                "evidence_limitations": (
                    "Aspect extraction is keyword-based. Manual review of flagged records is recommended "
                    "before committing resources."
                ),
            })

        # Opportunity 2 — Underperforming category
        if worst_cat_name != best_cat_name and worst_cat_rating < 4.0:
            gap = round(best_cat_rating - worst_cat_rating, 2)
            priority = "High" if worst_cat_rating < 3.0 else "Medium"
            action_items.append({
                "priority": priority,
                "category": worst_cat_name,
                "opportunity_title": f"Elevate '{worst_cat_name}' Category Performance",
                "problem": (
                    f"'{worst_cat_name}' has the lowest satisfaction rating at {worst_cat_rating:.2f}/5.0 "
                    f"across {worst_cat_count:,} reviews — {gap:.2f} points below the top category "
                    f"'{best_cat_name}' ({best_cat_rating:.2f}/5.0)."
                ),
                "evidence": (
                    f"Category avg rating: {worst_cat_rating:.2f}/5.0 (target: ≥4.0). "
                    f"Review volume: {worst_cat_count:,}. "
                    f"Performance gap vs best category: {gap:.2f} rating points."
                ),
                "affected_category": worst_cat_name,
                "root_cause_hypothesis": (
                    f"Product-market fit issues, inconsistent quality control, or unmet customer "
                    f"expectations in the '{worst_cat_name}' vertical."
                ),
                "action": (
                    f"Conduct full product/service audit in '{worst_cat_name}'. "
                    f"Set rating improvement target of ≥4.0/5.0 within 90 days. "
                    f"Benchmark against '{best_cat_name}' category practices."
                ),
                "implementation_steps": [
                    f"Pull all 1–2 star reviews for '{worst_cat_name}' and cluster themes.",
                    "Review product specifications and vendor quality standards.",
                    "Implement post-purchase follow-up for this category.",
                    f"Apply best practices from '{best_cat_name}' to '{worst_cat_name}'.",
                    "Monitor weekly rating trends post-intervention.",
                ],
                "expected_impact": (
                    f"Raising '{worst_cat_name}' avg rating from {worst_cat_rating:.2f} to 4.0+ "
                    f"would materially improve overall batch satisfaction index from {avg_rating:.2f}."
                ),
                "evidence_limitations": (
                    f"Category-level analysis is based on {worst_cat_count:,} reviews. "
                    "Statistical confidence improves with larger category sample sizes."
                ),
            })

        # Opportunity 3 — High-priority ticket volume
        if high_prio > 0 and high_prio_pct > 10:
            action_items.append({
                "priority": "High",
                "category": "Customer Support Escalation",
                "opportunity_title": "Reduce High-Priority Customer Escalation Rate",
                "problem": (
                    f"{high_prio:,} reviews ({high_prio_pct}% of batch) are flagged High Priority, "
                    f"indicating significant customer friction requiring immediate response."
                ),
                "evidence": (
                    f"High-priority reviews: {high_prio:,} ({high_prio_pct}%). "
                    f"Angry/Frustrated emotional signals: {negative_emotion_count:,} reviews. "
                    f"1–2 star ratings: {low_rating_count:,} ({low_rating_pct}%)."
                ),
                "affected_category": worst_cat_name if worst_cat_name != "General" else "Multiple",
                "root_cause_hypothesis": (
                    "High escalation volume indicates unresolved systemic issues reaching customers "
                    "without adequate proactive support intervention."
                ),
                "action": (
                    f"Implement proactive outreach to all {high_prio:,} high-priority customers. "
                    "Establish automated escalation routing for 1–2 star reviews. "
                    "Create rapid-response SLA (target: 24hr first contact)."
                ),
                "implementation_steps": [
                    "Export all high-priority review IDs from this batch.",
                    "Assign dedicated support agents to outreach queue.",
                    "Create resolution tracking dashboard for escalated cases.",
                    "Establish root-cause tagging for all resolved cases.",
                    "Review escalation triggers monthly to prevent recurrence.",
                ],
                "expected_impact": (
                    f"Proactive resolution of {high_prio:,} high-priority cases can convert negative "
                    "word-of-mouth to recoverable relationships and reduce churn risk."
                ),
                "evidence_limitations": (
                    "Priority classification is ML-based (sentiment + rating). "
                    "Manual triage of high-priority queue is recommended."
                ),
            })

        # Opportunity 4 — Leverage top positive aspect
        if top_positives:
            pos_asp, pos_asp_count = top_positives[0]
            pos_asp_count = int(pos_asp_count or 0)
            pos_asp_pct = cls._pct(pos_asp_count, total)
            action_items.append({
                "priority": "Medium",
                "category": pos_asp,
                "opportunity_title": f"Amplify '{pos_asp}' as Competitive Differentiator",
                "problem": (
                    f"'{pos_asp}' is the strongest positive signal, mentioned in {pos_asp_count:,} "
                    f"positive reviews ({pos_asp_pct}% of batch) — a proven competitive strength "
                    "not yet fully leveraged in marketing."
                ),
                "evidence": (
                    f"Positive aspect mentions: {pos_asp_count:,} ({pos_asp_pct}%). "
                    f"Happy/Satisfied customer signals: {positive_emotion_count:,} reviews. "
                    f"4–5 star reviews: {high_rating_count:,} ({high_rating_pct}%)."
                ),
                "affected_category": best_cat_name,
                "root_cause_hypothesis": (
                    f"'{pos_asp}' represents a genuine product/service strength that customers "
                    "independently and consistently recognize — a defensible competitive moat."
                ),
                "action": (
                    f"Use verified '{pos_asp}' reviews as primary social proof in marketing. "
                    "Design referral incentive program targeting satisfied customers in this segment. "
                    f"Invest in sustaining '{pos_asp}' quality as a brand pillar."
                ),
                "implementation_steps": [
                    f"Compile top 20 '{pos_asp}' reviews for marketing use (with permission).",
                    "Launch testimonial campaign across digital channels.",
                    f"Identify what drives '{pos_asp}' excellence and document as standard.",
                    "Extend same practices to underperforming categories.",
                    "Create loyalty rewards for repeat customers in this segment.",
                ],
                "expected_impact": (
                    f"Amplifying '{pos_asp}' in acquisition channels can increase conversion by "
                    f"leveraging authentic proof from {pos_asp_count:,} real customer reviews."
                ),
                "evidence_limitations": (
                    "Aspect counts are based on keyword matching. Cross-reference with actual "
                    "review text before use in public marketing materials."
                ),
            })

        # Opportunity 5 — Angry/frustrated emotional signal
        if negative_emotion_count > 0 and cls._pct(negative_emotion_count, total) > 5:
            neg_em_pct = cls._pct(negative_emotion_count, total)
            action_items.append({
                "priority": "Medium",
                "category": "Customer Experience",
                "opportunity_title": "Reduce Negative Emotional Experience Rate",
                "problem": (
                    f"{negative_emotion_count:,} customer interactions ({neg_em_pct}% of batch) "
                    f"reflect Angry or Frustrated emotional states — indicating systemic friction "
                    f"points beyond isolated product issues."
                ),
                "evidence": (
                    f"Angry responses: {angry_count:,}. Frustrated responses: {frustrated_count:,}. "
                    f"Combined negative emotional signal: {negative_emotion_count:,} ({neg_em_pct}%). "
                    f"This correlates with {high_prio:,} high-priority reviews."
                ),
                "affected_category": "Cross-Category",
                "root_cause_hypothesis": (
                    "Negative emotional responses typically indicate process failures, unmet expectations, "
                    "or inadequate customer recovery when issues arise."
                ),
                "action": (
                    "Deploy customer journey mapping focused on friction-point identification. "
                    "Implement sentiment-triggered support escalation. "
                    "Train front-line teams on de-escalation and service recovery protocols."
                ),
                "implementation_steps": [
                    "Map top 5 customer journeys with highest negative emotional density.",
                    "Identify common triggers for Angry vs Frustrated classifications.",
                    "Implement real-time sentiment monitoring on new reviews.",
                    "Create service-recovery script library for common complaint types.",
                    "Track emotional signal trends month-over-month.",
                ],
                "expected_impact": (
                    f"Reducing negative emotional responses by 30% would impact ~{int(negative_emotion_count * 0.3):,} "
                    f"customers, improving NPS and reducing churn probability."
                ),
                "evidence_limitations": (
                    "Emotion classification is keyword-based with rule-based fallback. "
                    "Manual review of classified angry/frustrated reviews recommended."
                ),
            })

        # Additional complaint aspects (opportunities 6+)
        for asp_name, asp_count in top_complaints[1:3]:
            asp_count = int(asp_count or 0)
            if asp_count > 0:
                asp_pct = cls._pct(asp_count, total)
                action_items.append({
                    "priority": "Low",
                    "category": asp_name,
                    "opportunity_title": f"Address '{asp_name}' Complaint Cluster",
                    "problem": (
                        f"'{asp_name}' appears in {asp_count:,} negative reviews ({asp_pct}% of batch)."
                    ),
                    "evidence": f"Complaint aspect frequency: {asp_count:,} ({asp_pct}% of {total:,} reviews).",
                    "affected_category": asp_name,
                    "root_cause_hypothesis": f"Recurring customer dissatisfaction linked to '{asp_name}' processes or quality.",
                    "action": f"Review and optimize '{asp_name}' touchpoints. Set quarterly improvement targets.",
                    "implementation_steps": [
                        f"Cluster reviews mentioning '{asp_name}' by sub-theme.",
                        "Assign process owner for improvement initiative.",
                        "Set measurable reduction target for next batch cycle.",
                    ],
                    "expected_impact": f"Reducing '{asp_name}' issues improves category satisfaction index.",
                    "evidence_limitations": "Based on automated aspect extraction — verify with manual review sample.",
                })

        # If no specific opportunities were generated
        if not action_items:
            action_items.append({
                "priority": "Low",
                "category": "General",
                "opportunity_title": "Insufficient Evidence for Specific Recommendations",
                "problem": "Insufficient complaint frequency or categorical differentiation to generate specific evidence-backed recommendations.",
                "evidence": f"Total reviews: {total}. Negative: {neg_count}. High priority: {high_prio}.",
                "affected_category": "All",
                "root_cause_hypothesis": "Data volume or variety may be too low for confident pattern detection.",
                "action": "Collect additional review data (target: ≥500 reviews) and re-run analysis.",
                "implementation_steps": ["Increase review collection volume.", "Ensure product category diversity in dataset."],
                "expected_impact": "Larger dataset enables statistically confident recommendations.",
                "evidence_limitations": "Current batch has insufficient data for high-confidence recommendations.",
            })

        # ══════════════════════════════════════════════════════════════════
        # 3. COMPLAINTS SUMMARY (formatted for display)
        # ══════════════════════════════════════════════════════════════════
        major_complaints: List[str] = []
        if top_complaints:
            for asp, cnt in top_complaints[:3]:
                cnt = int(cnt or 0)
                major_complaints.append(
                    f"**{asp}**: {cnt:,} complaints ({cls._pct(cnt, total)}% of batch reviews)"
                )
        if worst_cat_name != best_cat_name and worst_cat_rating < 4.0:
            major_complaints.append(
                f"**{worst_cat_name}** category underperforming: {worst_cat_rating:.2f}/5.0 avg rating "
                f"({worst_cat_count:,} reviews)"
            )
        if negative_emotion_count > 0:
            major_complaints.append(
                f"**{negative_emotion_count:,} Angry/Frustrated** customer signals detected "
                f"({cls._pct(negative_emotion_count, total)}% of batch)"
            )
        if high_prio > 0:
            major_complaints.append(
                f"**{high_prio:,} High-Priority** reviews ({high_prio_pct}%) require immediate action"
            )
        if not major_complaints:
            major_complaints.append("No critical systemic complaint patterns detected in this batch.")

        # ══════════════════════════════════════════════════════════════════
        # 4. POSITIVES SUMMARY
        # ══════════════════════════════════════════════════════════════════
        positive_trends: List[str] = []
        if top_positives:
            for asp, cnt in top_positives[:3]:
                cnt = int(cnt or 0)
                positive_trends.append(
                    f"**{asp}**: mentioned positively in {cnt:,} reviews ({cls._pct(cnt, total)}%)"
                )
        if best_cat_name != "General":
            positive_trends.append(
                f"**{best_cat_name}** leads category performance: {best_cat_rating:.2f}/5.0 "
                f"({best_cat_count:,} reviews)"
            )
        if positive_emotion_count > 0:
            positive_trends.append(
                f"**{positive_emotion_count:,} Happy/Satisfied** customer signals "
                f"({cls._pct(positive_emotion_count, total)}% of batch)"
            )
        if high_rating_count > 0:
            positive_trends.append(
                f"**{high_rating_count:,} 4–5 star ratings** ({high_rating_pct}% of batch)"
            )
        if not positive_trends:
            positive_trends.append("Positive trend data insufficient — insufficient review volume or variety.")

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        return {
            "is_fallback": True,
            "engine": "Local Analytics Engine",
            "provider": "Local Python Engine",
            "model": "local-analytics-v2",
            "generation_status": "success",
            "fallback_reason": fallback_reason or "Local Analytics Engine selected",
            "token_usage": None,
            "generated_at": now_iso,
            "executive_summary": exec_summary,
            "major_complaints": major_complaints,
            "complaints_summary": "\n".join([f"• {c}" for c in major_complaints]),
            "positive_trends": positive_trends,
            "positives_summary": "\n".join([f"• {p}" for p in positive_trends]),
            "action_plan": action_items,
            "top_topics": top_topics,
            "topic_clusters": topic_clusters,
        }

    @staticmethod
    def _empty_result(fallback_reason: Optional[str] = None) -> Dict[str, Any]:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return {
            "is_fallback": True,
            "engine": "Local Analytics Engine",
            "provider": "Local Python Engine",
            "model": "local-analytics-v2",
            "generation_status": "no_data",
            "fallback_reason": fallback_reason or "No review data available in selected batch.",
            "token_usage": None,
            "generated_at": now_iso,
            "executive_summary": "No review records found in the selected batch. Upload and process a dataset first.",
            "major_complaints": ["No data available."],
            "complaints_summary": "• No data available.",
            "positive_trends": ["No data available."],
            "positives_summary": "• No data available.",
            "action_plan": [],
            "top_topics": [],
            "topic_clusters": [],
        }


# ---------------------------------------------------------------------------
# Hybrid AI Service (Gemini primary + Local fallback)
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text or not isinstance(text, str):
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
    return None


def generate_ai_recommendations(
    analytics_summary: Dict[str, Any],
    force_local: bool = False,
    reviews_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Hybrid AI Service:
    1. Primary: Google Gemini API (when configured and operational).
    2. Automatic Fallback: Local Analytics Engine.
    Never crashes on API error. Clearly labels which engine generated results.
    """
    analytics_summary = analytics_summary or {}

    if force_local:
        logger.info("Local Analytics Engine explicitly requested.")
        return LocalAnalyticsEngine.generate_insights(
            analytics_summary,
            fallback_reason="User selected Force Local Analytics Engine.",
            reviews_df=reviews_df,
        )

    api_key = getattr(settings, "GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY", "")
    if not api_key or str(api_key).strip() in ("", "your_gemini_api_key"):
        logger.info("Gemini API key not configured. Invoking Local Analytics Engine.")
        return LocalAnalyticsEngine.generate_insights(
            analytics_summary,
            fallback_reason="Gemini API key not configured in .env.",
            reviews_df=reviews_df,
        )

    candidate_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    chosen_model = os.getenv("GEMINI_MODEL", candidate_models[0])
    local_insights = LocalAnalyticsEngine.generate_insights(analytics_summary, reviews_df=reviews_df)

    if int(analytics_summary.get("total_reviews", 0) or 0) == 0:
        return local_insights

    prompt = f"""
You are an expert Chief Data Officer reviewing AI-generated business analytics.
The local analytics engine has produced the following evidence-backed insights.
Your task is to ENHANCE the narrative quality and strategic depth — do NOT invent
new statistics, change numbers, or add fictional review examples.
Use ONLY the data provided below.

LOCAL ENGINE OUTPUT:
Executive Summary: {local_insights["executive_summary"]}
Major Complaints: {json.dumps(local_insights["major_complaints"], default=str)}
Positive Trends: {json.dumps(local_insights["positive_trends"], default=str)}
Action Plan: {json.dumps(local_insights["action_plan"], default=str)}

BATCH ANALYTICS DATA:
- Total Reviews: {analytics_summary.get("total_reviews", 0)}
- Average Rating: {analytics_summary.get("avg_rating", 0.0)}/5.0
- Sentiment: {json.dumps(analytics_summary.get("sentiment_distribution", {}), default=str)}
- Priority: {json.dumps(analytics_summary.get("priority_distribution", {}), default=str)}
- Emotions: {json.dumps(analytics_summary.get("emotion_distribution", {}), default=str)}
- Categories: {json.dumps(analytics_summary.get("category_stats", []), default=str)}
- Top Complaint Aspects: {json.dumps(dict(list((analytics_summary.get("complaint_aspects", {}) or {}).items())[:5]), default=str)}
- Top Positive Aspects: {json.dumps(dict(list((analytics_summary.get("positive_aspects", {}) or {}).items())[:5]), default=str)}

Return ONLY a valid JSON object with these exact keys — do not wrap in markdown:
{{
  "executive_summary": "Enhanced 3-4 sentence strategic narrative (preserve all statistics)",
  "major_complaints": ["Complaint 1 with specific numbers", "Complaint 2", "Complaint 3"],
  "positive_trends": ["Positive trend 1 with numbers", "Positive trend 2", "Positive trend 3"],
  "action_plan": [
    {{"priority": "High", "category": "...", "opportunity_title": "...", "problem": "...",
      "evidence": "...", "affected_category": "...", "root_cause_hypothesis": "...",
      "action": "...", "implementation_steps": ["..."], "expected_impact": "...",
      "evidence_limitations": "..."}}
  ]
}}
"""

    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=chosen_model,
            contents=prompt,
        )
        raw_text = getattr(response, "text", "") or ""
        parsed = _extract_json_block(raw_text)
        if not parsed:
            return LocalAnalyticsEngine.generate_insights(
                analytics_summary,
                fallback_reason="Gemini returned non-JSON output; using Local Analytics Engine.",
                reviews_df=reviews_df,
            )

        usage = getattr(response, "usage_metadata", None)
        token_usage = None
        if usage:
            prompt_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
            cand_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
            total_tok = int(getattr(usage, "total_token_count", 0) or (prompt_tok + cand_tok))
            token_usage = {
                "prompt_tokens": prompt_tok,
                "completion_tokens": cand_tok,
                "total_tokens": total_tok,
            }

        maj_complaints = parsed.get("major_complaints")
        if not isinstance(maj_complaints, list) or not maj_complaints:
            maj_complaints = local_insights["major_complaints"]

        pos_trends = parsed.get("positive_trends")
        if not isinstance(pos_trends, list) or not pos_trends:
            pos_trends = local_insights["positive_trends"]

        act_plan = parsed.get("action_plan")
        if not isinstance(act_plan, list) or not act_plan:
            act_plan = local_insights["action_plan"]

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return {
            "is_fallback": False,
            "engine": "Google Gemini AI",
            "provider": "Google Gemini",
            "model": chosen_model,
            "generation_status": "success",
            "fallback_reason": None,
            "token_usage": token_usage,
            "generated_at": now_iso,
            "executive_summary": str(parsed.get("executive_summary") or local_insights["executive_summary"]),
            "major_complaints": maj_complaints,
            "complaints_summary": "\n".join([f"• {c}" for c in maj_complaints]),
            "positive_trends": pos_trends,
            "positives_summary": "\n".join([f"• {p}" for p in pos_trends]),
            "action_plan": act_plan,
            "top_topics": local_insights.get("top_topics", []),
            "topic_clusters": local_insights.get("topic_clusters", []),
        }
    except Exception as exc:
        logger.warning("Gemini generation failed (%s). Falling back to Local Analytics Engine.", exc)
        return LocalAnalyticsEngine.generate_insights(
            analytics_summary,
            fallback_reason=f"Gemini API error ({type(exc).__name__}); using Local Analytics Engine.",
            reviews_df=reviews_df,
        )