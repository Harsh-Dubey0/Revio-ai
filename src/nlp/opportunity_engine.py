from typing import Any, Dict, List, Optional


def build_business_opportunities(
    summary: Dict[str, Any],
    topics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Convert NLP/ML batch summary and topic clusters into structured business insights."""
    total = int(summary.get("total_reviews", 0) or 0)
    if total <= 0:
        return {
            "executive_summary": "Upload customer reviews to uncover pain points, positive drivers, and growth opportunities.",
            "complaints_summary": "No customer feedback analyzed yet.",
            "positives_summary": "No positive highlights available yet.",
            "action_items": [],
            "top_topics": [],
        }

    avg_rating = float(summary.get("avg_rating", 0.0) or 0.0)
    sent = summary.get("sentiment_distribution", {}) or {}
    prio = summary.get("priority_distribution", {}) or {}
    emotions = summary.get("emotion_distribution", {}) or {}
    complaints = summary.get("complaint_aspects", {}) or {}
    positives = summary.get("positive_aspects", {}) or {}
    categories = summary.get("category_stats", []) or []

    pos_cnt = int(sent.get("Positive", 0))
    neg_cnt = int(sent.get("Negative", 0))
    high_prio_cnt = int(prio.get("High", 0))

    pos_pct = round((pos_cnt / total) * 100, 1)
    neg_pct = round((neg_cnt / total) * 100, 1)
    high_pct = round((high_prio_cnt / total) * 100, 1)

    sorted_complaints = sorted(complaints.items(), key=lambda x: x[1], reverse=True)
    sorted_positives = sorted(positives.items(), key=lambda x: x[1], reverse=True)

    topic_names = [t.get("topic") for t in (topics or []) if t.get("topic")][:4]
    if not topic_names:
        topic_names = [a for a, _ in sorted_complaints[:2]] + [a for a, _ in sorted_positives[:2]]

    action_items: List[Dict[str, Any]] = []

    if sorted_complaints:
        top_asp, top_cnt = sorted_complaints[0]
        asp_pct = round((top_cnt / total) * 100, 1)
        action_items.append({
            "priority": "High" if asp_pct >= 12 or high_pct >= 15 else "Medium",
            "category": top_asp,
            "opportunity_title": f"Fix Recurring '{top_asp}' Friction",
            "action": (
                f"Audit '{top_asp}' touchpoints flagged in {top_cnt:,} reviews ({asp_pct}% of feedback) "
                f"and set clear turnaround targets to reduce repeat complaints."
            ),
            "rationale": f"Targets {top_cnt:,} complaint mentions ({asp_pct}%) and helps lift customer satisfaction above {avg_rating:.2f}/5.0.",
        })

    if high_prio_cnt > 0:
        action_items.append({
            "priority": "High",
            "category": "Customer Retention",
            "opportunity_title": "Fast-Track Urgent Customer Follow-Ups",
            "action": (
                f"Prioritize outreach and resolution for {high_prio_cnt:,} high-priority reviews ({high_pct}% of batch) "
                f"before dissatisfied customers churn."
            ),
            "rationale": f"Protects high-risk accounts across {neg_cnt:,} negative reviews ({neg_pct}%).",
        })

    if categories and len(categories) > 1:
        sorted_cats = sorted(categories, key=lambda c: float(c.get("avg_rating", 0.0)))
        lowest_cat = sorted_cats[0]
        highest_cat = sorted_cats[-1]
        low_name = lowest_cat.get("product_category", "General")
        low_rating = float(lowest_cat.get("avg_rating", 0.0))
        high_name = highest_cat.get("product_category", "General")
        if low_name != high_name and low_rating < 4.0:
            action_items.append({
                "priority": "Medium",
                "category": low_name,
                "opportunity_title": f"Close Rating Gap in '{low_name}'",
                "action": (
                    f"Apply successful quality and service habits from '{high_name}' to improve "
                    f"'{low_name}' (currently {low_rating:.2f}/5.0 across {int(lowest_cat.get('count', 0)):,} reviews)."
                ),
                "rationale": f"Brings underperforming '{low_name}' closer to top-performing '{high_name}'.",
            })

    if sorted_positives:
        pos_asp, pos_cnt_asp = sorted_positives[0]
        pos_asp_pct = round((pos_cnt_asp / total) * 100, 1)
        action_items.append({
            "priority": "Medium",
            "category": pos_asp,
            "opportunity_title": f"Spotlight '{pos_asp}' in Customer Growth",
            "action": (
                f"Feature '{pos_asp}'—praised in {pos_cnt_asp:,} positive reviews ({pos_asp_pct}%)—"
                f"in product messaging, onboarding, and referral campaigns."
            ),
            "rationale": f"Builds on {pos_pct}% positive sentiment to convert happy customers into advocates.",
        })

    dominant_emo = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "Neutral"

    exec_summary = (
        f"Across {total:,} customer reviews (avg rating {avg_rating:.2f}/5.0), {pos_pct}% of feedback is positive "
        f"and {neg_pct}% is negative, with '{dominant_emo}' standing out as the main customer mood. "
        f"Addressing top friction areas while scaling proven strengths creates the clearest path to higher retention."
    )

    complaints_lines = [f"• **{asp}**: {cnt:,} complaints ({round(cnt / total * 100, 1)}%)" for asp, cnt in sorted_complaints[:4]]
    positives_lines = [f"• **{asp}**: {cnt:,} positive mentions ({round(cnt / total * 100, 1)}%)" for asp, cnt in sorted_positives[:4]]

    return {
        "executive_summary": exec_summary,
        "complaints_summary": "\n".join(complaints_lines) if complaints_lines else "No major recurring complaints identified.",
        "positives_summary": "\n".join(positives_lines) if positives_lines else "No standout positive highlights identified yet.",
        "action_items": action_items,
        "top_topics": topic_names,
    }