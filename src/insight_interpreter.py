"""Insight Interpreter module for PeopleOS.

Translates statistical metrics and visualizations into plain language insights.
LLM synthesis is optional: deterministic explanations are always available so
core analytics never depend on Ollama availability.
"""

from typing import Any
import json
import math

from src.logger import get_logger

logger = get_logger('insight_interpreter')


BENCHMARKS = {
    'turnover_rate': {'low': 0.10, 'high': 0.20, 'industry_avg': 0.15},
    'tenure_mean': {'low': 2.0, 'high': 5.0, 'industry_avg': 3.5},
    'rating_mean': {'low': 3.0, 'high': 4.0, 'target': 3.5},
    'f1_score': {'low': 0.5, 'good': 0.7, 'excellent': 0.85},
}


class InsightInterpreter:
    """Translate technical metrics into plain-language workforce context."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.llm_available = llm_client is not None and llm_client.is_available
        logger.info(f"InsightInterpreter initialized. LLM available: {self.llm_available}")

    def interpret_metric(self, metric_name: str, value: Any, context: dict = None) -> str:
        """Return an explanation without making the LLM a runtime dependency."""
        return self._get_deterministic_interpretation(metric_name, value, context)

    def _get_deterministic_interpretation(self, metric_name: str, value: Any, context: dict = None) -> str:
        if value is None:
            return "There is not enough data to interpret this metric reliably."

        label = metric_name.replace('_', ' ').strip()
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return f"{label.capitalize()} is available in the current dataset."

        if not math.isfinite(numeric):
            return 'There is not enough data to interpret this metric reliably.'
        if metric_name == 'headcount':
            return f"The active workforce contains {int(numeric):,} people. Use department and location breakdowns to understand where that workforce is concentrated."
        if metric_name == 'turnover_rate':
            return f"Observed attrition share is {numeric * 100:.1f}% among records with known outcomes. A period turnover rate requires dated events and a defined exposure denominator."
        if metric_name == 'tenure_mean':
            if numeric < BENCHMARKS['tenure_mean']['low']:
                return f"Average tenure is {numeric:.1f} years, suggesting a relatively new or fast-changing workforce. Check onboarding and early-tenure retention."
            return f"Average tenure is {numeric:.1f} years. Compare tenure by department to identify unusually new or stagnant parts of the organisation."
        if metric_name in {'lastrating_mean', 'rating_mean'}:
            if numeric < BENCHMARKS['rating_mean']['low']:
                return f"The average recorded rating is {numeric:.1f}/5, which warrants a closer look at performance support and rating patterns."
            if numeric >= BENCHMARKS['rating_mean']['high']:
                return f"The average recorded rating is {numeric:.1f}/5. Check distribution and calibration before treating the high average as uniformly strong performance."
            return f"The average recorded rating is {numeric:.1f}/5. Review team-level distribution for meaningful differences."
        if metric_name == 'f1_score':
            return f"The model F1 score is {numeric * 100:.1f}%. Treat model output as decision support and review the model's evaluation state before relying on predictions."

        return f"{label.capitalize()} is {numeric:,.2f}. Interpret it alongside the relevant workforce breakdown rather than in isolation."

    def _get_llm_interpretation(self, metric_name: str, value: Any, context: dict = None) -> str:
        prompt = f"""You are an HR advisor explaining analytics to a non-technical HR manager.

Explain this metric in exactly 1-2 sentences using plain language:
- Metric: {metric_name}
- Value: {value}
- Context: {json.dumps(context) if context else 'General workforce analysis'}

Guidelines:
- Be extremely concise and actionable
- Avoid statistical jargon and conversational filler
- Explain what this means for day-to-day HR decisions
- Mention if this is good, concerning, or needs attention

Plain language explanation (1-2 sentences only):"""
        response = self.llm_client.client.generate(
            model=self.llm_client.model,
            prompt=prompt,
            options={'num_predict': 200, 'temperature': 0.4},
        )
        explanation = response.get('response', '').strip()
        if ':' in explanation and len(explanation.split(':')[0].split()) < 10:
            potential = explanation.split(':', 1)[1].strip()
            if potential:
                explanation = potential
        if not explanation:
            raise RuntimeError("LLM interpretation returned empty content")
        if len(explanation) > 500:
            explanation = explanation[:497] + '...'
        return explanation

    def interpret_chart(self, chart_type: str, data_summary: dict) -> str:
        if not self.llm_available:
            return self._deterministic_chart_summary(chart_type, data_summary)
        try:
            prompt = f"""You are an HR advisor explaining a chart to a non-technical HR manager.

Chart type: {chart_type}
Data summary: {json.dumps(data_summary)}

In 1-2 sentences, explain the main pattern and the decision-relevant takeaway. Avoid technical terms."""
            response = self.llm_client.client.generate(
                model=self.llm_client.model,
                prompt=prompt,
                options={'num_predict': 150, 'temperature': 0.6},
            )
            explanation = response.get('response', '').strip()
            if explanation:
                return explanation
        except Exception as exc:
            logger.warning("Falling back to deterministic chart interpretation: %s", exc)
        return self._deterministic_chart_summary(chart_type, data_summary)

    @staticmethod
    def _deterministic_chart_summary(chart_type: str, data_summary: dict) -> str:
        if not data_summary:
            return "This view does not yet contain enough data for a reliable pattern summary."
        keys = list(data_summary)[:3]
        dimensions = ', '.join(str(key).replace('_', ' ') for key in keys)
        return f"This {chart_type} view compares {dimensions}. Use the largest differences as investigation prompts rather than treating the chart alone as a decision."

    def get_key_takeaways(self, analytics_data: dict) -> list:
        """Generate deterministic takeaways from either a summary dict or wrapper."""
        summary = analytics_data.get('summary', analytics_data)
        takeaways = []

        headcount = summary.get('headcount', 0)
        if headcount:
            takeaways.append(f"Current workforce: {int(headcount):,} people.")

        turnover = summary.get('turnover_rate')
        if turnover is not None and math.isfinite(float(turnover)):
            takeaways.append(f"Observed attrition share: {turnover * 100:.1f}% among known outcomes; no period turnover rate is established.")

        rating = summary.get('lastrating_mean')
        if rating is not None:
            takeaways.append(f"Average recorded performance rating: {rating:.1f}/5.")

        tenure = summary.get('tenure_mean')
        if tenure is not None:
            takeaways.append(f"Average tenure: {tenure:.1f} years.")

        return takeaways[:5]
