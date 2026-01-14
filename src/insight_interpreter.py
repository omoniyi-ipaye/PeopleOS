"""
Insight Interpreter module for PeopleOS.

Translates statistical metrics and visualizations into plain language
insights that HR professionals can immediately understand and act upon.
Uses local LLM (Ollama) for contextual explanations.
"""

from typing import Any
import json

from src.logger import get_logger

logger = get_logger('insight_interpreter')


# HR-friendly benchmark thresholds
BENCHMARKS = {
    'turnover_rate': {
        'low': 0.10,
        'high': 0.20,
        'industry_avg': 0.15
    },
    'tenure_mean': {
        'low': 2.0,
        'high': 5.0,
        'industry_avg': 3.5
    },
    'rating_mean': {
        'low': 3.0,
        'high': 4.0,
        'target': 3.5
    },
    'f1_score': {
        'low': 0.5,
        'good': 0.7,
        'excellent': 0.85
    }
}


class InsightInterpreter:
    """
    Translates technical HR metrics into plain language insights.
    Uses local LLM for contextual, human-friendly explanations.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the Insight Interpreter.
        
        Args:
            llm_client: Optional LLMClient instance for contextual explanations.
        """
        self.llm_client = llm_client
        self.llm_available = llm_client is not None and llm_client.is_available
        logger.info(f"InsightInterpreter initialized. LLM available: {self.llm_available}")

    def interpret_metric(self, metric_name: str, value: Any, context: dict = None) -> str:
        """
        Generate a plain language interpretation of a metric.
        
        Args:
            metric_name: Name of the metric (e.g., 'turnover_rate', 'f1_score').
            value: The metric value.
            context: Optional additional context (e.g., department, time period).
            
        Returns:
            Plain language explanation string.
        """
        if not self.llm_available:
            raise RuntimeError("Contextual insight unavailable: LLM client not connected")
        
        return self._get_llm_interpretation(metric_name, value, context)


    def _get_llm_interpretation(self, metric_name: str, value: Any, context: dict = None) -> str:
        """Get LLM-powered contextual interpretation."""
        try:
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
                options={'num_predict': 200, 'temperature': 0.4}
            )
            
            explanation = response.get('response', '').strip()
            
            # Clean up common AI preamble like "Here is the explanation..."
            if ":" in explanation and len(explanation.split(":")[0].split()) < 10:
                potential_pithy = explanation.split(":", 1)[1].strip()
                if potential_pithy:
                    explanation = potential_pithy

            # Error if LLM returns empty or too long (relaxed limit to 500)
            if not explanation:
                raise RuntimeError("LLM interpretation returned empty content")
            
            if len(explanation) > 500:
                logger.warning(f"LLM interpretation too long ({len(explanation)} chars). Truncating.")
                explanation = explanation[:497] + "..."
            
            return explanation

        except Exception as e:
            logger.error(f"LLM interpretation failed: {e}")
            raise RuntimeError(f"Contextual interpretation failed: {str(e)}")

    def interpret_chart(self, chart_type: str, data_summary: dict) -> str:
        """
        Generate a plain language summary of what a chart shows.
        
        Args:
            chart_type: Type of chart (e.g., 'bar', 'pie', 'scatter').
            data_summary: Summary statistics of the chart data.
            
        Returns:
            Plain language explanation of the chart's key insight.
        """
        if not self.llm_available:
            raise RuntimeError("Chart insight unavailable: LLM client not connected")
        
        try:
            prompt = f"""You are an HR advisor explaining a chart to a non-technical HR manager.

Chart type: {chart_type}
Data summary: {json.dumps(data_summary)}

In 1-2 sentences, explain:
1. What pattern or trend does this chart show?
2. What's the key takeaway for HR decisions?

Be concise and avoid technical terms. Plain language explanation:"""

            response = self.llm_client.client.generate(
                model=self.llm_client.model,
                prompt=prompt,
                options={'num_predict': 150, 'temperature': 0.6}
            )
            
            explanation = response.get('response', '').strip()
            if explanation:
                return explanation
            raise RuntimeError("LLM chart interpretation returned empty content")

        except Exception as e:
            logger.error(f"Chart interpretation failed: {e}")
            raise RuntimeError(f"Chart interpretation failed: {str(e)}")


    def get_key_takeaways(self, analytics_data: dict) -> list:
        """
        Generate key takeaways from overall analytics data.
        
        Args:
            analytics_data: Complete analytics results dictionary.
            
        Returns:
            List of 3-5 key insights in plain language.
        """
        takeaways = []
        
        summary = analytics_data.get('summary', {})
        
        # Workforce size context
        headcount = summary.get('headcount', 0)
        if headcount:
            takeaways.append(f"📊 You're managing a workforce of {headcount} employees.")
        
        # Turnover insight
        turnover = summary.get('turnover_rate')
        if turnover:
            if turnover > 0.20:
                takeaways.append(f"⚠️ Turnover at {turnover*100:.0f}% is above average - retention efforts needed.")
            elif turnover < 0.10:
                takeaways.append(f"✅ Low turnover ({turnover*100:.0f}%) indicates strong employee satisfaction.")
        
        # Performance insight
        rating = summary.get('lastrating_mean')
        if rating:
            if rating >= 4.0:
                takeaways.append(f"🌟 Team performance is strong (avg rating: {rating:.1f}/5).")
            elif rating < 3.0:
                takeaways.append(f"📈 Performance development needed (avg rating: {rating:.1f}/5).")
        
        # High risk insight
        high_risk = summary.get('attrition_count', 0)
        if high_risk and headcount:
            pct = (high_risk / headcount) * 100
            if pct > 15:
                takeaways.append(f"🔴 {high_risk} employees ({pct:.0f}%) at high attrition risk - prioritize engagement.")
        
        return takeaways[:5]  # Limit to 5 takeaways
