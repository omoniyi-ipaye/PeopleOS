"""Strategic advisor route handlers (LLM-powered insights)."""

from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.dependencies import get_app_state, AppState
from src.logger import get_logger

logger = get_logger('advisor_route')

router = APIRouter(prefix="/api/advisor", tags=["advisor"])


class StrategicSummary(BaseModel):
    """AI-generated strategic summary."""
    summary: str
    key_insights: List[str]
    action_items: List[str]
    generated_by: str


class AdvisorStatus(BaseModel):
    """Status of the AI advisor."""
    available: bool
    model: str | None
    reason: str | None


def require_llm(state: AppState = Depends(get_app_state)) -> AppState:
    """Dependency that requires LLM client."""
    if not state.has_data():
        if not state.load_from_database():
            raise HTTPException(
                status_code=400,
                detail="No data loaded. Please upload a file first."
            )

    if state.llm_client is None or not state.llm_client.is_available:
        raise HTTPException(
            status_code=503,
            detail="AI advisor not available. Ensure Ollama is running."
        )

    return state


@router.get("/status", response_model=AdvisorStatus)
async def get_advisor_status(
    state: AppState = Depends(get_app_state)
) -> AdvisorStatus:
    """
    Check if AI advisor (Ollama) is available.
    """
    if state.llm_client is None:
        return AdvisorStatus(
            available=False,
            model=None,
            reason="LLM client not initialized"
        )

    if not state.llm_client.is_available:
        return AdvisorStatus(
            available=False,
            model=None,
            reason="Ollama not running or not accessible"
        )

    return AdvisorStatus(
        available=True,
        model=state.llm_client.model,
        reason=None
    )


@router.get("/summary", response_model=StrategicSummary)
async def get_strategic_summary(
    state: AppState = Depends(require_llm)
) -> StrategicSummary:
    """
    Get AI-generated strategic summary of HR analytics.
    """
    # Gather context for the LLM
    context = _build_analytics_context(state)

    # Generate summary using LLM
    prompt = f"""
    You are an HR analytics expert. Based on the following HR metrics, provide a strategic summary
    with key insights and recommended actions.

    {context}

    Provide your response in the following format:
    SUMMARY: [2-3 sentence executive summary]
    KEY INSIGHTS:
    - [insight 1]
    - [insight 2]
    - [insight 3]
    ACTION ITEMS:
    - [action 1]
    - [action 2]
    - [action 3]

    Be specific, data-driven, and ensure each sentence is complete. Do not truncate your response.
    """

    try:
        response = state.llm_client.generate(prompt)

        # Parse response
        summary, insights, actions = _parse_llm_response(response)

        return StrategicSummary(
            summary=summary,
            key_insights=insights,
            action_items=actions,
            generated_by=state.llm_client.model
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


@router.post("/ask")
async def ask_advisor(
    question: str,
    state: AppState = Depends(require_llm)
) -> Dict[str, Any]:
    """
    Ask the AI advisor a specific question about the HR data.
    """
    # Gather context
    context = _build_analytics_context(state)

    prompt = f"""
    You are an HR analytics expert. Based on the following HR metrics, answer the user's question.

    HR Metrics:
    {context}

    User Question: {question}

    Provide a helpful, data-driven response.
    """

    try:
        response = state.llm_client.generate(prompt)

        return {
            'question': question,
            'answer': response,
            'model': state.llm_client.model
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")


def _build_analytics_context(state: AppState) -> str:
    """Build context string from current analytics."""
    context_parts = []

    # Summary stats
    if state.analytics_engine:
        stats = state.analytics_engine.get_summary_statistics()
        context_parts.append(f"Headcount: {stats.get('headcount', 'N/A')}")
        if stats.get('turnover_rate'):
            context_parts.append(f"Turnover Rate: {stats['turnover_rate']:.1%}")
        if stats.get('tenure_mean'):
            context_parts.append(f"Average Tenure: {stats['tenure_mean']:.1f} years")
        if stats.get('salary_mean'):
            context_parts.append(f"Average Salary: ${stats['salary_mean']:,.0f}")

    # Risk distribution
    if state.risk_scores is not None:
        high_risk = len(state.risk_scores[state.risk_scores['risk_category'] == 'High'])
        total = len(state.risk_scores)
        context_parts.append(f"High-Risk Employees: {high_risk} ({high_risk/total*100:.1f}%)")

    # Model performance
    if state.model_metrics:
        context_parts.append(f"Model F1 Score: {state.model_metrics.get('f1', 0):.2f}")

    return "\n".join(context_parts)


def _parse_llm_response(response: str) -> tuple[str, List[str], List[str]]:
    """Parse structured LLM response into components."""
    summary = ""
    insights = []
    actions = []

    current_section = 'summary'  # Assume summary starts first

    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue

        line_upper = line.upper()

        if 'SUMMARY:' in line_upper:
            current_section = 'summary'
            content = line.split(':', 1)[1].strip()
            if content:
                summary = content
        elif 'KEY INSIGHTS:' in line_upper:
            current_section = 'insights'
        elif 'ACTION ITEMS:' in line_upper:
            current_section = 'actions'
        elif line.startswith(('-', '•', '*', '1.', '2.', '3.')):
            item = line.lstrip('-•*0123456789. ').strip()
            if item:
                if current_section == 'insights':
                    insights.append(item)
                elif current_section == 'actions':
                    actions.append(item)
                else:
                    # If we found a bullet but no section, assume insights
                    insights.append(item)
        elif current_section == 'summary':
            if summary:
                summary += ' ' + line
            else:
                summary = line

    # Fallback: if summary is still empty, use first few lines
    if not summary and response:
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        summary = " ".join(lines[:2])

    # Final validation - be more lenient but log failures
    if not summary:
        summary = "Strategic summary unavailable. Review the key metrics below."
    
    if not insights and not actions:
        logger.warning(f"Failed to parse insights/actions from response: {response[:100]}")
        insights = ["Monitor workforce trends closely", "Review department-level turnover"]
        actions = ["Validate data consistency", "Schedule workforce review session"]

    # Sanitize markdown bold markers (**) from final strings
    summary = summary.replace('**', '')
    insights = [i.replace('**', '') for i in insights]
    actions = [a.replace('**', '') for a in actions]

    return summary, insights, actions
