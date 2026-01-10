"""
LLM Client module for PeopleOS.

Provides integration with Ollama for local LLM-powered insights.
Model-agnostic design allows any Ollama-compatible model.
"""

import json
from typing import Any, Optional

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('llm_client')



# Prohibited content patterns
PROHIBITED_PATTERNS = [
    "terminate", "termination", "fire", "dismiss", "dismissal",
    "disciplinary", "pip", "performance improvement plan",
    "salary reduction", "demotion", "punitive"
]


class LLMClientError(Exception):
    """Custom exception for LLM client errors."""
    pass


class LLMClient:
    """
    Client for interacting with local Ollama LLM.
    
    Model-agnostic design - works with any Ollama model.
    """
    
    def __init__(self):
        """Initialize the LLM Client with configuration."""
        self.config = load_config()
        self.ollama_config = self.config.get('ollama', {})
        self.host = self.ollama_config.get('host', 'http://localhost:11434')
        self.model = self.ollama_config.get('model', 'llama3')
        self.timeout = self.ollama_config.get('timeout', 30)
        self.max_retries = self.ollama_config.get('max_retries', 3)
        self.max_tokens = self.ollama_config.get('response_max_tokens', 500)
        self.is_available = False
        self._check_availability()
    
    def _check_availability(self) -> None:
        """Check if Ollama is available."""
        try:
            import ollama
            # Create client with configured host
            self.client = ollama.Client(host=self.host)
            # Try to list models to verify connection
            self.client.list()
            self.is_available = True
            logger.info(f"Ollama available at {self.host} with model {self.model}")
        except ImportError:
            logger.warning("Ollama package not installed")
            self.is_available = False
            self.client = None
        except Exception as e:
            logger.warning(f"Ollama not available: {str(e)}")
            self.is_available = False
            self.client = None

    def generate(self, prompt: str, **kwargs) -> Any:
        """
        Generate a completion using Ollama.
        
        Args:
            prompt: The text prompt.
            **kwargs: Additional options for generate call.
            
        Returns:
            The raw response string.
        """
        if not self.is_available or self.client is None:
            raise LLMClientError("LLM client not available")
            
        try:
            # Merge default options with kwargs
            options = {
                'num_predict': self.max_tokens,
                **kwargs.get('options', {})
            }
            
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options=options
            )
            return response.get('response', '')
        except Exception as e:
            logger.error(f"Generate failed: {str(e)}")
            raise LLMClientError(f"LLM generation failed: {str(e)}")
    
    def _validate_response(self, response: str) -> tuple[bool, str]:
        """
        Validate LLM response for prohibited content.
        
        Args:
            response: Raw LLM response text.
            
        Returns:
            Tuple of (is_valid, cleaned_response).
        """
        response_lower = response.lower()
        
        # Check for prohibited content
        for pattern in PROHIBITED_PATTERNS:
            if pattern in response_lower:
                logger.warning(f"Prohibited content detected: {pattern}")
                return False, ""
        
        # Limit to 3 bullet points
        lines = response.strip().split('\n')
        bullet_lines = [l for l in lines if l.strip().startswith(('-', '•', '*', '1', '2', '3'))]
        
        if len(bullet_lines) > 3:
            bullet_lines = bullet_lines[:3]
            response = '\n'.join(bullet_lines)
        
        return True, response
    
    def _build_prompt(self, request_type: str, metrics: dict, context: str = "attrition_risk_analysis") -> str:
        """
        Build a structured prompt for the LLM.
        
        Args:
            request_type: Type of request (strategic_summary, action_items, trend_interpretation).
            metrics: Dictionary of metrics to analyze.
            context: Analysis context.
            
        Returns:
            Formatted prompt string.
        """
        prompt_template = """You are a strategic HR advisor. Analyze the following workforce metrics and provide insights.

CONTEXT: {context}
REQUEST TYPE: {request_type}

METRICS:
{metrics_json}

CONSTRAINTS:
- Maximum 3 bullet points
- No termination or disciplinary recommendations
- No specific employee names
- Focus on systemic interventions
- Use professional HR terminology
- Be concise and actionable

Provide your analysis:"""
        
        metrics_json = json.dumps(metrics, indent=2)
        
        return prompt_template.format(
            context=context,
            request_type=request_type,
            metrics_json=metrics_json
        )
    
    def get_strategic_summary(self, metrics: dict) -> dict:
        """
        Get a strategic summary from the LLM.

        Args:
            metrics: Dictionary with workforce metrics.

        Returns:
            Dictionary with status and response.
        """
        if not self.is_available or self.client is None:
            raise LLMClientError("LLM client not available or not properly initialized")

        try:
            prompt = self._build_prompt("strategic_summary", metrics)

            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'num_predict': self.max_tokens,
                    'temperature': 0.7
                }
            )

            raw_response = response.get('response', '')
            is_valid, cleaned = self._validate_response(raw_response)

            if not is_valid:
                raise LLMClientError("LLM response failed safety validation (prohibited content detected)")

            return {
                "status": "success",
                "message": cleaned,
                "model": self.model
            }

        except Exception as e:
            logger.error(f"LLM request failed: {str(e)}")
            raise LLMClientError(f"LLM strategic summary generation failed: {str(e)}")
    
    def get_action_items(self, metrics: dict) -> dict:
        """
        Get recommended action items from the LLM.

        Args:
            metrics: Dictionary with workforce metrics.

        Returns:
            Dictionary with status and recommendations.
        """
        if not self.is_available or self.client is None:
            raise LLMClientError("LLM client not available for action items")

        try:
            prompt = self._build_prompt("action_items", metrics)

            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'num_predict': self.max_tokens,
                    'temperature': 0.5
                }
            )
            
            raw_response = response.get('response', '')
            is_valid, cleaned = self._validate_response(raw_response)
            
            if not is_valid:
                raise LLMClientError("LLM response for action items failed safety validation")
            
            # Parse bullet points into list
            lines = cleaned.strip().split('\n')
            recommendations = []
            for line in lines:
                line = line.strip()
                if line and line[0] in '-•*123':
                    # Remove bullet character
                    clean_line = line.lstrip('-•*0123456789. ')
                    if clean_line:
                        recommendations.append(clean_line)
            
            return {
                "status": "success",
                "recommendations": recommendations[:3],
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"LLM request failed: {str(e)}")
            raise LLMClientError(f"LLM action items generation failed: {str(e)}")
    
    def get_trend_interpretation(self, metrics: dict, trend_data: dict) -> dict:
        """
        Get interpretation of trend data from the LLM.

        Args:
            metrics: Dictionary with current metrics.
            trend_data: Dictionary with trend/historical data.

        Returns:
            Dictionary with status and interpretation.
        """
        if not self.is_available or self.client is None:
            raise LLMClientError("LLM client not available for trend interpretation")

        combined_metrics = {
            'current': metrics,
            'trends': trend_data
        }

        try:
            prompt = self._build_prompt("trend_interpretation", combined_metrics)

            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'num_predict': self.max_tokens,
                    'temperature': 0.6
                }
            )
            
            raw_response = response.get('response', '')
            is_valid, cleaned = self._validate_response(raw_response)
            
            if not is_valid:
                raise LLMClientError("LLM response for trend interpretation failed safety validation")
            
            return {
                "status": "success",
                "message": cleaned,
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"LLM request failed: {str(e)}")
            raise LLMClientError(f"LLM trend interpretation failed: {str(e)}")
    
    def generate_insight(self, request_type: str, data: dict) -> dict:
        """
        Generic method to generate insights.
        
        Args:
            request_type: Type of insight to generate.
            data: Data to analyze.
            
        Returns:
            Dictionary with insight response.
        """
        if request_type == "strategic_summary":
            return self.get_strategic_summary(data)
        elif request_type == "action_items":
            return self.get_action_items(data)
        elif request_type == "trend_interpretation":
            return self.get_trend_interpretation(data.get('metrics', {}), data.get('trends', {}))
        else:
            raise LLMClientError(f"Unknown request type: {request_type}")
    
    def is_connected(self) -> bool:
        """
        Check if LLM is connected.
        
        Returns:
            True if Ollama is available.
        """
        return self.is_available
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "model": self.model,
            "host": self.host,
            "available": self.is_available
        }

    def get_executive_briefing(self, metrics: dict, analytics_data: dict) -> dict:
        """
        Generate a structured executive briefing for leadership presentations.
        
        Args:
            metrics: Dictionary with workforce metrics.
            analytics_data: Additional analytics context.
            
        Returns:
            Dictionary with structured sections for executive presentation.
        """
        if not self.is_available or self.client is None:
            raise LLMClientError("LLM client not available for executive briefing")

        # Build comprehensive prompt
        turnover_str = f"{metrics.get('turnover_rate', 0) * 100:.1f}%" if metrics.get('turnover_rate') else 'N/A'
        tenure_str = f"{metrics.get('tenure_mean', 0):.1f} years" if metrics.get('tenure_mean') else 'N/A'
        rating_str = f"{metrics.get('lastrating_mean', 0):.1f}" if metrics.get('lastrating_mean') else 'N/A'
        
        prompt = f"""You are a Chief People Officer preparing a strategic workforce briefing for the executive leadership team.

WORKFORCE DATA:
- Total Headcount: {metrics.get('headcount', 'N/A')}
- Turnover Rate: {turnover_str}
- High-Risk Employees: {metrics.get('attrition_count', 'N/A')}
- Average Tenure: {tenure_str}
- Average Rating: {rating_str}
- Departments: {metrics.get('department_count', 'N/A')}


Provide a structured executive briefing with EXACTLY these four sections. Be specific, data-driven, and actionable.

SECTION 1 - SITUATION ANALYSIS (2-3 sentences):
Summarize the current workforce state. Reference specific metrics.

SECTION 2 - KEY RISKS (exactly 3 bullet points):
Identify the top 3 workforce risks requiring attention. Be specific about impact.

SECTION 3 - STRATEGIC OPPORTUNITIES (exactly 3 bullet points):
Highlight 3 opportunities for workforce optimization or growth.

SECTION 4 - RECOMMENDED ACTIONS (exactly 3 bullet points, prioritized):
Provide 3 specific, actionable recommendations with clear ownership suggestions.

CONSTRAINTS:
- Use executive language suitable for board presentations
- No termination or disciplinary recommendations
- Focus on systemic improvements, not individual employees
- Be concise but substantive

Begin your response:"""

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'num_predict': 1500,
                    'temperature': 0.7
                }
            )

            raw_response = response.get('response', '')
            
            # Parse sections from response
            sections = self._parse_executive_sections(raw_response)
            
            return {
                "status": "success",
                "sections": sections,
                "model": self.model
            }

        except Exception as e:
            logger.error(f"Executive briefing generation failed: {str(e)}")
            raise LLMClientError(f"Executive briefing generation failed: {str(e)}")

    def _parse_executive_sections(self, response: str) -> dict:
        """Parse LLM response into structured sections."""
        sections = {
            "situation": "",
            "risks": [],
            "opportunities": [],
            "actions": []
        }
        
        lines = response.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            
            # Detect section headers
            if 'situation' in line_lower and ('analysis' in line_lower or 'section 1' in line_lower):
                current_section = 'situation'
            elif 'risk' in line_lower and ('key' in line_lower or 'section 2' in line_lower):
                current_section = 'risks'
            elif 'opportunit' in line_lower and ('strategic' in line_lower or 'section 3' in line_lower):
                current_section = 'opportunities'
            elif 'action' in line_lower and ('recommend' in line_lower or 'section 4' in line_lower):
                current_section = 'actions'
            elif current_section:
                # Add content to current section
                if current_section == 'situation':
                    if sections['situation']:
                        sections['situation'] += ' ' + line
                    else:
                        sections['situation'] = line
                else:
                    # For list sections, extract bullet points
                    if line and line[0] in '-•*123456789':
                        clean_line = line.lstrip('-•*0123456789. ')
                        if clean_line:
                            sections[current_section].append(clean_line)
        
        # Ensure we have content
        if not sections['situation'] or not any(sections[k] for k in ['risks', 'opportunities', 'actions']):
            raise LLMClientError("Failed to parse meaningful sections from LLM response")
        
        return sections
