"""
NLP Engine module for PeopleOS.

Provides NLP-powered analysis of performance review text using Ollama.
Includes sentiment analysis, skill extraction, topic modeling, and AI summaries.
"""

import json
import re
from typing import Any, Optional

import pandas as pd
import numpy as np
from textblob import TextBlob

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('nlp_engine')


class NLPEngineError(Exception):
    """Custom exception for NLP engine errors."""
    pass


class NLPEngine:
    """
    NLP processing engine for performance review analysis.

    Uses Ollama LLM for text analysis including sentiment, skills, and topics.
    """

    def __init__(self, llm_client):
        """
        Initialize NLP Engine.

        Args:
            llm_client: Initialized LLMClient instance.
        """
        self.llm_client = llm_client
        self.config = load_config()
        self.nlp_config = self.config.get('nlp', {})
        self.batch_size = self.nlp_config.get('batch_size', 10)
        self.max_review_length = self.nlp_config.get('max_review_length', 500)
        self.topics_count = self.nlp_config.get('topics_count', 5)
        self.is_available = llm_client.is_available if llm_client else False

        logger.info(f"NLPEngine initialized. LLM available: {self.is_available}")

    def _truncate_text(self, text: str) -> str:
        """Truncate text to max length and scrub PII."""
        if not text or not isinstance(text, str):
            return ""
        scrubbed = self._scrub_pii(text)
        return scrubbed[:self.max_review_length]

    def _scrub_pii(self, text: str) -> str:
        """Scrub PII (emails, phones, SSN, credit cards) from text."""
        if not text:
            return ""
        # Scrub emails
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        # Scrub phones (various formats)
        text = re.sub(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b', '[PHONE]', text)
        # Scrub SSN (XXX-XX-XXXX or XXXXXXXXX)
        text = re.sub(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', '[SSN]', text)
        # Scrub credit card numbers (16 digits with optional separators)
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CREDIT_CARD]', text)
        return text

    def _parse_json_response(self, response: str) -> Optional[Any]:
        """Parse JSON from LLM response, handling common issues."""
        if not response:
            return None

        # Try to extract JSON from response
        try:
            # First try direct parsing
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in response
        json_patterns = [
            r'\[[\s\S]*\]',  # Array
            r'\{[\s\S]*\}',  # Object
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue

        logger.warning("Could not parse JSON from LLM response")
        return None

    def analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment of performance review texts.

        Args:
            df: DataFrame with PerformanceText column.

        Returns:
            DataFrame with EmployeeID, sentiment_score, sentiment_label columns.
        """
        if 'PerformanceText' not in df.columns:
            logger.warning("PerformanceText column not found")
            return pd.DataFrame(columns=['EmployeeID', 'sentiment_score', 'sentiment_label'])

        results = []

        if not self.is_available:
            logger.warning("Sentiment analysis skipped: LLM unavailable")
            return pd.DataFrame(columns=['EmployeeID', 'sentiment_score', 'sentiment_label'])
        
        # Use LLM for sentiment analysis
        texts = df['PerformanceText'].fillna('').tolist()
        employee_ids = df['EmployeeID'].tolist()

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            batch_ids = employee_ids[i:i+self.batch_size]

            try:
                batch_results = self._analyze_sentiment_batch(batch_texts, batch_ids)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Failed to analyze sentiment for batch: {e}")
                # Fallback to neutral for failed batch
                for eid in batch_ids:
                    results.append({
                        'EmployeeID': eid,
                        'sentiment_score': 0.5,
                        'sentiment_label': 'Neutral'
                    })

        return pd.DataFrame(results)


    def _analyze_sentiment_batch(self, texts: list, employee_ids: list) -> list:
        """Analyze sentiment for a batch of reviews using LLM."""
        prompt = self._build_sentiment_prompt(texts, employee_ids)

        try:
            response = self.llm_client.client.generate(
                model=self.llm_client.model,
                prompt=prompt,
                options={
                    'temperature': self.nlp_config.get('sentiment_temperature', 0.3),
                    'num_predict': 1000
                }
            )

            raw_response = response.get('response', '')
            parsed = self._parse_json_response(raw_response)

            if parsed and isinstance(parsed, list):
                return parsed
            else:
                raise NLPEngineError("Failed to parse LLM sentiment response")

        except Exception as e:
            logger.error(f"Sentiment batch analysis failed: {str(e)}")
            raise NLPEngineError(f"Sentiment batch analysis failed: {str(e)}")

    def _build_sentiment_prompt(self, texts: list, employee_ids: list) -> str:
        """Build prompt for sentiment analysis."""
        reviews_text = "\n".join([
            f"{eid}: {self._truncate_text(text)}"
            for eid, text in zip(employee_ids, texts)
        ])

        return f"""Analyze the sentiment of these performance reviews.

REVIEWS:
{reviews_text}

Return a JSON array with sentiment analysis for each review:
[
  {{"EmployeeID": "EMP0001", "sentiment_score": 0.85, "sentiment_label": "Positive"}},
  {{"EmployeeID": "EMP0002", "sentiment_score": 0.45, "sentiment_label": "Neutral"}}
]

Rules:
- sentiment_score: 0.0 (very negative) to 1.0 (very positive)
- sentiment_label: "Positive" (>0.6), "Neutral" (0.4-0.6), "Negative" (<0.4)
- Return ONLY the JSON array, no other text."""

    def extract_skills(self, df: pd.DataFrame) -> dict:
        """
        Extract skills mentioned in performance reviews.

        Args:
            df: DataFrame with PerformanceText column.

        Returns:
            Dictionary with technical_skills, soft_skills, and skill_counts.
        """
        if 'PerformanceText' not in df.columns:
            return {'technical_skills': [], 'soft_skills': [], 'skill_counts': {}}

        if not self.is_available:
            logger.warning("Skill extraction skipped: LLM unavailable")
            return {'technical_skills': [], 'soft_skills': [], 'skill_counts': {}}

        # Sample texts for skill extraction
        texts = df['PerformanceText'].fillna('').tolist()
        sample_size = min(50, len(texts))
        sample_texts = texts[:sample_size]

        prompt = self._build_skill_extraction_prompt(sample_texts)

        try:
            response = self.llm_client.client.generate(
                model=self.llm_client.model,
                prompt=prompt,
                options={
                    'temperature': self.nlp_config.get('skill_extraction_temperature', 0.2),
                    'num_predict': 800
                }
            )

            raw_response = response.get('response', '')
            parsed = self._parse_json_response(raw_response)

            if parsed and isinstance(parsed, dict):
                # Count skill occurrences across all texts
                skill_counts = self._count_skills_in_texts(texts, parsed)
                parsed['skill_counts'] = skill_counts
                return parsed
            else:
                raise NLPEngineError("Failed to parse LLM skill extraction response")

        except Exception as e:
            logger.error(f"Skill extraction failed: {str(e)}")
            raise NLPEngineError(f"Skill extraction failed: {str(e)}")


    def _build_skill_extraction_prompt(self, texts: list) -> str:
        """Build prompt for skill extraction."""
        combined = "\n---\n".join([self._truncate_text(t) for t in texts])

        return f"""Extract skills mentioned in these performance reviews.

REVIEWS:
{combined}

Return a JSON object with skills found:
{{
  "technical_skills": ["Python", "SQL", "Cloud Architecture"],
  "soft_skills": ["Leadership", "Communication", "Problem-solving"]
}}

Rules:
- List unique skills only (no duplicates)
- Use proper capitalization
- Maximum 15 skills per category
- Return ONLY the JSON object, no other text."""

    def _count_skills_in_texts(self, texts: list, skills_dict: dict) -> dict:
        """Count occurrences of each skill across all texts."""
        all_skills = (
            skills_dict.get('technical_skills', []) +
            skills_dict.get('soft_skills', [])
        )

        counts = {skill: 0 for skill in all_skills}

        for text in texts:
            text_lower = text.lower()
            for skill in all_skills:
                if skill.lower() in text_lower:
                    counts[skill] += 1

        return counts

    def extract_topics(self, df: pd.DataFrame) -> list:
        """
        Extract dominant topics from performance reviews.

        Args:
            df: DataFrame with PerformanceText column.

        Returns:
            List of topic dictionaries with name, description, prevalence.
        """
        if 'PerformanceText' not in df.columns:
            return []

        if not self.is_available:
            logger.warning("Topic extraction skipped: LLM unavailable")
            return []

        texts = df['PerformanceText'].fillna('').tolist()
        sample_size = min(50, len(texts))
        sample_texts = texts[:sample_size]

        prompt = self._build_topic_extraction_prompt(sample_texts)

        try:
            response = self.llm_client.client.generate(
                model=self.llm_client.model,
                prompt=prompt,
                options={
                    'temperature': self.nlp_config.get('topic_temperature', 0.5),
                    'num_predict': 1000
                }
            )

            raw_response = response.get('response', '')
            parsed = self._parse_json_response(raw_response)

            if parsed and isinstance(parsed, list):
                return parsed[:self.topics_count]
            elif parsed and isinstance(parsed, dict) and 'topics' in parsed:
                return parsed['topics'][:self.topics_count]
            else:
                raise NLPEngineError("Failed to parse LLM topic response")

        except Exception as e:
            logger.error(f"Topic extraction failed: {str(e)}")
            raise NLPEngineError(f"Topic extraction failed: {str(e)}")


    def _build_topic_extraction_prompt(self, texts: list) -> str:
        """Build prompt for topic extraction."""
        combined = "\n---\n".join([self._truncate_text(t) for t in texts])

        return f"""Identify the {self.topics_count} main themes in these performance reviews.

REVIEWS:
{combined}

Return a JSON array of topics:
[
  {{
    "name": "Theme Name",
    "description": "Brief description of this theme",
    "prevalence": "25%",
    "sentiment": "Positive"
  }}
]

Rules:
- Identify organizational themes, not individual issues
- Each theme should appear in multiple reviews
- Include approximate prevalence percentage
- Return ONLY the JSON array, no other text."""

    def generate_employee_summary(self, employee_data: dict) -> str:
        """
        Generate an AI summary for an individual employee.

        Args:
            employee_data: Dictionary with employee information.

        Returns:
            Natural language summary string.
        """
        if not self.is_available:
            raise NLPEngineError("NLP Engine unavailable: employee summary requires LLM")

        prompt = self._build_employee_summary_prompt(employee_data)

        try:
            response = self.llm_client.client.generate(
                model=self.llm_client.model,
                prompt=prompt,
                options={
                    'temperature': 0.6,
                    'num_predict': 200
                }
            )

            summary = response.get('response', '').strip()
            if summary:
                return summary
            raise NLPEngineError("LLM returned empty summary")
        except Exception as e:
            logger.error(f"Employee summary generation failed: {str(e)}")
            raise NLPEngineError(f"Employee summary generation failed: {str(e)}")


    def _build_employee_summary_prompt(self, employee_data: dict) -> str:
        """Build prompt for employee summary generation."""
        data_json = json.dumps(employee_data, indent=2)

        return f"""Generate a brief, professional summary for this employee based on their data.

EMPLOYEE DATA:
{data_json}

Rules:
- Maximum 2 sentences
- Focus on performance trends and strengths
- No termination or disciplinary language
- Be objective and professional
- Return ONLY the summary text, no other formatting."""

    def get_sentiment_summary(self, sentiment_df: pd.DataFrame) -> dict:
        """
        Get aggregated sentiment statistics.

        Args:
            sentiment_df: DataFrame with sentiment analysis results.

        Returns:
            Dictionary with sentiment summary statistics.
        """
        if sentiment_df.empty:
            return {
                'avg_sentiment': 0.5,
                'positive_count': 0,
                'neutral_count': 0,
                'negative_count': 0,
                'positive_pct': 0,
                'neutral_pct': 0,
                'negative_pct': 0
            }

        total = len(sentiment_df)
        positive = len(sentiment_df[sentiment_df['sentiment_label'] == 'Positive'])
        neutral = len(sentiment_df[sentiment_df['sentiment_label'] == 'Neutral'])
        negative = len(sentiment_df[sentiment_df['sentiment_label'] == 'Negative'])

        return {
            'avg_sentiment': round(sentiment_df['sentiment_score'].mean(), 2),
            'positive_count': positive,
            'neutral_count': neutral,
            'negative_count': negative,
            'positive_pct': round((positive / total) * 100, 1) if total > 0 else 0,
            'neutral_pct': round((neutral / total) * 100, 1) if total > 0 else 0,
            'negative_pct': round((negative / total) * 100, 1) if total > 0 else 0
        }

    def get_sentiment_by_department(self, df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get sentiment aggregated by department.

        Args:
            df: Original DataFrame with Dept column.
            sentiment_df: DataFrame with sentiment analysis results.

        Returns:
            DataFrame with department-level sentiment.
        """
        if sentiment_df.empty or 'Dept' not in df.columns:
            return pd.DataFrame()

        merged = df[['EmployeeID', 'Dept']].merge(sentiment_df, on='EmployeeID')

        dept_sentiment = merged.groupby('Dept').agg({
            'sentiment_score': 'mean',
            'sentiment_label': lambda x: (x == 'Positive').sum() / len(x) * 100
        }).round(2).reset_index()

        dept_sentiment.columns = ['Dept', 'AvgSentiment', 'PositivePct']

        return dept_sentiment.sort_values('AvgSentiment', ascending=False)

    def process_all(self, df: pd.DataFrame) -> dict:
        """
        Run full NLP pipeline on DataFrame.

        Args:
            df: DataFrame with PerformanceText column.

        Returns:
            Dictionary with all NLP analysis results.
        """
        logger.info("Starting full NLP processing pipeline")

        results = {
            'sentiment': pd.DataFrame(),
            'sentiment_summary': {},
            'sentiment_by_dept': pd.DataFrame(),
            'skills': {},
            'topics': [],
            'nlp_available': self.is_available
        }

        if 'PerformanceText' not in df.columns:
            logger.warning("PerformanceText column not found - NLP processing skipped")
            return results

        try:
            # Sentiment analysis
            logger.info("Running sentiment analysis...")
            sentiment_df = self.analyze_sentiment(df)
            results['sentiment'] = sentiment_df
            results['sentiment_summary'] = self.get_sentiment_summary(sentiment_df)
            results['sentiment_by_dept'] = self.get_sentiment_by_department(df, sentiment_df)

            # Skill extraction
            logger.info("Extracting skills...")
            results['skills'] = self.extract_skills(df)

            # Topic extraction
            logger.info("Extracting topics...")
            results['topics'] = self.extract_topics(df)

            logger.info("NLP processing complete")

        except Exception as e:
            logger.error(f"NLP processing failed: {str(e)}")
            raise e
