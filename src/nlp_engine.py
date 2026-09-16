"""
NLP Engine module for PeopleOS.

Provides NLP-powered analysis of performance review text using Ollama.
Includes sentiment analysis, skill extraction, topic modeling, and AI summaries.
"""

import json
from math import isfinite
from numbers import Real
import re
from typing import Any, Optional

import pandas as pd

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('nlp_engine')

_REVIEW_SAMPLE_SIZE = 10
_REVIEW_SAMPLE_SCOPE = f'first_{_REVIEW_SAMPLE_SIZE}_nonempty_unique_employee_reviews'


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
        self.batch_size = self._config_int('batch_size', 10, minimum=1, maximum=100)
        self.max_review_length = self._config_int('max_review_length', 500, minimum=1, maximum=10_000)
        self.topics_count = self._config_int('topics_count', 5, minimum=1, maximum=20)
        self.is_available = llm_client.is_available if llm_client else False
        self._last_sentiment_input_count = 0
        self._last_sentiment_observed_count = 0
        self._last_sentiment_unprocessed_count = 0
        self._last_sentiment_excluded_count = 0

        logger.info(f"NLPEngine initialized. LLM available: {self.is_available}")

    def _config_int(self, name: str, default: int, *, minimum: int, maximum: int) -> int:
        """Require bounded integer configuration before any model call is possible."""
        value = self.nlp_config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise NLPEngineError(
                f"NLP configuration {name} must be an integer from {minimum} to {maximum}"
            )
        return value

    @staticmethod
    def _empty_skills(reason: str | None = None, review_observations: int = 0) -> dict:
        """Return a schema-compatible, explicitly bounded skill result."""
        result = {
            'technical_skills': [],
            'soft_skills': [],
            'skill_counts': {},
            'skill_count_semantics': 'source_review_presence_count',
            'review_observations': review_observations,
        }
        if reason:
            result.update({'status': 'unavailable', 'unavailable_reason': reason})
        return result

    @staticmethod
    def _review_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Select one governed review per nonempty, valid employee identity."""
        if not {'EmployeeID', 'PerformanceText'}.issubset(df.columns):
            return pd.DataFrame(columns=['EmployeeID', 'PerformanceText'])
        reviews = df[['EmployeeID', 'PerformanceText']].copy(deep=True)
        ids = reviews['EmployeeID']
        valid_ids = ids.notna() & ids.astype(str).str.strip().ne('')
        valid_text = reviews['PerformanceText'].fillna('').astype(str).str.strip().ne('')
        reviews = reviews.loc[valid_ids & valid_text].copy()
        reviews['EmployeeID'] = reviews['EmployeeID'].astype(str)
        # Duplicate identities make row-level model output ambiguous; refuse inference.
        if reviews['EmployeeID'].duplicated().any():
            return pd.DataFrame(columns=['EmployeeID', 'PerformanceText'])
        return reviews

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
        self._last_sentiment_input_count = 0
        self._last_sentiment_observed_count = 0
        self._last_sentiment_unprocessed_count = 0
        self._last_sentiment_excluded_count = 0
        if 'PerformanceText' not in df.columns:
            logger.warning("PerformanceText column not found")
            return pd.DataFrame(columns=['EmployeeID', 'sentiment_score', 'sentiment_label'])

        results = []
        reviews = self._review_frame(df)
        self._last_sentiment_input_count = len(reviews)

        if not self.is_available:
            logger.warning("Sentiment analysis skipped: LLM unavailable")
            self._last_sentiment_unprocessed_count = len(reviews)
            return pd.DataFrame(columns=['EmployeeID', 'sentiment_score', 'sentiment_label'])
        
        # Keep the optional local-model path bounded for interactive use. All
        # review-model paths use the same small exploratory sample rather than
        # starting one model call per review in a large workforce upload.
        if len(reviews) > _REVIEW_SAMPLE_SIZE:
            self._last_sentiment_excluded_count = len(reviews) - _REVIEW_SAMPLE_SIZE
            reviews = reviews.head(_REVIEW_SAMPLE_SIZE).copy()

        # Use LLM for sentiment analysis
        texts = reviews['PerformanceText'].tolist()
        employee_ids = reviews['EmployeeID'].tolist()

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            batch_ids = employee_ids[i:i+self.batch_size]

            try:
                batch_results = self._analyze_sentiment_batch(batch_texts, batch_ids)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Failed to analyze sentiment for batch: {e}")
                # Failed inference is missing evidence, never neutral sentiment.
                continue

        self._last_sentiment_observed_count = len(results)
        self._last_sentiment_unprocessed_count = self._last_sentiment_excluded_count + len(employee_ids) - len(results)
        return pd.DataFrame(results, columns=['EmployeeID', 'sentiment_score', 'sentiment_label'])


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
                allowed = {str(value) for value in employee_ids}
                seen, valid = set(), []
                for item in parsed:
                    if not isinstance(item, dict):
                        raise NLPEngineError('Sentiment rows must be objects')
                    eid = str(item.get('EmployeeID'))
                    score = item.get('sentiment_score')
                    if (eid not in allowed or eid in seen or isinstance(score, bool) or
                            not isinstance(score, Real) or not isfinite(float(score)) or
                            not 0 <= score <= 1):
                        raise NLPEngineError('Sentiment response has invalid identity or score')
                    expected_label = 'Positive' if score > .6 else 'Negative' if score < .4 else 'Neutral'
                    if item.get('sentiment_label') != expected_label:
                        raise NLPEngineError('Sentiment label is invalid')
                    seen.add(eid)
                    valid.append(item)
                if seen != allowed:
                    raise NLPEngineError('Sentiment response must cover every requested review exactly once')
                return valid
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
            return self._empty_skills('PerformanceText column not found')

        reviews = self._review_frame(df)
        if not self.is_available:
            logger.warning("Skill extraction skipped: LLM unavailable")
            return self._empty_skills('LLM unavailable', len(reviews))

        # Sample texts for skill extraction
        texts = reviews['PerformanceText'].tolist()
        if not texts:
            return self._empty_skills('No valid review rows', 0)
        sample_size = min(_REVIEW_SAMPLE_SIZE, len(texts))
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
                for category in ('technical_skills', 'soft_skills'):
                    skills = parsed.get(category, [])
                    if not isinstance(skills, list) or any(not isinstance(skill, str) or not skill.strip() for skill in skills):
                        raise NLPEngineError('Skill categories require lists of nonempty strings')
                    normalized = list(dict.fromkeys(skill.strip() for skill in skills))
                    if len(normalized) > 15:
                        raise NLPEngineError('Skill response exceeds the 15-skill category limit')
                    if any(not self._skill_is_grounded(skill, texts) for skill in normalized):
                        raise NLPEngineError('Skill response contains a skill not literally supported by source review text')
                    parsed[category] = normalized
                # Count skill occurrences across all texts
                skill_counts = self._count_skills_in_texts(texts, parsed)
                return {
                    'technical_skills': parsed['technical_skills'],
                    'soft_skills': parsed['soft_skills'],
                    'skill_counts': skill_counts,
                    'skill_count_semantics': 'source_review_presence_count',
                    'review_observations': len(texts),
                    'status': 'available',
                }
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
        """Count source reviews containing each skill, not unverified token frequency."""
        all_skills = (
            skills_dict.get('technical_skills', []) +
            skills_dict.get('soft_skills', [])
        )

        counts = {skill: 0 for skill in all_skills}

        for text in texts:
            text_lower = text.lower()
            for skill in all_skills:
                if re.search(r'(?<!\w)' + re.escape(skill.lower()) + r'(?!\w)', text_lower):
                    counts[skill] += 1

        return counts

    @staticmethod
    def _skill_is_grounded(skill: str, texts: list[str]) -> bool:
        """Require a literal, case-insensitive source mention before returning a skill."""
        pattern = r'(?<!\w)' + re.escape(skill.lower()) + r'(?!\w)'
        return any(re.search(pattern, str(text).lower()) for text in texts)

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

        reviews = self._review_frame(df)
        if not self.is_available:
            logger.warning("Topic extraction skipped: LLM unavailable")
            return []

        texts = reviews['PerformanceText'].tolist()
        if not texts:
            return []
        sample_size = min(_REVIEW_SAMPLE_SIZE, len(texts))
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

            topics = parsed.get('topics') if isinstance(parsed, dict) else parsed
            if not isinstance(topics, list):
                raise NLPEngineError('Topic response must be a list')
            valid = []
            seen_names = set()
            for topic in topics:
                if not isinstance(topic, dict) or any(not isinstance(topic.get(key), str) or not topic[key].strip() for key in ('name', 'description')):
                    raise NLPEngineError('Topics require a name and description')
                if topic.get('sentiment') not in {'Positive', 'Neutral', 'Negative', 'Mixed'}:
                    raise NLPEngineError('Topic sentiment label is invalid')
                normalized_name = topic['name'].strip().casefold()
                if normalized_name in seen_names:
                    raise NLPEngineError('Topic response contains duplicate theme names')
                seen_names.add(normalized_name)
                # An LLM estimate is not a counted share of source reviews.
                valid.append({'name': topic['name'], 'description': topic['description'],
                              'sentiment': topic['sentiment'], 'prevalence': None,
                              'measurement_semantics': 'generated_theme_not_measured_prevalence',
                              'sample_size': sample_size,
                              'sample_scope': _REVIEW_SAMPLE_SCOPE})
            return valid[:self.topics_count]

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
  "sentiment": "Positive"
  }}
]

Rules:
- Identify organizational themes, not individual issues
- Each theme should appear in multiple reviews
- Do not estimate prevalence or coverage; prevalence is unavailable without a labeled review corpus and full-population counting.
- Return ONLY the JSON array, no other text."""

    def generate_employee_summary(self, employee_data: dict) -> str:
        """Retired: ungrounded LLM-generated individual summaries are outside scope."""
        return (
            'Unavailable: individual employee summaries are disabled because generated prose '
            'is not governed aggregate evidence.'
        )

    def get_sentiment_summary(self, sentiment_df: pd.DataFrame) -> dict:
        """
        Get aggregated sentiment statistics.

        Args:
            sentiment_df: DataFrame with sentiment analysis results.

        Returns:
            Dictionary with sentiment summary statistics.
        """
        required = {'sentiment_score', 'sentiment_label'}
        if sentiment_df.empty or not required.issubset(sentiment_df.columns):
            return {
                'avg_sentiment': None,
                'positive_count': 0,
                'neutral_count': 0,
                'negative_count': 0,
                'positive_pct': 0,
                'neutral_pct': 0,
                'negative_pct': 0,
                'sentiment_observations': 0,
                'excluded_sentiment_rows': int(len(sentiment_df)),
                'unprocessed_sentiment_rows': self._last_sentiment_unprocessed_count,
            }
        score = pd.to_numeric(sentiment_df['sentiment_score'], errors='coerce')
        expected = score.map(lambda value: 'Positive' if value > .6 else 'Negative' if value < .4 else 'Neutral')
        valid = score.notna() & score.between(0, 1) & sentiment_df['sentiment_label'].eq(expected)
        measured = sentiment_df.loc[valid].copy()
        measured['sentiment_score'] = score.loc[valid]
        total = len(measured)
        positive = len(measured[measured['sentiment_label'] == 'Positive'])
        neutral = len(measured[measured['sentiment_label'] == 'Neutral'])
        negative = len(measured[measured['sentiment_label'] == 'Negative'])

        return {
            'avg_sentiment': round(measured['sentiment_score'].mean(), 2) if total else None,
            'positive_count': positive,
            'neutral_count': neutral,
            'negative_count': negative,
            'positive_pct': round((positive / total) * 100, 1) if total > 0 else 0,
            'neutral_pct': round((neutral / total) * 100, 1) if total > 0 else 0,
            'negative_pct': round((negative / total) * 100, 1) if total > 0 else 0,
            'sentiment_observations': total,
            'excluded_sentiment_rows': int(len(sentiment_df) - total),
            'unprocessed_sentiment_rows': self._last_sentiment_unprocessed_count,
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

        from src.population import resolve_current_population
        current, _ = resolve_current_population(df)
        current['EmployeeID'] = current['EmployeeID'].astype(str)
        sentiment_df = sentiment_df.copy()
        sentiment_df['EmployeeID'] = sentiment_df['EmployeeID'].astype(str)
        required = {'EmployeeID', 'sentiment_score', 'sentiment_label'}
        if not required.issubset(sentiment_df.columns):
            return pd.DataFrame()
        score = pd.to_numeric(sentiment_df['sentiment_score'], errors='coerce')
        expected = score.map(lambda value: 'Positive' if value > .6 else 'Negative' if value < .4 else 'Neutral')
        valid = score.notna() & score.between(0, 1) & sentiment_df['sentiment_label'].eq(expected)
        sentiment_df = sentiment_df.loc[valid].copy()
        sentiment_df['sentiment_score'] = score.loc[valid]
        merged = current[['EmployeeID', 'Dept']].merge(sentiment_df.drop_duplicates('EmployeeID'), on='EmployeeID', validate='one_to_one')
        merged['Dept'] = merged['Dept'].fillna('Unknown')

        # Department sentiment is an aggregate result; suppress small cells.
        sizes = merged.groupby('Dept')['EmployeeID'].transform('count')
        merged = merged[sizes >= 10]
        if merged.empty:
            return pd.DataFrame()

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
            'nlp_available': self.is_available,
            'analysis_status': 'unavailable' if not self.is_available else 'available',
            'component_status': {},
        }

        if 'PerformanceText' not in df.columns:
            logger.warning("PerformanceText column not found - NLP processing skipped")
            results['analysis_status'] = 'unavailable'
            results['unavailable_reason'] = 'PerformanceText column not found'
            return results

        try:
            # Sentiment analysis
            logger.info("Running sentiment analysis...")
            sentiment_df = self.analyze_sentiment(df)
            results['sentiment'] = sentiment_df
            results['sentiment_summary'] = self.get_sentiment_summary(sentiment_df)
            results['sentiment_by_dept'] = self.get_sentiment_by_department(df, sentiment_df)
            results['component_status']['sentiment'] = {
                'status': 'available' if self._last_sentiment_input_count > 0 and self._last_sentiment_unprocessed_count == 0 else 'partial' if len(sentiment_df) else 'unavailable',
                'input_observations': self._last_sentiment_input_count,
                'observations': len(sentiment_df),
                'unprocessed_observations': self._last_sentiment_unprocessed_count,
                'sample_size': min(self._last_sentiment_input_count, _REVIEW_SAMPLE_SIZE),
                'excluded_observations': self._last_sentiment_excluded_count,
                'sample_scope': _REVIEW_SAMPLE_SCOPE if self._last_sentiment_excluded_count else 'all_nonempty_unique_employee_reviews',
            }

            if len(sentiment_df):
                # Skill and topic extraction are also sampled. If sentiment
                # produced no governed rows, avoid spending more model time
                # on follow-on calls that cannot produce a trustworthy result.
                logger.info("Extracting skills...")
                try:
                    results['skills'] = self.extract_skills(df)
                    results['component_status']['skills'] = {'status': results['skills'].get('status', 'available'), 'sample_size': min(len(self._review_frame(df)), _REVIEW_SAMPLE_SIZE)}
                except NLPEngineError as exc:
                    logger.warning("Skill extraction unavailable: %s", exc)
                    results['skills'] = self._empty_skills('Model output failed source-grounding validation', len(self._review_frame(df)))
                    results['component_status']['skills'] = {'status': 'unavailable', 'sample_size': min(len(self._review_frame(df)), _REVIEW_SAMPLE_SIZE)}

                logger.info("Extracting topics...")
                try:
                    results['topics'] = self.extract_topics(df)
                    results['component_status']['topics'] = {'status': 'available' if results['topics'] else 'unavailable', 'sample_size': min(len(self._review_frame(df)), _REVIEW_SAMPLE_SIZE)}
                except NLPEngineError as exc:
                    logger.warning("Topic extraction unavailable: %s", exc)
                    results['topics'] = []
                    results['component_status']['topics'] = {'status': 'unavailable', 'sample_size': min(len(self._review_frame(df)), _REVIEW_SAMPLE_SIZE)}
            else:
                reason = 'Skipped because sentiment inference returned no governed rows.'
                results['skills'] = self._empty_skills(reason, len(self._review_frame(df)))
                results['topics'] = []
                results['component_status']['skills'] = {'status': 'unavailable', 'unavailable_reason': reason}
                results['component_status']['topics'] = {'status': 'unavailable', 'unavailable_reason': reason}

            component_states = [item['status'] for item in results['component_status'].values()]
            results['analysis_status'] = 'available' if component_states and all(state == 'available' for state in component_states) else 'partial' if any(state in {'available', 'partial'} for state in component_states) else 'unavailable'

            logger.info("NLP processing complete")
            return results

        except Exception as e:
            logger.error(f"NLP processing failed: {str(e)}")
            raise e
