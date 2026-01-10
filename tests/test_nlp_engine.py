"""
Tests for the NLPEngine module.
"""

import pandas as pd
import pytest

from src.nlp_engine import NLPEngine


class TestNLPEngine:
    """Test cases for NLPEngine class."""

    def test_pii_scrubbing_email(self):
        """Test that emails are replaced with [EMAIL]."""
        engine = NLPEngine(None)
        
        raw = "Contact John at john.doe@company.com for details."
        scrubbed = engine._scrub_pii(raw)
        
        assert '[EMAIL]' in scrubbed
        assert 'john.doe@company.com' not in scrubbed

    def test_pii_scrubbing_phone(self):
        """Test that phone numbers are replaced with [PHONE]."""
        engine = NLPEngine(None)
        
        raw = "Call us at 555-123-4567 for more info."
        scrubbed = engine._scrub_pii(raw)
        
        assert '[PHONE]' in scrubbed
        assert '555-123-4567' not in scrubbed

    def test_pii_scrubbing_combined(self):
        """Test scrubbing both email and phone in one string."""
        engine = NLPEngine(None)
        
        raw = "Email jane@work.co.uk or call 123-456-7890."
        scrubbed = engine._scrub_pii(raw)
        
        assert '[EMAIL]' in scrubbed
        assert '[PHONE]' in scrubbed


    def test_truncate_text_scrubs_pii(self):
        """Test that _truncate_text also scrubs PII."""
        engine = NLPEngine(None)
        
        raw = "Report by admin@company.org (Cell: 999-888-7777)"
        result = engine._truncate_text(raw)
        
        assert '[EMAIL]' in result
        assert '[PHONE]' in result
