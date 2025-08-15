"""
app_utils package for Database Copilot
"""

from .database import db_manager
from .ai_assistant import DatabaseAssistant

__all__ = ['db_manager', 'DatabaseAssistant']