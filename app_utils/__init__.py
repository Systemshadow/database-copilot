"""
app_utils package for Database Copilot
"""

from .database import db_manager
from .ai_assistant import MultiTableDatabaseAssistant

__all__ = ['db_manager', 'MultiTableDatabaseAssistant']
