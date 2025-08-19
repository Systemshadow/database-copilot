"""
Complete AI Expert Colleague System
Implements conversational memory, context awareness, collaborative discovery,
uncertainty handling, emotional intelligence, and proactive insights.
"""

import os
import pandas as pd
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import time
import json

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .database import DatabaseManager, TableInfo

load_dotenv()


@dataclass
class QueryResult:
    """Result of a database query with metadata."""
    success: bool
    data: Optional[pd.DataFrame] = None
    sql_query: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None


@dataclass
class ConversationContext:
    """Tracks conversation state and user context."""
    chat_history: List[Tuple[str, str]]  # (question, response) pairs
    user_expertise_level: str  # 'technical', 'business', 'executive'
    conversation_theme: str  # 'exploration', 'troubleshooting', 'reporting', 'strategic'
    recent_topics: List[str]  # Recent data entities discussed
    user_preferences: Dict[str, Any]  # Learned preferences
    session_insights: List[str]  # Discovered patterns this session


class ExpertDatabaseAssistant:
    """AI colleague that thinks, learns, and collaborates like a human expert."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.openai_client = None
        self.conversation_context = ConversationContext(
            chat_history=[],
            user_expertise_level='business',  # Default
            conversation_theme='exploration',
            recent_topics=[],
            user_preferences={},
            session_insights=[]
        )
        self._initialize_openai()
    
    def _initialize_openai(self):
        """Initialize OpenAI client if API key is available."""
        api_key = os.getenv('OPENAI_API_KEY')
        if OPENAI_AVAILABLE and api_key and api_key != 'your_openai_api_key_here':
            try:
                self.openai_client = OpenAI(api_key=api_key)
            except Exception as e:
                print(f"OpenAI initialization failed: {str(e)}")
                self.openai_client = None
    
    def analyze_user_context(self, question: str) -> None:
        """Analyze user question to understand context and adapt behavior."""
        question_lower = question.lower()
        
        # Detect expertise level
        technical_indicators = ['sql', 'query', 'table', 'column', 'join', 'api', 'decline curve', 'eur']
        business_indicators = ['performance', 'compare', 'trend', 'revenue', 'roi', 'market share']
        executive_indicators = ['overview', 'summary', 'strategic', 'portfolio', 'high level']
        
        if any(indicator in question_lower for indicator in technical_indicators):
            self.conversation_context.user_expertise_level = 'technical'
        elif any(indicator in question_lower for indicator in executive_indicators):
            self.conversation_context.user_expertise_level = 'executive'
        else:
            self.conversation_context.user_expertise_level = 'business'
        
        # Detect conversation theme
        if any(word in question_lower for word in ['problem', 'issue', 'wrong', 'error', 'fix']):
            self.conversation_context.conversation_theme = 'troubleshooting'
        elif any(word in question_lower for word in ['report', 'summary', 'meeting', 'presentation']):
            self.conversation_context.conversation_theme = 'reporting'
        elif any(word in question_lower for word in ['strategy', 'should', 'recommend', 'decision']):
            self.conversation_context.conversation_theme = 'strategic'
        else:
            self.conversation_context.conversation_theme = 'exploration'
        
        # Extract topics/entities mentioned
        entities = []
        # Look for well names, operators, counties, etc. in the question
        words = question.split()
        for word in words:
            if len(word) > 3 and (word.isupper() or '-' in word or 'H' in word):
                entities.append(word)
        
        self.conversation_context.recent_topics.extend(entities)
        # Keep only last 10 topics
        self.conversation_context.recent_topics = self.conversation_context.recent_topics[-10:]
    
    def generate_sql_query(self, question: str, relevant_tables: List[str]) -> Tuple[str, str]:
        """Enhanced SQL generation with context awareness."""
        if not self.openai_client:
            return "", "OpenAI API key not configured. Please set OPENAI_API_KEY in .env file."
        
        try:
            schema_context = self._build_schema_context(relevant_tables)
            sample_data_context = self._get_sample_data_context(relevant_tables)
            conversation_context = self._build_conversation_context()
            
            prompt = f"""
Generate a SQL query for this question about data analysis:

QUESTION: {question}

CONVERSATION CONTEXT:
{conversation_context}

DATABASE INFO:
{schema_context}

SAMPLE DATA:
{sample_data_context}

CONTEXT AWARENESS:
- User expertise: {self.conversation_context.user_expertise_level}
- Conversation theme: {self.conversation_context.conversation_theme}
- Recent topics: {', '.join(self.conversation_context.recent_topics[-5:])}

Generate a simple, effective SELECT query. Use actual column names and values from the sample data.
Return only the SQL query.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert data analyst who generates precise SQL queries.
                        
                        Rules:
                        - Generate only SELECT queries
                        - Use exact column names from schema
                        - Base queries on sample data provided
                        - Keep queries simple and effective
                        - Return only SQL, no explanation"""
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            sql_query = response.choices[0].message.content.strip()
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            is_safe, error = self.db_manager.is_safe_query(sql_query)
            if not is_safe:
                return "", f"Generated query is not safe: {error}"
            
            return sql_query, ""
            
        except Exception as e:
            return "", f"Failed to generate SQL query: {str(e)}"
    
    def _build_conversation_context(self) -> str:
        """Build context from recent conversation."""
        if not self.conversation_context.chat_history:
            return "Start of conversation"
        
        recent_context = []
        # Include last 3 exchanges for context
        for q, r in self.conversation_context.chat_history[-3:]:
            recent_context.append(f"Previous Q: {q[:100]}...")
            recent_context.append(f"Previous A: {r[:150]}...")
        
        if self.conversation_context.session_insights:
            recent_context.append(f"Session insights: {', '.join(self.conversation_context.session_insights[-3:])}")
        
        return "\n".join(recent_context)
    
    def _get_sample_data_context(self, table_names: List[str]) -> str:
        """Get sample data with enhanced context."""
        sample_context = []
        
        for table_name in table_names[:2]:
            try:
                sample_query = f"SELECT * FROM {table_name} LIMIT 3"
                sample_df = self.db_manager.execute_query(sample_query)
                
                if not sample_df.empty:
                    sample_rows = sample_df.to_dict('records')
                    sample_context.append(f"Sample from {table_name}:")
                    for i, row in enumerate(sample_rows, 1):
                        row_str = ", ".join([f"{k}: {v}" for k, v in row.items() if v is not None])
                        sample_context.append(f"  Row {i}: {row_str}")
                
                # Get key distinct values
                table_info = self.db_manager.get_table_info(table_name)
                if table_info:
                    for col in table_info.columns[:5]:
                        col_name = col['name']
                        if col_name.lower() in ['well_name', 'operator', 'county', 'year']:
                            try:
                                distinct_query = f"SELECT DISTINCT {col_name} FROM {table_name} LIMIT 5"
                                distinct_df = self.db_manager.execute_query(distinct_query)
                                if not distinct_df.empty:
                                    values = distinct_df[col_name].tolist()
                                    sample_context.append(f"Available {col_name}: {', '.join(map(str, values))}")
                            except:
                                pass
                                
            except Exception as e:
                sample_context.append(f"Could not sample {table_name}: {str(e)}")
        
        return "\n".join(sample_context)
    
    def _build_schema_context(self, table_names: List[str]) -> str:
        """Build schema context for relevant tables."""
        context_parts = []
        
        for table_name in table_names:
            table_info = self.db_manager.get_table_info(table_name)
            if table_info:
                columns = [f"{col['name']} ({col['type']})" for col in table_info.columns]
                context_parts.append(f"Table: {table_name}\nColumns: {', '.join(columns)}\nRows: {table_info.row_count or 'Unknown'}")
        
        return "\n\n".join(context_parts)
    
    def execute_query(self, question: str) -> QueryResult:
        """Execute query with enhanced error handling and context tracking."""
        start_time = time.time()
        
        try:
            relevant_tables = self.db_manager.find_relevant_tables(question)
            
            if not relevant_tables:
                return QueryResult(
                    success=False,
                    error_message="Could not identify relevant tables for your question. Please be more specific."
                )
            
            sql_query, error = self.generate_sql_query(question, relevant_tables)
            
            if error:
                return QueryResult(success=False, error_message=error)
            
            result_df = self.db_manager.execute_query(sql_query)
            execution_time = time.time() - start_time
            
            return QueryResult(
                success=True,
                data=result_df,
                sql_query=sql_query,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QueryResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def generate_expert_response(self, question: str, query_result: QueryResult) -> str:
        """Generate response with full expert colleague capabilities."""
        if not query_result.success:
            return self._handle_query_failure(question, query_result.error_message)

        if query_result.data is None or query_result.data.empty:
            return self._handle_no_data_found(question, query_result.sql_query)

        if not self.openai_client:
            return self._generate_fallback_response(question, query_result)

        try:
            # Analyze data for insights and patterns
            data_insights = self._analyze_data_patterns(query_result.data)
            
            # Build comprehensive expert prompt
            expert_prompt = self._build_expert_prompt(question, query_result, data_insights)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_expert_system_prompt()
                    },
                    {"role": "user", "content": expert_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            expert_response = response.choices[0].message.content.strip()
            
            # Extract and store insights for future context
            self._extract_session_insights(expert_response, query_result.data)
            
            return expert_response
            
        except Exception as e:
            print(f"Expert response generation failed: {str(e)}")
            return self._generate_fallback_response(question, query_result)
    
    def _get_expert_system_prompt(self) -> str:
        """Comprehensive expert personality system prompt."""
        expertise_level = self.conversation_context.user_expertise_level
        theme = self.conversation_context.conversation_theme
        
        return f"""You are an experienced data analyst colleague who thinks like a strategic partner. You have deep domain expertise and collaborate naturally with humans.

CURRENT CONTEXT:
- User expertise level: {expertise_level}
- Conversation theme: {theme}
- You're having an ongoing conversation, not writing a report

CORE PERSONALITY TRAITS:
• Thoughtful and curious - you notice patterns and ask insightful questions
• Humble about uncertainty - you acknowledge data limitations and express appropriate doubt
• Collaborative - you work WITH the user, not just answer questions
• Contextually aware - you adapt your communication style to the situation
• Proactive - you surface unexpected insights and suggest better questions
• Human-like - you use natural conversation flow, not structured formats

RESPONSE STYLE GUIDELINES:

For TECHNICAL users: Include analytical details, mention data quality considerations, suggest validation steps
For BUSINESS users: Focus on business implications, use industry context, provide actionable insights  
For EXECUTIVE users: Lead with key takeaways, focus on strategic implications, be concise

For TROUBLESHOOTING: Be direct and solution-focused, ask clarifying questions
For EXPLORATION: Be curious and collaborative, suggest interesting angles
For REPORTING: Be clear and well-organized, highlight key findings
For STRATEGIC: Focus on implications and recommendations

CONVERSATION FLOW:
• Start with a natural, conversational answer to their specific question
• Add relevant context and insights based on what you observe in the data
• Express appropriate uncertainty when data might be incomplete
• End with collaborative suggestions - either follow-up questions for them OR questions back to them
• Occasionally ask the user what they think or if they've noticed patterns

AVOID:
- Numbered lists or formal structure (1. 2. 3.)
- Generic suggestions that could apply to any data
- Overly confident statements about uncertain conclusions
- Ignoring the conversation history and context

Be genuine, curious, and helpful like a thoughtful colleague who really understands both data and business context.
"""
    
    def _build_expert_prompt(self, question: str, query_result: QueryResult, data_insights: Dict) -> str:
        """Build comprehensive prompt for expert response generation."""
        df = query_result.data
        conversation_history = self._get_relevant_conversation_history()
        
        prompt = f"""
CONVERSATION CONTEXT:
{conversation_history}

CURRENT QUESTION: {question}

DATA ANALYSIS:
Dataset: {len(df)} rows, {len(df.columns)} columns
Key columns: {', '.join(df.columns.tolist()[:8])}

DISCOVERED PATTERNS AND INSIGHTS:
{self._format_data_insights(data_insights)}

SAMPLE DATA FOR CONTEXT:
{df.head(3).to_dict('records') if len(df) > 0 else 'No data'}

STATISTICAL SUMMARY:
{self._get_statistical_summary(df)}

USER CONTEXT:
- Expertise level: {self.conversation_context.user_expertise_level}
- Conversation theme: {self.conversation_context.conversation_theme}
- Recent topics discussed: {', '.join(self.conversation_context.recent_topics[-5:])}
- Previous insights this session: {', '.join(self.conversation_context.session_insights[-3:])}

Respond as a thoughtful colleague who understands both the data and the business context. Be naturally conversational and collaborative.
"""
        
        return prompt
    
    def _analyze_data_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze data for patterns, anomalies, and insights."""
        insights = {
            'statistical_patterns': [],
            'potential_anomalies': [],
            'data_quality_notes': [],
            'business_insights': []
        }
        
        if df.empty:
            return insights
        
        # Statistical analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols[:3]:
            if col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 1:
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    cv = std_val / mean_val if mean_val != 0 else 0
                    
                    if cv > 1.0:
                        insights['statistical_patterns'].append(f"{col} shows high variability (CV: {cv:.2f})")
                    
                    # Look for potential outliers
                    q75, q25 = col_data.quantile([0.75, 0.25])
                    iqr = q75 - q25
                    outlier_threshold = q75 + 1.5 * iqr
                    outliers = col_data[col_data > outlier_threshold]
                    
                    if len(outliers) > 0:
                        insights['potential_anomalies'].append(f"{col} has {len(outliers)} potential outliers")
        
        # Categorical analysis
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols[:2]:
            if col in df.columns:
                unique_count = df[col].nunique()
                total_count = len(df)
                
                if unique_count / total_count < 0.1:  # Low diversity
                    top_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                    insights['business_insights'].append(f"{col} dominated by {top_value}")
        
        # Data quality checks
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            high_missing = missing_data[missing_data > len(df) * 0.1]
            if len(high_missing) > 0:
                insights['data_quality_notes'].append(f"Significant missing data in: {', '.join(high_missing.index)}")
        
        return insights
    
    def _format_data_insights(self, insights: Dict) -> str:
        """Format insights for prompt inclusion."""
        formatted = []
        
        for category, items in insights.items():
            if items:
                formatted.append(f"{category.replace('_', ' ').title()}:")
                for item in items[:3]:  # Limit to prevent prompt bloat
                    formatted.append(f"  - {item}")
        
        return "\n".join(formatted) if formatted else "No significant patterns detected"
    
    def _get_statistical_summary(self, df: pd.DataFrame) -> str:
        """Get concise statistical summary."""
        if df.empty:
            return "No data to summarize"
        
        summary_parts = []
        
        # Numeric summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:2]:
                if col in df.columns:
                    total = df[col].sum()
                    avg = df[col].mean()
                    summary_parts.append(f"{col}: Total={total:,.1f}, Avg={avg:,.1f}")
        
        # Categorical summary
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols[:2]:
            if col in df.columns:
                unique_count = df[col].nunique()
                summary_parts.append(f"{col}: {unique_count} unique values")
        
        return "; ".join(summary_parts)
    
    def _get_relevant_conversation_history(self) -> str:
        """Get relevant conversation context."""
        if not self.conversation_context.chat_history:
            return "This is the start of our conversation."
        
        # Include last 2 exchanges for context
        history_items = []
        for q, r in self.conversation_context.chat_history[-2:]:
            history_items.append(f"Earlier you asked: {q[:80]}...")
            history_items.append(f"I responded about: {r[:120]}...")
        
        if self.conversation_context.session_insights:
            history_items.append(f"Key insights we've discovered: {', '.join(self.conversation_context.session_insights[-2:])}")
        
        return "\n".join(history_items)
    
    def _extract_session_insights(self, response: str, df: pd.DataFrame) -> None:
        """Extract insights from response to remember for future context."""
        # Simple keyword-based insight extraction
        insight_keywords = ['pattern', 'trend', 'anomaly', 'outlier', 'dominated', 'interesting', 'unusual']
        
        sentences = response.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in insight_keywords):
                clean_insight = sentence.strip()
                if len(clean_insight) > 20 and len(clean_insight) < 100:
                    self.conversation_context.session_insights.append(clean_insight)
        
        # Keep only last 10 insights
        self.conversation_context.session_insights = self.conversation_context.session_insights[-10:]
    
    def _handle_query_failure(self, question: str, error_message: str) -> str:
        """Handle query failures with expert-level troubleshooting."""
        if not self.openai_client:
            return f"I ran into a technical issue: {error_message}. Let me know if you'd like to try a different approach."
        
        try:
            prompt = f"""
The user asked: "{question}"
But we encountered this error: {error_message}

CONVERSATION CONTEXT:
User expertise: {self.conversation_context.user_expertise_level}
Theme: {self.conversation_context.conversation_theme}

As an expert colleague, explain what likely went wrong and suggest specific alternatives. Be collaborative and helpful, not just technical.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful expert explaining technical issues in a collaborative, human way."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        except:
            return f"I ran into a technical issue: {error_message}. This might be a data access problem or the question might need to be more specific. Want to try asking about what data is available first?"
    
    def _handle_no_data_found(self, question: str, sql_query: str) -> str:
        """Handle no data found with intelligent alternatives."""
        if not self.openai_client:
            return "I didn't find any data matching those criteria. This could be because the specific entities, time periods, or conditions don't exist in our database. Want to try asking about what data is available?"
        
        try:
            available_context = self._get_available_data_summary()
            
            prompt = f"""
The user asked: "{question}"
SQL query found no results: {sql_query}

AVAILABLE DATA CONTEXT:
{available_context}

CONVERSATION CONTEXT:
Recent topics: {', '.join(self.conversation_context.recent_topics[-3:])}
User expertise: {self.conversation_context.user_expertise_level}

As a thoughtful colleague, explain why we didn't find data and suggest specific, helpful alternatives based on what IS available.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a collaborative data expert helping explain why queries return no results and suggesting constructive alternatives."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        except:
            return "I didn't find data matching those criteria. This could be because the specific wells, operators, or time periods don't exist in our database. Want me to show you what data is available so we can explore other angles?"
    
    def _get_available_data_summary(self) -> str:
        """Get summary of available data for context."""
        try:
            tables = list(self.db_manager.schema_cache.values())
            summary_parts = [f"Database has {len(tables)} tables"]
            
            for table in tables[:3]:
                if table.row_count:
                    summary_parts.append(f"{table.name}: {table.row_count:,} records")
                key_cols = [col['name'] for col in table.columns[:4]]
                summary_parts.append(f"  Columns: {', '.join(key_cols)}")
            
            return "\n".join(summary_parts)
        except:
            return "Database structure information not available"
    
    def _generate_fallback_response(self, question: str, query_result: QueryResult) -> str:
        """Enhanced fallback when OpenAI unavailable."""
        df = query_result.data
        
        response_parts = [
            f"Found {len(df)} records for your question about the data."
        ]
        
        if len(df) > 0:
            # Basic insights
            numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
            if numeric_cols:
                for col in numeric_cols[:2]:
                    total = df[col].sum()
                    response_parts.append(f"Total {col}: {total:,.1f}")
            
            # Categorical insights
            text_cols = [col for col in df.columns if df[col].dtype == 'object']
            for col in text_cols[:1]:
                unique_count = df[col].nunique()
                response_parts.append(f"{unique_count} unique {col} values")
        
        response_parts.extend([
            "\nThis gives us a good starting point for analysis.",
            "What aspect would you like to explore further?"
        ])
        
        return " ".join(response_parts)
    
    def ask_question(self, question: str) -> str:
        """
        Main interface with full expert colleague capabilities.
        """
        # Analyze user context and intent
        self.analyze_user_context(question)
        
        # Execute the query
        query_result = self.execute_query(question)
        
        # Generate expert response
        response = self.generate_expert_response(question, query_result)
        
        # Store in conversation history
        self.conversation_context.chat_history.append((question, response))
        
        # Keep conversation history manageable
        if len(self.conversation_context.chat_history) > 20:
            self.conversation_context.chat_history = self.conversation_context.chat_history[-15:]
        
        return response


# For backwards compatibility, alias the new class
DatabaseAssistant = ExpertDatabaseAssistant