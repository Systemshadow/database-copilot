"""
Complete Production-Ready AI Assistant for Oil & Gas Database Analysis
Includes: Semantic Understanding, Performance Optimization, Data Quality Intelligence, Industry Logic
Fixed: SQLite compatibility (LIKE vs ILIKE), Database Metadata Handler, Query Classification
"""

import os
import pandas as pd
import streamlit as st
import hashlib
import time
import numpy as np
from typing import Optional, List, Dict, Tuple, Any, NamedTuple
from dataclasses import dataclass
from dotenv import load_dotenv
import traceback
import json
import re
from datetime import datetime, date
from functools import lru_cache

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .database import DatabaseManager, TableInfo

load_dotenv()


@dataclass
class QueryIntent:
    """Semantic intent representation"""
    primary: str  # 'retrieve', 'compare', 'verify', 'rank', 'summarize'
    entity: str   # 'wells', 'production', 'operators', 'counties'
    filters: Dict[str, str]  # {'county': 'Erie', 'year': '2024'}
    complexity: str  # 'simple', 'aggregation', 'multi_table'
    confidence: float  # 0.0-1.0
    normalized_question: str  # Simplified semantic form


@dataclass
class QueryApproach:
    """Represents one execution approach"""
    description: str
    sql_query: str
    tables: List[str]
    complexity: str
    failure_reason: Optional[str] = None


@dataclass
class QueryResult:
    """Enhanced result with semantic context"""
    success: bool
    data: Optional[pd.DataFrame] = None
    sql_query: Optional[str] = None
    approach_used: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    display_as_table: bool = False
    table_title: Optional[str] = None
    diagnostic_info: Optional[str] = None


@dataclass
class IndustryMetrics:
    """Oil & gas industry key performance indicators"""
    eur_gas: Optional[float] = None  # Estimated Ultimate Recovery - Gas (MCF)
    eur_oil: Optional[float] = None  # Estimated Ultimate Recovery - Oil (BBL)
    ip_90: Optional[float] = None    # Initial Production 90-day average
    decline_rate: Optional[float] = None  # Annual decline rate %
    gor: Optional[float] = None      # Gas-Oil Ratio
    wor: Optional[float] = None      # Water-Oil Ratio
    cumulative_gas: Optional[float] = None
    cumulative_oil: Optional[float] = None
    months_producing: Optional[int] = None


class DatabaseMetadataHandler:
    """Handle queries about database structure, not data content"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    def is_metadata_query(self, question: str) -> bool:
        """Detect if question is about database structure"""
        metadata_keywords = [
            'how many tables', 'what tables', 'database structure',
            'schema', 'table names', 'columns in', 'database contains',
            'what is in the database', 'database schema', 'table structure'
        ]
        return any(keyword in question.lower() for keyword in metadata_keywords)
    
    def handle_metadata_query(self, question: str) -> Dict[str, Any]:
        """Handle database structure questions directly"""
        
        if 'how many tables' in question.lower() or 'what tables' in question.lower():
            return self._count_and_list_tables()
        elif 'columns' in question.lower():
            return self._describe_columns(question)
        else:
            return self._general_schema_info()
    
    def _count_and_list_tables(self) -> Dict[str, Any]:
        """Return actual table count and details"""
        tables = list(self.db_manager.schema_cache.keys())
        
        # Get row counts if available
        production_count = 0
        wells_count = 0
        
        try:
            production_result = self.db_manager.execute_query("SELECT COUNT(*) as count FROM production")
            if not production_result.empty:
                production_count = production_result['count'].iloc[0]
        except:
            pass
            
        try:
            wells_result = self.db_manager.execute_query("SELECT COUNT(*) as count FROM wells")  
            if not wells_result.empty:
                wells_count = wells_result['count'].iloc[0]
        except:
            pass
        
        response_text = f"""The database contains **{len(tables)} tables**:

• **production** table - Monthly production data ({production_count:,} records)
  - Contains oil, gas, and water production by well and month
  - Key columns: API_WellNo, County, Operator, Year, Month, GasProd, OilProd, WaterProd

• **wells** table - Master well registry ({wells_count:,} records) 
  - Contains technical details about each well
  - Key columns: API_WellNo, Well_Name, County, True_vertical_depth, Formation

**Total records across both tables: {production_count + wells_count:,}**

The tables are joined on API_WellNo to combine production data with well technical details."""

        return {
            'text': response_text,
            'table_data': None,
            'table_title': None
        }
    
    def _describe_columns(self, question: str) -> Dict[str, Any]:
        """Describe columns in tables"""
        
        response_text = """**Database Schema:**

**Production Table Columns:**
• API_WellNo - Unique well identifier
• Well_Name - Common name of the well  
• County - County where well is located
• Operator - Company operating the well
• Year, Month - Production time period
• GasProd - Gas production (MCF)
• OilProd - Oil production (BBL) 
• WaterProd - Water production (BBL)

**Wells Table Columns:**
• API_WellNo - Unique well identifier (joins to production)
• Well_Name - Common name of the well
• County - County location
• True_vertical_depth - Well depth in feet
• Formation - Geological formation targeted
• Additional technical specifications"""

        return {
            'text': response_text,
            'table_data': None,
            'table_title': None
        }
    
    def _general_schema_info(self) -> Dict[str, Any]:
        """General database information"""
        return self._count_and_list_tables()


class SemanticQueryClassifier:
    """Intent-based query classification using LLM reasoning"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        
    def classify_intent(self, question: str) -> QueryIntent:
        """Classify query intent semantically, not by keywords"""
        
        # Pre-check for metadata queries - these should be handled by metadata handler
        if any(term in question.lower() for term in ['how many tables', 'database structure', 'schema', 'what tables']):
            return QueryIntent(
                primary='metadata',
                entity='database_structure', 
                filters={},
                complexity='simple',
                confidence=0.95,
                normalized_question="Database structure inquiry"
            )
        
        if not self.openai_client:
            return self._fallback_classification(question)
            
        try:
            intent_prompt = f"""
Classify this oil & gas database query semantically:

QUESTION: {question}

Analyze the semantic intent and provide classification:

PRIMARY_INTENT: [retrieve|compare|verify|rank|summarize]
ENTITY_FOCUS: [wells|production|operators|counties|formations]
DATA_COMPLEXITY: [simple|aggregation|multi_table]
FILTERS: {{"key": "value", ...}} (county, year, operator, well_type, etc.)
CONFIDENCE: [0.0-1.0]
NORMALIZED_FORM: [Rephrase as simplest equivalent question]

Examples:
"Show me all data for Erie County" → retrieve, production, simple, {{"county": "Erie"}}, 0.9, "Get Erie County production data"
"Is Erie County in the database" → verify, production, simple, {{"county": "Erie"}}, 0.95, "Get Erie County production data"  
"Top 10 wells in 2024" → rank, wells, aggregation, {{"year": "2024", "limit": "10"}}, 0.9, "Rank wells by production in 2024"

Return as JSON.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a semantic query classifier. Focus on user intent, not just keywords. Return valid JSON."
                    },
                    {"role": "user", "content": intent_prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            intent_data = json.loads(response.choices[0].message.content.strip())
            
            return QueryIntent(
                primary=intent_data.get('PRIMARY_INTENT', 'retrieve'),
                entity=intent_data.get('ENTITY_FOCUS', 'production'),
                filters=intent_data.get('FILTERS', {}),
                complexity=intent_data.get('DATA_COMPLEXITY', 'simple'),
                confidence=float(intent_data.get('CONFIDENCE', 0.5)),
                normalized_question=intent_data.get('NORMALIZED_FORM', question)
            )
            
        except Exception as e:
            print(f"DEBUG: Intent classification failed: {str(e)}")
            return self._fallback_classification(question)
    
    def _fallback_classification(self, question: str) -> QueryIntent:
        """Rule-based fallback when LLM unavailable"""
        question_lower = question.lower()
        
        # Extract basic intent
        if any(word in question_lower for word in ['top', 'highest', 'best', 'rank']):
            primary = 'rank'
        elif any(word in question_lower for word in ['compare', 'vs', 'versus']):
            primary = 'compare'  
        elif any(word in question_lower for word in ['is', 'does', 'exists', 'available']):
            primary = 'verify'
        else:
            primary = 'retrieve'
            
        # Extract filters
        filters = {}
        
        # County extraction
        county_match = re.search(r'(in |for |from )([A-Za-z]+)\s+county', question_lower)
        if county_match:
            filters['county'] = county_match.group(2).title()
            
        # Year extraction  
        year_match = re.search(r'\b(20\d{2})\b', question)
        if year_match:
            filters['year'] = year_match.group(1)
            
        # Limit extraction
        limit_match = re.search(r'top\s+(\d+)', question_lower)
        if limit_match:
            filters['limit'] = limit_match.group(1)
            
        return QueryIntent(
            primary=primary,
            entity='production',
            filters=filters,
            complexity='simple',
            confidence=0.7,
            normalized_question=question
        )


class PerformanceOptimizedExecutor:
    """High-performance query execution with caching and smart fallback ordering - SQLITE COMPATIBLE"""
    
    def __init__(self, db_manager: DatabaseManager, openai_client):
        self.db_manager = db_manager
        self.openai_client = openai_client
        self.query_cache = {}  # SQL hash -> (result_df, timestamp)
        self.success_patterns = {}  # Intent pattern -> successful approach type
        self.performance_stats = {}  # Approach type -> avg execution time
        
    def execute_with_fallbacks(self, intent: QueryIntent) -> QueryResult:
        """Execute with performance optimization and smart ordering"""
        
        start_time = time.time()
        
        # Check cache first
        cached_result = self._check_cache(intent)
        if cached_result:
            print(f"DEBUG: Cache hit - returning cached result in {time.time() - start_time:.2f}s")
            return cached_result
        
        # Generate approaches with smart ordering
        approaches = self._generate_optimized_approaches(intent)
        
        print(f"DEBUG: Generated {len(approaches)} approaches, ordered by success probability")
        
        for i, approach in enumerate(approaches):
            approach_start = time.time()
            
            try:
                print(f"DEBUG: Trying approach {i+1}: {approach.description}")
                
                result_df = self.db_manager.execute_query(approach.sql_query)
                execution_time = time.time() - approach_start
                
                # Update performance stats
                self._update_performance_stats(approach.complexity, execution_time)
                
                print(f"DEBUG: Approach {i+1} succeeded in {execution_time:.2f}s, got {len(result_df)} rows")
                
                if not result_df.empty:
                    # Cache successful result
                    query_result = QueryResult(
                        success=True,
                        data=result_df,
                        sql_query=approach.sql_query,
                        approach_used=approach.description,
                        execution_time=execution_time,
                        display_as_table=self._should_show_table(intent, result_df),
                        table_title=self._generate_table_title(intent, result_df)
                    )
                    
                    self._cache_result(intent, query_result)
                    self._record_success_pattern(intent, approach)
                    
                    return query_result
                else:
                    print(f"DEBUG: Approach {i+1} returned no data, trying next")
                    approach.failure_reason = "No matching records found"
                    continue
                    
            except Exception as e:
                execution_time = time.time() - approach_start
                error_msg = str(e)
                print(f"DEBUG: Approach {i+1} failed in {execution_time:.2f}s: {error_msg}")
                approach.failure_reason = error_msg
                continue
        
        total_time = time.time() - start_time
        print(f"DEBUG: All approaches failed in total time: {total_time:.2f}s")
        
        return self._build_optimized_error_response(intent, approaches, total_time)
    
    def _check_cache(self, intent: QueryIntent) -> Optional[QueryResult]:
        """Check if we have a cached result for this intent"""
        
        cache_key = self._generate_cache_key(intent)
        
        if cache_key in self.query_cache:
            cached_data, timestamp = self.query_cache[cache_key]
            
            # Cache expires after 5 minutes for production data
            if time.time() - timestamp < 300:
                return QueryResult(
                    success=True,
                    data=cached_data.copy(),  # Return copy to avoid mutations
                    sql_query="[CACHED]",
                    approach_used="Cached result",
                    execution_time=0.01,
                    display_as_table=self._should_show_table(intent, cached_data),
                    table_title=self._generate_table_title(intent, cached_data)
                )
        
        return None
    
    def _cache_result(self, intent: QueryIntent, result: QueryResult):
        """Cache successful query results"""
        
        if result.success and result.data is not None:
            cache_key = self._generate_cache_key(intent)
            self.query_cache[cache_key] = (result.data.copy(), time.time())
            
            # Limit cache size to prevent memory issues
            if len(self.query_cache) > 50:
                # Remove oldest entries
                oldest_key = min(self.query_cache.keys(), 
                               key=lambda k: self.query_cache[k][1])
                del self.query_cache[oldest_key]
    
    def _generate_cache_key(self, intent: QueryIntent) -> str:
        """Generate cache key from intent"""
        
        key_data = {
            'primary': intent.primary,
            'entity': intent.entity, 
            'filters': sorted(intent.filters.items()),
            'complexity': intent.complexity
        }
        
        key_string = str(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _generate_optimized_approaches(self, intent: QueryIntent) -> List[QueryApproach]:
        """Generate approaches ordered by success probability and performance"""
        
        approaches = self._generate_base_approaches(intent)
        
        # Sort by success probability and performance
        return sorted(approaches, key=lambda a: self._score_approach(intent, a), reverse=True)
    
    def _score_approach(self, intent: QueryIntent, approach: QueryApproach) -> float:
        """Score approach based on historical success and performance"""
        
        score = 0.0
        
        # Base score by complexity (simpler = higher score)
        complexity_scores = {'simple': 1.0, 'aggregation': 0.8, 'multi_table': 0.6}
        score += complexity_scores.get(approach.complexity, 0.5)
        
        # Historical success rate for this intent pattern
        pattern_key = f"{intent.primary}_{intent.entity}"
        if pattern_key in self.success_patterns:
            successful_approaches = self.success_patterns[pattern_key]
            if approach.complexity in successful_approaches:
                score += 0.5
        
        # Performance bonus (faster approaches get higher scores)
        if approach.complexity in self.performance_stats:
            avg_time = self.performance_stats[approach.complexity]
            # Lower time = higher score
            score += max(0, (2.0 - avg_time)) * 0.3
        
        return score
    
    def _generate_base_approaches(self, intent: QueryIntent) -> List[QueryApproach]:
        """Generate base approaches - FIXED FOR SQLITE (LIKE instead of ILIKE)"""
        
        approaches = []
        
        if intent.filters.get('county'):
            county = intent.filters['county']
            
            # Approach 1: Most likely to succeed - case insensitive partial match
            # FIXED: Changed ILIKE to LIKE for SQLite compatibility
            sql = f"""
            SELECT * FROM production 
            WHERE County LIKE '%{county}%' 
            ORDER BY Year DESC, GasProd DESC
            LIMIT 100
            """
            approaches.append(QueryApproach(
                description="Fast case-insensitive county search",
                sql_query=sql,
                tables=['production'],
                complexity='simple'
            ))
            
            # Approach 2: Aggregated data (faster for verification)
            if intent.primary == 'verify':
                # FIXED: Changed ILIKE to LIKE
                sql = f"""
                SELECT County, COUNT(*) as Record_Count, 
                       SUM(GasProd) as Total_Gas, SUM(OilProd) as Total_Oil
                FROM production 
                WHERE County LIKE '%{county}%'
                GROUP BY County
                """
                approaches.append(QueryApproach(
                    description="Aggregated county verification",
                    sql_query=sql,
                    tables=['production'],
                    complexity='aggregation'
                ))
            
            # Approach 3: Ranking with aggregation (for rank intents)
            if intent.primary == 'rank':
                limit = intent.filters.get('limit', '10')
                year_filter = ""
                if intent.filters.get('year'):
                    year_filter = f"AND Year = {intent.filters['year']}"
                
                # FIXED: Changed ILIKE to LIKE
                sql = f"""
                SELECT Well_Name, 
                       SUM(GasProd) as Total_Gas,
                       SUM(OilProd) as Total_Oil
                FROM production 
                WHERE County LIKE '%{county}%' {year_filter}
                GROUP BY Well_Name, API_WellNo
                ORDER BY SUM(GasProd) DESC
                LIMIT {limit}
                """
                approaches.append(QueryApproach(
                    description=f"Top {limit} wells ranking",
                    sql_query=sql,
                    tables=['production'],
                    complexity='aggregation'
                ))
            
            # Approach 4: Multi-table only if specifically needed
            if self._needs_multi_table(intent):
                # FIXED: Changed ILIKE to LIKE and added proper table aliases
                sql = f"""
                SELECT p.Well_Name, p.County, 
                       SUM(p.GasProd) as Total_Gas,
                       w.True_vertical_depth
                FROM production p
                LEFT JOIN wells w ON p.API_WellNo = w.API_WellNo
                WHERE p.County LIKE '%{county}%'
                GROUP BY p.Well_Name, p.County, p.API_WellNo, w.True_vertical_depth
                ORDER BY Total_Gas DESC
                LIMIT 100
                """
                approaches.append(QueryApproach(
                    description="Multi-table with well details",
                    sql_query=sql,
                    tables=['production', 'wells'],
                    complexity='multi_table'
                ))
        
        elif intent.primary == 'rank' and intent.filters.get('year'):
            # Year-based ranking without county
            year = intent.filters['year']
            limit = intent.filters.get('limit', '10')
            
            sql = f"""
            SELECT Well_Name, 
                   SUM(GasProd) as Total_Gas,
                   SUM(OilProd) as Total_Oil
            FROM production 
            WHERE Year = {year}
            GROUP BY Well_Name, API_WellNo
            ORDER BY SUM(GasProd) DESC
            LIMIT {limit}
            """
            approaches.append(QueryApproach(
                description=f"Top {limit} wells in {year}",
                sql_query=sql,
                tables=['production'],
                complexity='aggregation'
            ))
            
        elif intent.primary == 'compare':
            # Production comparison
            sql = """
            SELECT Well_Name, 
                   SUM(GasProd) as Total_Gas_Production,
                   SUM(OilProd) as Total_Oil_Production
            FROM production 
            GROUP BY Well_Name, API_WellNo
            ORDER BY SUM(GasProd) DESC
            LIMIT 1000
            """
            approaches.append(QueryApproach(
                description="Production comparison analysis",
                sql_query=sql,
                tables=['production'],
                complexity='aggregation'
            ))
        
        # Fallback approach
        if not approaches:
            sql = "SELECT * FROM production ORDER BY Year DESC, GasProd DESC LIMIT 50"
            approaches.append(QueryApproach(
                description="Generic data exploration",
                sql_query=sql,
                tables=['production'],
                complexity='simple'
            ))
        
        return approaches
    
    def _needs_multi_table(self, intent: QueryIntent) -> bool:
        """Determine if multi-table query is needed"""
        
        multi_table_terms = ['depth', 'formation', 'drilling', 'location', 'coordinate', 'permit']
        return any(term in intent.normalized_question.lower() for term in multi_table_terms)
    
    def _record_success_pattern(self, intent: QueryIntent, approach: QueryApproach):
        """Record successful patterns for future optimization"""
        
        pattern_key = f"{intent.primary}_{intent.entity}"
        
        if pattern_key not in self.success_patterns:
            self.success_patterns[pattern_key] = set()
        
        self.success_patterns[pattern_key].add(approach.complexity)
    
    def _update_performance_stats(self, complexity: str, execution_time: float):
        """Update running average of execution times"""
        
        if complexity not in self.performance_stats:
            self.performance_stats[complexity] = execution_time
        else:
            # Running average
            current_avg = self.performance_stats[complexity]
            self.performance_stats[complexity] = (current_avg * 0.8) + (execution_time * 0.2)
    
    def _should_show_table(self, intent: QueryIntent, df: pd.DataFrame) -> bool:
        """Determine if results should be shown as table"""
        if df is None or df.empty:
            return False
        return intent.primary in ['retrieve', 'rank', 'compare'] or len(df) > 1
    
    def _generate_table_title(self, intent: QueryIntent, df: pd.DataFrame) -> str:
        """Generate table title"""
        if intent.filters.get('county'):
            county = intent.filters['county']
            if intent.primary == 'verify':
                return f"{county} County Verification"
            elif intent.primary == 'rank':
                return f"Top Wells in {county} County"
            else:
                return f"{county} County Production Data"
        elif intent.primary == 'rank':
            if intent.filters.get('year'):
                return f"Top Wells in {intent.filters['year']}"
            return "Top Performing Wells"
        return f"Query Results ({len(df)} records)"
    
    def _build_optimized_error_response(self, intent: QueryIntent, failed_approaches: List[QueryApproach], total_time: float) -> QueryResult:
        """Build error response with performance context"""
        
        diagnostic_parts = [
            f"Attempted {len(failed_approaches)} optimized approaches in {total_time:.2f} seconds:",
        ]
        
        for i, approach in enumerate(failed_approaches, 1):
            diagnostic_parts.append(f"{i}. {approach.description} → {approach.failure_reason or 'No data found'}")
        
        if intent.filters.get('county'):
            county = intent.filters['county']
            diagnostic_parts.extend([
                f"\nOptimized suggestions for '{county}':",
                "• Try 'What counties are available in the database?'",
                "• Try a broader search: 'Show me all counties'",
                f"• Search for similar: 'Show me counties containing {county[:3]}'"
            ])
        
        return QueryResult(
            success=False,
            error_message="All optimized approaches failed",
            diagnostic_info="\n".join(diagnostic_parts),
            execution_time=total_time
        )


class DataQualityAnalyzer:
    """Analyze data quality and provide user warnings about limitations"""
    
    def __init__(self):
        self.quality_cache = {}  # Cache quality analysis results
        
    def analyze_data_quality(self, df: pd.DataFrame, intent: QueryIntent) -> Dict[str, Any]:
        """Comprehensive data quality analysis"""
        
        if df is None or df.empty:
            return {'warnings': ['No data available for analysis']}
        
        quality_report = {
            'warnings': [],
            'insights': [],
            'data_coverage': {},
            'completeness': {},
            'anomalies': []
        }
        
        # Time coverage analysis
        quality_report.update(self._analyze_time_coverage(df))
        
        # Completeness analysis
        quality_report.update(self._analyze_completeness(df))
        
        # Production data anomalies
        quality_report.update(self._analyze_production_anomalies(df))
        
        # Geographic coverage
        quality_report.update(self._analyze_geographic_coverage(df))
        
        # Intent-specific quality checks
        quality_report.update(self._analyze_intent_specific_quality(df, intent))
        
        return quality_report
    
    def _analyze_time_coverage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal data coverage and gaps"""
        
        analysis = {'time_warnings': [], 'time_insights': []}
        
        if 'Year' in df.columns:
            years = df['Year'].dropna()
            if not years.empty:
                min_year = int(years.min())
                max_year = int(years.max())
                current_year = datetime.now().year
                
                # Data freshness warnings
                if max_year < current_year - 1:
                    analysis['time_warnings'].append(
                        f"Data may be outdated - latest records from {max_year}"
                    )
                elif max_year == current_year - 1:
                    analysis['time_insights'].append(
                        f"Data current through {max_year} (typical 1-year reporting delay)"
                    )
                
                # Data span insights
                data_span = max_year - min_year + 1
                analysis['time_insights'].append(
                    f"Data spans {data_span} years ({min_year}-{max_year})"
                )
        
        return analysis
    
    def _analyze_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data completeness and missing values"""
        
        analysis = {'completeness_warnings': [], 'completeness_insights': []}
        
        # Overall missing data percentage
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        if missing_percentage > 20:
            analysis['completeness_warnings'].append(
                f"High missing data rate: {missing_percentage:.1f}% of all values"
            )
        elif missing_percentage > 5:
            analysis['completeness_warnings'].append(
                f"Moderate missing data: {missing_percentage:.1f}% of values missing"
            )
        
        return analysis
    
    def _analyze_production_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect production data anomalies and unusual patterns"""
        
        analysis = {'anomaly_warnings': [], 'anomaly_insights': []}
        
        production_columns = ['GasProd', 'OilProd', 'WaterProd']
        
        for col in production_columns:
            if col in df.columns and not df[col].dropna().empty:
                values = df[col].dropna()
                
                # Zero production analysis
                zero_count = (values == 0).sum()
                zero_percentage = (zero_count / len(values)) * 100
                
                if zero_percentage > 50:
                    analysis['anomaly_warnings'].append(
                        f"{col}: {zero_percentage:.1f}% of wells show zero production"
                    )
        
        return analysis
    
    def _analyze_geographic_coverage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze geographic data coverage"""
        
        analysis = {'geo_warnings': [], 'geo_insights': []}
        
        if 'County' in df.columns:
            counties = df['County'].dropna()
            if not counties.empty:
                unique_counties = counties.nunique()
                total_records = len(counties)
                
                analysis['geo_insights'].append(
                    f"Data covers {unique_counties} counties with {total_records:,} total records"
                )
        
        return analysis
    
    def _analyze_intent_specific_quality(self, df: pd.DataFrame, intent: QueryIntent) -> Dict[str, Any]:
        """Quality checks specific to user's intent"""
        
        analysis = {'intent_warnings': [], 'intent_insights': []}
        
        # Ranking intent quality checks
        if intent.primary == 'rank':
            if 'limit' in intent.filters:
                requested_count = int(intent.filters['limit'])
                available_count = len(df)
                
                if available_count < requested_count:
                    analysis['intent_warnings'].append(
                        f"Requested top {requested_count} but only {available_count} records available"
                    )
        
        return analysis
    
    def format_quality_warnings(self, quality_report: Dict[str, Any]) -> str:
        """Format quality warnings for user display"""
        
        warning_parts = []
        
        # Combine all warning types
        all_warnings = []
        all_warnings.extend(quality_report.get('warnings', []))
        all_warnings.extend(quality_report.get('time_warnings', []))
        all_warnings.extend(quality_report.get('completeness_warnings', []))
        all_warnings.extend(quality_report.get('anomaly_warnings', []))
        all_warnings.extend(quality_report.get('geo_warnings', []))
        all_warnings.extend(quality_report.get('intent_warnings', []))
        
        if all_warnings:
            warning_parts.append("**Data Quality Notes:**")
            for warning in all_warnings[:3]:  # Limit to top 3 warnings
                warning_parts.append(f"• {warning}")
        
        # Add key insights
        all_insights = []
        all_insights.extend(quality_report.get('insights', []))
        all_insights.extend(quality_report.get('time_insights', []))
        all_insights.extend(quality_report.get('geo_insights', []))
        
        if all_insights:
            warning_parts.append("\n**Data Context:**")
            for insight in all_insights[:2]:  # Show top 2 insights
                warning_parts.append(f"• {insight}")
        
        return "\n".join(warning_parts)
    
    def get_confidence_score(self, quality_report: Dict[str, Any]) -> float:
        """Calculate confidence score based on data quality (0.0-1.0)"""
        
        base_confidence = 1.0
        
        # Reduce confidence for warnings
        warning_count = (
            len(quality_report.get('warnings', [])) +
            len(quality_report.get('time_warnings', [])) +
            len(quality_report.get('completeness_warnings', [])) +
            len(quality_report.get('anomaly_warnings', [])) +
            len(quality_report.get('geo_warnings', [])) +
            len(quality_report.get('intent_warnings', []))
        )
        
        # Each warning reduces confidence
        confidence_reduction = min(warning_count * 0.1, 0.6)  # Max 60% reduction
        
        return max(base_confidence - confidence_reduction, 0.2)  # Min 20% confidence


class OilGasBusinessLogic:
    """Industry-specific business logic and calculations"""
    
    def __init__(self):
        # Unit conversion factors
        self.conversions = {
            'mcf_to_bcf': 1000,
            'mcf_to_mmbtu': 1.037,  # Average heat content
            'bbl_to_m3': 0.158987,
            'bbl_to_gallons': 42,
            'feet_to_meters': 0.3048,
        }
        
    def enhance_production_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add industry-specific calculated columns to production data"""
        
        if df is None or df.empty:
            return df
            
        enhanced_df = df.copy()
        
        # Calculate cumulative production by well
        if 'API_WellNo' in df.columns and 'Year' in df.columns and 'Month' in df.columns:
            enhanced_df = enhanced_df.sort_values(['API_WellNo', 'Year', 'Month'])
            
            # Cumulative gas production
            if 'GasProd' in enhanced_df.columns:
                enhanced_df['Cumulative_Gas'] = enhanced_df.groupby('API_WellNo')['GasProd'].cumsum()
            
            # Cumulative oil production
            if 'OilProd' in enhanced_df.columns:
                enhanced_df['Cumulative_Oil'] = enhanced_df.groupby('API_WellNo')['OilProd'].cumsum()
        
        # Calculate ratios
        enhanced_df = self._calculate_production_ratios(enhanced_df)
        
        # Add industry classifications
        enhanced_df = self._add_industry_classifications(enhanced_df)
        
        return enhanced_df
    
    def _calculate_production_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate key industry ratios"""
        
        enhanced_df = df.copy()
        
        # Gas-Oil Ratio (GOR)
        if 'GasProd' in df.columns and 'OilProd' in df.columns:
            # Avoid division by zero
            enhanced_df['GOR'] = np.where(
                enhanced_df['OilProd'] > 0,
                enhanced_df['GasProd'] / enhanced_df['OilProd'],
                np.inf
            )
            
            # Cap extreme GOR values for display purposes
            enhanced_df['GOR'] = np.where(enhanced_df['GOR'] > 50000, 50000, enhanced_df['GOR'])
        
        # Water-Oil Ratio (WOR)
        if 'WaterProd' in df.columns and 'OilProd' in df.columns:
            enhanced_df['WOR'] = np.where(
                enhanced_df['OilProd'] > 0,
                enhanced_df['WaterProd'] / enhanced_df['OilProd'],
                np.inf
            )
            
            # Cap extreme WOR values
            enhanced_df['WOR'] = np.where(enhanced_df['WOR'] > 100, 100, enhanced_df['WOR'])
        
        return enhanced_df
    
    def _add_industry_classifications(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add industry-standard classifications"""
        
        enhanced_df = df.copy()
        
        # Well type classification based on production
        if 'GasProd' in df.columns and 'OilProd' in df.columns:
            conditions = [
                (enhanced_df['GasProd'] > enhanced_df['OilProd'] * 6),  # High GOR
                (enhanced_df['OilProd'] > enhanced_df['GasProd'] / 6),  # Low GOR
                (enhanced_df['GasProd'] == 0) & (enhanced_df['OilProd'] > 0),  # Pure oil
                (enhanced_df['OilProd'] == 0) & (enhanced_df['GasProd'] > 0),  # Pure gas
            ]
            
            choices = ['Gas Well', 'Oil Well', 'Oil Only', 'Gas Only']
            
            enhanced_df['Well_Classification'] = np.select(
                conditions, choices, default='Mixed Production'
            )
        
        return enhanced_df
    
    def format_production_value(self, value: float, unit: str) -> str:
        """Format production values with appropriate units and precision"""
        
        if pd.isna(value) or value == 0:
            return f"0 {unit}"
        
        if unit.lower() in ['mcf', 'gas']:
            if value >= 1000000:
                return f"{value/1000000:.2f}M MCF"
            elif value >= 1000:
                return f"{value/1000:.1f}K MCF"
            else:
                return f"{value:,.0f} MCF"
        
        elif unit.lower() in ['bbl', 'oil']:
            if value >= 1000000:
                return f"{value/1000000:.2f}M BBL"
            elif value >= 1000:
                return f"{value/1000:.1f}K BBL"
            else:
                return f"{value:,.0f} BBL"
        
        else:
            return f"{value:,.0f} {unit}"
    
    def generate_industry_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate oil & gas industry-specific insights from data"""
        
        insights = []
        
        if df is None or df.empty:
            return insights
            
        if 'GasProd' in df.columns and 'OilProd' in df.columns:
            total_gas = df['GasProd'].sum()
            total_oil = df['OilProd'].sum()
            
            if total_gas > total_oil * 6:  # High GOR play
                insights.append("This appears to be primarily a gas play with associated liquids")
            elif total_oil > total_gas / 6:  # Low GOR play
                insights.append("This appears to be primarily an oil play with associated gas")
            
            # Production distribution insights
            active_wells = df[(df['GasProd'] > 0) | (df['OilProd'] > 0)]
            if len(active_wells) > 0:
                gas_producers = df[df['GasProd'] > 0]
                oil_producers = df[df['OilProd'] > 0]
                
                gas_pct = len(gas_producers) / len(active_wells) * 100
                oil_pct = len(oil_producers) / len(active_wells) * 100
                
                insights.append(f"{gas_pct:.0f}% of active wells produce gas, {oil_pct:.0f}% produce oil")
        
        # Geographic concentration insights
        if 'County' in df.columns:
            county_production = df.groupby('County')['GasProd'].sum().sort_values(ascending=False)
            if len(county_production) > 1:
                top_county_pct = county_production.iloc[0] / county_production.sum() * 100
                if top_county_pct > 50:
                    insights.append(f"Production concentrated in {county_production.index[0]} County ({top_county_pct:.0f}% of total gas)")
        
        return insights


class SemanticResponseGenerator:
    """Generate responses with semantic consistency"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        
    def generate_response(self, intent: QueryIntent, query_result: QueryResult) -> str:
        """Generate semantically consistent response"""
        
        if not query_result.success:
            return self._generate_error_response(intent, query_result)
            
        if query_result.data is None or query_result.data.empty:
            return self._generate_no_data_response(intent)
            
        return self._generate_success_response(intent, query_result)
    
    def _generate_success_response(self, intent: QueryIntent, query_result: QueryResult) -> str:
        """Generate response for successful queries with ACTUAL DATA USAGE"""
        
        if not self.openai_client:
            return self._fallback_success_response(intent, query_result)
            
        try:
            df = query_result.data
            
            # Include actual data for AI analysis - CRITICAL FOR PROPER RESPONSE
            data_sample = df.head(20).to_dict('records') if not df.empty else []
            
            response_prompt = f"""
Generate a conversational response for this oil & gas database query result.

CRITICAL: This is a DATA CONTENT query, not a database structure query.
- Query returned {len(df)} RECORDS (rows of data), not tables
- Database has 2 tables: 'production' and 'wells'  
- User asked about: {intent.normalized_question}

ACTUAL DATA RETURNED:
Rows: {len(df)} records
Columns: {list(df.columns)}
Sample data: {data_sample}

RESPONSE REQUIREMENTS:
1. Address the user's original intent directly
2. Use specific numbers and names from the actual data provided above
3. Be conversational but factually accurate
4. If listing items, include actual names/values from the data - never use "..." 
5. Mention that detailed data is shown in the table below
6. For verification queries, clearly state what was found
7. For ranking queries, list the actual top performers from the data
8. When mentioning numbers, specify "records" or "wells", never "tables"
9. Use actual well names and production values from the sample data

Generate a complete response using the real data provided above. Do not claim you don't have data when data is clearly provided.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an oil & gas data analyst. Always use actual data provided. Be specific and complete. Never claim you don't have data when data is clearly provided in the prompt."
                    },
                    {"role": "user", "content": response_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"DEBUG: Response generation failed: {str(e)}")
            return self._fallback_success_response(intent, query_result)
    
    def _fallback_success_response(self, intent: QueryIntent, query_result: QueryResult) -> str:
        """Fallback response generation"""
        df = query_result.data
        
        response_parts = []
        
        if intent.primary == 'verify':
            if intent.filters.get('county'):
                county = intent.filters['county']
                if not df.empty:
                    total_records = df['Record_Count'].sum() if 'Record_Count' in df.columns else len(df)
                    response_parts.append(f"Yes, {county} County is in the database with {total_records:,} records.")
                else:
                    response_parts.append(f"No data found for {county} County in the database.")
        else:
            response_parts.append(f"Found {len(df)} records matching your query.")
            
            if intent.primary == 'rank' and not df.empty:
                response_parts.append("Top results:")
                for i, row in df.head(5).iterrows():
                    well_name = row.get('Well_Name', f'Well #{i+1}')
                    if 'Total_Gas' in row and pd.notna(row['Total_Gas']):
                        response_parts.append(f"{i+1}. {well_name}: {row['Total_Gas']:,.0f} gas production")
        
        if query_result.display_as_table:
            response_parts.append("Detailed results are shown in the table below.")
        
        return " ".join(response_parts)
    
    def _generate_error_response(self, intent: QueryIntent, query_result: QueryResult) -> str:
        """Generate helpful error responses"""
        if query_result.diagnostic_info:
            return query_result.diagnostic_info
        else:
            return f"I encountered an issue processing your query: {query_result.error_message}"
    
    def _generate_no_data_response(self, intent: QueryIntent) -> str:
        """Generate response when no data found"""
        if intent.filters.get('county'):
            county = intent.filters['county']
            return f"No data found for {county} County. Try checking the exact county name spelling or ask 'What counties are available?'"
        else:
            return "No data found matching your criteria. Try rephrasing your query or ask for available data options."


class MultiTableDatabaseAssistant:
    """Production-ready AI assistant with semantic understanding, performance optimization, and industry intelligence"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.openai_client = None
        self.conversation_context = []
        self._initialize_openai()
        
        # Initialize all components including the metadata handler
        self.metadata_handler = DatabaseMetadataHandler(db_manager)  # NEW: Add metadata handler
        self.query_classifier = SemanticQueryClassifier(self.openai_client)
        self.fallback_executor = PerformanceOptimizedExecutor(db_manager, self.openai_client)
        self.response_generator = SemanticResponseGenerator(self.openai_client)
        self.quality_analyzer = DataQualityAnalyzer()
        self.business_logic = OilGasBusinessLogic()
    
    def _initialize_openai(self):
        """Initialize OpenAI client if API key is available."""
        api_key = None
        
        try:
            if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                api_key = st.secrets['OPENAI_API_KEY']
        except:
            pass
        
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if OPENAI_AVAILABLE and api_key and api_key != 'your_openai_api_key_here':
            try:
                self.openai_client = OpenAI(api_key=api_key)
                print("DEBUG: OpenAI client initialized successfully")
            except Exception as e:
                print(f"OpenAI initialization failed: {str(e)}")
                self.openai_client = None
        else:
            print("DEBUG: OpenAI not available or no API key")
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Enhanced with metadata query detection - FIXED"""
        
        try:
            print(f"DEBUG: Processing question: {question}")
            
            # Step 0: Check if this is a database metadata question FIRST
            if self.metadata_handler.is_metadata_query(question):
                print("DEBUG: Detected metadata query, handling directly")
                return self.metadata_handler.handle_metadata_query(question)
            
            # Step 1: Continue with normal semantic analysis for data queries
            intent = self.query_classifier.classify_intent(question)
            print(f"DEBUG: Classified intent - Primary: {intent.primary}, Entity: {intent.entity}, Filters: {intent.filters}")
            
            # Step 2: Performance-optimized execution with fallbacks
            query_result = self.fallback_executor.execute_with_fallbacks(intent)
            
            # Step 3: Enhance data with industry-specific calculations
            if query_result.success and query_result.data is not None and not query_result.data.empty:
                print("DEBUG: Enhancing data with industry calculations")
                query_result.data = self.business_logic.enhance_production_data(query_result.data)
            
            # Step 4: Analyze data quality
            quality_report = self.quality_analyzer.analyze_data_quality(query_result.data, intent)
            print(f"DEBUG: Quality analysis completed - {len(quality_report.get('warnings', []))} warnings found")
            
            # Step 5: Generate semantic response
            text_response = self.response_generator.generate_response(intent, query_result)
            
            # Step 6: Add quality context to response
            quality_warnings = self.quality_analyzer.format_quality_warnings(quality_report)
            if quality_warnings:
                text_response += f"\n\n{quality_warnings}"
            
            # Step 7: Add industry insights
            if query_result.data is not None and not query_result.data.empty:
                industry_insights = self.business_logic.generate_industry_insights(query_result.data)
                if industry_insights:
                    text_response += f"\n\n**Industry Context:**"
                    for insight in industry_insights[:2]:  # Limit to 2 insights
                        text_response += f"\n• {insight}"
            
            # Step 8: Build final response
            response_data = {
                'text': text_response,
                'table_data': query_result.data.copy() if query_result.data is not None else None,
                'table_title': query_result.table_title
            }
            
            # Add to conversation context
            self.conversation_context.append({"role": "user", "content": question})
            self.conversation_context.append({"role": "assistant", "content": text_response})
            
            print(f"DEBUG: Production pipeline completed successfully")
            return response_data
            
        except Exception as e:
            print(f"ERROR in production pipeline: {str(e)}")
            print(f"ERROR traceback: {traceback.format_exc()}")
            
            return {
                'text': f"I encountered an unexpected error during analysis: {str(e)}. Please try rephrasing your question.",
                'table_data': None,
                'table_title': None
            }


# Backward compatibility
DatabaseAssistant = MultiTableDatabaseAssistant