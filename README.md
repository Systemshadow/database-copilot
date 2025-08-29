# Database Copilot - AI-Powered Oil & Gas Data Analysis

Transform your oil & gas data into insights through natural language queries. Ask questions in plain English and get intelligent responses with professional table displays.

## Live Product Demo

**Try the interactive demo**: [https://database-copilot-ddmyeyubwl22mzb7jjgj9c.streamlit.app/](https://database-copilot-ddmyeyubwl22mzb7jjgj9c.streamlit.app/)

*Demo uses NY DEC oil & gas data to showcase product capabilities with your production and wells databases.*

## Product Features

- **Natural Language Interface**: Ask data questions in plain English
- **Multi-Database Support**: Works with SQL Server, PostgreSQL, MySQL, Oracle, and more
- **AI-Powered Analysis**: GPT-4 converts questions to optimized SQL queries
- **Professional Output**: Results displayed in formatted, exportable tables  
- **Cross-Dataset Intelligence**: Automatically joins production and wells data when needed
- **Industry Context**: Responses include relevant oil & gas domain knowledge
- **Enterprise Ready**: Secure, read-only database access with safety validation

## How It Works

1. **Connect to Your Database**: Point to your existing production/wells database
2. **Ask Questions Naturally**: "Show me top producing wells" or "Compare production by formation"  
3. **Get Intelligent Responses**: AI analyzes your data and provides insights with tables
4. **Export Results**: Download analysis as CSV for further use

## Demo Dataset (NY DEC)

The live demo showcases capabilities using New York Department of Environmental Conservation data:

- **67,812 production records** (2020-2024) - Oil, gas, water production by well/month
- **47,111 wells records** - Technical data including depths, formations, coordinates
- **Multi-table analysis** - Production metrics combined with well characteristics

## Example Business Questions

### Production Analysis
- "What are our top 10 producing wells this quarter?"
- "Show total gas production by county and operator"  
- "Which fields had declining production last year?"
- "Compare our oil vs gas production trends"

### Operational Intelligence
- "Show production for all horizontal wells in [Field Name]"
- "Which wells over 8,000 ft depth are underperforming?"
- "What's the average production by completion type?"
- "Find wells with unusual water production patterns"

### Strategic Analysis
- "Compare our performance vs regional averages"
- "Which formations show the best EUR potential?"
- "Analyze production decline curves by well type"
- "Show ROI by drilling program and vintage"

## Supported Database Types

- **Microsoft SQL Server** (Most common in O&G)
- **PostgreSQL**
- **MySQL/MariaDB**
- **Oracle Database**  
- **SQLite** (For testing/demos)

## Enterprise Deployment

### Database Connection
```env
DATABASE_TYPE=sqlserver
DATABASE_HOST=your-server.company.com
DATABASE_NAME=ProductionDB
DATABASE_USER=readonly_user
DATABASE_PASSWORD=secure_password
```

### Security Features
- **Read-Only Access**: Only SELECT queries allowed
- **SQL Injection Protection**: All queries validated before execution
- **Local Processing**: No data leaves your environment
- **Audit Trail**: All queries logged for compliance

## Quick Start (Evaluation)

### 1. Clone Repository
```bash
git clone [repository-url]
cd ai-oilwell-prototype
```

### 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3. Configure Database
Create `.env` file pointing to your database:
```env
OPENAI_API_KEY=your_api_key
DATABASE_TYPE=sqlserver
DATABASE_HOST=your_server.company.com  
DATABASE_NAME=your_production_db
DATABASE_USER=readonly_user
DATABASE_PASSWORD=your_password
```

### 4. Run Application
```bash
streamlit run app.py
```

## Architecture

**Multi-Database Design**
```
Database Copilot
├── Universal Database Connector (SQL Server, PostgreSQL, MySQL, Oracle)
├── AI Query Engine (GPT-4 powered SQL generation)  
├── Schema Discovery (Automatic table/column detection)
├── Safety Layer (Read-only validation and injection protection)
└── Professional UI (Tables, exports, industry context)
```

**Your Data Stays Secure**
- Connects directly to your existing database
- No data upload or cloud storage required
- All processing happens in your environment
- Uses your existing database security and permissions

## ROI for Oil & Gas Companies

- **Faster Analysis**: Minutes instead of hours for complex queries
- **Broader Access**: Non-technical staff can query production data  
- **Better Decisions**: AI provides context and identifies patterns
- **Reduced IT Load**: Self-service analytics reduces database admin requests
- **Standardized Reporting**: Consistent analysis across teams

## Demo vs Production

**This Demo Shows:**
- Core product functionality with real O&G data
- Natural language query capabilities  
- Multi-table analysis features
- Professional output formatting

**Your Production Deployment Gets:**
- Connection to YOUR database (SQL Server, PostgreSQL, etc.)
- YOUR data, YOUR wells, YOUR production history
- Customized for YOUR field names and data structure
- Integration with YOUR existing database security

## Technical Requirements

- **Database**: SQL Server 2016+, PostgreSQL 10+, MySQL 5.7+, Oracle 12c+
- **Network**: ODBC/JDBC connectivity to database server
- **Security**: Read-only database user account
- **API**: OpenAI API key for AI-powered analysis

## Contact for Enterprise Deployment

Ready to see Database Copilot analyze YOUR oil & gas data? 

- **Schedule Demo**: See it work with your actual database
- **Pilot Program**: 30-day evaluation with your data
- **Custom Integration**: Tailored for your data schema and workflows

---

**Database Copilot: Transform your oil & gas data from complex queries to conversational insights.**