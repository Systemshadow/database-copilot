# Database Copilot 🤖

AI-powered database assistant for oil & gas companies. Ask questions about your production data in plain English and get intelligent, conversational responses.

## ✨ Features

- **Natural Language Queries**: Ask questions like "What was the production for well ABC-123 in January 2023?"
- **Intelligent Responses**: Get conversational answers with context and insights
- **Multi-Database Support**: Works with SQL Server, PostgreSQL, MySQL, Oracle
- **Safe Querying**: Only allows read-only SELECT queries
- **Auto Schema Discovery**: Automatically learns your database structure

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file with your database and API credentials:

```env
# OpenAI API Configuration (required for AI responses)
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DATABASE_TYPE=sqlserver
DATABASE_HOST=your_db_server.company.com
DATABASE_PORT=1433
DATABASE_NAME=ProductionDB
DATABASE_USER=your_username
DATABASE_PASSWORD=your_password

# Optional: Company name for branding
COMPANY_NAME=Your Oil & Gas Company
```

### 3. Test Configuration
```bash
python test_connection.py
```

### 4. Launch Application
```bash
streamlit run app.py
```

## 🔧 Database Configuration

### SQL Server
```env
DATABASE_TYPE=sqlserver
DATABASE_HOST=server.company.com
DATABASE_PORT=1433
DATABASE_NAME=ProductionDB
DATABASE_USER=username
DATABASE_PASSWORD=password
```

### PostgreSQL
```env
DATABASE_TYPE=postgresql
DATABASE_HOST=server.company.com
DATABASE_PORT=5432
DATABASE_NAME=production_db
DATABASE_USER=username
DATABASE_PASSWORD=password
```

### MySQL
```env
DATABASE_TYPE=mysql
DATABASE_HOST=server.company.com
DATABASE_PORT=3306
DATABASE_NAME=production_db
DATABASE_USER=username
DATABASE_PASSWORD=password
```

### Windows Authentication (SQL Server)
```env
DATABASE_TYPE=sqlserver
DATABASE_HOST=server.company.com
DATABASE_NAME=ProductionDB
DATABASE_TRUSTED_CONNECTION=yes
```

## 💬 Example Questions

- "What was the oil production for well ABC-123 in January 2023?"
- "Show me the top 10 producing wells last month"
- "What's the total gas production for Chesapeake Energy in 2023?"
- "Which county had the highest water production last year?"
- "How did production change for well XYZ-456 over the last 6 months?"

## 📊 Database Schema Tips

The system works best with oil & gas databases that have common column patterns:

- **Well Identifiers**: `API_Well_No`, `Well_ID`, `UWI`
- **Dates**: `Production_Date`, `Report_Date`, `Month_Year`
- **Production**: `Oil_Prod`, `Gas_Prod`, `Water_Prod`
- **Operators**: `Operator_Name`, `Company_Name`
- **Locations**: `County`, `Field_Name`, `Township_Range`

## 🔒 Security

- **Read-Only**: Only SELECT queries are allowed
- **SQL Injection Protection**: All queries are validated for safety
- **Local Processing**: Data stays in your environment
- **Configurable Access**: Use database user with read-only permissions

## 🛠️ Troubleshooting

### Connection Issues
1. Verify your `.env` file has correct database credentials
2. Ensure database server is accessible from your network
3. Check firewall settings and database port accessibility
4. Run `python test_connection.py` to diagnose issues

### AI Response Issues
1. Verify `OPENAI_API_KEY` is set correctly
2. Check your OpenAI account has sufficient credits
3. Ensure internet connectivity for OpenAI API calls

### Query Issues
1. Check that your database has expected table/column names
2. Run "Discover Schema" to see available tables
3. Try more specific questions with exact table/column names

## 📁 Project Structure

```
database-copilot/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                   # Environment configuration
├── test_connection.py     # Configuration test script
└── app_utils/
    ├── database.py        # Database connection & schema discovery
    └── ai_assistant.py    # AI-powered query generation & responses
```

## 🔄 Development

To add new features or customize for your specific database:

1. **Custom Query Patterns**: Modify `ai_assistant.py` to add domain-specific query logic
2. **Database-Specific Optimizations**: Update `database.py` for your database type
3. **Response Formatting**: Customize the AI response generation in `DatabaseAssistant`

## 📞 Support

For issues or customization requests, check:
1. Configuration with `test_connection.py`
2. Streamlit logs for detailed error messages
3. Database connection logs
4. OpenAI API status and usage

## 🚀 Future Enhancements

- Interactive charts and visualizations
- Query result export (CSV, Excel)
- Query history and favorites
- Role-based access control
- Custom report generation