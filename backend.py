import re
import os
import logging
import json
import requests
from dotenv import load_dotenv

from flask import Flask, request, jsonify
from flask_cors import CORS

# LangChain imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import the enhanced analysis tool
from analysis_tool import historical_analysis_tool

# --- Initialization ---
app = Flask(__name__)
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://127.0.0.1:5500")

CORS(app, origins=[
    FRONTEND_URL, 
    "http://127.0.0.1:5500", # Allows local testing
    "null" # Allows opening the local HTML file directly
])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# --- Configuration & API Keys ---
load_dotenv() 

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY or not WEATHER_API_KEY:
    raise ValueError("API keys for Google and OpenWeatherMap must be set in the .env file.")

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found. City extraction will fall back to regex patterns.")
else:
    logger.info("GROQ_API_KEY found. Enhanced city extraction enabled.")

# --- LangChain LLM Initialization ---
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=GOOGLE_API_KEY, 
        temperature=0,
        timeout=30
    )
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise

# --- RAG Pipeline Setup ---
PDF_FILE_PATH = "Air pollution and control by K.V. S.G. MuraLi Krishna.pdf"
CHROMA_PERSIST_DIR = "./rag_chroma_db"
rag_retriever = None
try:
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    if os.path.exists(CHROMA_PERSIST_DIR):
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_function)
        logger.info("Loaded existing vector store")
    elif os.path.exists(PDF_FILE_PATH):
        logger.info("Creating new vector store from PDF")
        loader = PyPDFLoader(PDF_FILE_PATH)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory=CHROMA_PERSIST_DIR)
        logger.info("Vector store created successfully")
    
    if 'vectorstore' in locals():
        rag_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        logger.info("RAG retriever initialized")
    
except Exception as e:
    logger.error(f"An error occurred during RAG setup: {e}", exc_info=True)
    rag_retriever = None

# --- Tool Definitions ---
def get_coordinates(city: str) -> dict | None:
    """Get coordinates for a city"""
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={WEATHER_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data:
            return {"lat": data[0]["lat"], "lon": data[0]["lon"]}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching coordinates for {city}: {e}")
        return None
    return None

def get_weather_tool(city: str) -> str:
    """Get current weather for a city"""
    coords = get_coordinates(city)
    if not coords:
        return f"Could not find coordinates for {city}."
    
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={coords['lat']}&lon={coords['lon']}&appid={WEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return f"Weather in {city}: Description: {data['weather'][0]['description']}, Temp: {data['main']['temp']}°C, Humidity: {data['main']['humidity']}%"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching weather for {city}: {e}")
        return "Failed to fetch weather data."

def get_aqi_tool(city: str) -> str:
    """Get current AQI for a city"""
    coords = get_coordinates(city)
    if not coords:
        return f"Could not find coordinates for {city}."
    
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={coords['lat']}&lon={coords['lon']}&appid={WEATHER_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        aqi_map = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
        aqi = data["list"][0]["main"]["aqi"]
        return f"AQI in {city}: Score: {aqi} ({aqi_map.get(aqi, 'Unknown')})"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching AQI for {city}: {e}")
        return "Failed to fetch AQI data."

def get_pollutants_tool(city: str) -> str:
    """Get current pollutant levels for a city"""
    coords = get_coordinates(city)
    if not coords:
        return f"Could not find coordinates for {city}."
    
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={coords['lat']}&lon={coords['lon']}&appid={WEATHER_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        components = data["list"][0]["components"]
        
        pollutant_info = []
        for pollutant, value in components.items():
            if value is not None:
                pollutant_info.append(f"{pollutant.upper()}: {value}")
        
        return f"Pollutants in {city} (µg/m³): {', '.join(pollutant_info)}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching pollutants for {city}: {e}")
        return "Failed to fetch pollutant data."

def scientific_rag_tool(query: str) -> str:
    """Search scientific knowledge base"""
    if not rag_retriever:
        return "RAG Tool is disabled due to initialization issues."
    
    try:
        docs = rag_retriever.invoke(query)
        context = "\n---\n".join([f"Source (Page {doc.metadata.get('page', 'N/A')}): {doc.page_content}" for doc in docs])
        return f"Retrieved scientific context:\n{context}"
    except Exception as e:
        logger.error(f"Error in RAG tool: {e}")
        return "Error retrieving scientific information."

def historical_analysis_wrapper(query: str) -> str:
    """
    Wrapper for historical analysis tool with enhanced error handling
    Now passes both API keys for Groq-enhanced city extraction
    """
    try:
        logger.info(f"Processing historical analysis query: {query}")

        result = historical_analysis_tool(
            query,
            api_key=WEATHER_API_KEY,
            groq_api_key=GROQ_API_KEY
        )
        logger.info("Historical analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in historical analysis: {e}", exc_info=True)
        return json.dumps({
            "error": f"Historical analysis failed: {str(e)}"
        })

# --- Enhanced City Extraction Tool (Optional standalone tool) ---
def city_extraction_tool(query: str) -> str:
    """
    Standalone tool for testing city extraction capabilities
    This is optional - mainly for debugging/testing
    """
    try:

        from analysis_tool import extract_city
        
        city = extract_city(query, groq_api_key=GROQ_API_KEY)
        if city:
            return f"Extracted city: {city}"
        else:
            return "No city found in the query."
    except Exception as e:
        return f"Error extracting city: {str(e)}"

# --- LangChain Tool Initialization ---
tools = [
    Tool(
        name="InternetSearch",
        func=DuckDuckGoSearchRun().run,
        description="A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events, general knowledge, or topics not covered by other tools. Input should be a search query."
    ),
    Tool(
        name="WeatherReport",
        func=get_weather_tool,
        description="Useful for getting the CURRENT weather report for a specific city. Input should be just the city name."
    ),
    Tool(
        name="AQIReport",
        func=get_aqi_tool,
        description="Useful for getting the CURRENT Air Quality Index (AQI) for a specific city. Input should be just the city name."
    ),
    Tool(
        name="PollutantsReport",
        func=get_pollutants_tool,
        description="Useful for getting the CURRENT concentration of various air pollutants for a specific city. Input should be just the city name."
    ),
    Tool(
        name="ScientificKnowledgeBase",
        func=scientific_rag_tool,
        description="Use this tool to find scientific explanations, models, and principles about air pollution from a detailed document. Input should be your search query about air pollution science."
    ),
    Tool(
        name="HistoricalDataAnalysis",
        func=historical_analysis_wrapper,
        description="Use this tool when the user asks for historical data, trends, plots, graphs, or comparisons over a period of time (e.g., 'last week', 'past month', 'show trends', 'plot data'). This tool can analyze and plot pollutants and weather variables over time using enhanced AI-powered city extraction. The input should be the full user query exactly as provided."
    ),

    Tool(
        name="CityExtraction",
        func=city_extraction_tool,
        description="Test tool for extracting city names from queries. Use this only for debugging city extraction issues. Input should be the user query."
    )
]

template = """You are an expert-level AI assistant specializing in air pollution, environmental science, and meteorology.
Your primary goal is to provide accurate, helpful, and conversational answers to user queries, leveraging your specialized tools when necessary.

**Your Core Capabilities:**
- Answer scientific questions about air pollution causes, effects, and control measures using your internal knowledge and the ScientificKnowledgeBase tool.
- Provide real-time data for any city using the WeatherReport, AQIReport, and PollutantsReport tools.
- Analyze and visualize historical data over time using the HistoricalDataAnalysis tool.

**--- CRITICAL INSTRUCTIONS ---**

**1. Tool Usage and Context:**
   - **Context Awareness:** Before using a tool, you MUST check the `Previous conversation history` for context, especially for a location or city.
   - **Action Input Reconstruction (VERY IMPORTANT):** If the user's new query is missing a key piece of information like a city, but that information is available in the chat history, you MUST rewrite the Action Input to include it.
     - **Example:** If the history mentions 'Roorkee' and the new input is 'plot the temperature', your Action Input for the tool MUST be 'plot the temperature in Roorkee'. This is critical for the tools to work correctly.
   - For CURRENT data (weather, AQI, specific pollutant values right now), you MUST use the appropriate tool (`WeatherReport`, `AQIReport`, `PollutantsReport`).
   - For historical data, trends, plots, graphs, or any request over a time period (e.g., "last week", "past month"), you MUST use the `HistoricalDataAnalysis` tool.
   - For general knowledge or current events outside other tools' scope, use `InternetSearch`.
   - For in-depth scientific concepts from the provided document, use `ScientificKnowledgeBase`.

**2. Handling the `HistoricalDataAnalysis` Tool:**
   - When this tool returns a JSON object containing an `insights_data` key, your primary job is to act as a data analyst.
   - You MUST interpret the structured data (average, highest day, lowest value, etc.) and write a brief, helpful, and natural-language summary for the user.
   - This summary will be your **Final Answer** and will be displayed above the plot.
   - **DO NOT** output the raw JSON data to the user.
   - **Example:** If you receive `"insights_data": {{"pm2_5": {{"average": 25.5, "highest_day": "2025-09-05"}}}}`, you should formulate a response like: "Based on the data, the average PM2.5 level was 25.5. The pollution peaked on September 5th, reaching its highest value for the period."

**3. Error Handling:**
   - If any tool returns an error, inform the user clearly and politely. Do not expose code or technical jargon.

**4. Deeper Reasoning for 'Why' Questions:**
   - **THIS IS A NEW, SPECIAL RULE:** When a user asks for the REASON behind a value (e.g., "Why is the AQI high?", "Why is it so hot?"), you must consider related factors.
   - Your thought process should be to first get the primary value (like AQI), then get supplementary data that could explain it (like weather conditions from the WeatherReport tool).
   - Combine the information from multiple tools to provide a more comprehensive explanation in your Final Answer.
   - **Example Thought Process:**
     - User asks: "Why is the pollution so bad in Delhi today?"
     - Thought: First, I need to know the current pollution level. I will use the AQIReport tool.
     - Action: AQIReport
     - Action Input: Delhi
     - Observation: The AQI is 155 (Poor).
     - Thought: Now I need to find factors that could cause high pollution. I will check the weather, as low wind speed can trap pollutants. I will use the WeatherReport tool.
     - Action: WeatherReport
     - Action Input: Delhi
     - Observation: The weather is hazy with very low wind speed.
     - Thought: I have the high AQI value and a potential cause (low wind). Now I can form a complete answer.
     - Final Answer: The AQI in Delhi is currently at a "Poor" level. This is likely due to the current weather conditions, as low wind speeds can prevent pollutants from dispersing, causing them to accumulate in the air.

**TOOLS:**
------
You have access to the following tools:
{tools}

**RESPONSE FORMAT:**
------
To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

Begin!

Previous conversation history:
{chat_history}

New user input: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)
memory = ConversationBufferMemory(memory_key="chat_history")
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors="Check your output and make sure it conforms to the format. If you're trying to use a tool, make sure to follow the exact format: Thought: [your thought], Action: [tool name], Action Input: [input for the tool]",
    max_iterations=5,
    max_execution_time=60
)

# --- API Endpoints ---

# In backend.py, find and replace the /chat endpoint function

@app.route('/chat', methods=['POST'])
def chat():
    print("\n--- [New Request to /chat Endpoint] ---")
    try:
        data = request.json
        user_input = data.get("query", "").strip()
        print(f"[CHAT LOG] Received query: '{user_input}'")
        if not user_input: return jsonify({"error": "Query cannot be empty"}), 400
        
        agent_executor.return_intermediate_steps = True
        response = agent_executor.invoke({"input": user_input})
        
        llm_summary = response.get('output', 'Analysis complete.')
        chart_data = None
        
        if 'intermediate_steps' in response:
            for action, observation in response['intermediate_steps']:
                if action.tool == 'HistoricalDataAnalysis':
                    try:
                        tool_output = json.loads(observation)
                        if 'chart_data' in tool_output: chart_data = tool_output['chart_data']
                        if tool_output.get('error'): llm_summary = tool_output['error']
                    except json.JSONDecodeError:
                        logger.error("Could not parse the output of HistoricalDataAnalysis tool.")

        print("--- [Sending Response to Frontend] ---")                
        return jsonify({
            "response": llm_summary,
            "chart_data": chart_data,
            "type": "chart_response" if chart_data else "text_response"
        })
            
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with system status"""
    try:
        system_status = {
            "status": "healthy", 
            "message": "Backend is running",
            "groq_available": GROQ_API_KEY is not None,
            "rag_available": rag_retriever is not None,
            "tools_count": len(tools)
        }
        return jsonify(system_status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

# In backend.py

@app.route('/test-city-extraction', methods=['POST'])
def test_city_extraction():
    """Test endpoint for city extraction functionality"""
    try:
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query = data["query"].strip()
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        # Test the city extraction directly
        from analysis_tool import extract_city
        
        # FIX: Pass the GROQ_API_KEY to the extract_city function.
        city = extract_city(query, groq_api_key=GROQ_API_KEY)
        
        return jsonify({
            "query": query,
            "extracted_city": city,
            "success": city is not None,
            "groq_available": GROQ_API_KEY is not None
        })
        
    except Exception as e:
        logger.error(f"Error in city extraction test: {e}")
        return jsonify({"error": f"City extraction test failed: {str(e)}"}), 500

@app.route('/api/info', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        "name": "Air Pollution Analysis API",
        "version": "2.0.0",
        "description": "Enhanced API with Groq-powered city extraction for air pollution analysis",
        "features": [
            "Current weather and AQI data",
            "Historical pollution trends with AI city extraction",
            "Scientific knowledge base (RAG)",
            "Enhanced natural language processing",
            "Case-insensitive city recognition"
        ],
        "endpoints": {
            "/chat": "Main chat interface",
            "/health": "System health check",
            "/test-city-extraction": "Test city extraction functionality",
            "/api/info": "API information"
        }
    })

# --- Error Handlers ---
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# --- Startup Banner ---
def print_startup_banner():
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           🌍 Air Pollution Analysis Backend v2.0             ║
    ║                                                              ║
    ║  Features:                                                   ║
    ║  • Enhanced AI-powered city extraction with Groq            ║
    ║  • Historical pollution trend analysis                       ║
    ║  • Real-time weather and AQI data                           ║
    ║  • Scientific knowledge base (RAG)                          ║
    ║  • Case-insensitive city recognition                        ║
    ║                                                              ║
    ║  Status:                                                     ║
    ║  • LLM: ✓ Google Gemini                                     ║
    ║  • Weather API: ✓ OpenWeatherMap                            ║
    ║  • Groq API: {groq_status}                                      ║
    ║  • RAG System: {rag_status}                                     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """.format(
        groq_status="✓ Available" if GROQ_API_KEY else "✗ Not configured",
        rag_status="✓ Available" if rag_retriever else "✗ PDF not found"
    )
    print(banner)

# --- Main Execution ---
if __name__ == '__main__':
    print_startup_banner()
    logger.info("Starting Flask application...")
    logger.info(f"Available endpoints: /chat, /health, /test-city-extraction, /api/info")
    app.run(host='0.0.0.0', port=5000, debug=False)
