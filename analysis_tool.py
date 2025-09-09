import requests
import pandas as pd
import calendar
from datetime import datetime, timedelta
import dateparser
import json
import re
from groq import Groq

# --- Enhanced City Extraction (Your original function) ---
def extract_city(query: str, groq_api_key: str | None) -> str | None:
    if groq_api_key:
        try:
            print("-> Attempting city extraction with Groq API...")
            groq_client = Groq(api_key=groq_api_key)
            prompt = f"""
            Extract the city, state, country, or location name from the user query below.
            Return ONLY the location name, nothing else. If no clear location is found, return "NONE".
            Query: "{query}"
            Location:"""
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0, max_tokens=20
            )
            location = chat_completion.choices[0].message.content.strip()
            if location.upper() in ["NONE", ""] or len(location) < 2:                 
                print("   - Groq found no valid city, falling back to regex.")
            else:
                print(f"   + Groq extracted city: '{location}'")
            return location.replace("Location:", "").strip()
        except Exception as e:
            print(f"   - Groq API error: {e}. Falling back to regex.")
    
    # Fallback regex (Your original)
    print("-> Attempting city extraction with fallback regex...")
    patterns = [
        r'\b(?:in|from|at|near)\s+([a-zA-Z][a-zA-Z\s]{1,30}?)(?:\s+|$)',
        r'^([a-zA-Z][a-zA-Z\s]{1,30}?)\s+(?:pollution|weather|data|trend)'
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            city = match.group(1).strip()
            # --- LOGGING ---
            print(f"   + Regex extracted city: '{city}'")
            return city
    # --- LOGGING ---
    print("   - Regex found no city.")
    return None

# --- Data Fetching Functions (Your original functions) ---
def get_coordinates(city, api_key):
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data: raise ValueError(f"City '{city}' not found.")
    return data[0]["lat"], data[0]["lon"]

def get_pollutant_history(city, start_date, end_date, api_key):
    lat, lon = get_coordinates(city, api_key)
    start_unix = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_unix = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_unix}&end={end_unix}&appid={api_key}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if "list" not in data or not data["list"]:
        raise ValueError(f"No pollution data available for {city} in the specified period.")
    rows = []
    for entry in data["list"]:
        row_data = entry.get("components", {})
        row_data["datetime"] = datetime.utcfromtimestamp(entry["dt"])
        rows.append(row_data)
    if not rows: raise ValueError(f"No valid pollution data found for {city}")
    return pd.DataFrame(rows).set_index("datetime")

def get_weather_history(city, start_date, end_date, variables):
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    geo_resp = requests.get(geo_url, timeout=10)
    geo_resp.raise_for_status()
    geo_data = geo_resp.json()
    if "results" not in geo_data or not geo_data["results"]:
        raise ValueError(f"City '{city}' not found via Open-Meteo geocoding.")
    lat, lon = geo_data["results"][0]["latitude"], geo_data["results"][0]["longitude"]
    var_str = ",".join(variables)
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly={var_str}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if "hourly" not in data:
        raise ValueError(f"Could not retrieve weather data: {data.get('reason', 'Unknown error')}")
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")

# --- Core Logic Functions (Your original functions) ---
# In analysis_tool.py, replace the old function with this new one

def parse_time_range(query: str, groq_api_key: str | None) -> tuple[str, str]:
    """
    Uses Groq to parse the time range from a query, with a robust manual parser as a fallback.
    """
    if groq_api_key:
        try:
            print("-> Attempting time range extraction with Groq API...")
            groq_client = Groq(api_key=groq_api_key)
            current_date = datetime.now().strftime('%Y-%m-%d')
            prompt = f"""
            You are an expert date parsing assistant. Your task is to analyze a user's query and determine the start and end dates for a data request.
            The current date is: {current_date}.
            
            - Handle relative dates like "yesterday", "last week", "past 3 months".
            - Handle specific dates like "August 2025", "in 2024", or "on Sep 5th".
            - If a single day is mentioned, the start_date and end_date should be the same.
            - If no date is mentioned, use the last 7 days from the current date as the default.
            
            Return ONLY a valid JSON object with two keys: "start_date" and "end_date".
            The date format MUST be "YYYY-MM-DD". Do not add any other text, explanations, or markdown.

            Query: "{query}"
            JSON:"""
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0, max_tokens=100
            )
            response_text = chat_completion.choices[0].message.content.strip()
            
            # Validate the JSON response from the LLM
            date_json = json.loads(response_text)
            start_date_str = date_json.get("start_date")
            end_date_str = date_json.get("end_date")

            # Final validation of format and content
            if start_date_str and end_date_str:
                datetime.strptime(start_date_str, "%Y-%m-%d") # Check format
                datetime.strptime(end_date_str, "%Y-%m-%d")   # Check format
                print(f"   + Groq extracted dates: {start_date_str} to {end_date_str}")
                return start_date_str, end_date_str
            else:
                 raise ValueError("LLM response missing required date keys.")

        except Exception as e:
            print(f"   - Groq date parsing failed: {e}. Falling back to manual method.")
            return _parse_time_range_manually(query)
    
    # Default to manual parsing if no Groq key is provided
    print("-> No Groq key. Using manual time range extraction...")
    return _parse_time_range_manually(query)

def _parse_time_range_manually(query):
    """
    Handles all time range parsing cases manually using regex as a fallback.
    """
    now = datetime.now()
    query_lower = query.lower()

    # Case 1: Specific Month and Year (e.g., "August 2025", "jan 2024")
    month_year_match = re.search(
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(20[0-2][0-9])\b',
        query_lower
    )
    if month_year_match:
        month_name, year = month_year_match.group(1), int(month_year_match.group(2))
        try:
            month_num = datetime.strptime(month_name, "%B" if len(month_name) > 3 else "%b").month
            _, last_day = calendar.monthrange(year, month_num)
            start_date, end_date = datetime(year, month_num, 1), datetime(year, month_num, last_day)
            return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        except ValueError: pass

    # Case 2: Relative with numbers (e.g., "past 2 months", "last 90 days")
    relative_match = re.search(r'\b(last|past)\s+(\d+)\s+(day|week|month)s?\b', query_lower)
    if relative_match:
        value, unit = int(relative_match.group(2)), relative_match.group(3)
        end_date = now
        if unit == 'day': start_date = now - timedelta(days=value)
        elif unit == 'week': start_date = now - timedelta(weeks=value)
        elif unit == 'month': start_date = now - timedelta(days=int(value * 30.4))
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    # Case 3: Simple relative (e.g., "last month", "past year")
    if any(s in query_lower for s in ["last week", "past week"]):
        start_date, end_date = now - timedelta(days=7), now
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    if any(s in query_lower for s in ["last month", "past month"]):
        start_date, end_date = now - timedelta(days=30), now
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    if any(s in query_lower for s in ["last year", "past year"]):
        start_date, end_date = now - timedelta(days=365), now
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    # Case 4: Specific Year only (e.g., "in 2024", "for 2023")
    year_match = re.search(r'\b(20[0-2][0-9])\b', query_lower)
    if year_match:
        year = int(year_match.group(1))
        start_date, end_date = datetime(year, 1, 1), datetime(year, 12, 31)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    # Case 5: Default fallback
    start_date, end_date = now - timedelta(days=7), now
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

def parse_metrics(query):
    pollutant_map = {
        "pm2_5": ["pm2.5", "pm 2.5", "fine particles"],
        "pm10": ["pm10", "coarse particles"],
        "no2": ["no2", "nitrogen dioxide"],
        "so2": ["so2", "sulfur dioxide"],
        "co": ["co", "carbon monoxide"],
        "o3": ["o3", "ozone", "o-zone"],
        "aqi": ["aqi", "air quality", "air quality index"]
    }
    weather_map = {
        "temperature_2m": ["temperature", "temp"],
        "relative_humidity_2m": ["humidity"]
    }

    query_lower = query.lower()
    found_pollutants = []
    found_weather = []

    # NEW: Check for keywords to plot all pollutants
    if any(keyword in query_lower for keyword in ["all pollutants", "various pollutants", "pollutants"]):
        found_pollutants = list(pollutant_map.keys())
    else:
        # Original logic to find specific pollutants
        for api_value, common_names in pollutant_map.items():
            if any(name in query_lower for name in common_names):
                found_pollutants.append(api_value)

    # Logic for finding weather variables remains the same
    for api_value, common_names in weather_map.items():
        if any(name in query_lower for name in common_names):
            found_weather.append(api_value)

    if not found_pollutants and not found_weather:
        found_pollutants = ["pm2_5"]

    return sorted(list(set(found_pollutants))), sorted(list(set(found_weather)))

def choose_frequency(start_date, end_date):
    days = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days
    if days <= 30: return "D"
    elif days <= 366: return "W"
    else: return "M"

def prepare_chart_data(df, city, start_date, end_date):
    labels = [dt.strftime("%Y-%m-%d") for dt in df.index]
    colors = ["rgb(75, 192, 192)", "rgb(255, 99, 132)", "rgb(54, 162, 235)", "rgb(255, 205, 86)"]
    datasets = []
    for i, col in enumerate(df.columns):
        if pd.api.types.is_numeric_dtype(df[col]):
            datasets.append({
                "label": col.replace('_', ' ').title(),
                "data": [round(val, 2) if pd.notna(val) else None for val in df[col]],
                "borderColor": colors[i % len(colors)],
                "backgroundColor": colors[i % len(colors)].replace("rgb", "rgba").replace(")", ", 0.2)")
            })
    return {"type": "line", "title": f"Historical Analysis for {city} ({start_date} to {end_date})", "labels": labels, "datasets": datasets}

# --- REFACTORED: Function to generate structured data ---
def generate_programmatic_insights(df):
    insights_data = {}
    date_format = "%Y-%m-%d"
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isna().all():
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                insights_data[col] = {
                    "average": round(valid_data.mean(), 2),
                    "highest_value": round(valid_data.max(), 2),
                    "highest_day": valid_data.idxmax().strftime(date_format),
                    "lowest_value": round(valid_data.min(), 2),
                    "lowest_day": valid_data.idxmin().strftime(date_format)
                }
    return insights_data

# --- REFACTORED: Main tool function updated for the new workflow ---
def historical_analysis_tool(query: str, api_key: str, groq_api_key: str) -> str:
    print("\n--- [Historical Analysis Tool Triggered] ---")
    try:
        city = extract_city(query, groq_api_key=groq_api_key)
        if not city:
            return json.dumps({"error": "Could not identify a city in your request."})

        start_date, end_date = parse_time_range(query, groq_api_key=groq_api_key)
        pollutants, weather_vars = parse_metrics(query)
        freq = choose_frequency(start_date, end_date)
        
        print(f"-> Parsed Parameters: City='{city}', Start='{start_date}', End='{end_date}', Pollutants={pollutants}, Weather={weather_vars}")

        df_pollutants = pd.DataFrame()
        if pollutants:
            print("-> Fetching pollution data from OpenWeatherMap...")
            try: 
                df_pollutants = get_pollutant_history(city, start_date, end_date, api_key)
                print("   + Pollution data fetched successfully.")
            except Exception as e: 
                print(f"   - Pollution data error: {e}")

        df_weather = pd.DataFrame()
        if weather_vars:
            print("-> Fetching weather data from Open-Meteo...")
            try: 
                df_weather = get_weather_history(city, start_date, end_date, weather_vars)
                print("   + Weather data fetched successfully.")
            except Exception as e: 
                print(f"   - Weather data error: {e}")

        if df_pollutants.empty and df_weather.empty:
            return json.dumps({"error": f"Could not fetch any data for {city}."})
        
        print("-> Processing and combining data...")
        df_combined = pd.merge(df_pollutants, df_weather, left_index=True, right_index=True, how='outer') if not df_pollutants.empty and not df_weather.empty else (df_pollutants if not df_pollutants.empty else df_weather)
        
        all_vars = pollutants + weather_vars
        selected_cols = [col for col in all_vars if col in df_combined.columns]
        if not selected_cols:
             return json.dumps({"error": "Requested metrics were not available."})

        if freq in ['W', 'M']:
            # For weekly/monthly data, label with the start of the period
            df_final = df_combined[selected_cols].resample(freq, label='left').mean().dropna(how='all')
        else:
            # Daily data doesn't need a label change
            df_final = df_combined[selected_cols].resample(freq).mean().dropna(how='all')
            
        if df_final.empty:
             return json.dumps({"error": "No valid data points available after processing."})
        print(f"   + Data processed. Final shape: {df_final.shape}")
        
        print("-> Generating insights and chart data...")
        insights_data = generate_programmatic_insights(df_final)
        chart_data = prepare_chart_data(df_final, city, start_date, end_date)
        print("   + Insights and chart data created successfully.")
        
        print("--- [Historical Analysis Tool Finished] ---\n")
        return json.dumps({
            "insights_data": insights_data,
            "chart_data": chart_data,
        })

    except Exception as e:
        print(f"!!! CRITICAL ERROR in analysis tool: {e} !!!")
        return json.dumps({"error": f"An unexpected error occurred in the analysis tool: {str(e)}"})