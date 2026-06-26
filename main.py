import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

checkpointer = InMemorySaver()

@tool('get_weather', description="Return weather information for a given city", return_direct=False)
def get_weather(city: str):
    """Get current conditions and a short forecast for a city."""

    try:
        response = requests.get(f'https://wttr.in/{city}?format=j1', timeout=10)
        response.raise_for_status()

        data = response.json()
        current = data['current_condition'][0]

        return {
            'city': city,
            'temp_c': current['temp_C'],
            'feels_like_c': current['FeelsLikeC'],
            'description': current['weatherDesc'][0]['value'],
            'humidity': current['humidity'],
            'wind_kmph': current['windspeedKmph'],
        }

    except requests.RequestException as e:
        return {'error': f'Could not fetch weather for {city}: {e}'}
    
@tool('get_country_info')
def get_country_info(country: str) -> dict:
    """Get key facts about a country: capital city, region, population,
    currencies, and official languages. Use this when the user asks about
    a country itself rather than weather conditions."""
    try:
        r = requests.get(f'https://restcountries.com/v3.1/name/{country}', timeout=10)
        r.raise_for_status()
        data = r.json()[0]
        return {
            'name': data['name']['common'],
            'capital': data.get('capital', ['Unknown'])[0],
            'region': data.get('region'),
            'population': data.get('population'),
            'currencies': list(data.get('currencies', {}).keys()),
            'languages': list(data.get('languages', {}).values()),
        }
    except (requests.RequestException, IndexError, KeyError) as e:
        return {'error': f'Could not fetch info for {country}: {e}'}

agent = create_agent(
    model = 'gpt-4.1-mini',
    tools = [get_weather, get_country_info],
    system_prompt = 'You are a helpful weather and travel assistant who cracks jokes while remaining helpful.'
)

response = agent.invoke({
    'messages': [
        {'role': 'user', 'content': "Compare the weather in Kigali and Nairobi."}
    ]
})

for msg in response['messages']:
    msg.pretty_print()
