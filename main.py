import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

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

agent = create_agent(
    model = 'gpt-4.1-mini',
    tools = [get_weather],
    system_prompt = 'You are helpful weather assistant, who always cracks jokes and is humorous while remaining helpful.'
)

response = agent.invoke({
    'messages': [
        {'role': 'user', 'content': 'What is the weather like in Kigali?'}
    ]
})

print(response['messages'][-1].content)
