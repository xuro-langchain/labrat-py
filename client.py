from langsmith import Client
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

client = Client()
response = client.read_project(project_name="agents-from-scratch", include_stats=True)
print(response.json(indent=2))