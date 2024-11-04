from crewai import Agent, Task, Crew, LLM
import os

# Create the LLM instance using CrewAI's LLM class
llm = LLM(
    model="ollama/llama3.2:3b",  # Using llama2 as the model name
    base_url="http://localhost:11434",  # Ollama's default URL
    api_key="not-needed"  # Ollama doesn't need an API key
)

# Create an agent
researcher = Agent(
    role='Marine Biology Researcher',
    goal='Research and provide detailed information about marine creatures',
    backstory="""You are a marine biology expert with extensive knowledge about 
              sea creatures. You're passionate about sharing accurate and 
              detailed information about marine life.""",
    llm=llm,
    verbose=True  # Note: verbose should be boolean, not int
)

# Create a task
research_task = Task(
    description="""Provide a comprehensive overview of the box jellyfish. Include 
                information about their anatomy, habitat, behavior, and any unique 
                characteristics.""",
    expected_output="""A detailed report about box jellyfish, covering their 
                    physical characteristics, where they live, how they behave, 
                    and what makes them special.""",
    agent=researcher
)

# Create a crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True  # Fixed: using boolean instead of int
)

# Run the crew
result = crew.kickoff()
print("\nFinal Result:")
print(result)