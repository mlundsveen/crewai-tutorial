import os
from crewai import Agent, Task, Crew, Process
from pdf_analysis_tools import PDFDirectoryLoaderTool, PDFAnalysisTool

class PDFAnalysisCrew:
    def __init__(self, pdf_directory: str):
        self.pdf_directory = pdf_directory
        self.loader_tool = PDFDirectoryLoaderTool()
        self.analysis_tool = PDFAnalysisTool()

    def create_agents(self):
        # PDF Reader Agent
        self.reader_agent = Agent(
            role='PDF Reader and Organizer',
            goal='Thoroughly read and organize content from PDF files about Generative AI',
            backstory="""You are an expert at processing and organizing academic and technical 
            documents. With your advanced understanding of Generative AI, you can efficiently 
            extract and structure relevant information from multiple sources.""",
            tools=[self.loader_tool],
            verbose=True
        )

        # Analysis Agent
        self.analyst_agent = Agent(
            role='Research Analyst',
            goal='Analyze and identify key themes, methodologies, and findings across the PDF documents',
            backstory="""You are a seasoned research analyst with expertise in Generative AI. 
            Your strength lies in identifying patterns, comparing different approaches, and 
            synthesizing complex technical information into coherent insights.""",
            tools=[self.analysis_tool],
            verbose=True
        )

        # Literature Review Writer
        self.writer_agent = Agent(
            role='Technical Writer',
            goal='Create a comprehensive literature review that synthesizes the findings from all PDFs',
            backstory="""You are an accomplished technical writer specializing in AI and ML topics. 
            You excel at creating well-structured, academically rigorous literature reviews that 
            effectively communicate complex technical concepts.""",
            verbose=True
        )

    def create_tasks(self, pdf_contents=None):
        # Task 1: Read PDFs
        read_task = Task(
            description=f"""Load and process all PDF files from the directory: {self.pdf_directory}.
            Organize the content and create a brief overview of each document.""",
            agent=self.reader_agent,
            expected_output="""A structured list of all processed PDFs with their key topics 
            and brief summaries."""
        )

        # Task 2: Analyze Content
        analysis_task = Task(
            description="""Analyze the PDF contents to identify:
            1. Major themes and topics in Generative AI
            2. Key methodologies and approaches
            3. Important findings and conclusions
            4. Research gaps and future directions
            Use the analysis tool to perform semantic search when needed.""",
            agent=self.analyst_agent,
            expected_output="""A detailed analysis report covering the main themes, methodologies, 
            findings, and research directions identified across all documents."""
        )

        # Task 3: Write Literature Review
        writing_task = Task(
            description="""Create a comprehensive literature review that:
            1. Introduces the field of Generative AI
            2. Synthesizes the major themes and findings
            3. Discusses methodological approaches
            4. Identifies research gaps and future directions
            5. Concludes with the current state of the field
            Format the review in markdown with proper sections, citations, and references.""",
            agent=self.writer_agent,
            expected_output="""A well-structured markdown literature review document that 
            comprehensively covers the analyzed PDFs.""",
            output_file="literature_review.md"
        )

        return [read_task, analysis_task, writing_task]

    def run(self):
        # Create agents
        self.create_agents()
        
        # Create tasks
        tasks = self.create_tasks()
        
        # Create crew
        crew = Crew(
            agents=[self.reader_agent, self.analyst_agent, self.writer_agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            memory=True
        )
        
        # Execute the crew
        result = crew.kickoff()
        return result

# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Initialize and run the crew
    pdf_crew = PDFAnalysisCrew("path/to/your/pdf/directory")
    result = pdf_crew.run()
    print("Literature review has been generated!") 