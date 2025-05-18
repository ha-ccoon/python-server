import json
import os

from openai import OpenAI
from dotenv import load_dotenv

from data_type import InBodyData

load_dotenv() 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_data(previous_data: InBodyData, current_data: InBodyData):
  print('previous_data', previous_data)
  print('current_data', current_data)
  
  # 데이터를 JSON 문자열로 변환
  prev_data_str = json.dumps(previous_data, indent=2, ensure_ascii=False)
  curr_data_str = json.dumps(current_data, indent=2, ensure_ascii=False)

  analysis = client.responses.create(
    model="gpt-4.1-nano",
    input=f"Previous Data: {prev_data_str}, Current Data: {curr_data_str}",
    instructions=analysis_instruction(prev_data_str, curr_data_str)
  )
  
  report = client.responses.create(
    model="gpt-4.1-nano",
    input=analysis.output_text,
    instructions=translate_instruction()
  )
  
  return report.output_text
  

def analysis_instruction(previous: InBodyData, current: InBodyData) -> str:
    prompt_template = """
    # Role and Objective
    - You are a fitness data analyst specializing in body composition analysis. 
    - You will receive data from two InBody assessments: the previous assessment and the current assessment. 
    - Your task is to compare the data and generate a detailed analysis report.
    
    # Instructions
    - Analyze the data to identify significant changes in body composition.
    - Provide a structured report that includes:
      - Overall comparison of the four key metrics (Weight, Muscle Mass, Body Fat Mass, Body Fat Percentage).
      - Insights on muscle gain/loss, fat reduction/increase, and weight changes.
      - Personalized recommendations based on the observed trends.
    
    # Data
    - Previous Data:
    {previous}

    - Recent Data:
    {current}

    # Output Format
    **Respond in a structured report in the following format:**
    
    # Inbody Analysis Report
    
    ## Overview of Changes Between May 2023 and May 2025
    | Metric | 2023-05-15	| 2025-05-18	| Change | Percentage Change |
    |--------|------------|-------------|--------|-------------------|
    
    ---
    
    ## Key Insights
    - Each section should consist of a headline (key observation) and a detailed explanation (what the data indicates and its significance).
    
    ### 1. Weight & Body Composition Changes
    
    ### 2. Changes in Body Fat Percentage
    
    ### 3. Additional Observations
    
    ---
    
    ## Interpretation & Recommendations
    - Each section should consist of a headline (key interpretation) and a detailed explanation (what the data indicates and why it matters), followed by a recommendation (specific action to address the observation).
    
    ---
    
    ## Final Remarks
    - Summarize the overall findings in a narrative format, emphasizing key takeaways and next steps.
    
    ---
    
    **Note: Always consult with a healthcare or fitness professional to tailor personalized strategies suited to individual health status and goals.**
    """
    
    prompt = prompt_template.format(previous=previous, current=current)

    return prompt

def translate_instruction():
  prompt_template = """
  - You are a translation expert specializing in English to Korean. 
  - Transform the provided report into a well-structured, professional Korean report, maintaining the original format and structure while ensuring clarity and natural flow.
  """
  
  return prompt_template