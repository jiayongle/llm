import os
import json
from openai import OpenAI

client = OpenAI(
  api_key="your_openai_key",
)

student_custom_functions = [
    {
        'name': 'extract_student_info',
        'description': 'Get the student information from the body of the input text',
        'parameters': {
            'type': 'object',
            'properties': {
                'name': {
                    'type': 'string',
                    'description': 'Name of the person'
                },
                'major': {
                    'type': 'string',
                    'description': 'Major subject.'
                },
                'school': {
                    'type': 'string',
                    'description': 'The university name.'
                },
                'grades': {
                    'type': 'integer',
                    'description': 'GPA of the student.'
                },
                'club': {
                    'type': 'string',
                    'description': 'School club for extracurricular activities. '
                }
                
            }
        }
    }
]

student_1_description = "David Nguyen is a sophomore majoring in computer science at Stanford University. He is Asian American and has a 3.8 GPA. David is known for his programming skills and is an active member of the university's Robotics Club. He hopes to pursue a career in artificial intelligence after graduating."

student_2_description="Ravi Patel is a sophomore majoring in computer science at the University of Michigan. He is South Asian Indian American and has a 3.7 GPA. Ravi is an active member of the university's Chess Club and the South Asian Student Association. He hopes to pursue a career in software engineering after graduating."

student_description = [student_1_description,student_2_description]
for i in student_description:
    response = client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages = [{'role': 'user', 'content': i}],
        functions = student_custom_functions,
        function_call = 'auto'
    )

    # Loading the response as a JSON object
    json_response = json.loads(response.choices[0].message.function_call.arguments)
    print(json_response)
