from openai import OpenAI
import os

# client = OpenAI(
#     base_url=os.environ['LLM_API_ENDPOINT'],
#     api_key=os.environ['OPENROUTER_API_KEY'],
# )

# Make sure to set OPENAI_API_KEY environment variable
client = OpenAI()

# noinspection PyTypeChecker
stream = client.chat.completions.create(
    model="gpt-5.4-nano",
    messages=[
        {
          "role": "user",
          "content": "Who are you?"
        }
    ],
    stream=True
)

# print(completion.choices[0].message.content)

# Since we are streaming the model responses, we will have
# to accumulate all tool calls first before processing
final_tool_calls = {}

for event in stream:
    print(event)

    # delta_content = event.choices[0].delta.content
    # if delta_content:
    #     print(delta_content, end='')