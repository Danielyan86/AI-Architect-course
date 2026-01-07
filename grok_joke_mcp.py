import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url='https://space.ai-builders.com/backend/v1',
    api_key=os.getenv('AI_BUILDER_TOKEN')
)

try:
    print("正在通过 AI Builders MCP 让 Grok 讲笑话...\n")

    completion = client.chat.completions.create(
        model='grok-4-fast',
        messages=[
            {'role': 'user', 'content': '请讲一个幽默搞笑的笑话，不要冷笑话，要让人真正能笑出来的那种！'}
        ],
        temperature=0.9
    )

    joke = completion.choices[0].message.content

    print("=== Grok 讲的笑话 ===\n")
    print(joke)
    print("\n=====================")

except Exception as e:
    print(f"出错了: {e}")
