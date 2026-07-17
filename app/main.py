import argparse
import json
import os
import sys

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read and return the contents of the file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read",
                    }
                },
                "required": ["file_path"],
            },
        },
    }
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file",
            "parameters": {
            "type": "object",
            "required": ["file_path", "content"],
            "properties": {
                "file_path": {
                "type": "string",
                "description": "The path of the file to write to"
                },
                "content": {
                "type": "string",
                "description": "The content to write to the file"
                }
            }
            }
        }
    }
]


def execute_tool(name, arguments):
    if name == "Read":
        with open(arguments["file_path"], "r") as f:
            return f.read()
    raise RuntimeError(f"unknown tool: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    messages = [{"role": "user", "content": args.p}]

    max_iteration = 10
    iteration = 0

    while True:
        iteration += 1
        if iteration > max_iteration:
            raise RuntimeError("agent loop hit iteration cap")

        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=TOOLS,
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        message = chat.choices[0].message

        messages.append(message.model_dump(exclude_none=True))

        if message.tool_calls:
            for tc in message.tool_calls:
                name = tc.function.name
                arguments = json.loads(tc.function.arguments)
                result = execute_tool(name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
            continue
        else:
            print(message.content)
            break


if __name__ == "__main__":
    main()
