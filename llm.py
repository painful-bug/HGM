# This file is adapted from https://github.com/jennyzzt/dgm.

# Code adapted from https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/llm.py.
import json
import os
import re
import time

import anthropic
import backoff
import openai
from dotenv import load_dotenv

load_dotenv()

NIM_MODEL = "nim/nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_LLM = os.getenv("HGM_MODEL", os.getenv("DGM_MODEL", NIM_MODEL))
OLLAMA_FALLBACK = "ollama/qwen3.5:4b"
CHAT_PROVIDERS = ("nim/", "ollama/")

MAX_OUTPUT_TOKENS = 4096
AVAILABLE_LLMS = [
    NIM_MODEL,
    OLLAMA_FALLBACK,
    "gpt-5",
    "o4-mini",
    "o3",
    "deepseek/deepseek-chat-v3.1",
    "anthropic/claude-sonnet-4",
]


def create_client(model: str):
    if model.startswith("nim/"):
        if not os.getenv("NVIDIA_NIM_API_KEY"):
            print("[LLM fallback] NVIDIA_NIM_API_KEY is missing; trying Ollama.")
            return create_client(OLLAMA_FALLBACK)
        return openai.OpenAI(api_key=os.environ["NVIDIA_NIM_API_KEY"],
                             base_url="https://integrate.api.nvidia.com/v1",
                             max_retries=0, timeout=60), model
    elif model.startswith("ollama/"):
        return openai.OpenAI(api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
                             base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                             max_retries=0, timeout=180), model
    elif "gpt" in model or model.startswith("o"):
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif "vllm" in model.lower():
        print(f"Using vllm API with model {model}.")
        return (
            openai.OpenAI(base_url=f"http://{model[11:]}:8000/v1", api_key="dummy"),
            model,
        )
    else:
        print(f"Using OpenRouter API with model {model}.")
        return (
            openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OpenRouter_API_KEY"),
            ),
            model,
        )


def chat_completion_with_fallback(client, model, logging=print, **kwargs):
    """Retry a failed request once per provider, preserving its messages and tools."""
    models = [model]
    if model.startswith("nim/"):
        models += [OLLAMA_FALLBACK]
    attempted, failures = set(), []
    last_error = None
    for candidate in models:
        next_client, selected = client, candidate
        try:
            if candidate != model:
                next_client, selected = create_client(candidate)
            if selected in attempted:
                continue
            attempted.add(selected)
            request = dict(kwargs)
            if "max_completion_tokens" in request:
                request["max_tokens"] = request.pop("max_completion_tokens")
            if selected.startswith("nim/"):
                # NVIDIA recommends nonempty content for coding agents.
                request["extra_body"] = {"chat_template_kwargs": {
                    "force_nonempty_content": True}}
                request["reasoning_effort"] = "medium"
            started = time.monotonic()
            logging(f"[LLM] request: {selected}; messages={len(request.get('messages', []))}")
            response = next_client.chat.completions.create(model=selected.split("/", 1)[1], **request)
            usage = getattr(response, "usage", None)
            finish = getattr(response.choices[0], "finish_reason", None) if response.choices else None
            logging(f"[LLM] response: {selected}; seconds={time.monotonic() - started:.1f}; "
                    f"finish={finish}; prompt_tokens={getattr(usage, 'prompt_tokens', None)}; "
                    f"completion_tokens={getattr(usage, 'completion_tokens', None)}")
            if not response.choices:
                raise ValueError("Provider returned no choices")
            message = response.choices[0].message
            if not message.content and not getattr(message, "tool_calls", None):
                raise ValueError("Provider returned neither text nor tool calls")
            logging(f"[LLM] success: {selected}")
            return response
        except (openai.APIError, ValueError) as error:
            last_error = error
            failure = f"{selected}: {type(error).__name__} (status {getattr(error, 'status_code', 'n/a')})"
            failures.append(failure)
            logging(f"[LLM fallback] {failure}")
        finally:
            if next_client is not client:
                next_client.close()
    raise RuntimeError("All LLM providers failed: " + "; ".join(failures)) from last_error


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.APIStatusError,
    ),
    max_time=120,
)
def get_json_response_from_llm(
    msg,
    client,
    model,
    system_message,
):
    new_msg_history = [{"role": "user", "content": msg}]
    complete = chat_completion_with_fallback if model.startswith(CHAT_PROVIDERS) else None
    kwargs = dict(
        messages=[
            {"role": "system", "content": system_message},
            *new_msg_history,
        ],
        n=1,
        stop=None,
        seed=0,
        response_format={
            "type": "json_object",
        },
    )
    response = complete(client, model, **kwargs) if complete else client.chat.completions.create(model=model, **kwargs)
    content = response.choices[0].message.content
    import json

    content_json = json.loads(content)
    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

    return content_json, new_msg_history


def get_response_from_llm(
    msg,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
):
    if msg_history is None:
        msg_history = []

    if model.startswith(CHAT_PROVIDERS):
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = chat_completion_with_fallback(
            client, model, messages=[{"role": "system", "content": system_message}, *new_msg_history],
            temperature=temperature, max_tokens=MAX_OUTPUT_TOKENS, n=1, stop=None)
        content = response.choices[0].message.content
        new_msg_history += [{"role": "assistant", "content": content}]
    elif model.startswith("o"):
        new_msg_history = msg_history + [
            {"role": "user", "content": system_message + msg}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=new_msg_history,
            temperature=1,
            n=1,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif "gpt" in model.lower():
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            n=1,
            stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=client.models.list().data[0].id,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        print(f'User: {new_msg_history[-2]["content"]}')
        print(f'Assistant: {new_msg_history[-1]["content"]}')
        print("*" * 21 + " LLM END " + "*" * 21)
        print()
    return content, new_msg_history


def extract_json_between_markers(llm_output):
    inside_json_block = False
    json_lines = []

    # Split the output into lines and iterate
    for line in llm_output.split("\n"):
        striped_line = line.strip()

        # Check for start of JSON code block
        if striped_line.startswith("```json"):
            inside_json_block = True
            continue

        # Check for end of code block
        if inside_json_block and striped_line.startswith("```"):
            # We've reached the closing triple backticks.
            inside_json_block = False
            break

        # If we're inside the JSON block, collect the lines
        if inside_json_block:
            json_lines.append(line)

    # If we never found a JSON code block, fallback to any JSON-like content
    if not json_lines:
        # Fallback: Try a regex that finds any JSON-like object in the text
        fallback_pattern = r"\{.*?\}"
        matches = re.findall(fallback_pattern, llm_output, re.DOTALL)
        for candidate in matches:
            candidate = candidate.strip()
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Attempt to clean control characters and re-try
                    candidate_clean = re.sub(r"[\x00-\x1F\x7F]", "", candidate)
                    try:
                        return json.loads(candidate_clean)
                    except json.JSONDecodeError:
                        continue
        return None

    # Join all lines in the JSON block into a single string
    json_string = "\n".join(json_lines).strip()

    # Try to parse the collected JSON lines
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        # Attempt to remove invalid control characters and re-parse
        json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
        try:
            return json.loads(json_string_clean)
        except json.JSONDecodeError:
            return None
