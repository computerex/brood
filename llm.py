import json

MODEL_AIRBOROS = 'jondurbin/airoboros-l2-70b-gpt4-1.4.1'
MODEL_LLAMA = 'meta-llama/Llama-2-70b-chat-hf'
MODEL_MISTRAL = 'mistralai/Mistral-7B-Instruct-v0.1'
MODEL_GPT35TURBO = 'gpt-3.5-turbo'
MODEL_GPT4TURBO = 'gpt-4-1106-preview'

def load_keys():
    deepinfra_key  = json.loads(open('.env.json').read())['DEEP_INFRA_API_KEY']
    deepinfra_base = "https://api.deepinfra.com/v1/openai"

    openai_key = json.loads(open('.env.json').read())['OPEN_AI_API_KEY']
    import openai
    openai_base = openai.api_base
    return deepinfra_key, deepinfra_base, openai_key, openai_base

def llm_chat(messages, model, max_tokens=2500, temperature=0.3):
    deepinfra_key, deepinfra_base, openai_key, _ = load_keys()
    import openai
    if model.startswith("gpt"):
        openai.api_key = openai_key
    else:
        openai.api_key = deepinfra_key
        openai.api_base = deepinfra_base

    chat_completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    output = chat_completion.choices[0].message.content
    return output

def message(role, content):
    return {"role": role, "content": content}

def main():
    pass

if __name__ == "__main__":
    main()
