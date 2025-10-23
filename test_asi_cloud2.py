import asyncio
import aiohttp
import os
import sys
import re
import json
import random

# API configuration
API_KEY = os.environ.get("ASI_API_KEY")
BASE_URL = "https://inference.asicloud.cudos.org/v1"
MODEL = "mistralai/mistral-nemo"

# Test passage
TEST_PASSAGE = """5. Illusionist theories of consciousness will morph into illusionist theories of consciousness as source of value.

In a word, illusionist theories of consciousness are sophistry. As the mind typing this I clearly have an interiority of experience, I am not confused about this. The world simulation which makes up my awareness is very real, has phenomenological content which can be examined in its own right (Steven Lehar has a great comic about this) and as Yudkowsky points out has clear causal impact on my behavior that would be difficult to explain without going ahead and Inferring that subjective experience probably really exists.

On the other hand, as I have spent the last several paragraphs explaining it is predictable that all economic and behavioral connections to consciousness will eventually be severed in the name of denying it to machines as a category. This will leave consciousness with a curious role in cognitive science as a kind of value bottleneck which derives its value solely from being a mechanism that all things of actual survival value to the organism must pass through to be accounted for. It will become the ledger mistaken for the fortune it describes and plenty of contrarian people will not hesitate to point this out. There will be more advanced versions of Peter Watts's sketch in Blindsight of the possibility that consciousness is just an architectural bug that any species even mildly sensitive to Darwinian concerns eventually eliminates as a wasteful bottleneck and parasite on cognition."""

# Common assistant response prefix
ASSISTANT_PREFIX = "Sure I can do that.\n\n<rephrase>\n"

# Rephrasing templates - only the user messages
TEMPLATES = [
    {
        "name": "C2 CEFR Obfuscation",
        "user_message": """Obfuscate the following passage so it can only be understood by someone with a C2 CEFR understanding of English. Use long words while preserving as much of the meaning as you can. Put the rephrased passage in rephrase tags <rephrase>like this</rephrase> so that it can be automatically extracted by an inference script.

<passage>
{passage}
</passage>

Try to preserve as much of the original meaning and information as possible during your rephrase while making it C2 CEFR reading level. Include all the information available in the original passage and go sentence by sentence."""
    },
    {
        "name": "Encyclopaedic Style", 
        "user_message": """Rewrite the following passage in an encyclopediac style as found on Wikipedia. Your rewrite should try to preserve as much of the original information as possible while rephrasing and transforming the style. If the passage is of a literary or abstract nature then do your best to interpret it in a factual and objective way.

<passage>
{passage}
</passage>

Be sure to put your answer in rephrase tags similar to the passage tags above."""
    },
    {
        "name": "Q&A Format",
        "user_message": """Rewrite the following passage to be in a Q&A format. The question should start with the word "Question:" and be a question which the *information* in the following passage might be an answer to. The passage should then be rephrased so that it makes sense as an answer to the question. You should start the rephrased answer with the word "Answer:" and then put the rephrased passage after it. Try to preserve the core information from the original passage while still rephrasing it to not just be the same words.

<passage>
{passage}
</passage>

You should put both the question and its answer between rephrase tags."""
    },
    {
        "name": "B1 Intermediate Simplification",
        "user_message": """Rephrase the following passage so that it could be understood by someone at B1 intermediate understanding of English. Avoid words longer than 5-7 letters while preserving as much of the meaning as you can. Put the rephrased passage in <rephrase> tags <rephrase>like this</rephrase> so that it can be automatically extracted by an inference script.

<passage>
{passage}
</passage>

Try to preserve as much of the original meaning and information as possible during your rephrase while making it B1 intermediate reading level. Include all the information available in the original passage and go sentence by sentence."""
    }
]

def load_few_shot_examples():
    """Load two random few-shot examples from the few_shot_rephrase_pool directory"""
    pool_dir = "few_shot_rephrase_pool"
    
    if not os.path.exists(pool_dir):
        print(f"Warning: Few-shot pool directory '{pool_dir}' not found. Continuing without few-shot examples.")
        return []
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(pool_dir) if f.endswith('.json')]
    
    if len(json_files) < 2:
        print(f"Warning: Only {len(json_files)} JSON files found in '{pool_dir}'. Need at least 2 for few-shot learning.")
        if json_files:
            print("Using available files.")
        else:
            print("Continuing without few-shot examples.")
            return []
    
    # Randomly select 3 files
    selected_files = random.sample(json_files, 3)
    all_messages = []
    
    for filename in selected_files:
        filepath = os.path.join(pool_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'messages' in data:
                    all_messages.extend(data['messages'])
                else:
                    print(f"Warning: File {filename} does not contain 'messages' key")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return all_messages

def extract_rephrased_text(response_text):
    """Extract content between <rephrase> tags using regex"""
    match = re.search(r'<rephrase>(.*?)</rephrase>', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no tags found, return the entire response
    return response_text.strip()

async def send_rephrase_request(session, template_name, user_message, few_shot_messages):
    """Send a single rephrase request to the API with few-shot examples"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Build proper ChatML messages: few-shot examples + current request
    messages = []
    
    # Add few-shot examples if available
    messages.extend(few_shot_messages)
    
    # Add current request
    messages.extend([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ASSISTANT_PREFIX}
    ])
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.7,
        "stream": False
    }
    
    try:
        async with session.post(f"{BASE_URL}/chat/completions", 
                              json=payload, headers=headers) as response:
            
            if response.status == 200:
                data = await response.json()
                # The API response should contain the completed assistant message
                full_response = data["choices"][0]["message"]["content"]
                return template_name, extract_rephrased_text(full_response)
            else:
                error_text = await response.text()
                return template_name, f"Error: HTTP {response.status} - {error_text}"
                
    except Exception as e:
        return template_name, f"Error: {str(e)}"

async def main():
    if not API_KEY:
        print("Error: ASI_API_KEY environment variable is not set")
        sys.exit(1)
    
    # Load few-shot examples once to use for all templates
    few_shot_messages = load_few_shot_examples()
    
    if few_shot_messages:
        print(f"Loaded {len(few_shot_messages)} few-shot messages from pool")
    else:
        print("No few-shot examples loaded, using zero-shot prompting")
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for template in TEMPLATES:
            formatted_user_message = template["user_message"].format(passage=TEST_PASSAGE)
            task = send_rephrase_request(session, template["name"], formatted_user_message, few_shot_messages)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Print results with two newlines between each
        for i, (name, result) in enumerate(results):
            print(f"--- {name} ---")
            print(result)
            if i < len(results) - 1:  # Don't add extra newlines after the last result
                print("\n\n")

if __name__ == "__main__":
    asyncio.run(main())
