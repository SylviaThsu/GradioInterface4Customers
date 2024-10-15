import os
import openai
import sys
import utils
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
import json
from datetime import datetime
import threading

# Set your OpenAI API key from the environment variable
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") # 'ollama'  

# api_key = os.environ.get("OPENAI_API_KEY")
model = "gpt-4o-mini"
base_url = None

client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

# Create a lock for thread-safe logging
log_lock = threading.Lock()

def get_completion_from_messages(messages, 
                                 model="gpt-4o-mini", 
                                 temperature=0, 
                                 max_tokens=500):
    '''
    Encapsulate a function to access LLM

    Parameters: 
    messages: This is a list of messages, each message is a dictionary containing role and content. The role can be 'system', 'user' or 'assistant', and the content is the message of the role.
    model: The model to be called, default is gpt-4o-mini (ChatGPT) 
    temperature: This determines the randomness of the model output, default is 0, meaning the output will be very deterministic. Increasing temperature will make the output more random.
    max_tokens: This determines the maximum number of tokens in the model output.
    '''
    response = client.chat.completions.create(
        messages=messages,
        model = model, 
        temperature=temperature, # This determines the randomness of the model's output
        max_tokens=max_tokens, # This determines the maximum number of tokens in the model's output
    )

    return response.choices[0].message.content

def process_user_message(user_input, all_messages, debug=True):
    """
    Preprocess user messages
    
    Parameters:
    user_input : User input
    all_messages : Historical messages
    debug : Whether to enable DEBUG mode, enabled by default
    """
    # Delimiter
    delimiter = "```"
    
    # Step 1: Use OpenAI's Moderation API to check if the user input is compliant or an injected Prompt
    response = client.moderations.create(input=user_input)
    moderation_output = response.results[0]

    # The input is non-compliant after Moderation API check
    if moderation_output.flagged:
        print("Step 1: Input rejected by Moderation")
        return "Sorry, your request is non-compliant"

    # If DEBUG mode is enabled, print real-time progress
    if debug: 
        print("Step 1: Input passed Moderation check")
        print(f"\n**user_input**: {user_input} \n\n")
    
    # Step 2: Extract products and corresponding categories 
    category_and_product_response = utils.find_category_and_product_only(
        user_input, utils.get_products_and_category())
    #print(category_and_product_response)
    # Convert the extracted string to a list
    category_and_product_list = utils.read_string_to_list(category_and_product_response)
    print(category_and_product_list)

    if debug: print("Step 2: Extracted product list")

    # Step 3: Find corresponding product information
    product_information = utils.generate_output_string(category_and_product_list)
    print(product_information)
    if debug: 
        print("Step 3: Found information for extracted products")
        print(f"\n**user_input**: {user_input} \n\n")

    # Step 4: Generate answer based on information
    system_message = f"""
    You are a customer service assistant for a large electronic store. \
    Respond in a friendly and helpful tone, with concise answers. \
    Make sure to ask the user relevant follow-up questions.
    """
    # Insert message
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{delimiter}{user_input}{delimiter}"},
        {'role': 'assistant', 'content': f"Relevant product information:\n{product_information}"}
    ]
    # Get GPT3.5's answer
    # Implement multi-turn dialogue by appending all_messages
    final_response = get_completion_from_messages(all_messages + messages)
    if debug:print("Step 4: Generated user answer")
    # Add this round of information to historical messages
    all_messages = all_messages + messages[1:]

    # Step 5: Check if the output is compliant based on Moderation API
    response = client.moderations.create(input=final_response)
    moderation_output = response.results[0]
    print(moderation_output)
    # Output is non-compliant
    if moderation_output.flagged:
        if debug: print("Step 5: Output rejected by Moderation")
        return "Sorry, we cannot provide that information"

    if debug: print("Step 5: Output passed Moderation check")

    # Step 6: Model checks if the user's question is well answered
    user_message = f"""
    Customer message: {delimiter}{user_input}{delimiter}
    Agent response: {delimiter}{final_response}{delimiter}

    Does the response sufficiently answer the question? Answer Yes or no.
    """
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]
    # Request model to evaluate the answer
    evaluation_response = get_completion_from_messages(messages)
    if debug: print("Step 6: Model evaluated the answer")

    # Step 7: If evaluated as Y, output the answer; if evaluated as N, feedback that the answer will be manually corrected
    if "Y" in evaluation_response:  # Use 'in' to avoid the model possibly generating Yes
        if debug: print("Step 7: Model approved the answer.")
        return final_response, all_messages
    else:
        if debug: print("Step 7: Model disapproved the answer.")
        neg_str = "I apologize, but I cannot provide the information you need. I will transfer you to a human customer service representative for further assistance."
        return neg_str, all_messages

def log_interaction(user_input, ai_response, products, categories):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "ai_response": ai_response,
        "metadata": {
            "products": products,
            "categories": categories
        }
    }
    
    with log_lock:
        with open("interaction_log.json", "a") as log_file:
            json.dump(log_entry, log_file)
            log_file.write("\n")

def extract_products_and_categories(category_and_product_list):
    products = []
    categories = set()
    for item in category_and_product_list:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            categories.add(item[0])
            products.append(item[1])
        elif isinstance(item, dict):
            categories.add(item.get('category', ''))
            products.append(item.get('product', ''))
    return products, list(categories)

def chatbot(user_input, history):
    all_messages = [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': msg} for i, (_, msg) in enumerate(history)]
    
    try:
        response, updated_messages = process_user_message(user_input, all_messages)
        
        category_and_product_response = utils.find_category_and_product_only(
            user_input, utils.get_products_and_category())
        category_and_product_list = utils.read_string_to_list(category_and_product_response)
        
        products, categories = extract_products_and_categories(category_and_product_list)
        
        log_interaction(user_input, response, products, categories)
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        response = "I apologize, but an error occurred while processing your request. Please try again or contact support if the issue persists."
        products, categories = [], []
        
        log_interaction(user_input, f"Error: {str(e)}", products, categories)
    
    return response

# Create Gradio interface
iface = gr.ChatInterface(
    chatbot,
    title="AI Customer Service Assistant",
    description="Ask about our electronic products!",
    theme="soft",
    examples=[
        "Tell me about the SmartX ProPhone",
        "What TVs do you have?",
        "Do you have any cameras for professional photography?",
    ],
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()