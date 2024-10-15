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

# Set OpenAI API key from the environment variable
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY") # 'ollama'  

api_key = os.environ.get("OPENAI_API_KEY")
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

# Function to log each interaction between the user and the AI
def log_interaction(user_input, ai_response, products, categories):
    # Create a dictionary with all the information to log
    log_entry = {
        "timestamp": datetime.now().isoformat(),  # Current time in ISO format
        "user_input": user_input,                 # What the user asked
        "ai_response": ai_response,               # The AI's response
        "metadata": {
            "products": products,                 # Products mentioned in the interaction
            "categories": categories              # Categories mentioned in the interaction
        }
    }
    
    # Use a lock to ensure thread-safe file writing
    with log_lock:
        # Open the log file in append mode
        with open("interaction_log.json", "a") as log_file:
            # Write the log entry as a JSON object
            json.dump(log_entry, log_file)
            log_file.write("\n")  # Add a newline for readability

# Function to extract products and categories from the AI's parsed response
def extract_products_and_categories(category_and_product_list):
    products = []
    categories = set()  # Use a set to avoid duplicates
    for item in category_and_product_list:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            # If item is a list or tuple with at least 2 elements
            categories.add(item[0])  # First element is the category
            products.append(item[1])  # Second element is the product
        elif isinstance(item, dict):
            # If item is a dictionary
            categories.add(item.get('category', ''))  # Get category, or empty string if not found
            products.append(item.get('product', ''))  # Get product, or empty string if not found
    return products, list(categories)  # Convert categories back to a list

# Main chatbot function that processes user input and generates responses
def chatbot(user_input, history):
    # Convert chat history to a format suitable for the AI model
    all_messages = [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': msg} for i, (_, msg) in enumerate(history)]
    
    try:
        # Process the user's message and get AI response
        response, updated_messages = process_user_message(user_input, all_messages)
        
        # Extract product and category information from the user's input
        category_and_product_response = utils.find_category_and_product_only(
            user_input, utils.get_products_and_category())
        category_and_product_list = utils.read_string_to_list(category_and_product_response)
        
        # Extract products and categories from the parsed response
        products, categories = extract_products_and_categories(category_and_product_list)
        
        # Log the interaction
        log_interaction(user_input, response, products, categories)
    
    except Exception as e:
        # If an error occurs, print it and return an apologetic message
        print(f"An error occurred: {str(e)}")
        response = "I apologize, but an error occurred while processing your request. Please try again or contact support if the issue persists."
        products, categories = [], []
        
        # Log the error
        log_interaction(user_input, f"Error: {str(e)}", products, categories)
    
    return response

# Create Gradio interface for the chatbot
iface = gr.ChatInterface(
    chatbot,  # The main chatbot function
    title="AI Customer Service Assistant",
    description="Ask about our electronic products!",
    theme="soft",
    examples=[
        "Tell me about the SmartX ProPhone",
        "What TVs do you have?",
        "Do you have any cameras for professional photography?",
    ],
)

# Launch the interface if this script is run directly
if __name__ == "__main__":
    iface.launch()
