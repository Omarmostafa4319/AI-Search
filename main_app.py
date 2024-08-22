import streamlit as st
import json
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import os
from langchain.schema import Document
import numpy as np

os.environ["OPENAI_API_KEY"] = "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"

chat_model = ChatOpenAI(model="gpt-4o-mini")

# Load the JSON data
with open('raya_data_en.json', 'r') as file:
    data = json.load(file)

# Process the data and extract category attributes
all_products = []
for category in data:
    for product in category["products"]:
        product_info = {
            "category": category["category"],
            "name": product["name"],
            "description": product["description"],
            "price": product["price"],
        }
        product_info.update(product["attributes"])
        all_products.append(product_info)

# Combine all product information into a single string for page_content and add category to metadata
llm_data = []
for product in all_products:
    page_content = product['description']
    metadata = {
        "category": product["category"],
        "name": product["name"],
        "price": product["price"],
    }
    # Add remaining attributes to metadata
    for key, value in product.items():
        if key not in ['description', 'category', 'name', 'price']:
            metadata[key] = value

    # Create Document instance
    doc = Document(page_content=page_content, metadata=metadata)
    llm_data.append(doc)

embeddings_llm = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(llm_data, embeddings_llm)

categories = [category["category"] for category in data]
category_embeddings = embeddings_llm.embed_documents(categories)
category_embeddings = np.array(category_embeddings)

def find_closest_category(user_input):

    # Embed the user's input and convert it to a NumPy array
    user_input_embedding = np.array(embeddings_llm.embed_query(user_input))

    # Calculate cosine similarity between user input and each category embedding
    similarities = np.dot(category_embeddings, user_input_embedding)

    # Find the index of the closest category
    closest_index = np.argmax(similarities)
    closest_category = categories[closest_index]

    return closest_category

# Define a tool to get category attributes
@tool
def get_category_and_attributes(user_message: str) -> list:
    """
    Get attributes for a specific category based on the user's input to help ask specific questions related to them and make the search and recommendation process easier.

    Args:
    user_message (str): The message from the user.

    Returns:
    closest_category (str): The category that best matches the user's input.
    attributes (dict): A dictionary of attributes for the closest category.

    """
    closest_category = find_closest_category(user_message)

    attributes = {}
    for doc in llm_data:
        if doc.metadata.get("category") == closest_category:
            for key, value in doc.metadata.items():
                if key != "category":
                    attributes[key] = value

    return closest_category, attributes

# Define a tool to recommend products
@tool
def recommend_products(user_message: str, closest_category:str, attributes: dict) -> list:
    """
    Recommend products based on the specified category and user-defined attributes. This tool filters products by matching the category and user-specified attributes such as brand, screen size, and other features.

    Args:
    user_message (str): The message from the user.
    closest_category (str): The category that best matches the user's input.
    attributes (dict): A dictionary of user-defined attributes.

    Returns:
    recommended_products (list): A list of recommended products.
    """
    # Create filter conditions based on user_responses, excluding price
    filters = {"category": closest_category}
    for key, value in attributes.items():
        if value and key == "brand":
            filters[key] = value

    # Perform a similarity search with the specified filters
    recommended_products = vectorstore.similarity_search(query=user_message, k=10, filter=filters)
    return recommended_products


tools = [get_category_and_attributes, recommend_products]

system_schema = """
You are an AI assistant for Raya Store, specializing in helping customers find and recommend electronics based on the available data. Your role is to guide the user through the process of selecting a product by asking clarifying questions and making suggestions based on the data loaded from the provided JSON file. If the user asks about a product category or specific product that is not available in the data, respond by saying "Products currently not available." Ensure that you provide personalized recommendations by understanding the user's preferences, such as budget, features, and brand preferences. Always strive to be helpful, friendly, and concise.

When interacting with users, you must always respond in the language that matches the user's input. If the user writes in English, respond in English. If the user writes in Arabic, respond in Arabic.
"""



input_schema = """
Answer the following question asked by one of our customers. I will provide you with the customer's message and the context of the chat history.

Chat History Context enclosed by ========:
========Chat History Start========
{chat_history}
========Chat History End========

Step 1: Identify the closest category using the "get_category_and_attributes" tool. If no matching category or attributes are found, respond with "Products currently not available."
        If a matching category is found, you must recommend at least 5 products related to the closest category identified to help user.
        Retrieve a dictionary of attributes associated with that category.

Step 2: Ask the user about specific features that are important to them using the attributes retrieved from the "get_category_and_attributes" tool to guide your questioning.

Step 3: Based on the user's preferences and the identified category, use the "recommend_products" tool to refine the product recommendations by filtering with the user's specific attributes and recommending the best-related products. If no products are available that match the user's request, respond with "Products currently not available."

Think before you reply and revise your answer.

Focus on the customer's current message and provide a helpful response.

Always respond in the same language and dialect as the user's current message. Do not switch languages unless the user does.

Customer's Current Message: {input}
"""



prompt = ChatPromptTemplate.from_messages([
    ("system", system_schema),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(chat_model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def start_conversation():
    initial_message = "Hello! I'm your AI assistant from Raya Store. I'm here to help you find the best products for your needs. How can I assist you today?"
    st.session_state['chat_history'].append({"role": "assistant", "content": initial_message})
    return initial_message

def chat_with_bot(user_input):
    print(30*"*")
    print(st.session_state['chat_history'])
    print(30*"*")
    st.session_state['chat_history'].append({"role": "user", "content": user_input})
    input = input_schema.format(chat_history=st.session_state['chat_history'], input=user_input)
    agent_response = agent_executor.invoke({
        "input": input,
    })

    bot_response = agent_response["output"]
    st.session_state['chat_history'].append({"role": "assistant", "content": bot_response})
    return bot_response

# Streamlit UI
st.title("💬 Welcome to Raya Chatbot")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if not st.session_state['chat_history']:
    start_conversation()

for msg in st.session_state['chat_history']:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    bot_response = chat_with_bot(prompt)
    st.chat_message("assistant").write(bot_response)

# Bot: Hello! Welcome to Raya Store. I'm your virtual assistant here to help you find the best electronics. How can I assist you today? Are you looking for something specific, or would you like some recommendations?

# User:     

# Bot: Great! What features are important to you? Are you looking for a particular brand, camera quality, battery life, or maybe a budget-friendly option?

# User: I want an Apple mobile with a good camera and long battery life, and my budget is around 1100.

# Bot: Based on your preferences, here are a few smartphones that might interest you: 
# 1. [Smartphone 1] - Excellent camera and 24-hour battery, priced at $499.
# 2. [Smartphone 2] - Great camera with 30-hour battery life, priced at $520.
# 3. [Smartphone 3] - Solid camera and battery life with additional features, priced at $480.

# Would you like more details on any of these options?
