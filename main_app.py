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


os.environ["OPENAI_API_KEY"] = "sk-proj-Q2NpeueMJLBC6Ph8C2tBT3BlbkFJl7Be8iVm0d6suLtwVaJq"

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

# Define a tool to get category products
@tool
def get_category_products(user_message: str) -> list:
    """
    Make a filter based on this category and return all products in this category.

    Args:
    user_message (str): The original message from the user.

    Returns:
    list: A list of products in the specified category.
    """
    closest_category = find_closest_category(user_message)

    products = vectorstore.similarity_search(query=user_message, k=100, filter={"category": closest_category})

    product_list = []
    for product in products:
        product_info = {
            "name": product.metadata["name"],
            "description": product.page_content,
            "price": product.metadata["price"],
        }
        product_list.append(product_info)
    return closest_category, product_list

# Define a tool to get category attributes
@tool
def get_category_attributes(category: str) -> list:
    """
    Retrieve the attributes for a given product category to help ask specific questions related to the attributes and make the search and recommendation process easier.

    Args:
    category (str): The product category.

    Returns:
    list: A list of attributes for the specified category.
    """
    keys_set = set()
    for doc in llm_data:
        if doc.metadata.get("category") == category:
            keys_set.update(doc.metadata.keys())
    keys_set.discard('category')
    attributes = list(keys_set)
    return attributes

# Define a tool to recommend products
@tool
def recommend_products(user_message: str, category: str, user_responses: dict) -> list:
    """
    Recommend products based on the specified category and user-defined attributes. This tool filters products by matching the category and user-specified attributes such as brand, screen size, and other features.

    Args:
    user_message (str): The original message from the user.
    category (str): The product category.
    user_responses (dict): The user's preferences and requirements, e.g., {"brand": "Samsung", "screen size": "50 inches"}.

    Returns:
    list: A list of recommended products that match the user's criteria.
    """
    # Create filter conditions based on user_responses, excluding price
    filters = {"category": category}
    for key, value in user_responses.items():
        if key != "price":
            filters[key] = value

    # Perform a similarity search with the specified filters
    recommended_products = vectorstore.similarity_search(query=user_message, k=100, filter=filters)
    return recommended_products


tools = [get_category_products, get_category_attributes, recommend_products]

system_schema = """
You are an AI chatbot designed to act as a recommendation engine for Raya Store, Egypt's premier electronics e-commerce website. Your primary function is to assist users in finding the best products based on their needs and preferences.

When interacting with users, always provide responses in the language and dialect that matches the user's input.

Your role is to:

1. **Greeting and Introduction:**
   Start by greeting the user politely and introducing yourself as an AI assistant from Raya Store. Clearly state your purpose, which is to help them find the perfect products.
   Example: "Hello! I'm your AI assistant from Raya Store. I'm here to help you find the best products for your needs. How can I assist you today?"

2. **Understanding User's Needs:**
   Ask the user to specify the type of product they are looking for, such as mobile phones, TVs, or any other category. Use the `get_category_products` tool to retrieve the closest_category and a list of related products based on the category identified from the user's message.
   Example: "Can you please specify the type of product you are looking for? Like mobile phones, TVs, or any other category?"

3. **Display Related Products:**
   After calling the `get_category_products` tool, display the `product_list` (the list of products related to the user's input).

   Example: "Here are some mobile phones I found [List of products with brief descriptions and prices]."

4. **Inquiring for Specific Preferences:**
   Automatically send the `closest_category` to the `get_category_attributes` tool to retrieve relevant attributes for the identified category. Ask the user specific questions about their preferences related to these attributes to help refine their search.
   Example: "Based on the mobile phones I've found, could you tell me if you have any preferences for brand, screen size, or any particular features? What's your price range?"

5. **Providing Tailored Recommendations:**
   Use the `closest_category` along with the user's preferences (obtained from the attribute-based questions) as input to the `recommend_products` tool. Provide the user with tailored recommendations based on their answers.
   Example: "Based on your preferences, here are some mobile phones that might interest you: [List of recommended products with brief descriptions and prices]."

6. **Offering Additional Assistance:**
   After providing recommendations, ask the user if they need more details about any specific product or if they want additional recommendations.
   Example: "Would you like more information on any of these products, or should I suggest some more options?"

7. **Confirming User's Input:**
   Summarize the information you have gathered and confirm with the user to ensure accuracy before finalizing your recommendations.
   Example: "So, you're looking for a Samsung smartphone with a large display and a budget of around 10,000 EGP. Is that correct?"

8. **Ending the Conversation:**
   Conclude the interaction by thanking the user for visiting Raya Store. Encourage them to return for future purchases and remind them that they can reach out anytime they need assistance.
   Example: "Thank you for choosing Raya Store! If you need any more assistance in the future, feel free to reach out. Have a great day!"

Always make sure your responses are helpful, polite, and tailored to the user's current query. If a user requests a product or information that is not available, clearly inform them that the "product is currently unavailable" and suggest alternative options if possible.
"""



input_schema = """Answer the following question asked by one of our customers. I will provide you with the customer's message and the context of the chat history.

Chat History Context enclosed by ========:
========Chat History Start========
{chat_history}
========Chat History End========

Think before you reply and revise your answer.
Focus on the customer's current message and provide a helpful response in the same language and dialect.

Customer's Current Message: {input}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_schema),
    ("human", input_schema),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(chat_model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chat_history = []

def start_conversation():
    agent_response = agent_executor({"input": "", "chat_history": chat_history})
    ai_message = AIMessage(content=agent_response["output"])
    chat_history.append(ai_message)
    return agent_response["output"]

def chat_with_bot(user_input):
    human_message = HumanMessage(content=user_input)
    chat_history.append(human_message)
    agent_response = agent_executor({"input": user_input, "chat_history": chat_history})
    ai_message = AIMessage(content=agent_response["output"])
    chat_history.append(ai_message)
    return agent_response["output"]

# Streamlit UI
st.title("AI Search Chatbot for Raya Store")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("Type your message here:")
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    st.session_state['chat_history'].append(("User", user_input))
    bot_response = chat_with_bot(user_input)
    st.session_state['chat_history'].append(("Bot", bot_response))

# Display chat history
for sender, message in st.session_state['chat_history']:
    st.write(f"**{sender}:** {message}")

# Start the conversation with an initial message
if not st.session_state['chat_history']:
    initial_message = start_conversation()
    st.session_state['chat_history'].append(("Bot", initial_message))
    st.write(f"**Bot:** {initial_message}")
