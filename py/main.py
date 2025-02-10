from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

messages = [
    SystemMessage("You are now connected to the flight booking service."),
    HumanMessage("I would like to book a flight."),
]

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
# model = ChatMistralAI(model="mistral-large-latest")
# model = ChatAnthropic(model='claude-3.5-sonnet')
# model = ChatOpenAI(model='gpt-3.5-turbo')


# ====================== chat models ======================

# chat_history = []  

# system_message = SystemMessage(content="You are a helpful AI assistant.")
# chat_history.append(system_message)  # Add system message to chat history

# # Chat loop
# while True:
#     query = input("You: ")
#     if query.lower() == "exit":
#         break
#     chat_history.append(HumanMessage(content=query)) 

#     # Get AI response using history
#     result = model.invoke(chat_history)
#     response = result.content
#     chat_history.append(AIMessage(content=response))  

#     print(f"AI: {response}")


# print("---- Message History ----")
# print(chat_history)



# ====================== Prompt template ======================

# from langchain_core.prompts import ChatPromptTemplate

# template = "The user wants to book a flight from {city1} to {city2} on {date}. The user want flight schedule, the price, and the duration of the flight. The assistant should provide the user with the necessary information dummy data for now."

# prompt_template = ChatPromptTemplate.from_template(template)

# prompt = prompt_template.invoke({
#     "city1": "Mumbai",
#     "city2": "Delhi",
#     "date": "2025-03-01"
# })

# result = model.invoke(prompt)

# print(result.content)



# ====================== Chain ================================

# template = "The user wants to book a flight from {city1} to {city2} on {date}. The user want flight schedule, the price, and the duration of the flight. The assistant should provide the user with the necessary information provide single dummy data for now without mentioning its fake or dummy."
# prompt_template = ChatPromptTemplate.from_template(template)

# chain = prompt_template | model | StrOutputParser()
# result = chain.invoke({"city1": "mumbai", "city2": "delhi", "date": "2025-03-01"})
# print(result)

# ================= Chaining multiple chains =====================

# #  ===== sequential chaining =====

# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.output_parser import StrOutputParser
# from langchain.schema.runnable import RunnableLambda

# password_check_prompt = ChatPromptTemplate.from_template("""
# Check if this password meets basic requirements:
# Password: {password}
# Requirements:
# - At least 8 characters
# - Contains numbers
# - Contains special characters
# Respond with only "PASS" or "FAIL"
# """)

# feedback_prompt = ChatPromptTemplate.from_template("""
# Analyze why this password failed and provide feedback:
# Password: {password}

# Provide two short lines:
# 1. Why the password failed
# 2. How to improve it
# Keep it brief and clear.
# """)

# def create_password_chain():
#     password_validation = password_check_prompt | model | StrOutputParser()
#     feedback_generation = feedback_prompt | model | StrOutputParser()

#     def main_chain(inputs):
#         check_result = password_validation.invoke({
#             "password": inputs["password"]
#         })
        
#         if check_result.strip() == "FAIL":
#             feedback = feedback_generation.invoke({
#                 "password": inputs["password"]
#             })
#             return f"Result: {check_result}\nFeedback: {feedback}"
        
#         return f"Result: {check_result}"
    
#     return RunnableLambda(main_chain)

# password_chain = create_password_chain()

# print(password_chain.invoke({"password": "pass123"}))

# print(password_chain.invoke({"password": "P@ssw0rd123!"}))


#  ===== parallel chaining =====

# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.output_parser import StrOutputParser
# from langchain.schema.runnable import RunnablePassthrough, RunnableMap

# sentiment_prompt = ChatPromptTemplate.from_template("""
# Analyze the sentiment of this text. 
# Text: {text}
# Respond with only: POSITIVE, NEGATIVE, or NEUTRAL
# """)

# style_prompt = ChatPromptTemplate.from_template("""
# Analyze the writing style of this text.
# Text: {text}
# Respond with only: FORMAL, CASUAL, or TECHNICAL
# """)

# def create_analysis_chain():

#     sentiment_analysis = sentiment_prompt | model | StrOutputParser()
#     style_analysis = style_prompt | model | StrOutputParser()

#     parallel_chain = RunnableMap({
#         "sentiment": sentiment_analysis,
#         "style": style_analysis,
#         "original_text": RunnablePassthrough()
#     })

#     return parallel_chain

# analysis_chain = create_analysis_chain()

# result = analysis_chain.invoke({"text": "The quarterly financial report indicates a 15% revenue increase."})
    
# print(f"\nText: {result['original_text']['text']}")
# print(f"Sentiment: {result['sentiment']}")
# print(f"Style: {result['style']}")

# ========== conditional chaining =========

# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.output_parser import StrOutputParser
# from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

# age_prompt = ChatPromptTemplate.from_template("""
# Analyze the age {age} and categorize as:
# CHILD (0-12)
# TEEN (13-19)
# ADULT (20-59)
# SENIOR (60+)
# Respond with only the category name.
# """)

# child_prompt = ChatPromptTemplate.from_template("""
# Create a fun, simple message for a child aged {age}.
# Include a positive encouragement about learning.
# Keep it short and friendly.
# """)

# teen_prompt = ChatPromptTemplate.from_template("""
# Create a motivational message for a teenager aged {age}.
# Focus on growth and future opportunities.
# Keep it relatable and brief.
# """)

# adult_prompt = ChatPromptTemplate.from_template("""
# Provide a professional message for an adult aged {age}.
# Focus on work-life balance and personal development.
# Keep it concise and meaningful.
# """)

# senior_prompt = ChatPromptTemplate.from_template("""
# Create a respectful message for a senior aged {age}.
# Focus on wellness and life experience.
# Keep it warm and considerate.
# """)

# def create_age_based_chain():

#     age_analysis = age_prompt | model | StrOutputParser()
#     child_message = child_prompt | model | StrOutputParser()
#     teen_message = teen_prompt | model | StrOutputParser()
#     adult_message = adult_prompt | model | StrOutputParser()
#     senior_message = senior_prompt | model | StrOutputParser()

#     def route_by_age(inputs: dict):
#         age_category = age_analysis.invoke({"age": inputs["age"]})
        
#         if age_category == "CHILD":
#             message = child_message.invoke(inputs)
#             category = "Child Message"
#         elif age_category == "TEEN":
#             message = teen_message.invoke(inputs)
#             category = "Teen Message"
#         elif age_category == "ADULT":
#             message = adult_message.invoke(inputs)
#             category = "Adult Message"
#         else:  # SENIOR
#             message = senior_message.invoke(inputs)
#             category = "Senior Message"

#         return {
#             "age": inputs["age"],
#             "category": category,
#             "message": message
#         }

#     return RunnableLambda(route_by_age)

# age_chain = create_age_based_chain()

# result = age_chain.invoke({"age": 70})
# print(f"Age: {result['age']}")
# print(f"Category: {result['category']}")
# print(f"Message: {result['message']}")


