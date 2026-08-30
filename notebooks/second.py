import streamlit as st
import langchain_core
from langchain_classic.chains.sequential import SequentialChain
from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from config_loader import load_config_as_dict


llm_model: ChatOllama = None

def llm_initializer():
    global llm_model
    if llm_model:
        print("(*) LLM Model Already Initialized.")
        return
    llm_config = load_config_as_dict()['llm']
    llm_model = ChatOllama(
        model=llm_config['model'],
        temperature=llm_config['temperature'],
        num_predict=llm_config['max_tokens'],  # Ollama's equivalent of Groq's max_tokens
    )

def generate_name_and_menu(cuisine, food_preference):
    global llm_model
    if not llm_model:
        print(f"(*) Initializing LLM Model...")
        llm_initializer()
        print(f"(*) LLM Model Initialized.")

    query = "I want to open a restaurant for {cuisine} food especially for who prefers to have {food_preference} food category. Suggest a delicate yet fancy name for this. ** SHARE JUST THE BEST NAME**, only 1 name and nothing else."
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine', 'food_preference'],
        template=query
    )
    rest_name_chain = LLMChain(llm=llm_model, prompt=prompt_template_name, output_key='restaurant_name')

    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name', 'food_preference'],
        template="Suggest some menu items, total 5 items from {food_preference} category, for {restaurant_name}. Return it as a comma-separated list. Share ** just the items** nothing else."
    )
    food_items_chain = LLMChain(llm=llm_model, prompt=prompt_template_items, output_key='menu_items')

    chain = SequentialChain(
        chains=[rest_name_chain, food_items_chain],
        input_variables=['cuisine', 'food_preference'],
        output_variables=['restaurant_name', 'menu_items']
    )
    return chain.invoke({'cuisine': cuisine, 'food_preference': food_preference})

st.title("Restaurant Name & Menu Generator")
cuisine_name = st.sidebar.selectbox("Select A Cuisine", options=('Indian', 'Mexican', 'Arabic', 'Chinese', 'Korean'))
food_preference_categ = st.sidebar.selectbox("Select Your Food Preference", options=('Veg', 'Non-Veg', 'Vegan', 'Anything'))

if cuisine_name and food_preference_categ:
    # response = generate_name_and_menu(cuisine)
    response = generate_name_and_menu(cuisine_name, food_preference_categ)
    print(f"(*) Current Response: {response}")
    st.header(f"Welcome to {response['restaurant_name'].strip()} for {cuisine_name} cuisine🤗".title())
    st.write('Our Special Menu Items:')
    for item in response['menu_items'].strip().split(','):
        st.write(f'⟢ {item}')
