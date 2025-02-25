import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import datetime

# Initialize Groq API
groq_api_key = "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"
llm = ChatGroq(model_name="llama-3.3-70b-specdec", api_key=groq_api_key)

# Define Prompt Templates
travel_itinerary_prompt = PromptTemplate(
    input_variables=["destination", "days", "interests"],
    template="Create a {days}-day travel itinerary for {destination} focusing on {interests}."
)

flight_hotel_prompt = PromptTemplate(
    input_variables=["destination", "budget", "dates"],
    template="Find flights and hotels for {destination} within a {budget} budget for dates {dates}."
)

travel_tips_prompt = PromptTemplate(
    input_variables=["destination"],
    template="Give me essential travel tips for {destination}."
)

# Create Chains
itinerary_chain = LLMChain(llm=llm, prompt=travel_itinerary_prompt)
flight_hotel_chain = LLMChain(llm=llm, prompt=flight_hotel_prompt)
tips_chain = LLMChain(llm=llm, prompt=travel_tips_prompt)

# Streamlit UI
st.set_page_config(page_title="Travel Planning AI Assistant", layout="wide")
st.title("✈️ Travel Planning AI Assistant")

# Travel Itinerary Generator
st.header("🗺️ Plan Your Trip")
destination = st.text_input("Enter Destination:")
days = st.number_input("Number of Days", min_value=1, max_value=30, value=5)
interests = st.text_input("Enter Your Interests (e.g., adventure, culture, food):")
if st.button("Generate Itinerary"):
    itinerary = itinerary_chain.run({"destination": destination, "days": days, "interests": interests})
    st.write("### Your Travel Itinerary:")
    st.write(itinerary)

# Flight & Hotel Finder
st.header("🏨 Find Flights & Hotels")
budget = st.selectbox("Select Budget", ["Budget", "Mid-range", "Luxury"])
dates = st.date_input("Select Travel Dates", min_value=datetime.date.today())
if st.button("Find Flights & Hotels"):
    flight_hotel_info = flight_hotel_chain.run({"destination": destination, "budget": budget, "dates": dates})
    st.write("### Flight & Hotel Recommendations:")
    st.write(flight_hotel_info)

# Travel Tips
st.header("💡 Travel Tips")
if st.button("Get Travel Tips"):
    tips = tips_chain.run({"destination": destination})
    st.write("### Essential Travel Tips:")
    st.write(tips)

st.sidebar.header("🔗 Additional Features")
st.sidebar.write("- Weather forecast integration")
st.sidebar.write("- Currency exchange rates")
st.sidebar.write("- Local emergency contacts")
