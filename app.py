# Streamlit interface for entering passenger details and displaying predictions.
import streamlit as st
import pickle
import pandas as pd


df = pd.read_csv('data/train.csv')

df[['Deck', 'Cabin_num', 'Side']] = df['Cabin'].str.split('/', expand=True)

def load_model():
    with open('spaceship-titanic-model.pkl', 'rb') as f:
        return pickle.load(f)

st.header("Spaceship Titanic Transport Prediction App")
st.write("Enter passenger details and luxury amenity spending to predict whether they were transported to an alternate dimension.")

with st.sidebar:
    st.image("parami.jpg")
    st.subheader("Student Information")
    st.text("Student Name: Htet Khant Linn")
    st.text("Student ID: PIUS20230064")
    st.text("Course: CS 300 - Introduction to Machine Learning")
    st.text("Spaceship Titanic Transport Prediction App")
    st.text("Model: Tuned Gradient Boosting")
    st.text("Test Accuracy: 80.24%")

col1, col2 = st.columns(2)

with col1:
    home_planet = st.selectbox("Enter your HomePlanet", df['HomePlanet'].dropna().unique())
    cryo_sleep = st.selectbox("Enter your CryoSleep status", [False, True])
    destination = st.selectbox("Enter your Destination", df['Destination'].dropna().unique())
    vip = st.selectbox("Enter your VIP status", [False, True])
    deck = st.selectbox("Enter your Cabin Deck", sorted(df['Deck'].dropna().unique()))
    side = st.selectbox("Enter your Cabin Side (P = Port, S = Starboard)", sorted(df['Side'].dropna().unique()))

with col2:
    age = st.number_input("Enter your Age",
                            min_value=int(df['Age'].min()),
                            max_value=int(df['Age'].max()),
                            value=28)
    room_service = st.slider("Enter your RoomService spend",
                            min_value=float(df['RoomService'].min()),
                            max_value=float(df['RoomService'].max()),
                            value=0.0, step=50.0)
    food_court = st.slider("Enter your FoodCourt spend",
                            min_value=float(df['FoodCourt'].min()),
                            max_value=float(df['FoodCourt'].max()),
                            value=0.0, step=50.0)
    shopping_mall = st.slider("Enter your ShoppingMall spend",
                            min_value=float(df['ShoppingMall'].min()),
                            max_value=float(df['ShoppingMall'].max()),
                            value=0.0, step=50.0)
    spa = st.slider("Enter your Spa spend",
                    min_value=float(df['Spa'].min()),
                    max_value=float(df['Spa'].max()),
                    value=0.0, step=50.0)
    vr_deck = st.slider("Enter your VRDeck spend",
                        min_value=float(df['VRDeck'].min()),
                        max_value=float(df['VRDeck'].max()),
                        value=0.0, step=50.0)

# set if conditon for cryo_sleep passenger
if cryo_sleep == True:
    room_service = 0.0
    food_court = 0.0
    shopping_mall = 0.0
    spa = 0.0
    vr_deck = 0.0
    st.info("Note: Passengers in CryoSleep are confined to their cabins, so spending is set to 0.0.")

# calculate the total spending
total_spend = room_service + food_court + shopping_mall + spa + vr_deck
st.write(f"**Total Spend:** {total_spend}")

predict_button = st.button("Predict Transport Status")

if predict_button:
    bool_map = {True: "True", False: "False"}
    
    input_data = pd.DataFrame(
        [
            {
                "HomePlanet": home_planet,
                "CryoSleep": bool_map[cryo_sleep],
                "Destination": destination,
                "Age": float(age),
                "VIP": bool_map[vip],
                "RoomService": room_service,
                "FoodCourt": food_court,
                "ShoppingMall": shopping_mall,
                "Spa": spa,
                "VRDeck": vr_deck,
                "Deck": deck,
                "Side": side,
                "Total_Spend": total_spend
            }
        ]
    )

    model = load_model()
    result = model.predict(input_data)[0]
    
    if result == 1:
        st.success("Prediction: Transported (True) - Passenger was transported to an alternate dimension!")
    else:
        st.warning("Prediction: Not Transported (False) — Passenger remained safely onboard!")
