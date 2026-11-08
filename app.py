# Streamlit interface for entering passenger details and displaying predictions.
import streamlit as st
import pickle
import pandas as pd

# import the dataset
df = pd.read_csv('data/train.csv')

df[['Deck', 'Cabin_num', 'Side']] = df['Cabin'].str.split('/', expand=True)

# load the model
def load_model():
    with open('spaceship-titanic-model-v3.pkl', 'rb') as f:
        return pickle.load(f)

# create website
# create header and intro
st.image("spaceship_banner.png")
st.header("Spaceship Titanic Transport Prediction App")
st.subheader("About the Dataset")
st.markdown("""
It’s the year 2912. The Spaceship Titanic was carrying nearly 13,000
passengers to three distant planets when it encountered a spacetime
anomaly near Alpha Centauri. The ship remained intact, but nearly half
of its passengers were transported to another dimension.

This project uses recovered passenger records to predict who was    
transported. The target column, `Transported`, records the known outcome
for each passenger in the training dataset.
""")

# create expander for detailed dataset table
with st.expander("Explore the Original Training Dataset", expanded=False):
    original_df = df.drop(columns=["Deck", "Cabin_num", "Side"])
    st.caption(
        f"{len(original_df):,} passenger records - {original_df.shape[1]} columns. "
        "Scroll to explore the data or click a column header to sort."
    )
    st.dataframe(original_df, hide_index=True, height=450)

# create expander for Dataset Description
with st.expander("**Dataset Description**", expanded=False):
    st.markdown("""
This app uses Spaceship Titanic passenger records to predict whether
a passenger was transported to another dimension.

| Input | What it means |
|---|---|
| Home Planet | The planet the passenger departed from. |
| CryoSleep | Whether the passenger spent the voyage in suspended animation. Select `True` to set all spending to zero. |
| Destination | The planet where the passenger planned to disembark. |
| VIP | Whether the passenger paid for VIP service. |
| Cabin Deck | The deck where the passenger’s cabin was located. |
| Cabin Side | The side of the ship where the cabin was located. `P` is port (left), and `S` is starboard (right), facing forward. |
| Age | The passenger’s age in years. |
| Room Service | The amount charged for room service. |
| Food Court | The amount charged at the food court. |
| Shopping Mall | The amount charged at the shopping mall. |
| Spa | The amount charged for spa services. |
| VR Deck | The amount charged for virtual reality activities. |

**Total Spend** is calculated automatically by adding the five spending amounts.

**Transported** is the outcome the app predicts. In the original training
dataset, this column records what happened to each passenger.
""")


st.write("Enter passenger details and luxury amenity spending to predict whether they were transported to an alternate dimension.")

# create sidebar
with st.sidebar:
    st.image("parami.jpg")
    st.subheader("Student Information")
    st.markdown("Student Name: [Htet Khant Linn](https://htet-khant-linn.github.io/my-portfolio/)")
    st.text("Student ID: PIUS20230064")
    st.text("Course: CS 300 - Introduction to Machine Learning")
    st.text("Spaceship Titanic Transport Prediction App")
    st.text("Model: Tuned KNN Classifier")
    st.text("Test Accuracy: 78.61%")
    st.markdown("[View Project on GitHub](https://github.com/Htet-Khant-Linn/spaceship-titanic-ml-app)")

# create 2 cols for data inputs
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
                            value=0.0, step=50.0,
                            disabled=cryo_sleep)
    food_court = st.slider("Enter your FoodCourt spend",
                            min_value=float(df['FoodCourt'].min()),
                            max_value=float(df['FoodCourt'].max()),
                            value=0.0, step=50.0,
                            disabled=cryo_sleep)
    shopping_mall = st.slider("Enter your ShoppingMall spend",
                            min_value=float(df['ShoppingMall'].min()),
                            max_value=float(df['ShoppingMall'].max()),
                            value=0.0, step=50.0,
                            disabled=cryo_sleep)
    spa = st.slider("Enter your Spa spend",
                    min_value=float(df['Spa'].min()),
                    max_value=float(df['Spa'].max()),
                    value=0.0, step=50.0,
                    disabled=cryo_sleep)
    vr_deck = st.slider("Enter your VRDeck spend",
                        min_value=float(df['VRDeck'].min()),
                        max_value=float(df['VRDeck'].max()),
                        value=0.0, step=50.0,
                        disabled=cryo_sleep)


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

# create predict button
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