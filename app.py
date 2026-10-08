import streamlit as st
import pandas as pd
import joblib
from pandas.api.types import is_numeric_dtype


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Earthquake Damage Prediction",
    page_icon="🏚️",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")


model = load_model()


# --------------------------------------------------
# LOAD TRAINING DATA
# --------------------------------------------------

@st.cache_data
def load_training_data():
    return pd.read_csv("data/train_values.csv")


train_data = load_training_data()


# --------------------------------------------------
# LOAD LABELS
# --------------------------------------------------

@st.cache_data
def load_labels():
    return pd.read_csv("data/train_labels.csv")


labels = load_labels()


# --------------------------------------------------
# MERGE DATA FOR DEMO EXAMPLES
# --------------------------------------------------

demo_data = train_data.merge(
    labels,
    on="building_id"
)


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("🏚️ Earthquake Damage Prediction")

st.write(
    "Predict the damage level of a building using a "
    "Random Forest machine learning model trained on "
    "earthquake building data."
)

st.divider()


# ==================================================
# DEMO EXAMPLES
# ==================================================

st.header("🎯 Quick Demo Examples")

st.write(
    "Use these examples during the presentation to "
    "demonstrate predictions for different damage levels."
)


# --------------------------------------------------
# Find actual examples for each damage grade
# --------------------------------------------------

grade_1_examples = demo_data[demo_data["damage_grade"] == 1]
grade_2_examples = demo_data[demo_data["damage_grade"] == 2]
grade_3_examples = demo_data[demo_data["damage_grade"] == 3]


demo_col1, demo_col2, demo_col3 = st.columns(3)


with demo_col1:
    low_damage_button = st.button(
        "🟢 Load Low Damage Example",
        use_container_width=True
    )


with demo_col2:
    medium_damage_button = st.button(
        "🟠 Load Medium Damage Example",
        use_container_width=True
    )


with demo_col3:
    severe_damage_button = st.button(
        "🔴 Load Severe Damage Example",
        use_container_width=True
    )


# --------------------------------------------------
# Store selected demo example
# --------------------------------------------------

if "selected_demo" not in st.session_state:
    st.session_state.selected_demo = None


if low_damage_button and len(grade_1_examples) > 0:
    st.session_state.selected_demo = 1


if medium_damage_button and len(grade_2_examples) > 0:
    st.session_state.selected_demo = 2


if severe_damage_button and len(grade_3_examples) > 0:
    st.session_state.selected_demo = 3


# --------------------------------------------------
# Display selected demo information
# --------------------------------------------------

if st.session_state.selected_demo is not None:

    selected_grade = st.session_state.selected_demo

    if selected_grade == 1:
        selected_row = grade_1_examples.iloc[0]
        st.success("🟢 Low Damage example loaded.")

    elif selected_grade == 2:
        selected_row = grade_2_examples.iloc[0]
        st.warning("🟠 Medium Damage example loaded.")

    else:
        selected_row = grade_3_examples.iloc[0]
        st.error("🔴 Severe Damage example loaded.")

else:
    selected_row = None


st.divider()


# ==================================================
# BUILDING INFORMATION
# ==================================================

st.header("🏢 Building Information")

st.write(
    "Enter the building characteristics below and "
    "click Predict Damage."
)


feature_columns = list(model.feature_names_in_)

input_data = {}


# --------------------------------------------------
# CREATE INPUT FORM
# --------------------------------------------------

with st.form("prediction_form"):

    columns = st.columns(3)

    for index, column in enumerate(feature_columns):

        with columns[index % 3]:

            # --------------------------------------
            # If a demo example has been selected
            # --------------------------------------

            if selected_row is not None:

                demo_value = selected_row[column]

                # Numeric feature
                if is_numeric_dtype(train_data[column]):

                    if pd.isna(demo_value):
                        demo_value = train_data[column].median()

                    if (
                        pd.api.types.is_integer_dtype(train_data[column])
                        or float(demo_value).is_integer()
                    ):

                        input_data[column] = st.number_input(
                            column.replace("_", " ").title(),
                            value=int(demo_value),
                            step=1
                        )

                    else:

                        input_data[column] = st.number_input(
                            column.replace("_", " ").title(),
                            value=float(demo_value)
                        )

                # Categorical feature
                else:

                    values = (
                        train_data[column]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    values = sorted(values)

                    demo_value = str(demo_value)

                    if demo_value in values:

                        input_data[column] = st.selectbox(
                            column.replace("_", " ").title(),
                            values,
                            index=values.index(demo_value)
                        )

                    else:

                        input_data[column] = st.selectbox(
                            column.replace("_", " ").title(),
                            values
                        )

            # --------------------------------------
            # Normal user input
            # --------------------------------------

            else:

                if is_numeric_dtype(train_data[column]):

                    min_value = train_data[column].min()
                    max_value = train_data[column].max()
                    default_value = train_data[column].median()

                    if pd.isna(default_value):
                        default_value = min_value

                    if (
                        pd.api.types.is_integer_dtype(train_data[column])
                        or float(default_value).is_integer()
                    ):

                        input_data[column] = st.number_input(
                            column.replace("_", " ").title(),
                            min_value=int(min_value),
                            max_value=int(max_value),
                            value=int(default_value),
                            step=1
                        )

                    else:

                        input_data[column] = st.number_input(
                            column.replace("_", " ").title(),
                            min_value=float(min_value),
                            max_value=float(max_value),
                            value=float(default_value)
                        )

                else:

                    values = (
                        train_data[column]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    values = sorted(values)

                    if len(values) > 0:

                        input_data[column] = st.selectbox(
                            column.replace("_", " ").title(),
                            values
                        )

                    else:

                        input_data[column] = ""


    st.divider()

    predict_button = st.form_submit_button(
        "🔍 Predict Damage",
        use_container_width=True
    )


# ==================================================
# PREDICTION
# ==================================================

if predict_button:

    try:

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure exact feature order
        input_df = input_df[feature_columns]

        # Make prediction
        prediction = int(
            model.predict(input_df)[0]
        )


        # --------------------------------------------------
        # Damage labels
        # --------------------------------------------------

        damage_labels = {
            1: "Low Damage",
            2: "Medium Damage",
            3: "Almost Complete Destruction"
        }

        damage_description = damage_labels.get(
            prediction,
            "Unknown Damage Level"
        )


        # --------------------------------------------------
        # Display result
        # --------------------------------------------------

        st.divider()

        st.header("🎯 Prediction Result")


        if prediction == 1:

            st.success(
                f"## Damage Grade: {prediction}\n\n"
                f"### 🟢 {damage_description}"
            )

        elif prediction == 2:

            st.warning(
                f"## Damage Grade: {prediction}\n\n"
                f"### 🟠 {damage_description}"
            )

        elif prediction == 3:

            st.error(
                f"## Damage Grade: {prediction}\n\n"
                f"### 🔴 {damage_description}"
            )

        else:

            st.info(
                f"## Damage Grade: {prediction}"
            )


        # --------------------------------------------------
        # Show entered information
        # --------------------------------------------------

        with st.expander(
            "📋 View Entered Building Information"
        ):

            st.dataframe(
                input_df,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "Prediction could not be completed."
        )

        st.write(
            "Error:",
            str(e)
        )
