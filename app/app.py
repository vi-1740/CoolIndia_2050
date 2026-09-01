import os
import joblib
import pandas as pd
import streamlit as st
from datetime import date, timedelta

PROJECT_PATH = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

OUTPUT_PATH = os.path.join(
    PROJECT_PATH,
    "outputs"
)

DATA_PATH = os.path.join(
    PROJECT_PATH,
    "data",
    "monthly_temp_normal.csv"
)

lasso_model = joblib.load(
    os.path.join(
        OUTPUT_PATH,
        "lasso_model.pkl"
    )
)

ra_scaler = joblib.load(
    os.path.join(
        OUTPUT_PATH,
        "ra_scaler.pkl"
    )
)

monthly_temp_normal = pd.read_csv(
    DATA_PATH
)

monthly_temp_normal = monthly_temp_normal.rename(
    columns={
        "state": "electricity_name",
        "normal_temperature": "temp_normal"
    }
)

states = sorted(
    monthly_temp_normal[
        "electricity_name"
    ]
    .dropna()
    .unique()
)

st.set_page_config(
    page_title="PowerPulse India",
    page_icon="⚡",
    layout="wide"
)

st.title(
    "⚡ PowerPulse India"
)

st.subheader(
    "Future Electricity Demand Intelligence"
)

st.write(
    "Explore how a future temperature scenario may "
    "affect electricity demand compared with the "
    "historical normal level."
)

st.divider()

col1, col2 = st.columns(2)

with col1:

    state = st.selectbox(
        "📍 Select State",
        states
    )

with col2:

    tomorrow = (
        date.today()
        + timedelta(days=1)
    )

    future_date = st.date_input(
        "📅 Future Prediction Date",
        value=tomorrow,
        min_value=tomorrow
    )


max_temp = st.number_input(
    "🌡️ Expected Maximum Temperature (°C)",
    min_value=0.0,
    max_value=50.0,
    value=35.0,
    step=0.1
)


st.write("")

predict_button = st.button(
    "⚡ Analyze Future Demand",
    use_container_width=True
)

if predict_button:

    selected_date = pd.Timestamp(
        future_date
    )

    month = selected_date.month

    normal_row = monthly_temp_normal[
        (
            monthly_temp_normal[
                "electricity_name"
            ]
            == state
        )
        &
        (
            monthly_temp_normal[
                "month"
            ]
            == month
        )
    ]


    if normal_row.empty:

        st.error(
            "Historical temperature information is "
            "not available for the selected state "
            "and month."
        )

    else:

        normal_temperature = float(
            normal_row[
                "temp_normal"
            ].iloc[0]
        )

        temperature_anomaly = (
            max_temp
            - normal_temperature
        )

        squared_temperature_anomaly = (
            temperature_anomaly ** 2
        )

        heat_above_35 = max(
            max_temp - 35.0,
            0.0
        )

        recent_demand_anomaly = 0.0

        model_input = pd.DataFrame(
            [[
                max_temp,
                temperature_anomaly,
                squared_temperature_anomaly,
                heat_above_35,
                recent_demand_anomaly
            ]],
            columns=[
                "MaxTemp_C",
                "temp_anom",
                "temp2",
                "heat35",
                "lag_y"
            ]
        )

        scaled_input = ra_scaler.transform(
            model_input
        )

        prediction = float(
            lasso_model.predict(
                scaled_input
            )[0]
        )

        st.divider()

        st.subheader(
            "📊 Future Demand Outlook"
        )


        result1, result2, result3 = st.columns(3)


        with result1:

            st.metric(
                "Expected Demand Anomaly",
                f"{prediction:+.2f} MU"
            )

            st.caption(
                "MU means Million Units of electricity"
            )


        with result2:

            st.metric(
                "Expected Temperature",
                f"{max_temp:.1f} °C"
            )


        with result3:

            st.metric(
                "Historical Normal Temperature",
                f"{normal_temperature:.1f} °C"
            )


        st.caption(
            "A positive value indicates demand above "
            "the historical normal level. A negative "
            "value indicates demand below the historical "
            "normal level."
        )

        st.subheader(
            "💡 What does this mean?"
        )


        if prediction > 10:

            st.error(
                "🔴 High Demand Outlook"
            )

            st.write(
                "The selected temperature scenario is "
                "associated with substantially higher "
                "electricity demand than the historical "
                "normal level."
            )

            st.info(
                "Planning insight: electricity system "
                "planners may consider additional supply "
                "capacity and closer monitoring during "
                "potential peak demand conditions."
            )


        elif prediction > 2:

            st.warning(
                "🟡 Moderately Higher Demand"
            )

            st.write(
                "The selected temperature scenario is "
                "associated with moderately higher "
                "electricity demand than normal."
            )

            st.info(
                "Planning insight: increased monitoring "
                "of electricity demand may be useful "
                "under this temperature scenario."
            )


        elif prediction >= -2:

            st.success(
                "🟢 Demand Close to Normal"
            )

            st.write(
                "The selected temperature scenario is "
                "associated with electricity demand "
                "close to the historical normal level."
            )

            st.info(
                "Planning insight: demand conditions "
                "appear relatively close to the normal "
                "level for this scenario."
            )


        elif prediction >= -10:

            st.info(
                "🔵 Moderately Lower Demand"
            )

            st.write(
                "The selected temperature scenario is "
                "associated with moderately lower "
                "electricity demand than normal."
            )

            st.info(
                "Planning insight: electricity demand "
                "may remain below the historical normal "
                "level under this scenario."
            )


        else:

            st.info(
                "🔵 Low Demand Outlook"
            )

            st.write(
                "The selected temperature scenario is "
                "associated with substantially lower "
                "electricity demand than normal."
            )

            st.info(
                "Planning insight: demand conditions "
                "may remain considerably below the "
                "historical normal level."
            )

        st.subheader(
            "🌡️ Temperature Insight"
        )


        difference = (
            max_temp
            - normal_temperature
        )


        if difference > 0:

            st.write(
                f"The expected temperature is "
                f"**{difference:.1f} °C above** the "
                f"historical normal for {state} "
                f"during this month."
            )


        elif difference < 0:

            st.write(
                f"The expected temperature is "
                f"**{abs(difference):.1f} °C below** "
                f"the historical normal for {state} "
                f"during this month."
            )


        else:

            st.write(
                f"The expected temperature is equal to "
                f"the historical normal for {state} "
                f"during this month."
            )

        with st.expander(
            "🔬 View Technical Model Information"
        ):

            technical_data = pd.DataFrame(
                {
                    "Model Information": [
                        "Final Model",
                        "Model Test R²",
                        "State",
                        "Prediction Date",
                        "Expected Maximum Temperature",
                        "Temperature Anomaly",
                        "Squared Temperature Anomaly",
                        "Heat Above 35 °C",
                        "Recent Demand Assumption"
                    ],
                    "Value": [
                        "Lasso Regression",
                        "0.9042",
                        state,
                        selected_date.strftime(
                            "%Y-%m-%d"
                        ),
                        f"{max_temp:.2f} °C",
                        f"{temperature_anomaly:.2f} °C",
                        f"{squared_temperature_anomaly:.2f}",
                        f"{heat_above_35:.2f} °C",
                        "Normal demand level"
                    ]
                }
            )

            st.dataframe(
                technical_data,
                hide_index=True,
                use_container_width=True
            )

st.divider()

st.caption(
    "PowerPulse India is a research prototype based "
    "on a trained Lasso Regression model. The application "
    "uses historical temperature normals and a user "
    "defined future temperature scenario. A live "
    "operational forecasting system would require "
    "real time electricity demand data and reliable "
    "weather forecasts."
)
