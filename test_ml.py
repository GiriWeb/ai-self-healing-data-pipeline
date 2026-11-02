
import pandas as pd

from models.model_selection import train_suitable_models


# --------------------------------
# 1. Create Sample Vehicle Dataset
# --------------------------------

df = pd.DataFrame({
    "Brand": [
        "Honda", "Yamaha", "TVS", "Honda", "Suzuki",
        "Bajaj", "Yamaha", "TVS", "Honda", "Suzuki",
        "Bajaj", "Yamaha", "Honda", "TVS", "Suzuki",
        "Bajaj", "Honda", "Yamaha", "TVS", "Suzuki"
    ],

    "Year": [
        2020, 2021, 2019, 2022, 2020,
        2018, 2023, 2021, 2019, 2022,
        2020, 2018, 2023, 2022, 2019,
        2021, 2020, 2023, 2018, 2022
    ],

    "Mileage": [
        25000, 18000, 30000, 12000, 22000,
        35000, 10000, 16000, 28000, 14000,
        26000, 32000, 9000, 15000, 27000,
        19000, 24000, 11000, 34000, 13000
    ],

    "Engine_CC": [
        150, 125, 110, 160, 150,
        125, 155, 110, 150, 160,
        125, 110, 160, 110, 150,
        125, 150, 155, 110, 160
    ],

    "Vehicle_Value_INR": [
        65000, 72000, 45000, 85000, 60000,
        40000, 90000, 55000, 50000, 82000,
        48000, 38000, 95000, 70000, 52000,
        62000, 68000, 92000, 42000, 80000
    ]
})


# --------------------------------
# 2. Select Target Column
# --------------------------------

target_column = "Vehicle_Value_INR"


# --------------------------------
# 3. Automatic Model Selection
# --------------------------------

try:

    print("\nSTARTING INTELLIGENT ML MODEL SELECTION")

    output = train_suitable_models(
        df,
        target_column
    )

    # --------------------------------
    # 4. Display Problem Type
    # --------------------------------

    print("\nPROBLEM TYPE:")
    print(output["problem_type"])

    # --------------------------------
    # 5. Display Model Comparison
    # --------------------------------

    print("\nMODEL COMPARISON")

    print(output["results"].to_string(index=False))

    # --------------------------------
    # 6. Display Best Model
    # --------------------------------

    print("\nBEST MODEL:")

    print(output["best_model_name"])

    # --------------------------------
    # 7. Prediction Test
    # --------------------------------

    best_model = output["best_model"]

    sample_data = output["X_test"].iloc[[0]]

    prediction = best_model.predict(sample_data)

    print("\nSAMPLE PREDICTION:")

    print(prediction)

    print("\nML TEST SUCCESSFUL!")

except Exception as e:

    print("\nML TEST FAILED!")

    print("ERROR:", str(e))
