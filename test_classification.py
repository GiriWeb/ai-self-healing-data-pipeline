
import pandas as pd

from models.model_selection import train_suitable_models


# 1. Sample Student Dataset

df = pd.DataFrame({
    "Study_Hours": [
        2, 5, 1, 7, 3, 8, 4, 6, 2, 9,
        5, 1, 7, 3, 8, 4, 6, 2, 9, 5,
        3, 7, 1, 8, 4, 6, 2, 9, 5, 3
    ],

    "Attendance": [
        50, 85, 40, 95, 65, 98, 75, 90, 55, 99,
        80, 35, 92, 60, 96, 70, 88, 45, 97, 82,
        62, 94, 30, 99, 72, 89, 52, 98, 84, 68
    ],

    "Previous_Marks": [
        45, 80, 35, 90, 60, 95, 70, 85, 50, 98,
        78, 30, 92, 55, 96, 68, 88, 42, 97, 82,
        58, 93, 25, 99, 72, 86, 48, 96, 83, 65
    ],

    "Result": [
        "Fail", "Pass", "Fail", "Pass", "Fail",
        "Pass", "Pass", "Pass", "Fail", "Pass",
        "Pass", "Fail", "Pass", "Fail", "Pass",
        "Pass", "Pass", "Fail", "Pass", "Pass",
        "Fail", "Pass", "Fail", "Pass", "Pass",
        "Pass", "Fail", "Pass", "Pass", "Pass"
    ]
})


# 2. Target Column

target_column = "Result"


# 3. Automatic Model Selection

try:

    print("\nSTARTING CLASSIFICATION MODEL SELECTION")

    output = train_suitable_models(
        df,
        target_column
    )

    # 4. Problem Type

    print("\nPROBLEM TYPE:")
    print(output["problem_type"])

    # 5. Model Comparison

    print("\nCLASSIFICATION MODEL COMPARISON")

    print(output["results"].to_string(index=False))

    # 6. Best Model

    print("\nBEST MODEL:")

    print(output["best_model_name"])

    # 7. Prediction

    best_model = output["best_model"]

    sample_data = output["X_test"].iloc[[0]]

    prediction = best_model.predict(sample_data)

    print("\nSAMPLE STUDENT RESULT:")

    print(prediction[0])

    print("\nCLASSIFICATION TEST SUCCESSFUL!")

except Exception as e:

    print("\nCLASSIFICATION TEST FAILED!")

    print("ERROR:", str(e))
