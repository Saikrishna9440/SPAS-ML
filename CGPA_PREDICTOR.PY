import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    "sem1": [8.5, 7.8, 7.2, 8.0, 7.5, 6.9, 7.3, 8.1, 7.9, 7.0],
    "sem2": [8.4, 7.7, 7.0, 7.9, 7.4, 6.8, 7.2, 8.0, 7.8, 6.9],
    "sem3": [8.3, 7.6, 6.9, 7.8, 7.3, 6.7, 7.1, 7.9, 7.7, 6.8],
    "sem4": [8.2, 7.5, 6.8, 7.7, 7.2, 6.6, 7.0, 7.8, 7.6, 6.7],
    "gender": [1, 0, 1, 1, 0, 0, 1, 1, 0, 0],  # 1 for Male, 0 for Female
    "sem5": [8.1, 7.4, 6.7, 7.6, 7.1, 6.5, 6.9, 7.7, 7.5, 6.6]  # Target Variable
}

df = pd.DataFrame(data)
X = df[["sem1", "sem2", "sem3", "sem4", "gender"]] #train
y = df["sem5"] #test
reg = LinearRegression()
reg.fit(X,y)

sem1 = float(input("Enter your 1st Semester CGPA: "))
sem2 = float(input("Enter your 2nd Semester CGPA: "))
sem3 = float(input("Enter your 3rd Semester CGPA: "))
sem4 = float(input("Enter your 4th Semester CGPA: "))
gender = int(input("Enter 1 for 'Male' or 0 for 'Female': "))

# Make prediction
predicted_cgpa = reg.predict([[sem1, sem2, sem3, sem4, gender]])

print(f"Our Prediction for your 5th Semester CGPA is: {predicted_cgpa[0]:.2f}")
