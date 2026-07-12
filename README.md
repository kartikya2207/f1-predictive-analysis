# F1_Predictive_Analysis

Joint project with Shivakshi (github.com/Shivakshi26). Original repo: https://github.com/Shivakshi26/F1_Predictive_Analysis

F1 Predictive Analysis

A machine learning project that predicts Formula 1 driver points for upcoming races based on historical performance, track characteristics, and weather conditions.

Overview

This project uses historical Formula 1 data (from 1950 to 2022) to build a Linear Regression model that estimates the points a driver will score in a specific race. The model engineers features like cumulative driver skill, track difficulty (DNF rates, pit stop rates), and constructor reliability to make its predictions.

Dataset

The data is sourced from the Formula 1 World Championship (1950 - 2020) dataset on Kaggle, utilizing the kagglehub library for automated downloading.

Key Data Files Used:

results.csv - Race results and points.

races.csv - Race information (year, circuit).

pit_stops.csv - Pit stop durations and counts.

constructor_standings.csv - Team performance metrics.

qualifying.csv - Starting grid positions.

weather.csv - (External/Derived) Weather categories for races.

Feature Engineering

The model relies on several custom-engineered features to capture the complexity of F1 racing:
avg_driver_points: A proxy for "Absolute Skill," calculated as the driver's cumulative points divided by races participated in up to that date.

constructor_standing_points: Represents the current strength of the driver's team (car performance).

grid: The driver's starting position (Qualifying performance).

total_pit_stop_time: Total time spent in the pits during the race (Strategy/Efficiency).

avg_pit_stops_per_race: A proxy for track layout difficulty and tire degradation rates.

avg_dnf_rate: A proxy for track risk (likelihood of crashing or mechanical failure).

weather_category: Categorical data representing race conditions (e.g., Dry vs. Wet).

Model & Methodology

Algorithm: Linear Regression (sklearn.linear_model.LinearRegression).

Splitting Strategy: Time-series split (not random shuffle) to prevent data leakage.

Example Train: 2011 – 2017

Example Test: 2018 – 2020

Evaluation Metrics:

R² Score: ~0.50 (Explains ~50% of the variance in points).

RMSE (Root Mean Squared Error): ~5.16 points.

MAE (Mean Absolute Error): ~3.60 points.

Results Visualization

The project includes visualizations using matplotlib and seaborn:

Actual vs. Predicted Points: A comparison plot tracking model performance across the test set.

Residual Histogram: Analysis of prediction errors (normal distribution check).
