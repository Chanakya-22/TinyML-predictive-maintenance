Industrial Digital Twin and Predictive Maintenance System

Overview

This repository contains the implementation of a real-time predictive maintenance framework designed for industrial infrastructure. The system leverages edge computing principles and machine learning to monitor the health of rotating machinery. By analyzing high-frequency vibration and temperature telemetry, the application detects incipient mechanical faults, allowing for intervention prior to critical system failure.

Core Capabilities

Digital Twin Visualization: A live web interface mirroring the physical state of the monitored asset.

Explainable Diagnostic Engine: Rule-based heuristic logic layered over a machine learning classifier to provide actionable maintenance recommendations.

Physics-Based Simulation: A deterministic data generation pipeline that synthesizes accurate mechanical degradation signatures, including bearing wear and rotor unbalance.

Spectral Feature Engineering: Extraction of advanced signal characteristics such as Root Mean Square vibration, Kurtosis, and frequency-domain energy.

System Architecture

The architecture is divided into three primary modules:

Data Synthesis and Model Training: Python scripts utilizing NumPy and Scipy to generate training datasets and train a Gradient Boosting Classifier.

Edge Gateway Server: A Flask-based backend acting as the IoT gateway, managing state transitions and performing real-time inference.

Monitoring Dashboard: A frontend interface utilizing Chart.js for data visualization and asynchronous polling for continuous updates.

Execution Protocol

Prerequisites

Ensure a modern Python environment is active. Install the required dependencies using the provided configuration file.

pip install -r requirements.txt

Model Initialization

Generate the synthetic data profiles and compile the inference model.

python ml_model/train_final.py

Server Deployment

Initialize the local edge server to begin the diagnostic simulation.

python dashboard/app.py

Interface Access

Navigate to the local host address provided in the terminal output using a standard web browser to view the operational dashboard.