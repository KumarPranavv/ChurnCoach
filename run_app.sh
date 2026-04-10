#!/bin/bash

# 🎯 Quick Start Script for Churn Prediction App

echo "=================================="
echo "🎯 Churn Prediction Dashboard"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found in current directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if data file exists
if [ ! -f "data/WA_Fn-UseC_-Telco-Customer-Churn.csv" ]; then
    echo "❌ Error: Dataset not found"
    echo "Please ensure data/WA_Fn-UseC_-Telco-Customer-Churn.csv exists"
    exit 1
fi

echo "✅ Project files found"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✅ Python 3 is installed"
echo ""

# Install requirements
echo "📦 Installing required packages..."
pip install -q streamlit plotly scikit-learn xgboost lightgbm catboost pandas numpy

echo "✅ Packages installed successfully"
echo ""

# Run the app
echo "🚀 Starting Streamlit app..."
echo "=================================="
echo ""
echo "The app will open in your browser at:"
echo "👉 http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
