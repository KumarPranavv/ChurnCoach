# 🎯 Advanced Churn Prediction Dashboard

An interactive Streamlit web application for predicting customer churn using **advanced feature engineering** and **optimized gradient boosting models** with beautiful, modern visualizations.

## 🌟 Features

### � Advanced Data Processing
- **Advanced Feature Engineering**: 15+ engineered features including risk scores, interaction features, and composite metrics
- **SMOTE Balancing**: Handles class imbalance with SMOTE (Synthetic Minority Over-sampling Technique)
- **RobustScaler**: Better handling of outliers in feature scaling

### �📊 Model Comparison
- **Multiple ML Models**: XGBoost, LightGBM, CatBoost, Gradient Boosting
- **Optimized Hyperparameters**: Each model tuned for maximum performance
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Interactive Visualizations**: 
  - Bar charts for metric comparison
  - ROC curves for all models
  - Radar charts for performance overview
  - Model ranking visualizations

### 🎯 Predictions
- **Interactive Input Form**: Easy-to-use interface for entering customer data
- **Real-time Predictions**: Instant churn probability calculation
- **Model Selection**: Choose from any trained model
- **Visual Feedback**: 
  - Probability gauge
  - Risk level indicators
  - Retention recommendations

### 📈 Detailed Analysis
- **Confusion Matrices**: Visual representation of model predictions
- **Feature Importance**: Identify key factors affecting churn
- **Model-specific Metrics**: Deep dive into individual model performance

### 📋 Data Exploration
- **Dataset Preview**: Browse the raw data
- **Statistical Summary**: Understand data distributions
- **Churn Analysis**: Visual breakdown of churn patterns
- **Feature Distributions**: Explore numerical features by churn status

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager



## 📁 Project Structure

```
Churncoach/
├── app.py                                      # Main Streamlit application
├── requirements.txt                            # Python dependencies
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset
├── advanced_churn_prediction_90plus.ipynb     # Advanced model notebook
└── churn_model_comparison.ipynb                # Model comparison notebook
```

## 🎨 App Interface

### Tab 1: Model Comparison
- Performance metrics table with color-coded values
- Interactive bar charts comparing all metrics
- ROC curves for all models
- Performance radar chart
- Model ranking visualization

### Tab 2: Predictions
- Customer information input form
- Model selection dropdown
- Predict button for instant results
- Churn probability gauge
- Risk assessment and recommendations

### Tab 3: Detailed Analysis
- Model-specific confusion matrices
- Detailed metrics breakdown
- Feature importance analysis
- Top contributing features

### Tab 4: Data Exploration
- Dataset preview (first 100 rows)
- Statistical summary
- Churn distribution pie chart
- Feature distribution histograms

## 📊 Models Included

All models are trained with **SMOTE-balanced data** and **optimized hyperparameters**:

1. **XGBoost**
   - max_depth=6, learning_rate=0.1, n_estimators=300
   - Excellent performance on tabular data
   - Feature importance visualization

2. **LightGBM**
   - num_leaves=70, learning_rate=0.05, n_estimators=300
   - Fast training speed and high efficiency
   - Often achieves best accuracy (92%+)

3. **CatBoost**
   - depth=8, learning_rate=0.1, iterations=300
   - Handles categorical features efficiently
   - Robust to overfitting

4. **Gradient Boosting**
   - n_estimators=300, learning_rate=0.1, max_depth=4
   - Classic ensemble method
   - Reliable baseline performance

## 🎨 Enhanced UX Features

### Modern Design
- **Gradient Backgrounds**: Beautiful color gradients throughout the interface
- **Colorful Metric Cards**: Each metric displayed in vibrant, gradient cards
- **Smooth Animations**: Interactive hover effects and transitions
- **Professional Typography**: Clear hierarchy with bold, modern fonts

### Color Scheme
- **Primary Gradient**: Purple to Violet (#667eea to #764ba2)
- **Success**: Green gradient (#43e97b to #38f9d7)
- **Warning**: Pink to Red gradient (#f093fb to #f5576c)
- **Info**: Blue gradient (#4facfe to #00f2fe)
- **Accent**: Yellow gradient (#fa709a to #fee140)

### Visualization Style
- **No Grey on White**: All charts use vibrant colors with subtle backgrounds
- **Semi-transparent Backgrounds**: rgba(255, 255, 255, 0.9) for clean look
- **Colored Gridlines**: Subtle grey gridlines (rgba(200, 200, 200, 0.3))
- **Bold Text**: All chart text in dark blue (#1a365d, #2d3748)

## 🎯 How to Use

### Making Predictions

1. Navigate to the **🎯 Predictions** tab
2. Select your preferred model from the dropdown
3. Fill in customer information:
   - Demographics (gender, senior citizen, partner, dependents)
   - Service details (phone, internet, streaming services)
   - Account information (contract, billing, charges)
4. Click **🚀 Predict Churn**
5. Review the prediction results and recommendations

### Comparing Models

1. Go to the **📊 Model Comparison** tab
2. View the best performing model at the top
3. Explore interactive visualizations:
   - Hover over charts for detailed information
   - Compare metrics across all models
   - Analyze ROC curves and radar charts

### Analyzing Results

1. Switch to the **📈 Detailed Analysis** tab
2. Select a model for in-depth analysis
3. Examine the confusion matrix
4. Review feature importance rankings
5. Identify key factors driving predictions

## 🔧 Customization

### Adding New Models

To add a new model to the app:

1. Import the model in the imports section
2. Add training logic in the `train_all_models()` function
3. Calculate and store predictions and metrics
4. The visualizations will automatically include the new model

### Modifying Features

To change input features:

1. Update the feature inputs in Tab 2 (Predictions section)
2. Ensure the feature names match the dataset columns
3. Update the `input_data` dictionary accordingly

## 📈 Performance

The app automatically:
- Caches data loading and preprocessing
- Caches model training (models are trained once per session)
- Uses efficient Plotly visualizations
- Implements responsive layouts

## 🐛 Troubleshooting

### Issue: CatBoost not available
**Solution**: Install CatBoost separately:
```bash
pip install catboost
```

### Issue: Models training slowly
**Solution**: The first run trains all models, which may take time. Subsequent interactions use cached models.

### Issue: Data file not found
**Solution**: Ensure the data file is in the `data/` directory:
```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## 📝 Dataset Information

- **Source**: Telco Customer Churn Dataset
- **Samples**: 7,043 customers
- **Features**: 20 (including demographics, services, and account information)
- **Target**: Churn (Yes/No)
- **Churn Rate**: ~27%

## 🎓 Model Training

Models are trained with:
- **Train-Test Split**: 80-20
- **Stratified Sampling**: Maintains churn distribution
- **Feature Scaling**: StandardScaler for numerical features
- **Hyperparameter Tuning**: Optimized parameters for each model

## 💡 Tips

- **Best Accuracy**: Check the top metric in the Model Comparison tab
- **Risk Assessment**: Use the probability gauge for quick insights
- **Feature Analysis**: Review feature importance to understand key drivers
- **Data Quality**: Higher tenure and contract length typically reduce churn

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the notebook files for model details
3. Ensure all dependencies are installed correctly

## 🎉 Enjoy!

Start exploring customer churn patterns and making data-driven retention decisions!
