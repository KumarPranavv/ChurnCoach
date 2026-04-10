# 🎯 Churn Prediction Streamlit App - Project Summary



---

## 📁 Files Created

### Main Application
- **`app.py`** - Complete Streamlit application (800+ lines)

### Documentation
- **`README.md`** - Comprehensive project documentation
- **`APP_GUIDE.md`** - Detailed user guide with tips and best practices
- **`requirements.txt`** - Python dependencies
- **`run_app.sh`** - Quick start script (executable)

---

## 🎨 App Features

### 📊 **Tab 1: Model Comparison**
✅ Performance metrics for all models (Accuracy, Precision, Recall, F1, ROC-AUC)
✅ Interactive bar charts comparing metrics
✅ ROC curves for all models
✅ Performance radar chart
✅ Model ranking visualization
✅ Color-coded metrics table
✅ Best model highlighted

### 🎯 **Tab 2: Predictions**
✅ Dropdown to select model (Logistic Regression, XGBoost, CatBoost, LightGBM)
✅ User-friendly input form with 19 customer features
✅ Organized in 3 columns for easy data entry
✅ **Predict button** for instant predictions
✅ Visual results display:
   - Risk level indicator (High/Low)
   - Churn probability percentage
   - Retain probability percentage
   - Interactive probability gauge with color zones
✅ Actionable retention recommendations for high-risk customers

### 📈 **Tab 3: Detailed Analysis**
✅ Model selection for deep-dive analysis
✅ Confusion matrix heatmap
✅ Detailed metrics breakdown
✅ Feature importance visualization (Top 15 features)
✅ Interactive Plotly charts

### 📋 **Tab 4: Data Exploration**
✅ Dataset preview (first 100 rows)
✅ Statistical summary
✅ Churn distribution (pie chart & bar chart)
✅ Feature distribution by churn status
✅ Interactive histograms with box plots

---

## 🤖 Models Included

All models are trained automatically on app startup:

1. **Logistic Regression**
   - Fast baseline model
   - C=1.0, max_iter=1000

2. **XGBoost**
   - max_depth=5, learning_rate=0.1
   - n_estimators=200

3. **CatBoost**
   - depth=6, learning_rate=0.1
   - iterations=200

4. **LightGBM**
   - num_leaves=50, learning_rate=0.05
   - n_estimators=200, max_depth=7

---

## 📊 Visualizations (All with Plotly)

### Interactive Charts:
1. **Grouped Bar Chart** - Metrics comparison across models
2. **Horizontal Bar Chart** - Model ranking by accuracy
3. **ROC Curves** - All models overlaid with AUC scores
4. **Radar Chart** - 360° performance view
5. **Confusion Matrix** - Heatmap for predictions
6. **Feature Importance** - Color-coded horizontal bars
7. **Gauge Chart** - Churn probability indicator
8. **Pie Chart** - Churn distribution
9. **Histogram** - Feature distributions with box plots

### Visualization Features:
- ✨ Hover tooltips with detailed info
- 🔍 Zoom and pan capabilities
- 📸 Download as PNG
- 🎨 Custom color schemes
- 📱 Responsive design

---

## 🎯 Key Features

### Performance Optimizations:
- ✅ Data caching (`@st.cache_data`)
- ✅ Model caching (`@st.cache_resource`)
- ✅ Session state for trained models
- ✅ Efficient data preprocessing
- ✅ Lazy loading of visualizations

### User Experience:
- ✅ Clean, modern UI with custom CSS
- ✅ Gradient backgrounds and card designs
- ✅ Responsive layout (works on mobile)
- ✅ Loading spinners for better UX
- ✅ Color-coded metrics and alerts
- ✅ Informative sidebar
- ✅ Organized tabs for easy navigation

### Data Processing:
- ✅ Automatic handling of missing values
- ✅ Label encoding for categorical variables
- ✅ Standard scaling for numerical features
- ✅ 80-20 train-test split with stratification
- ✅ Proper data preprocessing pipeline

---

## 📈 Metrics Displayed

For each model:
1. **Accuracy** - Overall correctness
2. **Precision** - Positive prediction accuracy
3. **Recall** - True positive rate
4. **F1-Score** - Harmonic mean of precision & recall
5. **ROC-AUC** - Area under ROC curve

All metrics shown in:
- ✅ Table format (sortable, color-coded)
- ✅ Bar charts (grouped comparison)
- ✅ Radar chart (holistic view)
- ✅ Individual metric cards

---

## 🎨 Design Highlights

### Color Scheme:
- **Success/Low Risk**: Green (#43e97b)
- **Warning/Medium**: Purple (#f093fb)
- **Danger/High Risk**: Red (#fa709a)
- **Primary**: Blue (#667eea)
- **Accent**: Orange (#f39c12)

### Typography:
- Clean, modern fonts
- Bold headers for clarity
- Proper hierarchy
- Readable sizes

### Layout:
- Wide layout for data visibility
- Sidebar for configuration
- Metrics in rows of 5
- Charts in responsive columns

---

## 🚀 How to Use

### Starting the App:
```bash
# Option 1: Use the script
./run_app.sh

# Option 2: Run directly
streamlit run app.py

# Option 3: From anywhere
cd "/Users/pranav/Desktop/Copy of Churncoach"
streamlit run app.py
```

### Making Predictions:
1. Go to "🎯 Predictions" tab
2. Select a model from dropdown
3. Fill in customer information (19 fields)
4. Click "🚀 Predict Churn"
5. Review results and recommendations

### Comparing Models:
1. Go to "📊 Model Comparison" tab
2. View best model at top
3. Explore interactive charts
4. Hover for detailed metrics

---

## 📦 Dependencies Installed

All required packages are installed:
- ✅ streamlit (1.51.0)
- ✅ pandas (2.3.3)
- ✅ numpy (2.3.4)
- ✅ plotly (6.4.0)
- ✅ scikit-learn
- ✅ xgboost
- ✅ lightgbm
- ✅ catboost

---

## 🎓 Dataset Information

- **Source**: Telco Customer Churn Dataset
- **Total Samples**: 7,043 customers
- **Features**: 20 (19 input features + 1 target)
- **Target**: Churn (Yes/No)
- **Churn Rate**: ~27%
- **Feature Types**: 
  - Numerical: tenure, MonthlyCharges, TotalCharges
  - Categorical: gender, Contract, PaymentMethod, etc.

---

## 💡 Business Value

### For Data Scientists:
- Quick model comparison
- Feature importance analysis
- Performance metrics at a glance
- Easy experimentation

### For Business Users:
- Simple prediction interface
- Visual risk indicators
- Actionable recommendations
- No coding required

### For Stakeholders:
- Clear performance metrics
- ROI-focused insights
- Data-driven decisions
- Interactive exploration

---

## 🎯 Next Steps

### Immediate:
1. ✅ App is running - explore it!
2. ✅ Try making predictions
3. ✅ Compare model performance
4. ✅ Review feature importance

### Future Enhancements:
- 📊 Add more models (Neural Networks, Random Forest)
- 📈 Implement A/B testing
- 💾 Save predictions to database
- 📧 Email alert integration
- 📱 Mobile app version
- 🔐 User authentication
- 📊 Custom date range analysis
- 🎯 Batch predictions from CSV upload

---

## 📞 Support

### Documentation:
- **README.md** - Installation and setup
- **APP_GUIDE.md** - Detailed usage guide
- **requirements.txt** - Dependencies

### Troubleshooting:
- Ensure data file is in `data/` directory
- Check all dependencies are installed
- Refresh browser if models don't load
- First load takes ~30 seconds (model training)


---

## 🎉 **APP IS LIVE AND READY TO USE!**



The app will automatically:
1. Load the dataset
2. Preprocess the data
3. Train all 4 models
4. Display comprehensive visualizations
5. Enable instant predictions

**Enjoy your Churn Prediction Dashboard! 🎯**
