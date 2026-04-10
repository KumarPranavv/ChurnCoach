# 🎯 Streamlit App Usage Guide

## ✅ App is Successfully Running!

Your Churn Prediction Dashboard is now live at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://172.20.10.3:8501

---

## 📱 How to Use the App

### 1️⃣ **Model Comparison Tab**

This is your main dashboard for comparing all trained models.

**What you'll see:**
- 🏆 Best performing model highlighted at the top
- 📊 Performance metrics table with color-coded values
- 📈 Interactive charts:
  - Bar chart comparing all metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
  - Model ranking visualization
  - ROC curves for all models
  - Performance radar chart

**How to interact:**
- Hover over charts to see detailed values
- Click on legend items to show/hide specific models
- The table automatically highlights best performing metrics in green

---

### 2️⃣ **Predictions Tab** 

Make real-time churn predictions for individual customers.

**Steps to predict:**

1. **Select a Model**: Choose from the dropdown (Logistic Regression, XGBoost, CatBoost, LightGBM)

2. **Enter Customer Information**:
   - **Column 1**: Demographics (Gender, Senior Citizen, Partner, Dependents, Phone Services)
   - **Column 2**: Internet & Services (Internet Service, Security, Backup, etc.)
   - **Column 3**: Account Details (Contract, Billing, Charges, Tenure)

3. **Click "🚀 Predict Churn"**

4. **Review Results**:
   - ✅ **Low Risk** (Green) or ⚠️ **High Risk** (Red) indicator
   - **Churn Probability**: Percentage likelihood of customer leaving
   - **Retain Probability**: Percentage likelihood of customer staying
   - **Interactive Gauge**: Visual representation of risk level
     - Green (0-30%): Low risk
     - Purple (30-70%): Medium risk
     - Red (70-100%): High risk

5. **Get Recommendations**: If high risk is detected, you'll see actionable retention strategies

**Example Scenarios:**

**Low Risk Customer:**
- Long tenure (36+ months)
- Two-year contract
- Multiple services subscribed
- Automatic payment method

**High Risk Customer:**
- Short tenure (< 6 months)
- Month-to-month contract
- Few services
- Electronic check payment
- High monthly charges

---

### 3️⃣ **Detailed Analysis Tab**

Deep dive into specific model performance.

**Features:**
1. **Select Model**: Choose any model for detailed analysis
2. **Confusion Matrix**: 
   - See actual vs predicted classifications
   - Identify false positives and false negatives
3. **Metrics Breakdown**: View all 5 key metrics for the selected model
4. **Feature Importance**: 
   - Top 15 most important features
   - Color-coded by importance level
   - Helps understand what drives predictions

**How to read the confusion matrix:**
- **Top-left**: Correctly predicted NO churn
- **Top-right**: False positives (predicted churn but didn't)
- **Bottom-left**: False negatives (didn't predict churn but did)
- **Bottom-right**: Correctly predicted churn

---

### 4️⃣ **Data Exploration Tab**

Understand your dataset better.

**What's included:**
1. **Dataset Preview**: First 100 rows of raw data
2. **Statistical Summary**: Mean, std, min, max for all numerical features
3. **Churn Distribution**: 
   - Pie chart showing churn percentage
   - Bar chart showing counts
4. **Feature Distribution**: 
   - Select any numerical feature
   - View distribution by churn status
   - Box plot shows outliers

**Tips:**
- Look for patterns in churned vs non-churned customers
- Check tenure distribution - typically longer tenure = lower churn
- Monthly charges often correlate with churn risk

---

## 🎨 Interactive Features

### Charts are Interactive!
- **Hover**: See exact values
- **Zoom**: Click and drag on any chart
- **Pan**: Hold shift and drag
- **Reset**: Double-click
- **Download**: Use camera icon to save chart as PNG

### Responsive Design
- Works on desktop, tablet, and mobile
- Sidebar collapses on smaller screens
- Charts resize automatically

---

## 📊 Understanding the Metrics

### **Accuracy**
- Percentage of correct predictions (both churn and no-churn)
- Higher is better
- **Good**: > 80%

### **Precision**
- Of all predicted churns, how many were correct
- Important when false alarms are costly
- **Good**: > 75%

### **Recall (Sensitivity)**
- Of all actual churns, how many did we catch
- Important when missing churns is costly
- **Good**: > 70%

### **F1-Score**
- Harmonic mean of Precision and Recall
- Balanced metric
- **Good**: > 75%

### **ROC-AUC**
- Area Under ROC Curve
- Model's ability to distinguish classes
- **Good**: > 0.85
- **Excellent**: > 0.90

---

## 🎯 Best Practices

### For Predictions:
1. ✅ Use accurate, complete data
2. ✅ Try multiple models to compare results
3. ✅ Consider the probability score, not just yes/no
4. ✅ Act on high-risk customers proactively

### For Model Selection:
1. ✅ Check Model Comparison tab first
2. ✅ Consider your priority:
   - **Accuracy**: Overall correctness
   - **Recall**: Catching all churners (fewer missed)
   - **Precision**: Avoiding false alarms (fewer wrong predictions)
3. ✅ XGBoost and LightGBM typically perform best
4. ✅ Logistic Regression is fastest but may be less accurate

### For Analysis:
1. ✅ Review feature importance to understand key drivers
2. ✅ Check confusion matrix for model weaknesses
3. ✅ Compare multiple models on the same customer
4. ✅ Look for patterns in misclassified cases

---

## 💡 Pro Tips

### 🔥 Quick Insights
- Models train automatically on first load (takes ~30 seconds)
- Subsequent predictions are instant
- Refresh the page to retrain models with updated logic
- All visualizations are cached for fast performance

### 📈 Interpreting Results
- **High probability churn (>70%)**: Immediate intervention needed
- **Medium risk (30-70%)**: Monitor and engage
- **Low risk (<30%)**: Standard retention programs

### 🎯 Action Items Based on Predictions

**If Churn Probability > 70%:**
- 🎁 Offer immediate discounts or loyalty rewards
- 📞 Schedule personalized outreach call
- 🔄 Propose contract upgrade with benefits
- 💬 Send satisfaction survey

**If Churn Probability 30-70%:**
- 📧 Send engagement emails
- 🎯 Include in next marketing campaign
- 📊 Monitor usage patterns
- 🤝 Offer value-add services

**If Churn Probability < 30%:**
- ✅ Continue standard service
- 📈 Consider upsell opportunities
- 🌟 Use for case studies/testimonials

---

## 🔧 Customization Options

### In the Sidebar:
- **Dataset Overview**: See real-time stats
- **About Section**: App information and credits

### Model Training:
- Models are cached after first training
- To retrain, refresh the browser page
- Training uses 80-20 train-test split
- All models use optimized hyperparameters

---

## 📱 Mobile Usage

The app is fully responsive! On mobile:
1. Tap the arrow (>) to expand sidebar
2. Swipe left/right between tabs
3. Tap charts to interact
4. Pinch to zoom on visualizations

---

## 🐛 Troubleshooting

### App won't load?
- Check that you're at http://localhost:8501
- Try refreshing the page
- Restart the app: `streamlit run app.py`

### Models training slowly?
- First load trains all models (~30 seconds)
- Subsequent loads use cached models (instant)
- Patience on first run!

### Data not found?
- Ensure `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` exists
- Check file path is correct

### Visualizations not showing?
- Check browser console for errors
- Try a different browser (Chrome/Firefox recommended)
- Disable ad blockers

---

## 🎓 Learning More

### About the Models:

**Logistic Regression**
- Fast, interpretable
- Good baseline model
- Works well with linear relationships

**XGBoost**
- Powerful gradient boosting
- Handles non-linear patterns
- Often best accuracy

**CatBoost**
- Great with categorical features
- Robust to overfitting
- Fast training

**LightGBM**
- Very fast training
- Memory efficient
- Excellent for large datasets

### About the Features:

**Most Important Features** (typically):
1. **Tenure**: Longer tenure = lower churn
2. **Contract**: Two-year contracts churn less
3. **Monthly Charges**: Higher charges = higher risk
4. **Internet Service**: Fiber optic has higher churn
5. **Payment Method**: Electronic check = higher risk

---

## 🎉 Next Steps

1. ✅ Explore all four tabs
2. ✅ Make test predictions with different inputs
3. ✅ Compare model performance
4. ✅ Review feature importance
5. ✅ Use insights for business decisions

---

## 📞 Need Help?

- Review the README.md for installation issues
- Check the notebooks for model training details
- Ensure all dependencies are installed
- Try the example predictions first

---

**Enjoy predicting customer churn! 🎯**
