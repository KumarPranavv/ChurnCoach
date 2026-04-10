"""
🎯 ChurnCoach — AI-Powered Customer Churn Prediction Dashboard
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
from lightgbm import LGBMClassifier

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from rag_engine import RAGEngine
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnCoach · AI Churn Prediction",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero header ──────────────────────────────────────── */
.hero-container {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,.25);
}
.hero-container h1 {
    color: #ffffff !important;
    font-size: 2.6rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-container p {
    color: #a0d2db;
    font-size: 1.1rem;
    margin: .5rem 0 0;
}

/* ── Metric cards ─────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #1e293b, #334155);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,.15);
    border: 1px solid rgba(255,255,255,.06);
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 700;
    margin: .3rem 0;
}
.metric-card .metric-label {
    font-size: .85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-green  .metric-value { color: #22c55e; }
.metric-red    .metric-value { color: #ef4444; }
.metric-blue   .metric-value { color: #3b82f6; }
.metric-amber  .metric-value { color: #f59e0b; }
.metric-purple .metric-value { color: #a855f7; }

/* ── Section headers ──────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 2rem 0 1rem;
    padding-bottom: .6rem;
    border-bottom: 2px solid rgba(255,255,255,.08);
}
.section-header .icon { font-size: 1.5rem; }
.section-header h2 {
    margin: 0; font-size: 1.35rem; font-weight: 700;
    color: #e2e8f0;
}

/* ── Input form ───────────────────────────────────────── */
.form-section-title {
    font-size: .8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #64748b;
    margin-bottom: .4rem;
    padding-left: 2px;
}

/* ── Risk badge ───────────────────────────────────────── */
.risk-badge {
    display: inline-block;
    padding: .35rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: .9rem;
    letter-spacing: .5px;
}
.risk-high   { background: rgba(239,68,68,.2); color: #fca5a5; border: 1px solid rgba(239,68,68,.4); }
.risk-medium { background: rgba(245,158,11,.2); color: #fcd34d; border: 1px solid rgba(245,158,11,.4); }
.risk-low    { background: rgba(34,197,94,.2);  color: #86efac; border: 1px solid rgba(34,197,94,.4);  }

/* ── AI recommendation card ───────────────────────────── */
.ai-rec-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid rgba(139,92,246,.3);
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin: 1rem 0;
    box-shadow: 0 4px 24px rgba(139,92,246,.08);
}
.ai-rec-card h3, .ai-rec-card h4 {
    color: #c4b5fd;
}

/* ── Predict button ───────────────────────────────────── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: .9rem 2.2rem !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    width: 100% !important;
    letter-spacing: .5px;
    transition: all .25s ease;
    box-shadow: 0 4px 18px rgba(99,102,241,.35);
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99,102,241,.5) !important;
}

/* ── Sidebar ──────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b) !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label {
    color: #cbd5e1 !important;
}

/* ── Hide Streamlit branding ──────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Data overview cards ──────────────────────────────── */
.data-card {
    background: linear-gradient(135deg, #1e293b, #334155);
    border-radius: 14px;
    padding: 1.3rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,.06);
    transition: transform .2s, box-shadow .2s;
}
.data-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0,0,0,.3);
}
.data-card .number {
    font-size: 1.8rem;
    font-weight: 700;
    color: #38bdf8;
}
.data-card .label {
    font-size: .8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Tabs ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(15, 23, 42, 0.5);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 8px 20px;
    font-weight: 600;
    color: #94a3b8;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
}

/* ── Profile cards ────────────────────────────────────── */
.profile-card {
    background: linear-gradient(135deg, #1e293b, #334155);
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    border: 1px solid rgba(255,255,255,.06);
    min-height: 200px;
}
.profile-card h4 {
    color: #38bdf8;
    margin: 0 0 .8rem 0;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: .5px;
}
.profile-card p {
    line-height: 1.8;
}

/* ── Service pills ────────────────────────────────────── */
.svc-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 8px;
    font-size: .78rem;
    font-weight: 600;
    margin: 2px;
}
.svc-yes { background: rgba(34,197,94,.15); color: #86efac; border: 1px solid rgba(34,197,94,.3); }
.svc-no  { background: rgba(239,68,68,.1);  color: #fca5a5; border: 1px solid rgba(239,68,68,.2); }
</style>
""", unsafe_allow_html=True)


# ─── RAG Engine (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def init_rag_engine():
    if not RAG_AVAILABLE:
        return None
    try:
        return RAGEngine(index_dir="faiss_index")
    except Exception:
        return None

rag_engine = init_rag_engine()


# ─── Sidebar ─────────────────────────────────────────────────────────────────

# Auto-load API key from .env file
def _load_env_key():
    """Read GEMINI_API_KEY from .env if available."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == "GEMINI_API_KEY":
                        return v.strip()
    return os.environ.get("GEMINI_API_KEY", "")

_env_api_key = _load_env_key()

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    st.markdown("**🤖 AI Recommendation Engine**")
    llm_provider = st.selectbox(
        "LLM Provider",
        ["gemini", "anthropic", "openai"],
        index=0,
        help="Gemini (Google) · Claude (Anthropic) · GPT-4o-mini (OpenAI)",
    )
    llm_api_key = st.text_input(
        "API Key",
        value=_env_api_key,
        type="password",
        help="Auto-loaded from .env file. You can also paste a different key here.",
    )
    if llm_api_key:
        st.success("🔑 API key set", icon="✅")
    else:
        st.info("Enter API key for AI recommendations. Without it, rule-based fallback is used.")

    st.markdown("---")
    if rag_engine is not None:
        st.markdown("📚 **Knowledge base** · loaded")
    else:
        st.markdown("⚠️ **RAG unavailable** · rule-based mode")
    st.markdown("---")
    st.caption("ChurnCoach v2.0 · Built with ❤️")


# ─── Feature Engineering ─────────────────────────────────────────────────────
def create_advanced_features(df):
    df_processed = df.copy()
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    median_tc = df_processed['TotalCharges'].median()
    df_processed['TotalCharges'] = df_processed['TotalCharges'].fillna(median_tc)
    if 'customerID' in df_processed.columns:
        df_processed = df_processed.drop('customerID', axis=1)

    df_processed['is_new_customer'] = (df_processed['tenure'] <= 12).astype(int)
    df_processed['is_loyal_customer'] = (df_processed['tenure'] >= 48).astype(int)
    df_processed['tenure_squared'] = df_processed['tenure'] ** 2
    df_processed['charges_per_month'] = df_processed['TotalCharges'] / (df_processed['tenure'] + 1)
    df_processed['total_charges_log'] = np.log1p(df_processed['TotalCharges'])
    df_processed['monthly_charges_log'] = np.log1p(df_processed['MonthlyCharges'])
    df_processed['high_charges'] = (df_processed['MonthlyCharges'] > df_processed['MonthlyCharges'].median()).astype(int)
    df_processed['charges_ratio'] = df_processed['MonthlyCharges'] / (df_processed['TotalCharges'] + 1)

    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_processed['total_services'] = 0
    for col in service_cols:
        if col in df_processed.columns:
            df_processed['total_services'] += (df_processed[col] == 'Yes').astype(int)

    df_processed['has_internet'] = (df_processed['InternetService'] != 'No').astype(int)
    df_processed['has_phone'] = (df_processed['PhoneService'] == 'Yes').astype(int)
    df_processed['has_streaming'] = ((df_processed['StreamingTV'] == 'Yes') |
                                      (df_processed['StreamingMovies'] == 'Yes')).astype(int)
    df_processed['has_security'] = ((df_processed['OnlineSecurity'] == 'Yes') |
                                     (df_processed['DeviceProtection'] == 'Yes')).astype(int)
    df_processed['is_senior_with_partner'] = (df_processed['SeniorCitizen'] *
                                               (df_processed['Partner'] == 'Yes').astype(int))
    df_processed['has_family'] = ((df_processed['Partner'] == 'Yes') |
                                   (df_processed['Dependents'] == 'Yes')).astype(int)
    df_processed['senior_alone'] = ((df_processed['SeniorCitizen'] == 1) &
                                     (df_processed['Partner'] == 'No')).astype(int)
    df_processed['has_long_contract'] = (df_processed['Contract'] != 'Month-to-month').astype(int)
    df_processed['paperless_autopay'] = ((df_processed['PaperlessBilling'] == 'Yes') &
                                          (df_processed['PaymentMethod'].str.contains('automatic', case=False, na=False))).astype(int)
    df_processed['churn_risk_score'] = (
        (df_processed['Contract'] == 'Month-to-month').astype(int) * 3 +
        df_processed['is_new_customer'] * 2 +
        df_processed['high_charges'] * 1 +
        (df_processed['has_security'] == 0).astype(int) * 2 +
        df_processed['senior_alone'] * 1
    )

    # ── Interaction features ─────────────────────────────────────────────
    df_processed['tenure_x_monthly'] = df_processed['tenure'] * df_processed['MonthlyCharges']
    df_processed['commitment_x_tenure'] = df_processed['has_long_contract'] * df_processed['tenure']
    df_processed['services_x_charges'] = df_processed['total_services'] * df_processed['MonthlyCharges']
    df_processed['security_x_internet'] = df_processed['has_security'] * df_processed['has_internet']
    df_processed['new_no_contract'] = df_processed['is_new_customer'] * (1 - df_processed['has_long_contract'])
    df_processed['tenure_group'] = pd.cut(
        df_processed['tenure'], bins=[-1, 6, 12, 24, 48, 72], labels=False)
    df_processed['avg_monthly_spend'] = df_processed['TotalCharges'] / (
        df_processed['tenure'].clip(lower=1))

    return df_processed


@st.cache_data
def load_data():
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df_processed = create_advanced_features(df)
    y = df_processed['Churn'].map({'Yes': 1, 'No': 0})
    X = df_processed.drop('Churn', axis=1)
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    return df, X, y, label_encoders


@st.cache_resource
def train_models(_X_train, _X_test, _y_train, _y_test):
    models, predictions, metrics = {}, {}, {}

    # Class-weight ratio for handling imbalance without SMOTE
    _ratio = float((_y_train == 0).sum()) / float((_y_train == 1).sum())

    # ── XGBoost (optimised) ──────────────────────────────────────────────
    xgb_model = xgb.XGBClassifier(
        n_estimators=800, max_depth=5, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7,
        reg_lambda=5, reg_alpha=1, min_child_weight=10, gamma=0.3,
        scale_pos_weight=_ratio * 0.4,
        random_state=42, eval_metric='logloss', tree_method='hist')
    xgb_model.fit(_X_train, _y_train)
    models['XGBoost'] = xgb_model
    predictions['XGBoost'] = {
        'y_pred': xgb_model.predict(_X_test),
        'y_pred_proba': xgb_model.predict_proba(_X_test)[:, 1]}

    # ── LightGBM (optimised) ─────────────────────────────────────────────
    lgbm_model = LGBMClassifier(
        n_estimators=800, num_leaves=20, max_depth=5,
        learning_rate=0.02, subsample=0.8, colsample_bytree=0.7,
        reg_lambda=5, reg_alpha=1, min_child_samples=60,
        scale_pos_weight=_ratio * 0.4,
        random_state=42, verbose=-1, force_col_wise=True)
    lgbm_model.fit(_X_train, _y_train)
    models['LightGBM'] = lgbm_model
    predictions['LightGBM'] = {
        'y_pred': lgbm_model.predict(_X_test),
        'y_pred_proba': lgbm_model.predict_proba(_X_test)[:, 1]}

    # ── CatBoost (if available) ──────────────────────────────────────────
    if CATBOOST_AVAILABLE:
        cat_model = CatBoostClassifier(
            learning_rate=0.02, l2_leaf_reg=5, iterations=800,
            depth=6, border_count=128, bagging_temperature=0.5,
            auto_class_weights='Balanced',
            random_state=42, verbose=0)
        cat_model.fit(_X_train, _y_train)
        models['CatBoost'] = cat_model
        predictions['CatBoost'] = {
            'y_pred': cat_model.predict(_X_test),
            'y_pred_proba': cat_model.predict_proba(_X_test)[:, 1]}

    # ── GradientBoosting (optimised) ─────────────────────────────────────
    _sw = compute_sample_weight({0: 1.0, 1: _ratio * 0.4}, _y_train)
    gb_model = GradientBoostingClassifier(
        n_estimators=800, max_depth=4, learning_rate=0.02,
        min_samples_split=10, min_samples_leaf=8, max_features='sqrt',
        subsample=0.8, random_state=42)
    gb_model.fit(_X_train, _y_train, sample_weight=_sw)
    models['GradientBoosting'] = gb_model
    predictions['GradientBoosting'] = {
        'y_pred': gb_model.predict(_X_test),
        'y_pred_proba': gb_model.predict_proba(_X_test)[:, 1]}

    for name in predictions:
        y_pred = predictions[name]['y_pred']
        y_proba = predictions[name]['y_pred_proba']
        metrics[name] = {
            'Accuracy': accuracy_score(_y_test, y_pred),
            'Precision': precision_score(_y_test, y_pred),
            'Recall': recall_score(_y_test, y_pred),
            'F1-Score': f1_score(_y_test, y_pred),
            'ROC-AUC': roc_auc_score(_y_test, y_proba)}
    return models, predictions, metrics


def _confusion_chart(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    labels = [['True Negative', 'False Positive'], ['False Negative', 'True Positive']]
    text_display = [[f'{labels[i][j]}<br><b>{cm[i][j]}</b>' for j in range(2)] for i in range(2)]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=['Predicted Stay', 'Predicted Churn'],
        y=['Actual Stay', 'Actual Churn'],
        colorscale=[[0, '#1e293b'], [0.5, '#4338ca'], [1, '#8b5cf6']],
        text=text_display, texttemplate='%{text}', textfont={"size": 16, "color": "#fff"},
        showscale=False, hovertemplate='%{y} / %{x}<br>Count: %{z}<extra></extra>'))
    fig.update_layout(
        title=dict(text=f'{name} — Confusion Matrix', font=dict(size=16, color='#e2e8f0')),
        height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'), xaxis=dict(side='bottom'),
        margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-container">
        <h1>🎯 ChurnCoach</h1>
        <p>AI-Powered Customer Churn Prediction &amp; Retention Intelligence</p>
        <p style="color:#64748b;font-size:.85rem;margin-top:.3rem">Powered by Ensemble ML Models · RAG Knowledge Base · Personalised Action Plans</p>
    </div>""", unsafe_allow_html=True)

    # ── Load & train ──────────────────────────────────────────────────────────
    with st.spinner("Loading data & training models …"):
        df_original, X, y, label_encoders = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if 'models' not in st.session_state:
        with st.spinner("Training ensemble models …"):
            models, predictions, metrics = train_models(
                X_train_scaled, X_test_scaled, y_train, y_test)
            st.session_state.update(dict(
                models=models, predictions=predictions, metrics=metrics,
                scaler=scaler, feature_names=X.columns.tolist(),
                y_test=y_test, label_encoders=label_encoders,
                df_original=df_original))

    models = st.session_state.models
    predictions = st.session_state.predictions
    metrics = st.session_state.metrics

    # ── Dataset overview ──────────────────────────────────────────────────────
    total = len(df_original)
    churned = int((df_original['Churn'] == 'Yes').sum())
    retained = total - churned
    churn_rate = churned / total * 100
    best_name = max(metrics, key=lambda m: metrics[m]['ROC-AUC'])

    st.markdown('<div class="section-header"><span class="icon">📊</span>'
                '<h2>Dataset Overview</h2></div>', unsafe_allow_html=True)
    ov = st.columns(5)
    cards = [
        (f'{total:,}', 'Total Customers', '#38bdf8'),
        (f'{retained:,}', 'Retained', '#22c55e'),
        (f'{churned:,}', 'Churned', '#ef4444'),
        (f'{churn_rate:.1f}%', 'Churn Rate', '#f59e0b'),
        (str(len(models)), 'Models Trained', '#a855f7'),
    ]
    for col, (val, lbl, clr) in zip(ov, cards):
        with col:
            st.markdown(f'<div class="data-card"><div class="number" style="color:{clr}">'
                        f'{val}</div><div class="label">{lbl}</div></div>',
                        unsafe_allow_html=True)

    # ── Churn distribution + Tenure histogram ─────────────────────────────────
    st.markdown('<div class="section-header"><span class="icon">📉</span>'
                '<h2>Churn Analysis</h2></div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        cc = df_original['Churn'].value_counts()
        fig_cd = go.Figure(data=[go.Pie(
            labels=['Retained', 'Churned'],
            values=[cc.get('No', 0), cc.get('Yes', 0)],
            marker=dict(colors=['#22c55e', '#ef4444'],
                        line=dict(color='#0f172a', width=3)),
            hole=0.6, textinfo='label+percent', textfont_size=14,
            hovertemplate='%{label}: %{value:,} customers (%{percent})<extra></extra>')])
        fig_cd.update_layout(
            title=dict(text='Customer Churn Distribution', font=dict(size=15, color='#e2e8f0')),
            height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'), showlegend=False,
            annotations=[dict(text=f'<b>{churn_rate:.1f}%</b><br><span style="font-size:12px">Churn Rate</span>',
                              x=.5, y=.5, font_size=24, font_color='#ef4444', showarrow=False)])
        st.plotly_chart(fig_cd, key="cd", width="stretch")

    with d2:
        # Tenure distribution with KDE-like smooth overlay
        bins_retained = df_original[df_original['Churn'] == 'No']['tenure']
        bins_churned = df_original[df_original['Churn'] == 'Yes']['tenure']
        fig_t = go.Figure()
        fig_t.add_trace(go.Histogram(
            x=bins_retained, name='Retained', marker_color='rgba(34,197,94,0.6)',
            nbinsx=24, hovertemplate='Tenure: %{x} months<br>Count: %{y}<extra>Retained</extra>'))
        fig_t.add_trace(go.Histogram(
            x=bins_churned, name='Churned', marker_color='rgba(239,68,68,0.6)',
            nbinsx=24, hovertemplate='Tenure: %{x} months<br>Count: %{y}<extra>Churned</extra>'))
        fig_t.update_layout(
            title=dict(text='Customer Tenure Distribution', font=dict(size=15, color='#e2e8f0')),
            barmode='overlay', height=380,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis_title='Tenure (months)', yaxis_title='Customer Count',
            xaxis=dict(gridcolor='rgba(255,255,255,.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,.05)'),
            legend=dict(orientation='h', y=1.12, x=.5, xanchor='center',
                        bgcolor='rgba(0,0,0,0)', font=dict(size=12)))
        st.plotly_chart(fig_t, key="tenure", width="stretch")

    # ── Service adoption + Contract churn + Payment churn ─────────────────────
    mc1, mc2 = st.columns(2)
    with mc1:
        # Service adoption heatmap by churn status
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies']
        svc_data = []
        for svc in service_cols:
            for churn_val, label in [('No', 'Retained'), ('Yes', 'Churned')]:
                subset = df_original[df_original['Churn'] == churn_val]
                adoption_rate = (subset[svc] == 'Yes').sum() / len(subset) * 100
                svc_data.append({'Service': svc.replace('Online', 'Online ').replace('Streaming', 'Stream. ').replace('Device', 'Device ').replace('Tech', 'Tech '),
                                 'Status': label, 'Adoption %': adoption_rate})
        svc_df = pd.DataFrame(svc_data)
        fig_svc = px.bar(svc_df, x='Service', y='Adoption %', color='Status',
                         barmode='group',
                         color_discrete_map={'Retained': '#22c55e', 'Churned': '#ef4444'})
        fig_svc.update_layout(
            title=dict(text='Service Adoption: Retained vs Churned', font=dict(size=15, color='#e2e8f0')),
            height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(gridcolor='rgba(255,255,255,.05)', tickangle=-25),
            yaxis=dict(gridcolor='rgba(255,255,255,.05)', title='Adoption Rate (%)'),
            legend=dict(orientation='h', y=1.12, x=.5, xanchor='center', bgcolor='rgba(0,0,0,0)'))
        fig_svc.update_traces(texttemplate='%{y:.0f}%', textposition='outside', textfont_size=10)
        st.plotly_chart(fig_svc, key="svc_adoption", width="stretch")

    with mc2:
        # Churn rate by contract type (horizontal bar with gradient)
        contract_churn = df_original.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=True)
        contract_total = df_original.groupby('Contract')['Churn'].count()
        fig_cc = go.Figure(go.Bar(
            x=contract_churn.values, y=contract_churn.index, orientation='h',
            marker=dict(
                color=contract_churn.values,
                colorscale=[[0, '#22c55e'], [0.5, '#f59e0b'], [1, '#ef4444']],
                cmin=0, cmax=50,
                line=dict(width=0)),
            text=[f'{v:.1f}% ({contract_total[idx]:,} customers)' for v, idx in zip(contract_churn.values, contract_churn.index)],
            textposition='outside', textfont=dict(color='#e2e8f0', size=11),
            hovertemplate='%{y}<br>Churn Rate: %{x:.1f}%<extra></extra>'))
        fig_cc.update_layout(
            title=dict(text='Churn Rate by Contract Type', font=dict(size=15, color='#e2e8f0')),
            height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(gridcolor='rgba(255,255,255,.05)', title='Churn Rate (%)', range=[0, 55]),
            margin=dict(l=10, r=80))
        st.plotly_chart(fig_cc, key="contract_churn", width="stretch")

    # ── Monthly charges boxplot + Payment method churn ────────────────────────
    mc3, mc4 = st.columns(2)
    with mc3:
        fig_box = go.Figure()
        for churn_val, clr, nm in [('No', '#22c55e', 'Retained'), ('Yes', '#ef4444', 'Churned')]:
            subset = df_original[df_original['Churn'] == churn_val]
            fig_box.add_trace(go.Box(
                y=subset['MonthlyCharges'], x=subset['Contract'],
                name=nm, marker_color=clr, boxmean=True,
                line=dict(width=1.5),
                hovertemplate='Contract: %{x}<br>Monthly Charges: $%{y:.2f}<extra>%{fullData.name}</extra>'))
        fig_box.update_layout(
            title=dict(text='Monthly Charges by Contract & Churn', font=dict(size=15, color='#e2e8f0')),
            height=400, boxmode='group',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            yaxis=dict(gridcolor='rgba(255,255,255,.05)', title='Monthly Charges ($)'),
            legend=dict(orientation='h', y=1.12, x=.5, xanchor='center', bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_box, key="box_charges", width="stretch")

    with mc4:
        pm_churn = df_original.groupby('PaymentMethod')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=True)
        pm_total = df_original.groupby('PaymentMethod')['Churn'].count()
        # Shorten labels
        pm_labels = [lbl.replace(' (automatic)', '\n(auto)').replace('Bank transfer', 'Bank Xfer').replace('Credit card', 'Credit Card').replace('Electronic check', 'E-Check').replace('Mailed check', 'Mail Check') for lbl in pm_churn.index]
        fig_pm = go.Figure(go.Bar(
            x=pm_churn.values, y=pm_labels, orientation='h',
            marker=dict(
                color=pm_churn.values,
                colorscale=[[0, '#22c55e'], [0.35, '#f59e0b'], [1, '#ef4444']],
                cmin=0, cmax=50,
                line=dict(width=0)),
            text=[f'{v:.1f}%' for v in pm_churn.values],
            textposition='outside', textfont=dict(color='#e2e8f0', size=11),
            hovertemplate='%{y}<br>Churn Rate: %{x:.1f}%<extra></extra>'))
        fig_pm.update_layout(
            title=dict(text='Churn Rate by Payment Method', font=dict(size=15, color='#e2e8f0')),
            height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(gridcolor='rgba(255,255,255,.05)', title='Churn Rate (%)', range=[0, 55]),
            margin=dict(l=10, r=60))
        st.plotly_chart(fig_pm, key="pm_churn", width="stretch")

    # ── Model Performance Comparison ──────────────────────────────────────────
    st.markdown('<div class="section-header"><span class="icon">🏆</span>'
                '<h2>Model Performance</h2></div>', unsafe_allow_html=True)

    model_colors = {'XGBoost': '#3b82f6', 'LightGBM': '#22c55e',
                    'CatBoost': '#f59e0b', 'GradientBoosting': '#a855f7'}

    # Model cards
    mc_cols = st.columns(len(models))
    for idx, (name, m) in enumerate(metrics.items()):
        clr = model_colors.get(name, '#38bdf8')
        best_badge = ' 👑' if name == best_name else ''
        with mc_cols[idx]:
            st.markdown(f"""
            <div class="metric-card" style="border-top:3px solid {clr}">
                <div class="metric-label">{name}{best_badge}</div>
                <div class="metric-value" style="color:{clr}">{m['ROC-AUC']:.2%}</div>
                <div style="color:#94a3b8;font-size:.75rem">ROC-AUC</div>
                <div style="color:#64748b;font-size:.7rem;margin-top:.3rem">
                    F1: {m['F1-Score']:.3f} · Recall: {m['Recall']:.3f}
                </div>
            </div>""", unsafe_allow_html=True)

    # Radar + ROC side by side
    metric_names = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    p1, p2 = st.columns(2)
    with p1:
        fig_r = go.Figure()
        for name, m in metrics.items():
            clr = model_colors.get(name, '#38bdf8')
            vals = [m[mn] for mn in metric_names] + [m[metric_names[0]]]
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=metric_names + [metric_names[0]],
                name=name, line=dict(color=clr, width=2.5),
                fill='toself',
                fillcolor=clr.replace(')', ',0.1)').replace('rgb', 'rgba'),
                hovertemplate='%{theta}: %{r:.4f}<extra>%{fullData.name}</extra>'))
        fig_r.update_layout(
            polar=dict(bgcolor='rgba(0,0,0,0)',
                       radialaxis=dict(visible=True, range=[0.4, 1],
                                       gridcolor='rgba(255,255,255,.1)',
                                       tickfont=dict(size=10, color='#64748b')),
                       angularaxis=dict(gridcolor='rgba(255,255,255,.1)',
                                        tickfont=dict(size=12, color='#cbd5e1'))),
            title=dict(text='Model Performance Radar', font=dict(size=15, color='#e2e8f0')),
            height=440, paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            legend=dict(orientation='h', y=-0.12, x=.5, xanchor='center',
                        bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_r, key="radar", width="stretch")

    with p2:
        # ROC curves
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                      line=dict(color='rgba(255,255,255,.15)', dash='dash', width=1),
                                      showlegend=False, hoverinfo='skip'))
        for name in predictions:
            clr = model_colors.get(name, '#38bdf8')
            fpr, tpr, _ = roc_curve(y_test, predictions[name]['y_pred_proba'])
            auc_val = metrics[name]['ROC-AUC']
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f'{name} ({auc_val:.4f})',
                line=dict(color=clr, width=2.5),
                hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>%{fullData.name}</extra>'))
        fig_roc.update_layout(
            title=dict(text='ROC Curves — All Models', font=dict(size=15, color='#e2e8f0')),
            xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
            height=440, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(gridcolor='rgba(255,255,255,.05)', zeroline=False),
            yaxis=dict(gridcolor='rgba(255,255,255,.05)', zeroline=False),
            legend=dict(orientation='h', y=-0.12, x=.5, xanchor='center',
                        bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_roc, key="roc", width="stretch")

    # ══════════════════════════════════════════════════════════════════════════
    # PREDICTION SECTION
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header"><span class="icon">🎯</span>'
                '<h2>Customer Churn Prediction</h2></div>', unsafe_allow_html=True)

    model_choice = st.selectbox("Select Prediction Model", list(models.keys()), index=0)

    # ── Grouped inputs ────────────────────────────────────────────────────────
    st.markdown('<p class="form-section-title">👤 Customer Demographics</p>',
                unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1: gender = st.selectbox("Gender", df_original['gender'].unique())
    with d2: senior_citizen = st.selectbox("Senior Citizen", [0, 1],
                                            format_func=lambda x: "Yes" if x else "No")
    with d3: partner = st.selectbox("Partner", df_original['Partner'].unique())
    with d4: dependents = st.selectbox("Dependents", df_original['Dependents'].unique())

    st.markdown('<p class="form-section-title">📡 Services</p>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        phone_service = st.selectbox("Phone Service", df_original['PhoneService'].unique())
        multiple_lines = st.selectbox("Multiple Lines", df_original['MultipleLines'].unique())
        internet_service = st.selectbox("Internet Service", df_original['InternetService'].unique())
    with s2:
        online_security = st.selectbox("Online Security", df_original['OnlineSecurity'].unique())
        online_backup = st.selectbox("Online Backup", df_original['OnlineBackup'].unique())
        device_protection = st.selectbox("Device Protection", df_original['DeviceProtection'].unique())
    with s3:
        tech_support = st.selectbox("Tech Support", df_original['TechSupport'].unique())
        streaming_tv = st.selectbox("Streaming TV", df_original['StreamingTV'].unique())
        streaming_movies = st.selectbox("Streaming Movies", df_original['StreamingMovies'].unique())

    st.markdown('<p class="form-section-title">💳 Billing & Contract</p>', unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    with b1: contract = st.selectbox("Contract", df_original['Contract'].unique())
    with b2: paperless_billing = st.selectbox("Paperless Billing", df_original['PaperlessBilling'].unique())
    with b3: payment_method = st.selectbox("Payment Method", df_original['PaymentMethod'].unique())
    with b4: tenure = st.slider("Tenure (months)", 0, 72, 12)

    ch1, ch2 = st.columns(2)
    with ch1: monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0, step=5.0)
    with ch2: total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0, step=50.0)

    # ── Predict ───────────────────────────────────────────────────────────────
    if st.button("🚀 Predict Churn & Generate Action Plan", type="primary"):
        input_data = {
            'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner,
            'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone_service,
            'MultipleLines': multiple_lines, 'InternetService': internet_service,
            'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
            'DeviceProtection': device_protection, 'TechSupport': tech_support,
            'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
            'Contract': contract, 'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges}

        input_df = pd.DataFrame([input_data])
        input_processed = create_advanced_features(input_df)
        for col in input_processed.select_dtypes(include=['object']).columns:
            if col in st.session_state.label_encoders:
                input_processed[col] = st.session_state.label_encoders[col].transform(
                    input_processed[col].astype(str))
        for col in st.session_state.feature_names:
            if col not in input_processed.columns:
                input_processed[col] = 0
        input_processed = input_processed[st.session_state.feature_names]
        input_scaled = st.session_state.scaler.transform(input_processed)

        selected_model = models[model_choice]
        prediction = selected_model.predict(input_scaled)[0]
        prediction_proba = selected_model.predict_proba(input_scaled)[0]
        churn_prob = float(prediction_proba[1])
        retain_prob = float(prediction_proba[0])

        # Risk classification
        if churn_prob > 0.7:
            risk_level, risk_cls, risk_emoji = "HIGH RISK", "risk-high", "🚨"
        elif churn_prob > 0.3:
            risk_level, risk_cls, risk_emoji = "MODERATE RISK", "risk-medium", "⚠️"
        else:
            risk_level, risk_cls, risk_emoji = "LOW RISK", "risk-low", "✅"

        st.markdown("---")

        # Risk badge
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:1rem">
            <span class="risk-badge {risk_cls}" style="font-size:1.1rem;padding:.5rem 1.5rem">
                {risk_emoji}  {risk_level}
            </span>
        </div>""", unsafe_allow_html=True)

        # Metric cards
        r1, r2, r3, r4 = st.columns(4)
        pred_text = "WILL CHURN" if prediction == 1 else "WILL STAY"
        pred_cls = "metric-red" if prediction == 1 else "metric-green"
        with r1:
            st.markdown(f'<div class="metric-card {pred_cls}"><div class="metric-label">Prediction</div>'
                        f'<div class="metric-value">{pred_text}</div></div>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="metric-card metric-red"><div class="metric-label">Churn Probability</div>'
                        f'<div class="metric-value">{churn_prob:.1%}</div></div>', unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="metric-card metric-green"><div class="metric-label">Retain Probability</div>'
                        f'<div class="metric-value">{retain_prob:.1%}</div></div>', unsafe_allow_html=True)
        with r4:
            st.markdown(f'<div class="metric-card metric-blue"><div class="metric-label">Model Used</div>'
                        f'<div class="metric-value" style="font-size:1.2rem">{model_choice}</div></div>',
                        unsafe_allow_html=True)

        # Gauge + Donut
        g1, g2 = st.columns(2)
        with g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=churn_prob * 100,
                number={'suffix': '%', 'font': {'size': 42, 'color': '#e2e8f0'}},
                delta={'reference': 50,
                       'increasing': {'color': '#ef4444'},
                       'decreasing': {'color': '#22c55e'}},
                title={'text': "Churn Risk Score",
                       'font': {'size': 16, 'color': '#94a3b8'}},
                gauge={'axis': {'range': [0, 100], 'tickcolor': '#475569'},
                       'bar': {'color': '#ef4444' if churn_prob > .5 else '#22c55e',
                               'thickness': .3},
                       'bgcolor': '#1e293b', 'borderwidth': 0,
                       'steps': [
                           {'range': [0, 30], 'color': 'rgba(34,197,94,.15)'},
                           {'range': [30, 70], 'color': 'rgba(245,158,11,.15)'},
                           {'range': [70, 100], 'color': 'rgba(239,68,68,.15)'}],
                       'threshold': {'line': {'color': '#f59e0b', 'width': 3},
                                     'thickness': .8, 'value': 50}}))
            fig_gauge.update_layout(height=370, paper_bgcolor='rgba(0,0,0,0)',
                                     font=dict(color='#e2e8f0'))
            st.plotly_chart(fig_gauge, key="gauge")

        with g2:
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Churn Risk', 'Retention Likelihood'],
                values=[churn_prob, retain_prob],
                marker=dict(colors=['#ef4444', '#22c55e'],
                            line=dict(color='#0f172a', width=3)),
                hole=.6, textinfo='label+percent',
                textfont_size=13, textfont_color='#e2e8f0')])
            fig_donut.update_layout(
                title=dict(text='Risk Breakdown', font=dict(size=15, color='#e2e8f0')),
                height=370, paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'), showlegend=False,
                annotations=[dict(text=f"<b>{churn_prob:.0%}</b><br>Churn", x=.5, y=.5,
                                  font_size=20,
                                  font_color='#ef4444' if churn_prob > .5 else '#22c55e',
                                  showarrow=False)])
            st.plotly_chart(fig_donut, key="donut")

        # Customer profile summary
        st.markdown('<div class="section-header"><span class="icon">👤</span>'
                    '<h2>Customer Profile Summary</h2></div>', unsafe_allow_html=True)
        pr1, pr2, pr3 = st.columns(3)

        active_svc = sum([phone_service == 'Yes', online_security == 'Yes',
                          online_backup == 'Yes', device_protection == 'Yes',
                          tech_support == 'Yes', streaming_tv == 'Yes',
                          streaming_movies == 'Yes'])

        with pr1:
            st.markdown(f"""<div class="profile-card">
<h4>👤 Demographics</h4>

| Attribute | Value |
|---|---|
| Gender | `{gender}` |
| Senior Citizen | `{'Yes' if senior_citizen else 'No'}` |
| Partner | `{partner}` |
| Dependents | `{dependents}` |
| Tenure | `{tenure} months` |
</div>""", unsafe_allow_html=True)
        with pr2:
            svc_list = [
                ('Phone', phone_service), ('Internet', internet_service),
                ('Security', online_security), ('Backup', online_backup),
                ('Protection', device_protection), ('Support', tech_support),
                ('TV', streaming_tv), ('Movies', streaming_movies)]
            pills = ""
            for svc_name, svc_val in svc_list:
                cls = "svc-yes" if svc_val == "Yes" else "svc-no" if svc_val == "No" else "svc-no"
                pills += f'<span class="svc-pill {cls}">{svc_name}: {svc_val}</span> '
            st.markdown(f"""<div class="profile-card">
<h4>📡 Services ({active_svc} active)</h4>
<div style="margin-top:.5rem">{pills}</div>
</div>""", unsafe_allow_html=True)

        with pr3:
            st.markdown(f"""<div class="profile-card">
<h4>💳 Billing</h4>

| Attribute | Value |
|---|---|
| Contract | `{contract}` |
| Paperless | `{paperless_billing}` |
| Payment | `{payment_method}` |
| Monthly | `${monthly_charges:.2f}` |
| Total | `${total_charges:.2f}` |
</div>""", unsafe_allow_html=True)

        # Churn risk factor breakdown
        st.markdown('<div class="section-header"><span class="icon">⚡</span>'
                    '<h2>Churn Risk Factor Analysis</h2></div>', unsafe_allow_html=True)
        factors = {}
        if contract == 'Month-to-month':
            factors['📋 Month-to-month contract'] = 0.85
        if tenure <= 12:
            factors[f'🆕 New customer ({tenure} months)'] = 0.75
        if online_security == 'No' and internet_service != 'No':
            factors['🔓 No online security'] = 0.65
        if monthly_charges > 80:
            factors[f'💰 High charges (${monthly_charges:.0f}/mo)'] = 0.60
        if 'Electronic check' in payment_method:
            factors['💳 Electronic check payment'] = 0.55
        if device_protection == 'No' and internet_service != 'No':
            factors['🛡️ No device protection'] = 0.55
        if tech_support == 'No' and internet_service != 'No':
            factors['🔧 No tech support'] = 0.50
        if senior_citizen == 1 and partner == 'No':
            factors['👴 Senior, no partner'] = 0.50
        if internet_service == 'Fiber optic':
            factors['⚡ Fiber optic (higher churn)'] = 0.40
        if paperless_billing == 'Yes':
            factors['📄 Paperless billing'] = 0.30
        if partner == 'No' and dependents == 'No':
            factors['👤 Single, no family'] = 0.25
        if not factors:
            factors['✅ No significant risk factors'] = 0.05

        f_df = pd.DataFrame({'Factor': list(factors.keys()),
                              'Risk Impact': list(factors.values())}).sort_values(
            'Risk Impact', ascending=True)

        # Color each bar based on impact level
        bar_colors = []
        for v in f_df['Risk Impact']:
            if v >= 0.7:
                bar_colors.append('#ef4444')
            elif v >= 0.5:
                bar_colors.append('#f59e0b')
            elif v >= 0.3:
                bar_colors.append('#38bdf8')
            else:
                bar_colors.append('#22c55e')

        fig_f = go.Figure(go.Bar(
            x=f_df['Risk Impact'], y=f_df['Factor'], orientation='h',
            marker=dict(color=bar_colors,
                        line=dict(width=0)),
            text=[f'{v:.0%}' for v in f_df['Risk Impact']],
            textposition='outside', textfont=dict(size=12, color='#e2e8f0'),
            hovertemplate='%{y}<br>Impact Score: %{x:.0%}<extra></extra>'))
        fig_f.update_layout(
            title=dict(text='Risk Factors — Impact Score', font=dict(size=15, color='#e2e8f0')),
            height=max(300, len(factors) * 50 + 80),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(range=[0, 1.15], gridcolor='rgba(255,255,255,.05)',
                       title='Impact Score', tickformat='.0%'),
            margin=dict(l=10, r=50))
        st.plotly_chart(fig_f, key="risk_factors", width="stretch")

        # ── AI Recommendations (RAG-only) ────────────────────────────────────
        st.markdown('<div class="section-header"><span class="icon">🤖</span>'
                    '<h2>AI-Powered Retention Action Plan</h2></div>',
                    unsafe_allow_html=True)

        customer_profile = dict(input_data)
        llm_connected = False

        if rag_engine is not None and llm_api_key:
            try:
                llm_connected = rag_engine.connect_llm(llm_api_key, provider=llm_provider)
            except Exception as e:
                st.warning(f"⚠️ Could not connect to LLM: {e}")
                llm_connected = False

        if rag_engine is not None:
            with st.spinner("🤖 Generating personalised retention plan from knowledge base …"):
                try:
                    rec = rag_engine.get_recommendation(customer_profile, churn_prob)
                except Exception as e:
                    st.warning(f"⚠️ AI generation failed: {e}. Using data-driven fallback.")
                    rec = RAGEngine._fallback_recommendation(customer_profile, churn_prob)
        else:
            # RAG engine not available — use static fallback
            rec = RAGEngine._fallback_recommendation(customer_profile, churn_prob)

        st.markdown(f'<div class="ai-rec-card">\n\n{rec}\n\n</div>',
                    unsafe_allow_html=True)

        if llm_connected:
            src = {"gemini": "Google Gemini", "anthropic": "Anthropic Claude",
                   "openai": "OpenAI GPT-4o"}.get(llm_provider, llm_provider)
            st.caption(f"🧠 Generated by {src} · RAG knowledge base · Customer-specific analysis")
        elif rag_engine is not None and not llm_api_key:
            st.caption("📋 Data-driven recommendation — enter your API key in the sidebar for AI-enhanced plans")
        else:
            st.caption("📋 Data-driven recommendation based on customer profile analysis")

        # ── Feature importance + Confusion Matrix ─────────────────────────────
        if hasattr(selected_model, 'feature_importances_'):
            fi_col, cm_col = st.columns(2)
            with fi_col:
                st.markdown('<div class="section-header"><span class="icon">🔍</span>'
                            '<h2>Top Prediction Factors</h2></div>', unsafe_allow_html=True)
                imp = selected_model.feature_importances_
                fi = pd.DataFrame({'Feature': st.session_state.feature_names,
                                    'Importance': imp}).sort_values(
                    'Importance', ascending=False).head(10)
                # Color gradient for importance
                fi_sorted = fi.sort_values('Importance', ascending=True)
                fig_imp = go.Figure(go.Bar(
                    x=fi_sorted['Importance'], y=fi_sorted['Feature'], orientation='h',
                    marker=dict(color=fi_sorted['Importance'],
                                colorscale=[[0, '#1e40af'], [0.5, '#6366f1'], [1, '#c084fc']],
                                line=dict(width=0)),
                    text=[f'{v:.4f}' for v in fi_sorted['Importance']],
                    textposition='outside', textfont=dict(size=10, color='#e2e8f0'),
                    hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'))
                fig_imp.update_layout(
                    title=dict(text=f'{model_choice} — Feature Importance',
                               font=dict(size=15, color='#e2e8f0')),
                    height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0'),
                    xaxis=dict(gridcolor='rgba(255,255,255,.05)', title='Importance Score'),
                    margin=dict(l=10, r=60))
                st.plotly_chart(fig_imp, key="feat_imp", width="stretch")

            with cm_col:
                st.markdown('<div class="section-header"><span class="icon">📊</span>'
                            '<h2>Confusion Matrix</h2></div>', unsafe_allow_html=True)
                st.plotly_chart(_confusion_chart(
                    st.session_state.y_test, predictions[model_choice]['y_pred'],
                    model_choice), key="conf", width="stretch")


if __name__ == "__main__":
    main()
