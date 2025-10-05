"""
Fraud Detection Streamlit App

Interactive dashboard for predicting return fraud with:
- CSV upload with pre-computed features
- Manual input for single transactions
- Adjustable prediction threshold
- Real-time metrics and visualizations
- High-risk transaction filtering
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

# Page configuration
st.set_page_config(
    page_title="Return Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_PATH = "models/random_forest_model.pkl"
FEATURE_INFO_PATH = "models/feature_info.pkl"

@st.cache_resource
def load_model_artifacts():
    """Load trained model and feature info."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(FEATURE_INFO_PATH, 'rb') as f:
            feature_info = pickle.load(f)
        return model, feature_info
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please run save_model.py first: {e}")
        return None, None

def preprocess_data(df, imputer):
    """Preprocess uploaded data: handle missing values and add engineered features."""
    base_features = [
        'return_rate', 'num_pur', 'num_ret', 'amt_ret', 'amount',
        'num_true_ret', 'return_to_purchase_ratio', 'days_since_last_return',
        'item_repeat_return_ratio', 'return_acceleration', 'avg_return_amount',
        'high_value_return_flag'
    ]
    
    X = df[base_features].copy()
    X['is_first_return'] = X['days_since_last_return'].isnull().astype(int)
    
    X_imputed = pd.DataFrame(
        imputer.transform(X),
        columns=X.columns,
        index=X.index
    )
    return X_imputed

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate classification metrics."""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_prob)
    }

def plot_confusion_matrix(y_true, y_pred):
    """Create confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Non-Fraud', 'Fraud'],
        y=['Non-Fraud', 'Fraud'],
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )
    return fig

def plot_feature_importance(model, feature_names, top_n=10):
    """Plot top N feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig = go.Figure(go.Bar(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Feature Importances',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig

def plot_threshold_metrics(y_true, y_prob, thresholds):
    """Plot metrics across different thresholds."""
    metrics_data = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        metrics_data.append({
            'Threshold': threshold,
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1 Score': metrics['F1 Score']
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_metrics['Threshold'], y=df_metrics['Precision'], 
                             mode='lines', name='Precision'))
    fig.add_trace(go.Scatter(x=df_metrics['Threshold'], y=df_metrics['Recall'], 
                             mode='lines', name='Recall'))
    fig.add_trace(go.Scatter(x=df_metrics['Threshold'], y=df_metrics['F1 Score'], 
                             mode='lines', name='F1 Score'))
    
    fig.update_layout(
        title='Metrics vs Threshold',
        xaxis_title='Threshold',
        yaxis_title='Score',
        height=400,
        hovermode='x unified'
    )
    return fig

def main():
    st.title("🔍 Return Fraud Detection Dashboard")
    st.markdown("**Upload transaction data and detect fraudulent returns with adjustable confidence threshold**")
    
    # Load model
    model, feature_info = load_model_artifacts()
    if model is None:
        st.stop()
    
    # Get imputer and expected features from feature_info
    imputer = feature_info.get('imputer')
    expected_features = feature_info.get('features', [])
    feature_names = feature_info.get('feature_names', [])
    
    # Sidebar - Configuration only
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        threshold = st.slider(
            "Fraud Probability Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Transactions with probability ≥ threshold are flagged as fraud"
        )
        
        st.markdown("---")
        
        top_n = st.slider(
            "Top N Features to Display",
            min_value=5,
            max_value=15,
            value=10,
            step=1
        )
        
        st.markdown("---")
        st.markdown("### Model Info")
        st.info(f"**Model:** Random Forest\n\n**Features:** {len(feature_names)}")
    
    # Main content - Input method selection
    st.header("📥 Input Method")
    input_tab1, input_tab2 = st.tabs(["📊 Upload CSV", "✏️ Manual Input"])
    
    uploaded_file = None
    manual_input_df = None
    
    with input_tab1:
        st.markdown("### Upload Transaction Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file with transaction features",
            type=['csv'],
            help="Upload a CSV file containing the required features"
        )
        
        if uploaded_file is None:
            st.info("💡 Upload a CSV file with the following columns:")
            st.code('\n'.join(expected_features), language=None)
    
    with input_tab2:
        st.markdown("### Enter Transaction Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            return_rate = st.number_input("Return Rate", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
            num_pur = st.number_input("Number of Purchases", min_value=0.0, value=15.0, step=1.0)
            num_ret = st.number_input("Number of Returns", min_value=0.0, value=15.0, step=1.0)
            amt_ret = st.number_input("Return Amount ($)", min_value=0.0, value=15004.86, step=10.0)
        
        with col2:
            amount = st.number_input("Transaction Amount ($)", min_value=-3000.0, value=-999.99, step=10.0)
            num_true_ret = st.number_input("True Returns (excl. voids)", min_value=0.0, value=7.0, step=1.0)
            return_to_purchase_ratio = st.number_input("Return/Purchase Ratio", min_value=0.0, value=0.47, step=0.01)
            days_since_last_return = st.number_input("Days Since Last Return", min_value=0.0, value=73.88, step=1.0)
        
        with col3:
            item_repeat_return_ratio = st.number_input("Item Repeat Return Ratio", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
            return_acceleration = st.number_input("Return Acceleration", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
            avg_return_amount = st.number_input("Avg Return Amount ($)", min_value=0.0, value=2143.55, step=10.0)
            high_value_return_flag = st.selectbox("High Value Return?", [0, 1], index=0)
        
        if st.button("🔍 Predict Fraud", type="primary"):
            manual_input_df = pd.DataFrame([{
                'return_rate': return_rate,
                'num_pur': num_pur,
                'num_ret': num_ret,
                'amt_ret': amt_ret,
                'amount': amount,
                'num_true_ret': num_true_ret,
                'return_to_purchase_ratio': return_to_purchase_ratio,
                'days_since_last_return': days_since_last_return,
                'item_repeat_return_ratio': item_repeat_return_ratio,
                'return_acceleration': return_acceleration,
                'avg_return_amount': avg_return_amount,
                'high_value_return_flag': high_value_return_flag
            }])
    
    # Determine data source and load data
    df = None
    data_source = None
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df):,} transactions from CSV")
        data_source = "csv"
    elif manual_input_df is not None:
        df = manual_input_df
        st.success(f"✅ Created transaction from manual input")
        data_source = "manual"
    
    # Only process if we have data
    if df is not None:
        # Validate columns
        missing_features = set(expected_features) - set(df.columns)
        if missing_features:
            st.error(f"❌ Missing required features: {missing_features}")
            st.stop()
        
        # Check if ground truth available
        has_labels = 'return_fraud' in df.columns
        
        # Preprocess and predict
        X = preprocess_data(df, imputer)
        
        # Get predictions
        fraud_prob = model.predict_proba(X)[:, 1]
        fraud_pred = (fraud_prob >= threshold).astype(int)
        
        # Add predictions to dataframe
        df['fraud_probability'] = fraud_prob
        df['fraud_prediction'] = fraud_pred
        
        # For manual input, show immediate result at the top
        if data_source == "manual":
            st.markdown("---")
            st.header("🎯 Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                prob = fraud_prob[0]
                st.metric("Fraud Probability", f"{prob:.1%}")
            
            with col2:
                prediction = "🚨 FRAUD" if fraud_pred[0] == 1 else "✅ LEGITIMATE"
                color = "red" if fraud_pred[0] == 1 else "green"
                st.markdown(f"### Prediction: :{color}[{prediction}]")
            
            with col3:
                st.metric("Threshold Used", f"{threshold:.2f}")
            
            # Show feature values
            with st.expander("📋 View Input Features"):
                st.dataframe(df[expected_features].T.rename(columns={0: 'Value'}), use_container_width=True)
            
            st.markdown("---")
            st.info("💡 Adjust the threshold in the sidebar to see how the prediction changes")
        
        # Always show detailed analysis tabs
        if True:
            st.markdown("---")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Overview", 
                "🎯 Performance Metrics", 
                "📈 Feature Importance", 
                "⚠️ High-Risk Transactions"
            ])
            
            with tab1:
                st.header("Prediction Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Transactions", f"{len(df):,}")
                with col2:
                    fraud_count = fraud_pred.sum()
                    st.metric("Predicted Frauds", f"{fraud_count:,}")
                with col3:
                    fraud_rate = fraud_count / len(df) * 100 if len(df) > 0 else 0
                    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                with col4:
                    avg_prob = fraud_prob.mean()
                    st.metric("Avg Fraud Probability", f"{avg_prob:.2%}")
                
                if len(df) > 1:
                    st.subheader("Fraud Probability Distribution")
                    fig = px.histogram(
                        df, 
                        x='fraud_probability',
                        nbins=50,
                        title='Distribution of Fraud Probabilities',
                        labels={'fraud_probability': 'Fraud Probability'},
                        color_discrete_sequence=['steelblue']
                    )
                    fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                                 annotation_text=f"Threshold: {threshold}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Upload multiple transactions via CSV to see probability distribution")
            
            with tab2:
                if has_labels:
                    st.header("Model Performance Metrics")
                    
                    y_true = df['return_fraud']
                    metrics = calculate_metrics(y_true, fraud_pred, fraud_prob)
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    cols = [col1, col2, col3, col4, col5]
                    
                    for i, (metric_name, value) in enumerate(metrics.items()):
                        with cols[i]:
                            st.metric(metric_name, f"{value:.3f}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_cm = plot_confusion_matrix(y_true, fraud_pred)
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    with col2:
                        thresholds = np.linspace(0, 1, 101)
                        fig_thresh = plot_threshold_metrics(y_true, fraud_prob, thresholds)
                        st.plotly_chart(fig_thresh, use_container_width=True)
                else:
                    st.warning("⚠️ No ground truth labels (return_fraud) found in data. Upload labeled data to see performance metrics.")
            
            with tab3:
                st.header("Feature Importance Analysis")
                
                fig_importance = plot_feature_importance(model, feature_names, top_n=top_n)
                st.plotly_chart(fig_importance, use_container_width=True)
                
                st.subheader("Feature Importance Values")
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.dataframe(
                    importance_df.style.background_gradient(cmap='Blues', subset=['Importance']),
                    use_container_width=True,
                    height=400
                )
            
            with tab4:
                st.header("High-Risk Transactions")
                
                if len(df) > 1:
                    col1, col2 = st.columns(2)
                    with col1:
                        min_prob = st.slider(
                            "Minimum Fraud Probability",
                            min_value=0.0,
                            max_value=1.0,
                            value=threshold,
                            step=0.05
                        )
                    with col2:
                        show_only_fraud = st.checkbox("Show only predicted frauds", value=True)
                    
                    filtered_df = df[df['fraud_probability'] >= min_prob].copy()
                    if show_only_fraud:
                        filtered_df = filtered_df[filtered_df['fraud_prediction'] == 1]
                else:
                    filtered_df = df.copy()
                    st.info("Showing single transaction from manual input")
                
                filtered_df = filtered_df.sort_values('fraud_probability', ascending=False)
                
                st.subheader(f"Showing {len(filtered_df):,} transactions")
                
                display_cols = ['fraud_probability', 'fraud_prediction']
                
                for col in expected_features:
                    if col in filtered_df.columns:
                        display_cols.append(col)
                
                if has_labels and 'return_fraud' in filtered_df.columns:
                    display_cols = ['return_fraud'] + display_cols
                
                st.dataframe(
                    filtered_df[display_cols].style.background_gradient(
                        cmap='Reds', 
                        subset=['fraud_probability']
                    ),
                    use_container_width=True,
                    height=500
                )
                
                if len(df) > 1:
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download High-Risk Transactions",
                        data=csv,
                        file_name=f"high_risk_transactions_threshold_{threshold}.csv",
                        mime="text/csv"
                    )
    else:
        st.info("👆 Choose an input method above to get started")

if __name__ == "__main__":
    main()