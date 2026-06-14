import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. PLATFORM INITIALIZATION & CSS INJECTION
# ==========================================
st.set_page_config(
    page_title="Enterprise Churn Analytics Console",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate Minimalist UI Styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            -webkit-font-smoothing: antialiased;
        }
        .platform-header {
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            line-height: 1.2;
            color: var(--text-color, #1E3A8A);
            margin-bottom: 0.5rem;
        }
        .platform-subtitle {
            font-size: 1.1rem;
            font-weight: 400;
            color: #6B7280;
            line-height: 1.6;
            margin-bottom: 2.5rem;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(249, 250, 251, 0.05);
            border: 1px solid rgba(229, 231, 235, 0.3);
            padding: 1.25rem 1.5rem;
            border-radius: 8px;
        }
        div.stButton > button {
            background: #1E40AF !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            padding: 0.75rem 2rem !important;
            border-radius: 6px !important;
            border: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CACHED PIPELINE ASSET LOADING
# ==========================================
@st.cache_resource
def load_churn_workspace_assets():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler_churn.pkl")
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Critical Error: Required pipeline binary file is missing from your project folder. Details: {e}")
        st.stop()

churn_model, churn_scaler = load_churn_workspace_assets()

# ==========================================
# 3. WORKSPACE VIEW NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("## Workspace Console")
    workspace_view = st.selectbox(
        "Select Operational View",
        ["Real-Time Diagnostic Input", "Batch Exploratory Insights"]
    )
    st.markdown("---")

# ==========================================
# VIEW A: REAL-TIME DIAGNOSTIC INPUT
# ==========================================
if workspace_view == "Real-Time Diagnostic Input":
    with st.sidebar:
        st.markdown("### Customer Attributes")
        # Removed sliders - replaced with professional numeric input boxes
        age = st.number_input("Customer Age Profile", min_value=12, max_value=100, value=44, step=1)
        gender_input = st.selectbox("Customer Gender Identity", ["Male", "Female"])
        
        st.markdown("---")
        st.markdown("### Subscription Metrics")
        tenure = st.number_input("Tenure Length (Months with Brand)", min_value=0, max_value=150, value=18, step=1)
        monthly_charges = st.number_input("Active Monthly Contract Fee ($)", min_value=30, max_value=200, value=74, step=1)

    gender_encoded = 1 if gender_input == "Female" else 0

    st.markdown("<div class='platform-header'>Account Retention & Risk Portal</div>", unsafe_allow_html=True)
    st.markdown("<div class='platform-subtitle'>Run single-user variables instantly through the linear Support Vector Machine classifier boundary.</div>", unsafe_allow_html=True)
    
    if st.button("Execute Diagnostic Risk Assessment", use_container_width=True):
        input_data = np.array([[age, gender_encoded, tenure, monthly_charges]])
        scaled_input = churn_scaler.transform(input_data)
        prediction = churn_model.predict(scaled_input)[0]
        
        st.markdown("### Diagnostic Assessment Report")
        m_col1, m_col2 = st.columns(2)
        
        if prediction == 1:
            with m_col1: st.metric("Predictive Status", "High Attrition Risk")
            with m_col2: st.metric("Account Security Status", "Flagged / Critical")
            
            st.error("🚨 Warning: This customer vector falls within the model's Churn Decision Space.")
            st.markdown("""
                <div style="background-color: rgba(220, 38, 38, 0.03); padding: 1.5rem; border-radius: 6px; border: 1px solid rgba(220, 38, 38, 0.15);">
                    <strong>Recommended Go-To-Market Strategy:</strong> This individual is highly sensitive to pricing metrics. 
                    Assign this user account directly to a customer success representative and trigger an active retention promo loop.
                </div>
            """, unsafe_allow_html=True)
            # Balloons removed for clean corporate UX
        else:
            with m_col1: st.metric("Predictive Status", "Retained / Active")
            with m_col2: st.metric("Account Security Status", "Verified Secure")
            
            st.success("✅ Account Status Verified: This customer coordinates fall within the safe Retention Zone.")
            st.markdown("""
                <div style="background-color: rgba(5, 150, 105, 0.03); padding: 1.5rem; border-radius: 6px; border: 1px solid rgba(5, 150, 105, 0.15);">
                    <strong>Standard Maintenance Action:</strong> User metrics show stable product engagement tracks. Let account continue running through standard automated loyalty touchpoints.
                </div>
            """, unsafe_allow_html=True)

# ==========================================
# VIEW B: BATCH EXPLORATORY INSIGHTS
# ==========================================
else:
    st.markdown("<div class='platform-header'>Batch Exploratory Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='platform-subtitle'>Upload historical databases to inspect continuous probability density metrics and model confusion performance.</div>", unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader("Upload Historical Database (.csv)", type="csv")
    
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        
        required_cols = ["Age", "Gender", "Tenure", "MonthlyCharges", "Churn"]
        if all(col in raw_df.columns for col in required_cols):
            
            if raw_df["Churn"].dtype == object:
                raw_df["Churn_Numeric"] = raw_df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
            else:
                raw_df["Churn_Numeric"] = raw_df["Churn"]
                raw_df["Churn"] = raw_df["Churn_Numeric"].map({1: "Yes", 0: "No"})
            
            # --- ROW 1: BEHAVIORAL DISTRIBUTION DENSITY GRIDS ---
            st.markdown("### Behavioral Feature Space Distributions across Target Populations")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
            sns.set_theme(style="ticks")
            
            sns.kdeplot(
                data=raw_df, x="MonthlyCharges", hue="Churn", fill=True,
                palette={"Yes": "#DC2626", "No": "#059669"}, alpha=0.35, linewidth=2, ax=axes[0]
            )
            axes[0].set_title("Customer Lifeline Trajectory Density by Contract Price ($)", fontsize=11, fontweight="bold")
            axes[0].set_xlabel("Monthly Contract Charges ($)")
            axes[0].set_ylabel("Probability Density")
            
            sns.kdeplot(
                data=raw_df, x="Tenure", hue="Churn", fill=True,
                palette={"Yes": "#DC2626", "No": "#059669"}, alpha=0.35, linewidth=2, ax=axes[1]
            )
            axes[1].set_title("Lifespan Retention Saturation Profile across Tenure Length (Months)", fontsize=11, fontweight="bold")
            axes[1].set_xlabel("Tenure (Months)")
            axes[1].set_ylabel("")
            
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("---")
            
            # --- ROW 2: LIVE ERROR AND PERFORMANCE AUDITS ---
            col_b1, col_b2 = st.columns([1, 1.2])
            
            # Process batch features defensively matching exact lambda configurations
            batch_features = raw_df[["Age", "Gender", "Tenure", "MonthlyCharges"]].copy()
            batch_features["Gender"] = batch_features["Gender"].apply(lambda x: 1 if x == "Female" else 0)
            
            scaled_batch = churn_scaler.transform(batch_features)
            y_batch_preds = churn_model.predict(scaled_batch)
            
            raw_df["SVM_Prediction_Class"] = y_batch_preds
            raw_df["SVM_Prediction"] = raw_df["SVM_Prediction_Class"].map({1: "Predicted Churn", 0: "Predicted Retained"})
            
            with col_b1:
                st.markdown("### SVM Confusion Matrix")
                st.caption("Inspect absolute True/False metrics to flag where decision boundaries conflict.")
                
                cm = confusion_matrix(raw_df["Churn_Numeric"], y_batch_preds)
                cm_df = pd.DataFrame(
                    cm, 
                    index=["Actual Active", "Actual Churned"],
                    columns=["Predicted Active", "Predicted Churned"]
                )
                
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 12, "weight": "bold"})
                plt.title("Active Model Decision Mistakes Heatmap", fontsize=11, fontweight="bold", pad=12)
                plt.tight_layout()
                st.pyplot(fig_cm)
                
            with col_b2:
                st.markdown("### Interactive Structural Database View")
                st.caption("Audit processed coordinates and corresponding classifications simultaneously.")
                
                st.dataframe(
                    raw_df[["Age", "Gender", "Tenure", "MonthlyCharges", "Churn", "SVM_Prediction"]],
                    use_container_width=True,
                    height=300
                )
                
            # --- NEW ADDITION: TARGETED STRATEGIC EXPORT PANEL ---
            st.markdown("---")
            st.markdown("### 📥 Automated Marketing Campaign Target Tool")
            st.caption("Filter down the database dynamically based on the model's predictions to download highly specific high-risk contact lists.")
            
            export_filter = st.selectbox(
                "Filter Target Population for Strategic Export", 
                options=["Show All Customers", "Show Only Predicted Churn Accounts", "Show Only Predicted Secure Accounts"]
            )
            
            if export_filter == "Show Only Predicted Churn Accounts":
                filtered_df = raw_df[raw_df["SVM_Prediction_Class"] == 1]
            elif export_filter == "Show Only Predicted Secure Accounts":
                filtered_df = raw_df[raw_df["SVM_Prediction_Class"] == 0]
            else:
                filtered_df = raw_df
                
            st.dataframe(
                filtered_df[["Age", "Gender", "Tenure", "MonthlyCharges", "Churn", "SVM_Prediction"]],
                use_container_width=True,
                height=250
            )
            
            # Export Action Button
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Export List ({len(filtered_df)} Records) to CSV",
                data=csv_data,
                file_name="targeted_churn_risk_list.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        else:
            st.error("Uploaded Database file attributes do not match core training expected index labels. Please look at header titles.")
    else:
        st.info("Please drag and drop your evaluation base database `.csv` layout file onto the left file upload configuration tray to execute multi-view interactive analytic charts.")