import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title=" Hogwarts Sorting Hat",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "Hogwarts House Classifier"}
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 100%);
    }
    
    .main { background: transparent; }
    
    .app-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        padding: 50px 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        border: 1px solid rgba(226, 201, 126, 0.2);
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .app-header h1 {
        color: #e2c97e;
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: 5px;
        margin: 0;
        font-family: 'Cinzel', serif;
        text-shadow: 2px 2px 10px rgba(226, 201, 126, 0.3);
    }
    
    .app-header p {
        color: #a8b2c1;
        font-size: 1.1rem;
        margin: 15px 0 0 0;
        letter-spacing: 2px;
    }
    
    .house-card {
        padding: 40px 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
        border: 2px solid;
        animation: slideIn 0.5s ease-out;
        transition: transform 0.3s ease;
    }
    
    .house-card:hover {
        transform: translateY(-5px);
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    
    .house-card h2 {
        margin: 0 0 10px 0;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: 4px;
        font-family: 'Cinzel', serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a 0%, #2d3e50 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #e2c97e 0%, #d4a574 100%);
        color: #1a1a2e;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 15px 30px;
        border-radius: 10px;
        border: none;
        letter-spacing: 2px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(226, 201, 126, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(226, 201, 126, 0.5);
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(226, 201, 126, 0.2);
    }
    
    section[data-testid="stSidebar"] * {
        color: #c8d0dc !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 42, 58, 0.5);
        border-radius: 10px 10px 0 0;
        padding: 15px 25px;
        color: #a8b2c1;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e2c97e 0%, #d4a574 100%);
        color: #1a1a2e !important;
    }
    
    .section-title {
        color: #e2c97e;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 30px 0 20px 0;
        font-family: 'Cinzel', serif;
        letter-spacing: 3px;
    }
</style>
""", unsafe_allow_html=True)

HOUSE_COLORS = {
    "Gryffindor": {"bg": "#7B0D1E", "text": "#F0C040", "accent": "#DC143C", "border": "#F0C040"},
    "Hufflepuff": {"bg": "#4A3728", "text": "#F5C518", "accent": "#D4A017", "border": "#F5C518"},
    "Ravenclaw": {"bg": "#0E1A40", "text": "#C5B358", "accent": "#4169E1", "border": "#C5B358"},
    "Slytherin": {"bg": "#1A472A", "text": "#8DB48E", "accent": "#2E8B57", "border": "#8DB48E"}
}

@st.cache_resource
def load_model():
    try:
        model_path = "models/best_model_logistic_regression.joblib"
        le_path = "models/label_encoder.joblib"
        
        cwd = os.getcwd()
        files_in_cwd = os.listdir(cwd)
        
        if not os.path.exists("models"):
            st.error(f" Models directory not found. CWD: {cwd}, Files: {files_in_cwd}")
            return None, None, None
            
        if not os.path.exists(model_path):
            st.error(f" Model file not found: {model_path}")
            return None, None, None
            
        if not os.path.exists(le_path):
            st.error(f" Label encoder not found: {le_path}")
            return None, None, None
            
        model = joblib.load(model_path)
        le = joblib.load(le_path)
        
        from sklearn.linear_model import LogisticRegression
        steps = list(model.named_steps.values()) if hasattr(model, "named_steps") else [model]
        for step in steps:
            if isinstance(step, LogisticRegression):
                if not hasattr(step, "multi_class"):
                    step.__dict__["multi_class"] = "auto"
                if not hasattr(step, "l1_ratio"):
                    step.__dict__["l1_ratio"] = None
        
        return model, le, "Logistic Regression"
    except Exception as e:
        import traceback
        st.error(f" Error loading model: {e}")
        st.code(traceback.format_exc())
        return None, None, None

@st.cache_data
def load_dataset():
    return None

model, le, model_name = load_model()
df = load_dataset()

st.markdown("""
<div class="app-header">
    <h1> THE SORTING HAT</h1>
    <p>Discover Your Hogwarts House Through Engineering Magic</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### About This Project")
    st.markdown("""
    This classifier uses machine learning to predict which Hogwarts house you belong to based on your personality traits and skills.
    
    **Technology Stack**
    - Logistic Regression Model
    - Scikit-learn Pipeline
    - Streamlit Interface
    - Plotly Visualizations
    
    **Model Performance**
    - Accuracy: 89%
    - F1-Score: 87%
    - 4-class classification
    """)
    
    if model:
        st.markdown("---")
        st.markdown(f"**Active Model:** {model_name}")
        st.markdown(f"**Houses:** {len(le.classes_)}")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([" Sorting Ceremony", "Data Insights", "Model Info", "About", "Credits"])

# TAB 1 - PREDICTION
with tab1:
    st.markdown('<p class="section-title">Step Into The Great Hall</p>', unsafe_allow_html=True)
    
    if model:
        col_left, col_right = st.columns([1, 1], gap="large")
        
        with col_left:
            st.markdown("### Your Magical Profile")
            
            with st.container():
                name = st.text_input("What is your name?", placeholder="Enter your name...", key="name")
                blood_status = st.selectbox("Blood Status", ["Pure-blood", "Half-blood", "Muggle-born"], index=1)
            
            st.markdown("---")
            st.markdown("### Rate Your Abilities (1-10)")
            
            col1, col2 = st.columns(2)
            with col1:
                bravery = st.slider("Bravery", 1, 10, 5, help="Your courage in facing danger")
                intelligence = st.slider("Intelligence", 1, 10, 5, help="Your wit and learning")
                loyalty = st.slider("Loyalty", 1, 10, 5, help="Your dedication to friends")
                ambition = st.slider("Ambition", 1, 10, 5, help="Your drive for success")
            
            with col2:
                dark_arts = st.slider("Dark Arts Knowledge", 1, 10, 3, help="Understanding of dark magic")
                quidditch = st.slider("Quidditch Skills", 1, 10, 5, help="Flying and sports ability")
                dueling = st.slider("Dueling Skills", 1, 10, 5, help="Combat magic proficiency")
                creativity = st.slider("Creativity", 1, 10, 5, help="Innovative thinking")
            
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("THE SORTING HAT DECIDES", use_container_width=True, type="primary")
        
        with col_right:
            if not predict_btn:
                st.markdown(f"""
                <div style="padding:80px 40px; border:2px dashed rgba(226, 201, 126, 0.3); 
                            border-radius:20px; text-align:center; margin-top:60px;
                            background: rgba(30, 42, 58, 0.3);">
                    <h2 style="color:#e2c97e; font-family:'Cinzel', serif; margin-bottom:20px;">
                        The Sorting Hat Awaits
                    </h2>
                    <p style="color:#a8b2c1; font-size:1.1rem; line-height:1.8;">
                        Complete your magical profile<br>
                        and let the Sorting Hat reveal<br>
                        your true Hogwarts house.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                try:
                    student_data = pd.DataFrame([{
                        "Blood Status": blood_status,
                        "Bravery": bravery,
                        "Intelligence": intelligence,
                        "Loyalty": loyalty,
                        "Ambition": ambition,
                        "Dark Arts Knowledge": dark_arts,
                        "Quidditch Skills": quidditch,
                        "Dueling Skills": dueling,
                        "Creativity": creativity,
                    }])
                    
                    pred_encoded = model.predict(student_data)[0]
                    pred_house = le.inverse_transform([pred_encoded])[0]
                    probs = model.predict_proba(student_data)[0]
                    max_prob = max(probs)
                    
                    colors = HOUSE_COLORS[pred_house]
                    display_name = name.strip() if name.strip() else "You"
                    
                    st.markdown(f"""
                    <div class="house-card" style="background:{colors['bg']}; border-color:{colors['border']};">
                        <p style="color:{colors['text']}; font-size:0.9rem; letter-spacing:3px; 
                                  font-weight:600; text-transform:uppercase; margin-bottom:10px; opacity:.7;">
                            The Sorting Hat Has Spoken
                        </p>
                        <h2 style="color:{colors['text']};">{pred_house.upper()}</h2>
                        <p style="color:{colors['text']}; font-size:1.2rem; margin-top:15px;">
                            {display_name} belongs in <strong>{pred_house}</strong>!
                        </p>
                        <p style="color:{colors['accent']}; font-size:1.1rem; margin-top:20px; font-weight:700;">
                            Confidence: {max_prob:.1%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Probability chart
                    prob_df = pd.DataFrame({
                        "House": le.classes_,
                        "Probability": probs
                    }).sort_values("Probability", ascending=False)
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=prob_df["Probability"],
                            y=prob_df["House"],
                            orientation='h',
                            marker=dict(
                                color=[HOUSE_COLORS[h]["accent"] for h in prob_df["House"]],
                                line=dict(color='rgba(255,255,255,0.3)', width=2)
                            ),
                            text=[f"{p:.1%}" for p in prob_df["Probability"]],
                            textposition='outside',
                        )
                    ])
                    
                    fig.update_layout(
                        title="House Probability Distribution",
                        xaxis_title="Probability",
                        yaxis_title="",
                        template="plotly_dark",
                        height=300,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter", size=12, color="#a8b2c1"),
                        title_font=dict(size=16, color="#e2c97e"),
                        xaxis=dict(range=[0, 1.1])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Skills radar
                    skills = {
                        "Bravery": bravery, "Intelligence": intelligence,
                        "Loyalty": loyalty, "Ambition": ambition,
                        "Dark Arts": dark_arts, "Quidditch": quidditch,
                        "Dueling": dueling, "Creativity": creativity
                    }
                    
                    fig2 = go.Figure(data=go.Scatterpolar(
                        r=list(skills.values()),
                        theta=list(skills.keys()),
                        fill='toself',
                        fillcolor=f"rgba{tuple(list(int(HOUSE_COLORS[pred_house]['accent'][i:i+2], 16) for i in (1, 3, 5)) + [0.3])}",
                        line=dict(color=HOUSE_COLORS[pred_house]["accent"], width=2)
                    ))
                    
                    fig2.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 10], gridcolor='rgba(255,255,255,0.1)'),
                            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                        ),
                        template="plotly_dark",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title="Your Magical Skills Profile",
                        title_font=dict(size=16, color="#e2c97e"),
                        font=dict(family="Inter", size=11, color="#a8b2c1")
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    else:
        st.error("Model not loaded. Please ensure model files exist in the models/ directory.")

# TAB 2 - DATA INSIGHTS
with tab2:
    st.markdown('<p class="section-title">Dataset Exploration</p>', unsafe_allow_html=True)
    
    st.info("Dataset exploration is available in the Jupyter notebook. The deployed app focuses on predictions.")
    
    st.markdown("### Model Training Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#e2c97e; margin:0;"> 1,000+</h3>
            <p style="color:#a8b2c1; margin:5px 0 0 0;">Training Records</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#e2c97e; margin:0;"> 9</h3>
            <p style="color:#a8b2c1; margin:5px 0 0 0;">Input Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#e2c97e; margin:0;"> 4</h3>
            <p style="color:#a8b2c1; margin:5px 0 0 0;">House Classes</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#e2c97e; margin:0;">89%</h3>
            <p style="color:#a8b2c1; margin:5px 0 0 0;">Test Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("### House Distribution")
    st.markdown("""
    The training dataset contains a balanced distribution across all four Hogwarts houses:
    - **Gryffindor**: ~25% (Bravery & Courage)
    - **Hufflepuff**: ~25% (Loyalty & Hard Work)
    - **Ravenclaw**: ~25% (Intelligence & Wisdom)
    - **Slytherin**: ~25% (Ambition & Cunning)
    """)
    
    st.markdown("### Key Features")
    features_df = pd.DataFrame({
        "Feature": ["Blood Status", "Bravery", "Intelligence", "Loyalty", "Ambition", 
                    "Dark Arts Knowledge", "Quidditch Skills", "Dueling Skills", "Creativity"],
        "Type": ["Categorical", "Numeric (1-10)", "Numeric (1-10)", "Numeric (1-10)", "Numeric (1-10)",
                 "Numeric (1-10)", "Numeric (1-10)", "Numeric (1-10)", "Numeric (1-10)"],
        "Importance": ["Medium", "High", "High", "High", "High", "Medium", "Low", "Medium", "Medium"]
    })
    st.dataframe(features_df, use_container_width=True, hide_index=True)

# TAB 3 - MODEL INFO
with tab3:
    st.markdown('<p class="section-title">Model Architecture</p>', unsafe_allow_html=True)
    
    if model:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color:#e2c97e; margin:0;">Logistic Regression</h3>
                <p style="color:#a8b2c1; margin:5px 0 0 0;">Model Type</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color:#e2c97e; margin:0;">{len(le.classes_)}</h3>
                <p style="color:#a8b2c1; margin:5px 0 0 0;">Output Classes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color:#e2c97e; margin:0;">9</h3>
                <p style="color:#a8b2c1; margin:5px 0 0 0;">Input Features</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("###  Model Performance")
            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "F1-Score", "Balanced Accuracy"],
                "Train": ["92%", "90%", "90%"],
                "Test": ["89%", "87%", "88%"]
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("###  Target Classes")
            houses_df = pd.DataFrame({
                "House": ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"],
                "Core Trait": ["Bravery", "Loyalty", "Intelligence", "Ambition"]
            })
            st.dataframe(houses_df, use_container_width=True, hide_index=True)
        
        st.markdown("###  Feature Engineering Pipeline")
        st.markdown("""
        1. **Imputation**: Median for numeric, mode for categorical
        2. **Scaling**: StandardScaler for numeric features
        3. **Encoding**: OneHotEncoder for categorical features
        4. **Classification**: Logistic Regression 
        """)
    else:
        st.error("Model not loaded.")

# TAB 4 - ABOUT
with tab4:
    st.markdown('<p class="section-title">About This Project</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ###  Project Overview
        
        This application demonstrates the power of machine learning in personality classification. 
        Using a dataset of Hogwarts students, we've trained a model to predict which house 
        best suits an individual based on their traits and abilities.
        
        ###  Methodology
        
        1. **Data Collection**: Harry Potter Sorting Dataset from Kaggle
        2. **Exploratory Analysis**: Understanding feature distributions and correlations
        3. **Preprocessing**: Handling missing values, scaling, and encoding
        4. **Model Training**: Multiple algorithms tested with cross-validation
        5. **Evaluation**: F1-score, accuracy, and balanced accuracy metrics
        6. **Deployment**: Streamlit web application on Google Cloud Run
        
        ### Technology Stack
        
        - **ML Framework**: Scikit-learn
        - **Visualization**: Plotly
        - **Web Framework**: Streamlit
        - **Deployment**: Docker + Google Cloud Run
        - **Language**: Python 3.9+
        """)
    
    with col2:
        st.markdown("""
        ### The Four Houses
        
        **Gryffindor**
        Values bravery, courage, and chivalry. Known for producing bold wizards and witches.
        
        **Hufflepuff**
        Values loyalty, hard work, and fair play. Known for dedication and patience.
        
        **Ravenclaw**
        Values intelligence, creativity, and wisdom. Known for scholarly pursuits.
        
        **Slytherin**
        Values ambition, cunning, and resourcefulness. Known for producing powerful wizards.
        
        ### Dataset Information
        
        - **Source**: Kaggle - Harry Potter Sorting Dataset
        - **Records**: 1,000+ students
        - **Features**: 9 (1 categorical, 8 numeric)
        - **Target**: 4 classes (houses)
        - **Balance**: Relatively balanced across classes
        
        """)

# TAB 5 - CREDITS
with tab5:
    st.markdown('<p class="section-title">Project Credits</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align:center; padding:40px 20px; background: linear-gradient(135deg, #1e2a3a 0%, #2d3e50 100%); 
                border-radius:20px; margin:20px 0; border: 1px solid rgba(226, 201, 126, 0.2);">
        <h2 style="color:#e2c97e; font-family:'Cinzel', serif; margin-bottom:30px; font-size:2rem;">
            Escuela de Ingeniería
        </h2>
        <h3 style="color:#a8b2c1; margin-bottom:40px; font-size:1.3rem;">
            Ingeniería en Ciencias Computacionales
        </h3>
        <h3 style="color:#e2c97e; margin-bottom:40px; font-size:1.5rem;">
            Inteligencia Computacional
        </h3>
        <h4 style="color:#a8b2c1; margin-bottom:50px; font-size:1.2rem;">
            Proyecto 1 - Aplicación con funcionalidades de ML
        </h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Team Members")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="text-align:left; padding:30px;">
            <h3 style="color:#e2c97e; margin-bottom:10px;">David Alexandro García Morales</h3>
            <p style="color:#a8b2c1; font-size:1.1rem;">ID: 31624</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card" style="text-align:left; padding:30px;">
            <h3 style="color:#e2c97e; margin-bottom:10px;">Edgar Daniel De la Torre Reza</h3>
            <p style="color:#a8b2c1; font-size:1.1rem;">ID: 34887</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align:left; padding:30px;">
            <h3 style="color:#e2c97e; margin-bottom:10px;">Luis Bernardo Bremer Ortega</h3>
            <p style="color:#a8b2c1; font-size:1.1rem;">ID: 32366</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card" style="text-align:left; padding:30px;">
            <h3 style="color:#e2c97e; margin-bottom:10px;">Hugo German Manzano López</h3>
            <p style="color:#a8b2c1; font-size:1.1rem;">ID: 36231</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align:center; padding:30px; background: rgba(30, 42, 58, 0.3); 
                border-radius:15px; border: 1px solid rgba(226, 201, 126, 0.1);">
        <p style="color:#a8b2c1; font-size:1rem; line-height:1.8;">
            This project demonstrates the application of machine learning techniques<br>
            for multi-class classification using the Harry Potter Sorting Dataset.<br>
            Developed as part of the Computational Intelligence course.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#9aa3af; font-size:0.9rem; padding:20px;">
     Hogwarts Sorting Hat Classifier | Built using Streamlit & Scikit-learn | Deployed on Google Cloud Run
</div>
""", unsafe_allow_html=True)