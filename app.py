import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
import time
import random
from utils.data_processing import process_tapping_data, load_symptom_checklist
from utils.ml_models import predict_risk, load_model
from utils.rag_system import get_response

# Set page config
st.set_page_config(
    page_title="NeuroBridge",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # Continue without custom styling

load_css()

# Initialize session state variables
if 'risk_score' not in st.session_state:
    st.session_state.risk_score = None
if 'symptoms' not in st.session_state:
    st.session_state.symptoms = {}
if 'tapping_data' not in st.session_state:
    st.session_state.tapping_data = None
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False

# Main app
def main():
    st.title("🧠 NeuroBridge – Detect • Connect • Personalize")
    st.markdown("""
    An AI-powered tool for early detection of neurological conditions, connecting patients with caregivers, and personalizing care plans.
    """)
    
    # Navigation
    menu = ["Home", "Detect", "Connect", "Personalize", "About"]
    choice = st.sidebar.selectbox("Navigation", menu)
    
    if choice == "Home":
        show_home()
    elif choice == "Detect":
        show_detect()
    elif choice == "Connect":
        show_connect()
    elif choice == "Personalize":
        show_personalize()
    elif choice == "About":
        show_about()

def show_home():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Welcome to NeuroBridge")
        st.markdown("""
        NeuroBridge helps you:
        - **Detect** early signs of neurological conditions through simple tests
        - **Connect** with healthcare providers by generating shareable reports
        - **Personalize** your care plan with AI-driven recommendations
        
        ### How it works
        1. Complete a quick symptom checklist and tapping test in the **Detect** section
        2. Generate a health report in the **Connect** section to share with your doctor
        3. Get personalized recommendations in the **Personalize** section
        
        Get started by navigating to the **Detect** section!
        """)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=600&q=80", 
                 caption="Early detection leads to better outcomes")

def show_detect():
    st.header("Detect Early Signs")
    
    # Symptom Checklist
    st.subheader("Symptom Checklist")
    st.info("Please answer the following questions based on your experiences over the past few weeks.")
    
    symptoms = load_symptom_checklist()
    for symptom, question in symptoms.items():
        st.session_state.symptoms[symptom] = st.radio(
            question, 
            options=["No", "Yes"], 
            key=symptom,
            horizontal=True
        )
    
    # Tapping Test
    st.subheader("Tapping Test")
    st.markdown("""
    Research has shown that finger tapping patterns can reveal early signs of neurological conditions like Parkinson's disease.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Instructions:**")
        st.write("1. Click the 'Start Tapping Test' button below")
        st.write("2. Tap the spacebar or click the button as fast and steadily as possible for 10 seconds")
        st.write("3. We'll analyze your tapping rhythm and consistency")
        
        if st.button("Start Tapping Test", key="start_tapping"):
            st.session_state.tapping_data = run_tapping_test()
    
    with col2:
        if st.session_state.tapping_data:
            st.write("**Tapping Results:**")
            st.write(f"Number of taps: {len(st.session_state.tapping_data)}")
            
            # Calculate and display metrics
            intervals = np.diff(st.session_state.tapping_data)
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            st.metric("Average time between taps", f"{mean_interval:.3f} seconds")
            st.metric("Consistency (standard deviation)", f"{std_interval:.3f} seconds")
            
            # Simple visualization
            fig = px.line(
                x=range(len(intervals)), 
                y=intervals,
                labels={"x": "Tap Number", "y": "Time Between Taps (seconds)"},
                title="Tapping Rhythm Pattern"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Analysis button
    if st.button("Analyze Results", type="primary"):
        with st.spinner("Analyzing your results..."):
            # Calculate symptom score
            symptom_score = sum(1 for s in st.session_state.symptoms.values() if s == "Yes")
            
            # Calculate risk score (simplified for demo)
            risk_score = symptom_score * 10
            
            if st.session_state.tapping_data:
                # Analyze tapping data
                tapping_analysis = process_tapping_data(st.session_state.tapping_data)
                risk_score += tapping_analysis['risk_contribution']
                risk_score = min(100, risk_score)
            
            st.session_state.risk_score = risk_score
            
            # Display results
            st.success("Analysis complete!")
            
            if risk_score < 30:
                st.balloons()
                st.success(f"Your risk score is {risk_score}%. This suggests low risk. Keep monitoring your health regularly.")
            elif risk_score < 70:
                st.warning(f"Your risk score is {risk_score}%. This suggests moderate risk. Consider discussing these results with a healthcare provider.")
            else:
                st.error(f"Your risk score is {risk_score}%. This suggests higher risk. Please consult with a healthcare professional for a proper evaluation.")
            
            st.info("**Remember:** This is a screening tool, not a medical diagnosis. Always consult with healthcare professionals for medical advice.")

def show_connect():
    st.header("Connect with Caregivers")
    
    if st.session_state.risk_score is None:
        st.warning("Please complete the assessment in the Detect section first.")
        return
    
    st.subheader("Your Health Report")
    
    # Generate report
    report = generate_report(st.session_state.symptoms, st.session_state.risk_score)
    
    # Display report
    col1, col2 = st.columns(2)
    
    with col1:
        st.json(report)
    
    with col2:
        st.info("""
        **How to use this report:**
        - Download and share with your healthcare provider
        - Keep for your records to track changes over time
        - Use as a basis for discussion about your neurological health
        """)
    
    # Download buttons
    st.download_button(
        label="Download Report as JSON",
        data=report,
        file_name="neurobridge_health_report.json",
        mime="application/json"
    )
    
    # Generate a simple text summary
    text_report = generate_text_report(st.session_state.symptoms, st.session_state.risk_score)
    st.download_button(
        label="Download Summary as Text",
        data=text_report,
        file_name="neurobridge_summary.txt",
        mime="text/plain"
    )
    
    st.session_state.report_generated = True
    st.success("Report generated successfully!")
def show_personalize():
    st.header("Personalized Care Plan")

    if not st.session_state.report_generated:
        st.warning("Please generate a report in the Connect section first.")
        return

    st.subheader("Care Plan Assistant")
    st.info("Ask questions about your results or get personalized recommendations for improving your neurological health.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.divider()
    st.subheader("💡 Suggested Questions")

    questions = [
        "What lifestyle changes can I make to improve my neurological health?",
        "How accurate is this assessment?",
        "What are the early signs of Parkinson's disease?",
        "Should I see a specialist based on my results?",
        "What exercises can help with coordination?",
        "How often should I monitor my symptoms?"
    ]

    # Suggested question buttons
    cols = st.columns(2)  # two buttons per row
    for i, question in enumerate(questions):
        with cols[i % 2]:
            if st.button(question, key=f"q_{i}"):
                handle_user_query(question)

    st.divider()

    # Normal chat input
    if prompt := st.chat_input("Ask a question about your care plan..."):
        handle_user_query(prompt)

    # Clear chat option
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


def handle_user_query(query: str):
    """Helper to handle both button questions and chat input uniformly"""
    # Show user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(query, st.session_state.risk_score)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})



def show_about():
    st.header("About NeuroBridge")
    st.markdown("""
    NeuroBridge is a tool designed to help with early detection of neurological conditions like Parkinson's disease.
    
    ### How it Works
    - **Detect**: We use a combination of symptom tracking and motor function tests (like finger tapping) to identify potential early signs
    - **Connect**: We generate easy-to-share reports that help facilitate conversations with healthcare providers
    - **Personalize**: Our AI system provides tailored recommendations based on your specific results
    
    ### Disclaimer
    NeuroBridge is a screening tool, not a diagnostic device. Always consult with qualified healthcare professionals for medical advice and diagnosis.
    
    ### For Developers
    This project was built for the NeuraViaHacks hackathon using:
    - Streamlit for the web interface
    - Python for data processing and analysis
    - Plotly for visualizations
    
    The source code is available on [GitHub](https://github.com/yourusername/neurobridge).
    """)

def run_tapping_test():
    """Simulate a tapping test by recording timestamps"""
    st.info("Get ready to tap! Click the button below as fast as you can for 10 seconds.")
    
    # In a real implementation, you would use:
    # 1. JavaScript with Streamlit components for real tapping capture
    # 2. Or streamlit-webrtc for mobile camera-based tapping detection
    
    # For demo purposes, we'll simulate tapping data
    if st.button("Tap here repeatedly", key="tap_button"):
        start_time = time.time()
        taps = []
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate tapping for 10 seconds
        while time.time() - start_time < 10:
            # Wait for user to click again (simulated)
            time.sleep(0.2)
            taps.append(time.time())
            progress = (time.time() - start_time) / 10
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Time remaining: {10 - (time.time() - start_time):.1f} seconds")
        
        progress_bar.empty()
        status_text.empty()
        st.success("Tapping test completed!")
        return taps
    
    return None

def generate_report(symptoms, risk_score):
    """Generate a JSON health report"""
    report = {
        "risk_score": risk_score,
        "symptoms": {s: v for s, v in symptoms.items()},
        "date": str(datetime.now()),
        "recommendations": [
            "Discuss these results with a healthcare provider",
            "Monitor symptoms regularly",
            "Consider lifestyle modifications like regular exercise and a balanced diet"
        ]
    }
    return json.dumps(report, indent=2)

def generate_text_report(symptoms, risk_score):
    """Generate a text summary report"""
    positive_symptoms = [s for s, v in symptoms.items() if v == "Yes"]
    
    report = f"""NeuroBridge Health Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}

Risk Score: {risk_score}%

Symptoms reported:
{', '.join(positive_symptoms) if positive_symptoms else 'None'}

Recommendations:
- Discuss these results with a healthcare provider
- Monitor symptoms regularly
- Consider lifestyle modifications like regular exercise and a balanced diet

Note: This is a screening tool, not a medical diagnosis.
"""
    return report

if __name__ == "__main__":
    main()