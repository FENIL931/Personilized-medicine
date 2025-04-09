import streamlit as st
import numpy as np
import pandas as pd

def show_drug_interaction_page():
    # Set page configuration for the drug interaction page
    # Removed to avoid warnings since it should only be called once

    # Add a back button
    if st.button("← Back to Main App"):
        st.session_state['drug_interaction_page'] = False
        st.experimental_rerun()
    
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: #1565c0;'>💊 Drug Interaction Analyzer</h1>
            <p style='color: #666; font-size: 1.2em;'>
                Analyze potential interactions between medications
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        drug1 = st.selectbox(
            "Select Primary Drug",
            ["Drug A", "Drug B", "Drug C", "Drug D"]
        )
        
    with col2:
        drug2 = st.selectbox(
            "Select Secondary Drug",
            ["Drug X", "Drug Y", "Drug Z"]
        )
    
    # Additional options
    st.subheader("Additional Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dosage1 = st.number_input("Primary Drug Dosage (mg)", 10, 1000, 100)
    
    with col2:
        dosage2 = st.number_input("Secondary Drug Dosage (mg)", 10, 1000, 100)
    
    with col3:
        patient_age = st.number_input("Patient Age", 18, 100, 50)
    
    # Validate user inputs
    if dosage1 <= 0 or dosage2 <= 0:
        st.error("Dosage must be greater than 0.")
        return
    if patient_age < 18:
        st.error("Patient age must be at least 18.")
        return
    
    # Use a unique key for this button to avoid conflicts with other buttons
    if st.button("Check Interaction", key="drug_interaction_check"):
        with st.spinner("Analyzing potential interactions..."):
            # Placeholder for actual analysis logic
            # Replace the following with real interaction analysis
            interaction_level = "Low"  # Placeholder
            confidence = 0.85  # Placeholder
            
            st.markdown("""
                <div style='background: linear-gradient(45deg, #f8f9fa, #e9ecef);
                          padding: 20px; border-radius: 15px; margin-top: 20px;'>
                <h4 style='color: #1565c0;'>Interaction Analysis Results</h4>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Risk Level", interaction_level)
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            
            # Detailed analysis
            st.subheader("Detailed Analysis")
            
            # Create a sample interaction details dataframe
            interaction_details = pd.DataFrame({
                'Effect': ['Increased toxicity', 'Reduced efficacy', 'Metabolic inhibition'],
                'Severity': ['Low', 'Medium', 'High'],
                'Evidence Level': ['Strong', 'Moderate', 'Limited']
            })
            
            st.dataframe(interaction_details, use_container_width=True)
            
            # Recommendations
            st.info("Recommended Action: Monitor patient closely for potential side effects.")
            
            # Additional visualizations
            st.subheader("Interaction Timeline")
            
            timeline_data = pd.DataFrame({
                'Time (hours)': range(0, 24, 4),
                'Interaction Strength': np.random.rand(6) * 100
            })
            
            st.line_chart(timeline_data.set_index('Time (hours)'))
