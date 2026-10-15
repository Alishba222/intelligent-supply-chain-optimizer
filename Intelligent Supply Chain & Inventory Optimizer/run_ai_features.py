#!/usr/bin/env python3
"""
INTELLIGENT SUPPLY CHAIN OPTIMIZER - AI FEATURES DEMO
Demonstrates all 3 AI features:
1. Delay Predictor (ML Classification)
2. NLP Chatbot (Natural Language Processing)
3. Sentiment Analyzer (Emotion Detection)
"""

import os
import sys
from delay_predictor import DelayPredictor
from nlp_chatbot import SupplyChainChatbot
from sentiment_analyzer import DriverSentimentAnalyzer

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_ai_features():
    """Run all AI features"""
    
    print_header("🤖 INTELLIGENT SUPPLY CHAIN OPTIMIZER - AI FEATURES")
    
    # Load data once
    print("📂 Loading data...")
    import pandas as pd
    
    try:
        df = pd.read_csv('integrated_data_warehouse.csv')
        print(f"✅ Integrated warehouse loaded: {df.shape[0]} rows")
    except:
        try:
            df = pd.read_csv('dataco_postgresql_structured.csv')
            print(f"✅ PostgreSQL CSV loaded: {df.shape[0]} rows")
        except:
            print("❌ ERROR: No data found")
            return
    
    print("\n" + "="*70)
    print("  AVAILABLE AI FEATURES")
    print("="*70)
    print("""
    1️⃣  DELAY PREDICTOR
        - Predicts order delays using Machine Learning
        - Random Forest classifier
        - Risk scoring: 0-100%
        - Decision: Recommend action based on risk
    
    2️⃣  NLP CHATBOT
        - Natural Language queries about supply chain
        - Ask: "Which city has most delays?"
        - Answers: City analysis, metrics, performance
    
    3️⃣  SENTIMENT ANALYZER
        - Analyzes driver mood/stress from logs
        - Detects: Fatigue, Frustration, Safety Risks
        - Actions: Alerts and recommendations
    
    4️⃣  ALL FEATURES (Run everything)
    5️⃣  EXIT
    """)
    
    while True:
        choice = input("\n👉 Select feature (1-5): ").strip()
        
        if choice == '1':
            print_header("1. DELAY PREDICTOR - MACHINE LEARNING MODEL")
            run_delay_predictor(df)
        
        elif choice == '2':
            print_header("2. NLP CHATBOT - NATURAL LANGUAGE QUERIES")
            run_nlp_chatbot(df)
        
        elif choice == '3':
            print_header("3. SENTIMENT ANALYZER - DRIVER EMOTION DETECTION")
            run_sentiment_analyzer(df)
        
        elif choice == '4':
            print_header("RUNNING ALL AI FEATURES")
            run_delay_predictor(df)
            input("\n➡️ Press Enter to continue to NLP Chatbot...")
            run_nlp_chatbot(df)
            input("\n➡️ Press Enter to continue to Sentiment Analyzer...")
            run_sentiment_analyzer(df)
        
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Try again.")


def run_delay_predictor(df):
    """Run delay predictor on the ENTIRE dataset"""
    predictor = DelayPredictor()
    
    # Train model (using balanced data logic we added)
    predictor.train(df)
    
    # --- UPDATED SECTION: Full Dataset Processing ---
    print("\n🚀 Processing ALL 1,000,000+ orders... AI is thinking, please wait.")
    
    # 1. Poore dataset par prediction run karein
    predictions = predictor.predict_batch(df) 
    
    if predictions is not None:
        # 2. Poore results ko aik nayi file mein save kar dein
        predictions.to_csv('all_order_predictions_full.csv', index=False)
        print("✅ SUCCESS: AI has predicted delays for the entire dataset!")
        print("📂 Full report saved as: all_order_predictions_full.csv")

        # 3. Terminal mein variety dikhane ke liye random 50 samples
        print("\n--- PREDICTION VARIETY (Random Samples from 1M records) ---")
        # Check if we have at least 50 rows, otherwise take what we have
        sample_size = min(50, len(predictions))
        print(predictions.sample(sample_size)[['delay_percentage']].to_string())
        
        print("\n💡 Tip: Open 'all_order_predictions_full.csv' to see the risk score for every single order.")
    else:
        print("❌ ERROR: Prediction failed.")

def run_nlp_chatbot(df):
    """Run NLP chatbot"""
    chatbot = SupplyChainChatbot(df)
    
    print("🤖 CHATBOT - Ask questions about your supply chain\n")
    
    # Sample questions
    sample_questions = [
        "Which city has the most delays?",
        "What is the average shipping time?",
        "What is our delay rate?",
        "Show performance metrics"
    ]
    
    print("📋 Sample Queries:\n")
    for q in sample_questions:
        print(f"  👤 You: {q}")
        answer = chatbot.answer_query(q)
        print(f"  🤖 Bot: {answer}\n")
    
    # Interactive mode
    interactive = input("\n🎯 Start interactive chat? (y/n): ").lower()
    if interactive == 'y':
        chatbot.interactive_chat()


def run_sentiment_analyzer(df):
    """Run sentiment analyzer"""
    analyzer = DriverSentimentAnalyzer()
    
    print("💭 SENTIMENT ANALYZER - Analyzing driver emotions\n")
    
    # Analyze logs
    results_df = analyzer.analyze_batch(df)
    
    if results_df is not None:
        # Show summary
        print(analyzer.get_summary_report())
        
        # Show top risk factors
        print("\n🚨 TOP RISK FACTORS:")
        for risk, count in analyzer.get_top_risk_factors():
            print(f"  {risk}: {count} occurrences")
        
        # Show critical drivers
        critical_drivers = analyzer.get_critical_drivers()
        if critical_drivers:
            print(f"\n⚠️ CRITICAL DRIVERS DETECTED: {len(critical_drivers)}")
            print("\nSample critical drivers:")
            for i, driver in enumerate(critical_drivers[:3]):
                print(f"\n  Driver {i+1}:")
                print(f"    Stress: {driver['stress_score']}/100 ({driver['stress_level']})")
                print(f"    Emotions: {', '.join(driver['emotions'][:3])}")
                print(f"    Action: {driver['recommendation']}")
        
        # Save results
        results_df.to_csv('driver_sentiment_analysis.csv', index=False)
        print(f"\n✅ Analysis saved to driver_sentiment_analysis.csv")


if __name__ == "__main__":
    try:
        run_ai_features()
    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
