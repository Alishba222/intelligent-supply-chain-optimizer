#!/usr/bin/env python3
"""Enhanced NLP Chatbot Demo - Test Any Query"""

from nlp_chatbot import SupplyChainChatbot

print("="*70)
print("🤖 ENHANCED NLP CHATBOT - NOW ANSWERS ANYTHING!")
print("="*70)

# Initialize
chatbot = SupplyChainChatbot()
if not chatbot.load_data():
    print("❌ Failed to load data")
    exit(1)

print("\n✅ Data loaded successfully!")
print(f"   {len(chatbot.df):,} rows, {len(chatbot.df.columns)} columns\n")

# Sample questions showing new capabilities
sample_questions = [
    "How many total orders?",
    "Show metrics",
    "What's the delay rate?",
    "Top cities with delays",
    "Average shipping days",
    "Which products sell most?",
    "Revenue metrics",
    "Shipping modes",
    "Top customers",
    "Show me data from Karachi",
    "What columns available?",
    "Performance summary",
    "Humidity levels",
    "Tell me everything",
]

print("="*70)
print("TESTING ENHANCED CHATBOT WITH VARIOUS QUESTIONS:")
print("="*70)

for i, question in enumerate(sample_questions, 1):
    print(f"\n{i}. 👤 You: {question}")
    answer = chatbot.answer_query(question)
    print(f"   🤖 Bot: {answer}")
    print("   " + "-"*66)

print(f"\n\n✅ Successfully answered {chatbot.queries_answered} different questions!")
print("\n" + "="*70)
print("STARTING INTERACTIVE MODE - Ask anything you want!")
print("="*70)
chatbot.interactive_chat()
