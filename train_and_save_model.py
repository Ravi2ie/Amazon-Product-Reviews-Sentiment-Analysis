# # train_and_save_model.py

# import pandas as pd
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# from sklearn.model_selection import train_test_split
# from transformers import DataCollatorWithPadding
# import torch

# # Load dataset
# dataset = load_dataset("amazon_reviews_multi", "en", split="train[:2%]")  # small sample

# # Keep relevant columns
# df = dataset.to_pandas()[['review_body', 'stars']].dropna()
# df['label'] = df['stars'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))  # 0: Negative, 1: Neutral, 2: Positive

# # Train-test split
# train_texts, val_texts, train_labels, val_labels = train_test_split(df['review_body'], df['label'], test_size=0.2)

# # Tokenizer
# checkpoint = "distilbert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
# val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

# class ReviewDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels
#     def __getitem__(self, idx):
#         return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {'labels': torch.tensor(self.labels[idx])}
#     def __len__(self):
#         return len(self.labels)

# train_dataset = ReviewDataset(train_encodings, list(train_labels))
# val_dataset = ReviewDataset(val_encodings, list(val_labels))

# # Model
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

# training_args = TrainingArguments(
#     output_dir="./amazon_review_model",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     num_train_epochs=2,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     logging_dir="./logs",
#     save_total_limit=1,
#     load_best_model_at_end=True,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     tokenizer=tokenizer,
#     data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
# )

# trainer.train()

# # Save model
# model.save_pretrained("saved_model")
# tokenizer.save_pretrained("saved_model")

import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Load sentiment model and tokenizer from local directory ---
@st.cache_resource
def load_sentiment_model():
    local_model_path = "./local_model"  # Ensure this folder contains config.json, pytorch_model.bin, vocab.txt, etc.
    tokenizer = BertTokenizer.from_pretrained(local_model_path)
    model = BertForSequenceClassification.from_pretrained(local_model_path)
    return tokenizer, model

tokenizer, model = load_sentiment_model()

# --- Load product reviews data ---
@st.cache_data
def load_data():
    df = pd.read_csv("../../Amazon product reviews/1429_1.csv")
    df['reviews.date'] = pd.to_datetime(df['reviews.date'], errors='coerce')  # Ensure proper datetime format
    return df

df = load_data()

# --- Helper to map prediction to star labels ---
def map_sentiment(label):
    label_map = {
        0: "1 star",
        1: "2 stars",
        2: "3 stars",
        3: "4 stars",
        4: "5 stars"
    }
    return label_map[label]

# --- Streamlit App Layout ---
st.set_page_config(page_title="Amazon Reviews Sentiment Explorer", layout="wide")
st.title("📦 Amazon Product Reviews Explorer with Sentiment Analysis")

# Input
asin_input = st.text_input("🔍 Enter ASIN ID to view reviews and sentiment analysis")

# If ASIN is entered
if asin_input:
    asin_reviews = df[
        df['asins'].astype(str).str.contains(asin_input, na=False) &
        df['reviews.date'].notna()
    ]

    if not asin_reviews.empty:
        # Earliest review
        earliest_review_date = asin_reviews['reviews.date'].min()
        st.success(f"🕐 Earliest review for ASIN `{asin_input}` was on **{earliest_review_date.date()}**")

        # Sort by date
        asin_reviews_sorted = asin_reviews.sort_values(by='reviews.date')

        # Sentiment Analysis
        st.markdown("### 🧠 Running Sentiment Analysis...")
        sentiment_labels = []
        sentiment_scores = []

        progress_text = "🔍 Analyzing sentiment of reviews..."
        progress_bar = st.progress(0, text=progress_text)

        total_reviews = len(asin_reviews_sorted)
        for i, review in enumerate(asin_reviews_sorted['reviews.text']):
            inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_id = torch.argmax(logits, dim=1).item()

            sentiment_label = map_sentiment(predicted_class_id)
            sentiment_labels.append(sentiment_label)
            sentiment_scores.append(predicted_class_id)

            progress_bar.progress((i + 1) / total_reviews, text=f"{progress_text} ({i + 1}/{total_reviews})")

        # Add results to DataFrame
        asin_reviews_sorted['sentiment_label'] = sentiment_labels
        asin_reviews_sorted['sentiment_score'] = sentiment_scores

        # --- Show Results in Expanders ---
        st.markdown(f"### 📋 Reviews with Sentiment for ASIN `{asin_input}`")
        with st.expander("📄 View All Reviews Table"):
            st.dataframe(asin_reviews_sorted[['reviews.date', 'reviews.rating', 'sentiment_label', 'reviews.text', 'reviews.username']])

        # --- 📈 Sentiment Trend Over Time ---
        st.markdown("### 📈 Sentiment Trend Over Time")
        sentiment_trend = asin_reviews_sorted.groupby('reviews.date')['sentiment_score'].mean().reset_index()
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(sentiment_trend['reviews.date'], sentiment_trend['sentiment_score'], marker='o', color='b')
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Avg Sentiment Score")
        ax1.set_title(f"Sentiment Score Trend for ASIN: {asin_input}")
        ax1.grid(True)
        st.pyplot(fig1)

        # --- 📊 Sentiment Score Distribution ---
        st.markdown("### 📊 Sentiment Score Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.histplot(asin_reviews_sorted['sentiment_score'], bins=5, kde=True, ax=ax2)
        ax2.set_xlabel("Sentiment Score (0 = 1 star, 4 = 5 stars)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Sentiment Scores")
        st.pyplot(fig2)

        # --- 🧩 Review Rating vs Sentiment ---
        st.markdown("### 🧩 Review Rating vs Sentiment Score")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=asin_reviews_sorted, x='reviews.rating', y='sentiment_score', ax=ax3, color='purple')
        ax3.set_xlabel("Review Rating (1-5)")
        ax3.set_ylabel("Sentiment Score (0-4)")
        ax3.set_title("Rating vs. Sentiment Score")
        st.pyplot(fig3)

    else:
        st.warning("No reviews found for the given ASIN.")
