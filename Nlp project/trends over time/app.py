import streamlit as st
import requests
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#https://www.amazon.in/Umiwe-Womens-Shoulder-Sleeve-Blouse/dp/B0B9Y48Y7V
# --- Configuration --- (Gemini API Key)
GEMINI_API_KEY = "AIzaSyBvDqbIz9iov6yWMuEw3vC9j_yUkEkOyDo"

# --- Helper Functions for Product Finder ---
def is_url(text):
    return re.match(r'https?://', text.strip()) is not None

def extract_keywords_from_url(url):
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    prompt = f"Extract the main product name or keywords from this e-commerce URL: {url}"

    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(endpoint, headers=headers, json=data)

    try:
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return "Failed to extract keywords."

def get_detailed_related_products(keywords, platforms):
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    platform_lines = "\n".join([f"- {p}" for p in platforms])
    prompt = (
        f"Find 5 similar or related products to '{keywords}' from each of the following Indian e-commerce platforms:\n"
        f"{platform_lines}\n\n"
        "For each product, provide the following details:\n"
        "1. Product Name\n"
        "2. Direct URL\n"
        "3. Price (in ₹)\n"
        "4. Rating (out of 5)\n"
        "5. Short Description (1 line only)\n\n"
        "Format the result in plain text like this:\n"
        "Platform: Amazon India\n"
        "Product 1:\n"
        "🛍️ Name: <Product Name>\n"
        "🔗 Link: <URL>\n"
        "💰 Price: ₹<Price>\n"
        "⭐ Rating: <X>/5 (<Y> reviews)\n"
        "📦 Description: <One-line description>\n"
        "---"
    )

    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(endpoint, headers=headers, json=data)

    try:
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return "Failed to fetch related products."

def extract_links_by_platform(text):
    sections = re.split(r'Platform: ', text)
    platform_data = {}
    for section in sections[1:]:
        lines = section.strip().splitlines()
        if not lines:
            continue
        platform_name = lines[0].strip()
        links = re.findall(r"🔗 Link:\s*(https?://[^\s]+)", section)
        platform_data[platform_name] = links
    return platform_data

# --- Load Sentiment Data ---
@st.cache_data
def load_data():
    data = pd.read_csv("./final_sentiment_reviews.csv")
    data['reviews.date'] = pd.to_datetime(data['reviews.date'], errors='coerce')
    return data

df = load_data()

# --- Streamlit Layout ---
st.set_page_config(page_title="Product Finder & Sentiment Dashboard", layout="wide")

# --- Sidebar for Navigation ---
with st.sidebar:
    st.header("🧭 Navigation Panel")
    app_mode = st.radio("Select App Mode", ("Product Finder", "Product Sentiment Analysis"))

# --- Product Finder Section ---
if app_mode == "Product Finder":
    st.title("🛒 Product Finder")
    user_input = st.text_input(
        "🔎 Enter product name or URL",
        placeholder="E.g., 'wireless earbuds' or Amazon link"
    )

    available_platforms = ["Amazon India", "Flipkart", "Myntra", "Ajio", "Meesho"]
    selected_platforms = st.multiselect(
        "🛍️ Choose platforms",
        available_platforms,
        default=available_platforms
    )

    search_clicked = st.button("🔍 Find Related Products")

    if search_clicked and user_input.strip():
        if is_url(user_input):
            with st.spinner("Extracting keywords from URL..."):
                keywords = extract_keywords_from_url(user_input)
        else:
            keywords = user_input.strip()

        if "Failed" in keywords or not keywords:
            st.error("❌ Could not extract keywords. Please check your input.")
        else:
            st.success(f"📝 Using Product Keywords: {keywords}")

            with st.spinner("Searching for related products..."):
                raw_result = get_detailed_related_products(keywords, selected_platforms)

            if "Failed" in raw_result or not raw_result:
                st.error("❌ Could not fetch related products.")
            else:
                platform_links = extract_links_by_platform(raw_result)
                if platform_links:
                    st.markdown("## 🔗 Product Links by Platform")
                    for platform, links in platform_links.items():
                        with st.expander(f"🛒 {platform} ({len(links)} products)") :
                            for i, link in enumerate(links, 1):
                                st.markdown(
                                    f"""
                                    <div style='padding: 8px 0;'>
                                        <a href="{link}" target="_blank" style="text-decoration: none; font-weight: bold; color: #0072C6;">
                                            👉 {link}
                                        </a>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                st.markdown("## 🔗 Show Raw Product Details")
                with st.expander("📋 Show Raw Product Details"):
                    st.markdown(f"```text\n{raw_result}\n```")

# --- Product Sentiment Analysis Section ---
if app_mode == "Product Sentiment Analysis":
    st.title("📦 Amazon Product Sentiment Analysis Dashboard")

    # --- ASIN + Product Dropdown ---
    df['display_name'] = df['name'].fillna('Unknown') + " | " + df['asins']
    unique_products = df[['asins', 'display_name']].drop_duplicates().dropna()

    selected_display = st.selectbox("🔍 Select a Product (Name | ASIN)", unique_products['display_name'].tolist())

    selected_asin = selected_display.split(" | ")[-1]

    asin_df = df[df['asins'] == selected_asin].copy()

    # --- Filter by Date ---
    if asin_df.empty:
        st.warning("No data found for the selected ASIN.")
    else:
        min_date = asin_df['reviews.date'].min()
        max_date = asin_df['reviews.date'].max()

        start_date, end_date = st.date_input("📅 Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

        if start_date > end_date:
            st.error("❗ Start date must be before end date.")
        else:
            asin_df['reviews.date'] = asin_df['reviews.date'].dt.tz_localize(None)
            filtered_df = asin_df[(asin_df['reviews.date'] >= pd.to_datetime(start_date)) &
                                  (asin_df['reviews.date'] <= pd.to_datetime(end_date))]

            if filtered_df.empty:
                st.warning("No reviews available in selected date range.")
            else:
                st.success(f"Showing {len(filtered_df)} reviews from {start_date} to {end_date}")

                # --- Analysis Scorecard ---
                with st.expander("📋 Analysis Scorecard", expanded=True):
                    sentiment_counts = filtered_df['Sentiment'].value_counts()

                    total_reviews = len(filtered_df)
                    avg_rating = filtered_df['reviews.rating'].mean()
                    #avg_sentiment = filtered_df['Sentiment Rate'].mean()

                    positive_reviews = filtered_df[filtered_df['Sentiment'] == 'Positive']
                    negative_reviews = filtered_df[filtered_df['Sentiment'] == 'Negative']
                    pos_percent = (len(positive_reviews) / total_reviews) * 100
                    neg_percent = (len(negative_reviews) / total_reviews) * 100

                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("📝 Total Reviews", total_reviews)
                    col2.metric("⭐ Avg Rating", f"{avg_rating:.2f}")
                    #col3.metric("💬 Avg Sentiment", f"{avg_sentiment:.2f}")
                    col3.metric("👍 % Positive", f"{pos_percent:.1f}%")
                    col4.metric("👎 % Negative", f"{neg_percent:.1f}%")

                # --- Sentiment Trend ---
                with st.expander("📈 Sentiment Trend Over Time"):
                    sentiment_trend = filtered_df.groupby('reviews.date')['Sentiment Rate'].mean().reset_index()
                    fig1, ax1 = plt.subplots(figsize=(10, 4))
                    ax1.plot(sentiment_trend['reviews.date'], sentiment_trend['Sentiment Rate'], marker='o', color='teal')
                    ax1.set_title("Sentiment Trend Over Time")
                    ax1.set_xlabel("Date")
                    ax1.set_ylabel("Average Sentiment Rate")
                    ax1.grid(True)
                    st.pyplot(fig1)

                # --- Sentiment Distribution ---
                with st.expander("📊 Sentiment Distribution"):
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    sns.histplot(filtered_df['Sentiment Rate'], bins=5, kde=True, ax=ax2, color='royalblue')
                    ax2.set_title("Distribution of Sentiment Scores")
                    ax2.set_xlabel("Sentiment Rate")
                    st.pyplot(fig2)

                # --- Rating vs Sentiment ---
                with st.expander("🧩 Rating vs Sentiment"):
                    fig3, ax3 = plt.subplots(figsize=(8, 4))
                    sns.scatterplot(data=filtered_df, x='reviews.rating', y='Sentiment Rate', ax=ax3, color='darkorange')
                    ax3.set_title("Review Rating vs Sentiment")
                    ax3.set_xlabel("Review Rating")
                    ax3.set_ylabel("Sentiment Rate")
                    st.pyplot(fig3)

                # --- Positive vs Negative Sentiment ---
                with st.expander("🧠 Positive vs Negative Sentiment"):
                    sentiment_counts = filtered_df['Sentiment'].value_counts()
                    fig4, ax4 = plt.subplots()
                    ax4.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=["#4CAF50", "#F44336"])
                    ax4.set_title("Sentiment Breakdown")
                    st.pyplot(fig4)

                # --- Full Data Table ---
                with st.expander("📋 Show Filtered Reviews Table"):
                    st.dataframe(filtered_df[['reviews.date', 'reviews.username', 'reviews.rating',
                                              'Sentiment', 'Sentiment Rate', 'reviews.text']])




# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification
# import torch
# import os

# # --- Load sentiment model and tokenizer from local directory ---
# def load_sentiment_model():
#     local_model_path = "./local_model"
#     tokenizer = BertTokenizer.from_pretrained(local_model_path)
#     model = BertForSequenceClassification.from_pretrained(local_model_path)
#     return tokenizer, model

# tokenizer, model = load_sentiment_model()

# # --- Load reviews CSV ---
# df = pd.read_csv("../../Amazon product reviews/1429_1.csv")
# df['reviews.date'] = pd.to_datetime(df['reviews.date'], errors='coerce')

# # Filter rows with valid review text
# df = df[df['reviews.text'].notna()].reset_index(drop=True)

# # --- Map model output to readable label and category ---
# def map_sentiment_label(label):
#     label_map = {
#         0: "1 star",
#         1: "2 stars",
#         2: "3 stars",
#         3: "4 stars",
#         4: "5 stars"
#     }
#     return label_map.get(label, "Unknown")

# def sentiment_category(score):
#     return "Positive" if score >= 3 else "Negative"

# # --- Sentiment Analysis ---
# sentiment_labels = []
# sentiment_scores = []
# sentiment_types = []
# sentiment_rate = []

# print("Starting sentiment analysis...\n")

# for i, review in enumerate(df['reviews.text']):
#     inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         predicted_class_id = torch.argmax(logits, dim=1).item()

#     label = map_sentiment_label(predicted_class_id)
#     category = sentiment_category(predicted_class_id)
#     star_rating = predicted_class_id + 1

#     sentiment_labels.append(label)
#     sentiment_scores.append(predicted_class_id)
#     sentiment_types.append(category)
#     sentiment_rate.append(star_rating)

#     print(f"[{i+1}/{len(df)}] Processed review => Sentiment: {label}, Category: {category}")

# # --- Add to DataFrame ---
# df['sentiment_label'] = sentiment_labels
# df['sentiment_score'] = sentiment_scores
# df['Sentiment'] = sentiment_types
# df['Sentiment Rate'] = sentiment_rate

# # --- Save to CSV ---
# output_file = "final_sentiment_reviews.csv"
# df.to_csv(output_file, index=False)

# print(f"\n✅ Sentiment analysis complete. Results saved to `{output_file}`.")
