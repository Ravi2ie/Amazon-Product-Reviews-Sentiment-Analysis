# import streamlit as st

# # Must be the first Streamlit command
# st.set_page_config(page_title="Amazon Review Finder", layout="wide")

# import pandas as pd
# import re

# # --- Function to extract ASIN from a URL ---
# def extract_asin(url):
#     match = re.search(r'/dp/([A-Z0-9]{10})|/gp/product/([A-Z0-9]{10})', url)
#     if match:
#         return match.group(1) or match.group(2)
#     return None

# # --- Load dataset ---
# @st.cache_data
# def load_data():
#     df = pd.read_csv("../../Amazon product reviews/1429_1.csv")
#     df['reviews.date'] = pd.to_datetime(df['reviews.date'], errors='coerce')
#     return df

# df = load_data()

# # --- Streamlit UI ---
# st.title("🔍 Amazon Product Review Explorer by URL")

# url_input = st.text_input("Paste the Amazon product URL:")

# if url_input:
#     asin = extract_asin(url_input)
    
#     if asin:
#         st.success(f"✅ ASIN extracted: `{asin}`")

#         asin_reviews = df[df['asins'].astype(str).str.contains(asin, na=False)]

#         if not asin_reviews.empty:
#             st.markdown(f"### 📋 Showing reviews for ASIN: `{asin}`")
#             st.dataframe(asin_reviews[['reviews.date', 'reviews.rating', 'reviews.text', 'reviews.username']])
#         else:
#             st.warning("❌ No reviews found for this ASIN in the dataset.")
#     else:
#         st.error("❌ Could not extract ASIN from the URL. Please make sure it's a valid Amazon product URL.")
import streamlit as st
import requests
from bs4 import BeautifulSoup

# Function to extract ASIN from Amazon product URL
def extract_asin(url):
    parts = url.split("/")
    if "dp" in parts:
        return parts[parts.index("dp") + 1]
    elif "product" in parts:
        return parts[parts.index("product") + 1]
    return None

# Function to get reviews using ScraperAPI
def get_reviews(asin, api_key):
    review_url = f"https://www.amazon.in/product-reviews/{asin}"
    
    # ScraperAPI request
    params = {
        "api_key": api_key,
        "url": review_url
    }
    
    response = requests.get("http://api.scraperapi.com", params=params)

    if response.status_code != 200:
        return ["Failed to fetch reviews. Try a different URL or check your connection."]
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")
    reviews = soup.find_all("span", {"data-hook": "review-body"})

    return [review.get_text(strip=True) for review in reviews]

# Streamlit app
st.title("🛒 Amazon Product Review Extractor with ScraperAPI")
st.write("Enter an Amazon product URL to fetch and display its customer reviews.")

# Input for URL
url = st.text_input("🔗 Amazon Product URL", "")

# Input for ScraperAPI key
api_key = "bc49030572f59c343484de3828a27dd9"

# On button click
if st.button("Get Reviews") and url and api_key:
    # Extract ASIN from URL
    asin = extract_asin(url)
    
    if asin:
        with st.spinner("Fetching reviews..."):
            reviews = get_reviews(asin, api_key)
            if reviews:
                st.success(f"Found {len(reviews)} reviews.")
                for i, review in enumerate(reviews[:10], 1):  # Show only first 10 reviews
                    st.markdown(f"**Review {i}:** {review}")
            else:
                st.warning("No reviews found or could not parse the page.")
    else:
        st.error("❌ Could not extract ASIN from the URL. Please make sure it's a valid Amazon product link.")
