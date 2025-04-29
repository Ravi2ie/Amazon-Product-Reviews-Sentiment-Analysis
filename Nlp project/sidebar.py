import streamlit as st
import requests
import re

# --- Configuration ---
GEMINI_API_KEY = "AIzaSyBvDqbIz9iov6yWMuEw3vC9j_yUkEkOyDo"

# --- Helper Functions ---
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

# --- Streamlit UI ---
st.set_page_config(page_title="Product Finder", layout="wide")

with st.sidebar:
    st.header("🧭 Product Finder Panel")
    user_input = st.text_input(
        "🔎 Enter product name or URL",
        placeholder="E.g., 'wireless earbuds' or Amazon link"
    )

    available_platforms = ["Amazon India", "Flipkart", "Myntra", "Ajio", "Meesho"]
    selected_platforms = st.multiselect(
        "🛒 Choose platforms",
        available_platforms,
        default=available_platforms
    )

    search_clicked = st.button("🔍 Find Related Products")

# Main content area
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
            st.markdown(
                """
                <style>
                .slidein {
                    animation: slide-in 0.5s forwards;
                }
                @keyframes slide-in {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                </style>
                <div class='slidein'>
                """,
                unsafe_allow_html=True
            )

            platform_links = extract_links_by_platform(raw_result)
            if platform_links:
                st.markdown("## 🔗 Product Links by Platform")
                for platform, links in platform_links.items():
                    with st.expander(f"🛒 {platform} ({len(links)} products)"):
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
            else:
                st.warning("⚠️ No product links were found in the response.")

            # Product Details in expander
            st.markdown("## 🔗  Show Raw Product Details")
            with st.expander("📋 Show Raw Product Details"):
                st.markdown(f"```text\n{raw_result}\n```")

            st.markdown("</div>", unsafe_allow_html=True)
