import streamlit as st
import requests
import json
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from fpdf import FPDF
import os
import re
from constants import states_cities

# Set up logging
log_filename = "log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_filename,
    filemode="w",
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

# Define available states and cities
states_cities = states_cities

neighborhoods = {
    "New York City": ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
    "Los Angeles": ["Hollywood", "Beverly Hills", "Downtown", "Santa Monica"],
    "San Francisco": ["Financial District", "Mission District", "SoMa", "Chinatown"],
    "Miami": ["Downtown", "Wynwood", "South Beach", "Little Havana"],
}

# Set custom Streamlit styles
st.markdown(
    """
    <style>
    .main {
        background-color: #00171f;
    }
    .stButton>button {
        background-color: #003459;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        margin-top: 10px;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: black;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #cccccc;
    }
    .stSlider>div>div>div>div {
        background: linear-gradient(90deg, #007ea7, #00a8e8);
        border-radius: 8px;
    }
    .stSelectbox>div>div>div>div {
        background-color: #ffffff;
        color: black;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #cccccc;
    }
    .stMarkdown {
        color: #ffffff;
        font-size: 16px;
    }
    .css-1v0mbdj.e16nr0p33 {
        background-color: #ffffff;
        color: #007ea7;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #007ea7;
    }
    .css-17e8joj.e1fqkh3o10 {
        background-color: #003459;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
    }
    .kpi {
        background-color: #003459;
        color: #ffffff;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
        font-size: 24px;
    }
    .photo-container {
        display: flex;
        overflow-x: auto;
    }
    .photo-container img {
        margin-right: 10px;
        border-radius: 8px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def search_bars(api_key, search_terms, city, state, neighborhood, limit):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"

    all_results = []
    total_results = 0

    query = f"{search_terms} in {neighborhood}, {city}, {state}"
    st.write(f"Searching for '{query}'...")
    params = {"query": query, "key": api_key}

    next_page_token = None

    while True:
        if next_page_token:
            params["pagetoken"] = next_page_token

        response = requests.get(url, params=params)
        data = json.loads(response.text)

        if "results" in data:
            new_results = len(data["results"])
            total_results += new_results
            st.write(
                f"Found {new_results} results for '{query}'. Total so far: {total_results}"
            )
            for place in data["results"]:
                st.write(f"Getting details for: {place['name']}")
                place_details = get_place_details(place["place_id"], api_key)
                if place_details:
                    place_details["search_term"] = search_terms
                    place_details["search_city"] = city
                    place_details["search_state"] = state
                    place_details["search_neighborhood"] = neighborhood
                    all_results.append(place_details)
                    st.write(f"Processed details for: {place_details['name']}")
                    if len(all_results) >= limit:
                        st.write(
                            f"Reached limit of {limit} businesses. Stopping search."
                        )
                        return all_results
                time.sleep(2)  # Avoid hitting rate limits

        if "next_page_token" in data:
            next_page_token = data["next_page_token"]
            st.write("More results available. Waiting before next request...")
            time.sleep(2)  # Required delay before using next_page_token
        else:
            st.write(f"No more results for '{query}'")
            break

    return all_results


def get_place_details(place_id, api_key):
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,formatted_address,address_components,website,rating,reviews,user_ratings_total,opening_hours,formatted_phone_number,geometry,photos,price_level,types",
        "key": api_key,
    }

    response = requests.get(url, params=params)
    data = json.loads(response.text)

    if "result" in data:
        result = data["result"]
        city, state = extract_city_state(result.get("address_components", []))
        price_level = interpret_price_level(result.get("price_level"))
        business_types = ", ".join(result.get("types", []))  # Get business types
        return {
            "name": result.get("name"),
            "address": result.get("formatted_address"),
            "city": city,
            "state": state,
            "website": result.get("website"),
            "rating": result.get("rating"),
            "reviews_count": result.get("user_ratings_total"),
            "reviews": result.get("reviews"),
            "opening_hours": result.get("opening_hours", {}).get("weekday_text"),
            "phone_number": result.get("formatted_phone_number"),
            "latitude": result["geometry"]["location"]["lat"],
            "longitude": result["geometry"]["location"]["lng"],
            "photos": get_place_photos(place_id, api_key),
            "price_level": price_level,
            "price_level_str": price_level_to_str(
                price_level
            ),  # Add descriptive price level
            "popularity_score": calculate_popularity_score(
                result.get("rating"), result.get("user_ratings_total")
            ),
            "types": business_types,  # Include business types
        }
    return None


def extract_city_state(address_components):
    city = next(
        (
            component["long_name"]
            for component in address_components
            if "locality" in component["types"]
        ),
        None,
    )
    state = next(
        (
            component["short_name"]
            for component in address_components
            if "administrative_area_level_1" in component["types"]
        ),
        None,
    )
    return city, state


def interpret_price_level(price_level):
    if price_level is None:
        return -1  # Use -1 for "Price information not available"
    return price_level


def get_place_photos(place_id, api_key, max_photos=5):
    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
    photo_url = "https://maps.googleapis.com/maps/api/place/photo"

    details_params = {"place_id": place_id, "fields": "photos", "key": api_key}
    details_response = requests.get(details_url, params=details_params)
    details_data = details_response.json()

    photos = []
    if "result" in details_data and "photos" in details_data["result"]:
        for photo in details_data["result"]["photos"][:max_photos]:
            photo_params = {
                "maxwidth": 400,
                "photoreference": photo["photo_reference"],
                "key": api_key,
            }
            photo_response = requests.get(photo_url, params=photo_params)
            if photo_response.status_code == 200:
                photos.append(photo_response.url)

    st.write(f"Retrieved {len(photos)} photos")
    return photos


def calculate_popularity_score(rating, reviews_count):
    if rating is None or reviews_count is None:
        return 0
    return (rating * 2) + (min(reviews_count, 1000) / 100)  # Max score: 10 + 10 = 20


def append_to_json(data, filename):
    if os.path.exists(filename):
        st.write(f"Appending to existing file: {filename}")
        with open(filename, "r+") as file:
            file_data = json.load(file)
            file_data.extend(data)
            file.seek(0)
            json.dump(file_data, file, indent=4)
    else:
        st.write(f"Creating new file: {filename}")
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
    st.write(f"Data saved to {filename}")
    logging.info(f"Data saved to {filename}")


def search_your_business(api_key, business_name):
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": business_name,
        "inputtype": "textquery",
        "fields": "place_id",
        "key": api_key,
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "candidates" in data and data["candidates"]:
        place_id = data["candidates"][0]["place_id"]
        place_details = get_place_details(place_id, api_key)
        return place_details
    return None


class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Business Analysis Report", 0, 1, "C")

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(4)

    def chapter_body(self, body):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_chapter(self, title, body):
        self.add_page()
        self.chapter_title(title)
        self.chapter_body(body)


def clean_text(text):
    # Remove or replace any characters that cannot be encoded in 'latin-1'
    return re.sub(r"[^\x00-\x7F]+", "", text)


def create_pdf_report(business_details, comparison_data, price_level_comparison):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Business details
    pdf.chapter_title("Your Business Details:")
    for key, value in business_details.items():
        if isinstance(value, list):
            # Ensure value is a list of strings
            value = [
                clean_text(str(v)) if not isinstance(v, dict) else json.dumps(v)
                for v in value
            ]
            value = ", ".join(value)
        else:
            value = clean_text(str(value))
        pdf.chapter_body(f"{key}: {value}")

    # Comparison data
    pdf.chapter_title("Comparison with Other Businesses:")
    for key, value in comparison_data.items():
        pdf.chapter_body(f"{key}: {clean_text(str(value))}")

    # Price level comparison
    pdf.chapter_title("Price Level Comparison:")
    pdf.chapter_body(
        f"Your Business Price Level: {clean_text(business_details['price_level_str'])}"
    )
    pdf.chapter_body(
        f"Average Competitors' Price Level: {clean_text(price_level_comparison)}"
    )

    # Save the PDF
    pdf_filename = "business_analysis_report.pdf"
    pdf.output(pdf_filename, "F")

    return pdf_filename


def price_level_to_str(price_level):
    price_map = {
        -1: "Price information not available",
        0: "Free",
        1: "Inexpensive",
        2: "Moderate",
        3: "Expensive",
        4: "Very Expensive",
    }
    return price_map.get(price_level, "Unknown")


def plot_comparison(df, business_details):
    sns.set(style="whitegrid")

    # Highlight user's business
    df["Color"] = df["BusinessType"].apply(lambda x: "red" if x == 1 else "blue")

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Average Rating Comparison
    sns.barplot(x="Rating", y="Business", data=df, ax=axes[0, 0], palette=df["Color"])
    axes[0, 0].set_title("Average Rating Comparison")
    axes[0, 0].set_xlabel("Average Rating")
    axes[0, 0].set_ylabel("Business")

    # Plot 2: Reviews Count
    sns.barplot(
        x="Reviews Count", y="Business", data=df, ax=axes[0, 1], palette=df["Color"]
    )
    axes[0, 1].set_title("Reviews Count Comparison")
    axes[0, 1].set_xlabel("Reviews Count")
    axes[0, 1].set_ylabel("Business")

    # Plot 3: Popularity Score
    sns.barplot(
        x="Popularity Score", y="Business", data=df, ax=axes[1, 0], palette=df["Color"]
    )
    axes[1, 0].set_title("Popularity Score Comparison")
    axes[1, 0].set_xlabel("Popularity Score")
    axes[1, 0].set_ylabel("Business")

    # Plot 4: Price Level Distribution
    sns.countplot(x="Price Level", data=df, ax=axes[1, 1], palette="viridis")
    axes[1, 1].set_title("Price Level Distribution")
    axes[1, 1].set_xlabel("Price Level")
    axes[1, 1].set_ylabel("Count")

    plt.tight_layout()
    st.pyplot(fig)


# Streamlit app
st.title("🚀 Business Competitive Analysis Dashboard")
st.markdown(
    """
    Welcome to the **Business Competitive Analysis Dashboard**! This powerful tool helps you:
    - 🔍 **Compare your business** with others in your area.
    - 📈 **Analyze trends** and gain insights into your competitors.
    - 🌐 **Improve your SEO presence** by understanding the market landscape.

    **Get started by entering your details below!** 💼
"""
)

api_key = st.text_input("🔑 Enter your Google API Key", type="password")

st.markdown("### 📊 Look Up Your Business")

business_name = st.text_input("Enter the name of your business")

if "business_details" not in st.session_state:
    st.session_state.business_details = None

if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None

if st.button("Look Up", key="lookup_button"):
    if api_key and business_name:
        with st.spinner("Looking up your business details..."):
            st.write(f"Looking up details for '{business_name}'")
            business_details = search_your_business(api_key, business_name)
            time.sleep(1)  # Simulating some delay

        if business_details:
            st.session_state.business_details = business_details

            st.write("### Your Business Details")
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown(
                f'<div class="kpi">Rating: {business_details["rating"]}</div>',
                unsafe_allow_html=True,
            )
            col2.markdown(
                f'<div class="kpi">Price: {business_details["price_level_str"]}</div>',
                unsafe_allow_html=True,
            )
            col3.markdown(
                f'<div class="kpi">Reviews: {business_details["reviews_count"]}</div>',
                unsafe_allow_html=True,
            )
            col4.markdown(
                f'<div class="kpi">Popularity: {business_details["popularity_score"]:.2f}</div>',
                unsafe_allow_html=True,
            )

            with st.expander("Reviews"):
                for review in business_details["reviews"]:
                    st.write(review["text"])

            with st.expander("Opening Hours"):
                for hours in business_details["opening_hours"]:
                    st.write(hours)

            # Map
            m = folium.Map(
                location=[business_details["latitude"], business_details["longitude"]],
                zoom_start=15,
            )
            folium.Marker(
                [business_details["latitude"], business_details["longitude"]],
                popup=business_details["name"],
            ).add_to(m)
            st_folium(m, width=500)  # Made the map smaller

            # Photos
            st.write("### Photos")
            st.markdown('<div class="photo-container">', unsafe_allow_html=True)
            for photo in business_details["photos"]:
                st.image(photo, width=200)
            st.markdown("</div>", unsafe_allow_html=True)

            # Extract business types
            business_types = business_details["types"].split(", ")
            st.session_state.business_types = business_types

if st.session_state.business_details:
    st.write("### Your Business Details")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f'<div class="kpi">Rating: {st.session_state.business_details["rating"]}</div>',
        unsafe_allow_html=True,
    )
    col2.markdown(
        f'<div class="kpi">Price: {st.session_state.business_details["price_level_str"]}</div>',
        unsafe_allow_html=True,
    )
    col3.markdown(
        f'<div class="kpi">Reviews: {st.session_state.business_details["reviews_count"]}</div>',
        unsafe_allow_html=True,
    )
    col4.markdown(
        f'<div class="kpi">Popularity: {st.session_state.business_details["popularity_score"]:.2f}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Reviews"):
        for review in st.session_state.business_details["reviews"]:
            st.write(review["text"])

    with st.expander("Opening Hours"):
        for hours in st.session_state.business_details["opening_hours"]:
            st.write(hours)

    # Map
    m = folium.Map(
        location=[
            st.session_state.business_details["latitude"],
            st.session_state.business_details["longitude"],
        ],
        zoom_start=15,
    )
    folium.Marker(
        [
            st.session_state.business_details["latitude"],
            st.session_state.business_details["longitude"],
        ],
        popup=st.session_state.business_details["name"],
    ).add_to(m)
    st_folium(m, width=500)  # Made the map smaller

    # Photos
    st.write("### Photos")
    st.markdown('<div class="photo-container">', unsafe_allow_html=True)
    for photo in st.session_state.business_details["photos"]:
        st.image(photo, width=200)
    st.markdown("</div>", unsafe_allow_html=True)

    # Allow user to select type for comparison
    selected_type = st.selectbox(
        "Select a business type to compare", st.session_state.business_types
    )

    # Additional input for custom business type
    custom_type = st.text_input("Enter a custom business type (optional)")

    state = st.selectbox("Select your state", list(states_cities.keys()))
    city = st.selectbox("Select your city", states_cities[state])
    neighborhood = st.selectbox("Select your neighborhood", neighborhoods.get(city, []))

    limit = st.slider(
        "Select the number of businesses to compare",
        min_value=1,
        max_value=50,
        value=10,
    )

    if st.button("Compare with Similar Businesses"):
        search_terms = selected_type
        if custom_type:
            search_terms = custom_type

        st.session_state.search_terms = search_terms
        st.session_state.city = city
        st.session_state.state = state
        st.session_state.neighborhood = neighborhood
        st.session_state.limit = limit

        with st.spinner("Comparing with similar businesses..."):
            all_bars = search_bars(
                api_key, search_terms, city, state, neighborhood, limit
            )
            st.session_state.comparison_results = all_bars
            time.sleep(1)  # Simulating some delay

if st.session_state.comparison_results:
    st.write("### Comparison Results")
    comparison_results = st.session_state.comparison_results

    # Append user's business details to comparison results
    user_business = st.session_state.business_details
    user_business["BusinessType"] = 1
    for result in comparison_results:
        result["BusinessType"] = 0

    comparison_results.append(user_business)

    # Ensure all dictionaries have the 'search_term' key
    for bar in comparison_results:
        bar.setdefault("search_term", "")

    # Convert results to DataFrame
    df = pd.DataFrame(comparison_results).drop_duplicates(subset=["name"])
    st.write("### Results", df)

    # Provide option to download as CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name="business_comparison_results.csv",
        mime="text/csv",
    )

    st.write("All data appended to JSON file")
    append_to_json(comparison_results, "specialty_bars_us_cities.json")
    st.write("Search complete. All data saved.")
    logging.info("Search complete. All data saved.")

    # Update column names for plotting
    df = df.rename(
        columns={
            "name": "Business",
            "rating": "Rating",
            "reviews_count": "Reviews Count",
            "popularity_score": "Popularity Score",
            "price_level": "Price Level",
        }
    )

    # Box plots and histograms
    plot_comparison(df, st.session_state.business_details)

    # Calculate averages for comparison metrics
    avg_rating = df["Rating"].mean()
    avg_reviews_count = df["Reviews Count"].mean()
    avg_popularity_score = df["Popularity Score"].mean()

    # Price level comparison
    avg_price_level = df["Price Level"].replace(-1, pd.NA).dropna().astype(int).mean()
    avg_price_level_str = price_level_to_str(int(round(avg_price_level)))

    st.write(f"### Comparison with Averages")
    st.write(f"**Your Business Rating:** {st.session_state.business_details['rating']}")
    st.write(f"**Average Rating:** {avg_rating}")
    st.write(
        f"**Your Business Review Count:** {st.session_state.business_details['reviews_count']}"
    )
    st.write(f"**Average Review Count:** {avg_reviews_count}")
    st.write(
        f"**Your Business Popularity Score:** {st.session_state.business_details['popularity_score']}"
    )
    st.write(f"**Average Competitors' Popularity Score:** {avg_popularity_score}")
    st.write(
        f"**Your Business Price Level:** {st.session_state.business_details['price_level_str']}"
    )
    st.write(f"**Average Competitors' Price Level:** {avg_price_level_str}")

    # Create PDF report
    pdf_filename = create_pdf_report(
        st.session_state.business_details, df.describe(), avg_price_level_str
    )
    st.download_button(
        label="Download PDF Report",
        data=open(pdf_filename, "rb").read(),
        file_name=pdf_filename,
        mime="application/pdf",
    )
