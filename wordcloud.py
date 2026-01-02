# streamlit_app.py

import streamlit as st
import pandas as pd
import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Helper Functions ---
def fetch_reddit_posts(topic, limit=500):
    # Reddit RSS search URL for a topic
    url = f"https://www.reddit.com/r/all/search.rss?q={topic}&limit={limit}"
    feed = feedparser.parse(url)
    titles = [entry.title for entry in feed.entries]
    return titles

def fetch_twitter_posts(topic, limit=500):
    # Placeholder: Replace with Tweepy / Twitter API code
    # Should return a list of text tweets for the given topic
    st.warning("Twitter API not configured. Please add your API keys and fetching code.")
    return ["Sample tweet about " + topic] * limit

def fetch_facebook_posts(topic, limit=500):
    # Placeholder: Replace with Facebook Graph API code
    st.warning("Facebook API not configured. Please add your API keys and fetching code.")
    return ["Sample Facebook post about " + topic] * limit

def generate_wordcloud(text_list):
    df = pd.DataFrame(text_list, columns=["text"])
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    scores = tfidf_matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()
    tfidf_dict = dict(zip(words, scores))

    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(tfidf_dict)
    return wc

# --- Streamlit App ---
st.set_page_config(page_title="Social Media WordCloud Generator", layout="wide")
st.title("Social Media WordCloud Generator")

# Tabs for social media
tab1, tab2, tab3 = st.tabs(["Reddit", "Twitter", "Facebook"])

with tab1:
    st.header("Reddit WordCloud")
    topic = st.text_input("Enter topic for Reddit:", "silver")
    num_posts = st.slider("Number of posts to fetch:", 500, 5000, 1000, step=100)
    if st.button("Generate Reddit WordCloud"):
        with st.spinner("Fetching Reddit posts..."):
            posts = fetch_reddit_posts(topic, limit=num_posts)
            if posts:
                wc = generate_wordcloud(posts)
                st.pyplot(fig=plt.figure(figsize=(12,6)))
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                st.pyplot()
            else:
                st.warning("No posts found for this topic.")

with tab2:
    st.header("Twitter WordCloud")
    topic = st.text_input("Enter topic for Twitter:", "silver")
    num_posts = st.slider("Number of tweets to fetch:", 500, 5000, 1000, step=100, key="tw")
    if st.button("Generate Twitter WordCloud"):
        with st.spinner("Fetching Twitter posts..."):
            posts = fetch_twitter_posts(topic, limit=num_posts)
            wc = generate_wordcloud(posts)
            plt.figure(figsize=(12,6))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            st.pyplot()

with tab3:
    st.header("Facebook WordCloud")
    topic = st.text_input("Enter topic for Facebook:", "silver")
    num_posts = st.slider("Number of posts to fetch:", 500, 5000, 1000, step=100, key="fb")
    if st.button("Generate Facebook WordCloud"):
        with st.spinner("Fetching Facebook posts..."):
            posts = fetch_facebook_posts(topic, limit=num_posts)
            wc = generate_wordcloud(posts)
            plt.figure(figsize=(12,6))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            st.pyplot()
