import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import time
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_DIR = Path(__file__).parent

naive_bayes_model = joblib.load(
    MODEL_DIR / "naive_bayes_model.pkl"
)

tfidf_vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

rnn_model = load_model(
    MODEL_DIR / "sentiment_rnn.keras"
)

tokenizer = joblib.load(
    MODEL_DIR / "tokenizer.pkl"
)

label_encoder = joblib.load(
    MODEL_DIR / "label_encoder.pkl"
)



def clean_tweet(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol
    text = re.sub(r"#", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)


st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: gray;
        font-size: 17px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">💬 Twitter Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze Twitter text using Machine Learning and Deep Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

st.subheader("⚡ Try a Sample Tweet")

sample1, sample2, sample3 = st.columns(3)

if "tweet_input" not in st.session_state:
    st.session_state.tweet_input = ""


with sample1:
    if st.button(
        "🔥 Loved the service!",
        use_container_width=True
    ):
        st.session_state.tweet_input = (
            "I absolutely loved the service! 🔥"
        )


with sample2:
    if st.button(
        "😡 Worst delay ever!",
        use_container_width=True
    ):
        st.session_state.tweet_input = (
            "This is the worst delay ever! 😡"
        )


with sample3:
    if st.button(
        "😊 Amazing experience!",
        use_container_width=True
    ):
        st.session_state.tweet_input = (
            "I had an amazing experience! 😊"
        )



st.subheader("🤖 Select Model")

model_choice = st.selectbox(
    "Choose the model for prediction:",
    [
        "ML - Naive Bayes",
        "DL - SimpleRNN"
    ]
)


tweet = st.text_area(
    "📝 Enter your tweet:",
    value=st.session_state.tweet_input,
    placeholder="Example: I really love this game! 😍",
    height=130
)


if st.button(
    "🔍 Predict Sentiment",
    use_container_width=True
):

    if not tweet.strip():

        st.warning("⚠️ Please enter a tweet first.")

    else:


        cleaned_tweet = clean_tweet(tweet)


        with st.expander(
            "🔎 View Text Preprocessing"
        ):

            st.write("**Original Tweet:**")

            st.code(tweet)

            st.write("**Cleaned Tweet:**")

            st.code(cleaned_tweet)


        if model_choice == "ML - Naive Bayes":

            start_time = time.perf_counter()

            tweet_tfidf = tfidf_vectorizer.transform(
                [cleaned_tweet]
            )

            prediction = naive_bayes_model.predict(
                tweet_tfidf
            )

            probabilities = (
                naive_bayes_model.predict_proba(
                    tweet_tfidf
                )[0]
            )

            end_time = time.perf_counter()

            sentiment = label_encoder.inverse_transform(
                prediction
            )[0]

            inference_time = (
                end_time - start_time
            ) * 1000


        else:

            start_time = time.perf_counter()

            sequence = tokenizer.texts_to_sequences(
                [cleaned_tweet]
            )

            padded_sequence = pad_sequences(
                sequence,
                maxlen=100,
                padding="post",
                truncating="post"
            )

            prediction_probabilities = (
                rnn_model.predict(
                    padded_sequence,
                    verbose=0
                )
            )

            prediction = np.argmax(
                prediction_probabilities,
                axis=1
            )

            probabilities = (
                prediction_probabilities[0]
            )

            end_time = time.perf_counter()

            sentiment = label_encoder.inverse_transform(
                prediction
            )[0]

            inference_time = (
                end_time - start_time
            ) * 1000

        st.divider()

        st.subheader("🎯 Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Sentiment",
                sentiment
            )

        with col2:
            st.metric(
                "Model",
                model_choice
            )

        with col3:
            st.metric(
                "Inference Time",
                f"{inference_time:.2f} ms"
            )

        confidence = np.max(probabilities) * 100

        st.write(
            f"### 🎯 Confidence: {confidence:.2f}%"
        )


        st.subheader("📊 Sentiment Probability")

        class_names = label_encoder.classes_

        probability_percent = probabilities * 100

        probability_df = pd.DataFrame(
            {
                "Sentiment": class_names,
                "Probability (%)": probability_percent
            }
        )

        probability_df = probability_df.sort_values(
            "Probability (%)",
            ascending=False
        )

        st.bar_chart(
            probability_df.set_index("Sentiment")
        )


        st.subheader("📋 Probability Details")

        display_df = probability_df.copy()

        display_df["Probability (%)"] = (
            display_df["Probability (%)"]
            .round(2)
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("💡 Interpretation")

        st.write(
            f"The **{model_choice}** model classified "
            f"the tweet as **{sentiment}** with a "
            f"confidence of **{confidence:.2f}%**."
        )

st.divider()

with st.expander("ℹ️ About This Project"):

    st.write(
        """
        **Twitter Sentiment Analyzer** is a Natural Language
        Processing (NLP) application that classifies tweets
        into four sentiment categories:

        - Positive
        - Negative
        - Neutral
        - Irrelevant

        The application uses two trained models:

        **Machine Learning:** Naive Bayes

        **Deep Learning:** SimpleRNN

        Text preprocessing includes converting text to lowercase,
        removing URLs, removing mentions, removing hashtags,
        and removing unnecessary characters.
        """
    )
