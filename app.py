import streamlit as st
import joblib
import json
import os
import mysql.connector
import bcrypt
@st.cache_resource
def load_model():
    model = joblib.load("final_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf
model, tfidf = load_model()
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="sakshamsharma@2007",
        database="FAKENEWS"
    )

SETTINGS_FILE = "settings.json"

FONT_COLORS = {
    "White": "#FFFFFF",
    "Black": "#000000",
    "Red": "#FF0000",
    "Green": "#00FF00",
    "Yellow": "#FFFF00"
}

BG_COLORS = {
    "Default": "#111827",
    "White": "#FFFFFF",
    "Black": "#000000",
    "Red": "#FF0000",
    "Green": "#00FF00",
    "Yellow": "#FFFF00"
}


def load_settings():

    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)

    return {
        "font_color": "White",
        "bg_color": "#111827"
    }


def save_settings(font_color, bg_color):

    settings = {
        "font_color": font_color,
        "bg_color": bg_color
    }

    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)


settings = load_settings()

font_color = FONT_COLORS[settings["font_color"]]
bg_color = settings["bg_color"]

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg_color};
    }}

    h1, h2, h3, h4, h5, h6, p, label {{
        color: {font_color} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
if "page" not in st.session_state:
    st.session_state.page = "home"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if st.session_state.page == "home":    
    col1,col2,col3,col4,col5 = st.columns([4,1,1.1,1,1.1]) 
    with col1:
        if st.button("About", key="about"):
            st.session_state.page = "about"
            st.rerun()
    with col2:  
        if st.button("History", key="history"):
            st.session_state.page = "history"
            st.rerun()
    with col3:  
        if st.button("Example", key="example"):
            st.session_state.page = "example"
            st.rerun()    
    with col4:
        if st.button("Login", key="login"):
            st.session_state.page = "login"
            st.rerun()    
    with col5:
        if st.button("Settings",key="settings"):
            st.session_state.page = "settings"
            st.rerun() 
    st.title("Fake News Checker")
    text=st.text_area("Enter a News Article(minimum 20 words): ",height=300)
    if st.button("Check News", use_container_width=True):
        if text.strip():
            if len(text.split()) < 20:
                st.warning("Please enter a longer article.")
            else:
                with st.spinner("Analyzing the news article..."):
                    text_vector=tfidf.transform([text])
                    prediction=model.predict(text_vector)[0]
                    probability=model.predict_proba(text_vector)[0]
                    confidence=probability.max()
                if prediction == 1:
                    st.success("Likely Real News")
                    st.write(f"Confidence: {confidence * 100:.2f}%")
                    if confidence < 0.60:
                        st.warning(f"Low confidence. ""Please verify this article using reliable sources.")           
                    st.success("Article Analysed Successfully")
                else:
                    st.error("Likely Fake News")
                    st.write(f"Confidence: {confidence * 100:.2f}%")
                    if confidence < 0.60:
                        st.warning("Low confidence. ""Please verify this article using reliable sources.")
                    st.success("Article Analysed Successfully")
                if st.session_state.logged_in:
                    try:
                        connection=get_connection()
                        cursor=connection.cursor()
                        cursor.execute("""INSERT INTO predictions (user_id, article, prediction,
                            confidence) VALUES (%s,%s,%s,%s)""",
                                       (st.session_state.user_id,text,"Real" if prediction==1 else "Fake",confidence*100))
                        connection.commit()
                    except mysql.connector.Error as e:
                            st.error(f"Could not save prediction: {e}")
                    finally:
                        if "cursor" in locals():
                                cursor.close()
                        if "connection" in locals() and connection.is_connected():
                                connection.close()
                else:
                        st.info("Log in to save this prediction to your history.")
        else:
            st.warning("Please enter an article first.")
elif st.session_state.page=="login":
    st.title("Login")
    c1=st.columns([2])
    with c1[0]:
        username=st.text_input("Username:")
        password=st.text_input("Password: ")
    if st.button("Login",key="login_submit"):
        if username and password:
            try:
                connection = get_connection()
                cursor = connection.cursor()
                cursor.execute(
                "SELECT id, password_hash FROM USERS WHERE username = %s",
                (username,))
                user = cursor.fetchone()
                if user:
                    user_id = user[0]
                    stored_hash = user[1]
                    if bcrypt.checkpw(
                        password.encode("utf-8"),
                        stored_hash.encode("utf-8")
                    ):
                        st.success("Login successful!")
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        st.session_state.page = "home"
                        st.rerun()
                    else:
                        st.error("Sorry, incorrect password.")
                else:
                    st.error("Sorry, username not found.")
            except mysql.connector.Error as e:
                st.error(f"Database error: {e}")
            finally:
                if "cursor" in locals():
                    cursor.close()
                if "connection" in locals() and connection.is_connected():
                    connection.close()
        else:
                st.warning("Please enter all username, password.")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    col6,col7=st.columns([6,1])
    with col6:
        if st.button("Back",key="login_back"):
                st.session_state.page = "home"
                st.rerun()
    with col7:
        if st.button("Sign up",key="sign_up"):
            st.session_state.page = "sign_up"
            st.rerun()
    
elif st.session_state.page=="about":
    st.title("About")
    st.write(
        "Fake News Checker is a machine learning application "
        "that analyzes news articles and predicts whether they "
        "are likely to be real or fake."
    )
    st.subheader("How it works")
    st.write("""
    1. Enter a news article.
    2. TF-IDF converts the text into numerical features.
    3. The trained machine learning model analyzes the text.
    4. The model produces a prediction and confidence score.
    """)
    st.subheader("Model Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Model:", "Logistic Regression")
    with col2:
        st.write("Vectorizer:", "TF-IDF")
    with col3:
        st.write("Accuracy:", "98.68%")
    st.subheader("Remarks")
    st.write("""
    This application provides an ML-based prediction and should
    not be treated as a definitive fact-checking system.
    The confidence score represents the model's confidence in its
    prediction, not the actual probability that the news is true.
    For important information, always verify the article using
    reliable and independent news sources.
    """)
    if st.button("Back",key="about_back"):
        st.session_state.page = "home"
        st.rerun()
elif st.session_state.page == "history":

    st.title("Prediction History")

    if not st.session_state.logged_in:

        st.warning("Please log in to view your prediction history.")
    else:
        try:
            connection=get_connection()
            cursor=connection.cursor()
            cursor.execute("""SELECT prediction,confidence,article,created_at from predictions WHERE user_id=
            %s ORDER BY created_at DESC""",(st.session_state.user_id,))
            history=cursor.fetchall()
            if not history:
                st.info("No prediction yet")
            else:
                for i, result in enumerate(history, start=1):
                    prediction = result[0]
                    confidence = result[1]
                    article = result[2]
                    created_at = result[3]
                    st.subheader(f"Prediction {i}")
                    st.write(f"**Result:** {prediction}")
                    st.write(f"**Confidence:** {confidence:.2f}%")
                    st.write(f"**Date:** {created_at}")
                    st.write(f"**Article:** {article}")
                    st.divider()
        except mysql.connector.Error as e:
            st.error(f"Database error: {e}")
        finally:
            if "cursor" in locals():
                cursor.close()
            if "connection" in locals() and connection.is_connected():
                connection.close()
        if st.button("Clear History", key="clear_history"):
            try:
                connection = get_connection()
                cursor = connection.cursor()
                cursor.execute(
                    """
                    DELETE FROM predictions
                    WHERE user_id = %s
                    """,
                    (st.session_state.user_id,)
                )
                connection.commit()
                st.success("History cleared!")
                st.rerun()
            except mysql.connector.Error as e:
                st.error(f"Database error: {e}")
            finally:
                if "cursor" in locals():
                    cursor.close()
                if "connection" in locals() and connection.is_connected():
                    connection.close()
    if st.button("Back", key="history_back"):
        st.session_state.page = "home"
        st.rerun()
elif st.session_state.page == "settings":
    st.title(" Settings")
    st.subheader("Change Font Color")
    font_choice = st.radio(
        "Select font color:",
        list(FONT_COLORS.keys()),
        index=list(FONT_COLORS.keys()).index(settings["font_color"]),
        key="font_choice"
    )

    st.subheader("Change Background Color")

    current_bg_name = "Default"

    for name, value in BG_COLORS.items():
        if value == settings["bg_color"]:
            current_bg_name = name
            break

    background_choice = st.radio(
        "Select background color:",
        list(BG_COLORS.keys()),
        index=list(BG_COLORS.keys()).index(
            current_bg_name
        ),
        key="background_choice"
    )
    st.subheader("Preview")
    st.markdown(
        f"""
        <div style="
            background-color: {BG_COLORS[background_choice]};
            color: {FONT_COLORS[font_choice]};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        ">
            This is how your colors will look.
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Save Settings", key="save_settings"):

        save_settings(
            font_choice,
            BG_COLORS[background_choice]
        )
        st.success("Settings saved permanently!")
        st.rerun()
    if st.button("Back", key="settings_back"):

        st.session_state.page = "home"
        st.rerun()
elif st.session_state.page=="example":
    st.header("Examples for Fake and Real News")
    st.subheader("Examples of Fake News: ")
    st.write("""Scientists Claim New Smartphone Feature Can Eliminate the Need for Sleep A
      team of researchers has reportedly developed a smartphone application that allows users
        to remain fully energized after just 30 minutes of sleep each day. The researchers claim
          the app uses specially designed sound waves and artificial intelligence to “recharge” 
          the human brain while the user sleeps. According to the report, the technology will be 
          available on all smartphones by the end of the year and could completely eliminate traditional sleep schedules.""")
    st.write("For more fake articles click below button: ")
    st.link_button("Fake News","https://library-nd.libguides.com/fakenews/examples")
    st.subheader("Examples of Real News: ")
    st.write("""India's foreign exchange reserves hit a record $729.33 billion as of August 21, 2026,
    according to the Reserve Bank of India. The increase was driven by capital inflows, including 
    foreign-currency deposits and other measures introduced to strengthen India's balance of payments.""")
    st.write("For more Real articles click below button: ")
    st.link_button("Real News","https://www.bbc.com/")
    if st.button("Back", key="example_back"):
            st.session_state.page = "home"
            st.rerun()
elif st.session_state.page=="sign_up":
    st.title("Sign up")
    username=st.text_input("Username:")
    password=st.text_input("Password: ")
    confirm_password=st.text_input("Confirm Password: ")
    if st.button("Sign up", key="signup_submit"):
        if username and password and confirm_password:
            if confirm_password==password:
                if len(password)>=8:
                    try:
                        connection = get_connection()
                        cursor = connection.cursor()
                        cursor.execute(
                        "SELECT id FROM USERS WHERE username = %s",
                        (username,))
                        existing_user = cursor.fetchone()
                        if existing_user:
                            st.error("Sorry, username already exists.")
                        else:
                            password_hash = bcrypt.hashpw(
                            password.encode("utf-8"),
                            bcrypt.gensalt()).decode("utf-8")
                            cursor.execute(
                            """
                            INSERT INTO USERS (username, password_hash)
                            VALUES (%s, %s)
                            """,
                            (username, password_hash))
                            connection.commit()
                            st.success("Account created successfully!")
                    except mysql.connector.Error as e:
                        st.error(f"Database error: {e}")
                    finally:
                        if "cursor" in locals():
                            cursor.close()
                        if "connection" in locals() and connection.is_connected():
                            connection.close()
                else:
                    st.warning("try again! use password more than 7 characters")
            else:
                st.warning("try again! Confirm password and password dont match")
        else:
            st.warning("Please enter all username, password and confirm password.")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    if st.button("Back", key="signup_back"):
        st.session_state.page = "login"
        st.rerun()