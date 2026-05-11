import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import segment_users, recommend_courses

st.set_page_config(page_title="EduPro Dashboard", layout="wide")

# ---------------- LOAD ----------------
features, df = segment_users()

# ---------------- TITLE ----------------
st.title("🎓 EduPro Course Recommendation System")

# ---------------- KPI ----------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Users", len(features))
col2.metric("Avg Spending", round(features["Amount"].mean(), 2))
col3.metric("Avg Rating", round(features["CourseRating"].mean(), 2))

# ---------------- PIE CHART ----------------
st.subheader("🥧 User Segmentation")

segment_counts = features["Segment"].value_counts().sort_index()

plt.figure()
wedges, texts, autotexts = plt.pie(
    segment_counts.values,
    labels=[f"Segment {i}" for i in segment_counts.index],
    autopct='%1.1f%%',
    startangle=90
)

plt.legend(wedges, [f"Segment {i}" for i in segment_counts.index],
           title="Segments", loc="center left", bbox_to_anchor=(1, 0.5))

st.pyplot(plt)

# ---------------- USER SELECT ----------------
st.subheader("👤 Select User")

user_list = features["UserID"].tolist()
selected_user = st.selectbox("Choose User ID", user_list)

# ---------------- USER VS SEGMENT ----------------
def user_vs_segment(user_id):
    user_data = features[features["UserID"] == user_id]
    seg = user_data["Segment"].values[0]

    segment_avg = features[features["Segment"] == seg][
        ["Amount", "CourseDuration", "CourseRating"]
    ].mean()

    return user_data, segment_avg

if selected_user:
    st.subheader("📊 User vs Segment Comparison")

    user_data, seg_avg = user_vs_segment(selected_user)

    comparison = pd.DataFrame({
        "User": user_data.iloc[0][["Amount", "CourseDuration", "CourseRating"]],
        "Segment Avg": seg_avg
    })

    st.dataframe(comparison)

# ---------------- RECOMMEND ----------------
if selected_user:
    st.subheader("🎯 Recommended Courses")

    rec = recommend_courses(selected_user)

    if rec is None or rec.empty:
        st.warning("No recommendations available")
    else:
        rec = rec.reset_index()
        st.dataframe(rec)

        # chart
        st.subheader("📊 Course Ratings")

        plt.figure()
        plt.bar(rec["CourseName"], rec["CourseRating"])
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # download
        csv = rec.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download", csv, "recommendations.csv", "text/csv")

        st.info("Recommended based on similar users, preferences, and ratings.")