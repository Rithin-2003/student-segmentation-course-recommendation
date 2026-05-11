import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------- LOAD DATA ----------------
def load_data():
    file = "EduPro Online Platform.xlsx"

    users = pd.read_excel(file, sheet_name="Users")
    courses = pd.read_excel(file, sheet_name="Courses")
    transactions = pd.read_excel(file, sheet_name="Transactions")

    return users, courses, transactions


# ---------------- PREPROCESS ----------------
def preprocess():
    users, courses, transactions = load_data()

    df = transactions.merge(users, on="UserID")
    df = df.merge(courses, on="CourseID")

    features = df.groupby("UserID").agg({
        "Amount": "sum",
        "CourseDuration": "sum",
        "CourseRating": "mean"
    }).reset_index()

    return features, df


# ---------------- SEGMENT USERS ----------------
def segment_users():
    features, df = preprocess()

    scaler = StandardScaler()
    X = scaler.fit_transform(features[["Amount", "CourseDuration", "CourseRating"]])

    kmeans = KMeans(n_clusters=4, random_state=42)
    features["Segment"] = kmeans.fit_predict(X)

    return features, df


# ---------------- RECOMMENDATION ----------------
def recommend_courses(user_id):
    features, df = segment_users()

    user_row = features[features["UserID"] == user_id]

    if user_row.empty:
        return None

    segment = user_row["Segment"].values[0]

    # similar users
    similar_users = features[features["Segment"] == segment]["UserID"]

    # courses already taken
    user_courses = df[df["UserID"] == user_id]["CourseName"]

    # similar users data
    rec = df[df["UserID"].isin(similar_users)].copy()

    # remove already taken
    rec = rec[~rec["CourseName"].isin(user_courses)]

    # -------- PERSONALIZATION --------
    user_pref = df[df["UserID"] == user_id]["CourseCategory"].mode()

    if not user_pref.empty:
        pref_category = user_pref[0]
        rec["PreferenceScore"] = np.where(
            rec["CourseCategory"] == pref_category, 1.5, 1
        )
    else:
        rec["PreferenceScore"] = 1

    # final scoring
    rec["FinalScore"] = (
        rec["CourseRating"] * 2 +
        (rec["Amount"] / rec["Amount"].max()) +
        rec["PreferenceScore"]
    )

    # randomness to avoid same output
    rec["FinalScore"] += np.random.uniform(0, 0.5, len(rec))

    # group and rank
    recommendations = rec.groupby("CourseName").agg({
        "CourseRating": "mean",
        "Amount": "sum",
        "FinalScore": "mean"
    }).sort_values(by="FinalScore", ascending=False)

    return recommendations.head(5)