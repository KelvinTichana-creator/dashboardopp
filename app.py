import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Opportunities Page Analytics", layout="wide")
st.title("Opportunities Page Analytics Dashboard")
st.markdown("Upload your CSV export and explore engagement by country and region.")

# ---------------------------
# Helpers
# ---------------------------
NUMERIC_COLS = [
    "Total users",
    "Returning users",
    "Active users",
    "Event count per active user",
    "Event count",
    "Engaged sessions",
    "Engagement rate",
    "Engaged sessions per active user",
    "Average engagement time per active user (sec)",
]

REQUIRED_COLS = [
    "Country",
    "Region",
    "Page path and screen class",
] + NUMERIC_COLS


def to_number(series: pd.Series) -> pd.Series:
    """Convert strings like '6,461' to numeric safely."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def load_and_clean(csv_file) -> pd.DataFrame:
    df = pd.read_csv(csv_file)

    # Normalize column names (avoid surprises)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing columns in your CSV: {missing}")
        st.stop()

    # Convert numeric columns
    for col in NUMERIC_COLS:
        df[col] = to_number(df[col])

    # Drop rows that have no country/region
    df = df.dropna(subset=["Country", "Region"])

    # Clean text columns
    df["Country"] = df["Country"].astype(str).str.strip()
    df["Region"] = df["Region"].astype(str).str.strip()
    df["Page path and screen class"] = df["Page path and screen class"].astype(str).str.strip()

    return df


# ---------------------------
# Upload
# ---------------------------
st.sidebar.header("Upload")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if not uploaded:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = load_and_clean(uploaded)

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.header("Filters")

regions = sorted(df["Region"].unique())
countries = sorted(df["Country"].unique())
pages = sorted(df["Page path and screen class"].unique())

region_filter = st.sidebar.multiselect("Region", options=regions, default=regions)
country_filter = st.sidebar.multiselect("Country", options=countries, default=countries)
page_filter = st.sidebar.multiselect("Page path", options=pages, default=pages)

filtered_df = df[
    (df["Region"].isin(region_filter))
    & (df["Country"].isin(country_filter))
    & (df["Page path and screen class"].isin(page_filter))
].copy()

if filtered_df.empty:
    st.warning("No rows match your filters. Adjust the selections in the sidebar.")
    st.stop()

# ---------------------------
# KPIs
# ---------------------------
st.subheader("Key Performance Indicators")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Total Users", f"{int(filtered_df['Total users'].sum()):,}")
with c2:
    st.metric("Active Users", f"{int(filtered_df['Active users'].sum()):,}")
with c3:
    st.metric("Returning Users", f"{int(filtered_df['Returning users'].sum()):,}")
with c4:
    st.metric("Engaged Sessions", f"{int(filtered_df['Engaged sessions'].sum()):,}")
with c5:
    st.metric(
        "Average Engagement Time (sec)",
        f"{filtered_df['Average engagement time per active user (sec)'].mean():.1f}",
    )

# ---------------------------
# Data Table
# ---------------------------
st.subheader("Data Overview")
st.dataframe(filtered_df, use_container_width=True)

# ---------------------------
# Users by Country
# ---------------------------
st.subheader("Users by Country")

users_df = filtered_df.groupby(["Country", "Region"], as_index=False)["Total users"].sum()

fig_users = px.bar(
    users_df,
    x="Country",
    y="Total users",
    color="Region",
    title="Total Users per Country",
)
st.plotly_chart(fig_users, use_container_width=True)

# ---------------------------
# Engagement Intensity
# ---------------------------
st.subheader("Engagement Intensity (per active user)")

metric_choice = st.multiselect(
    "Choose metrics to compare",
    ["Event count per active user", "Engaged sessions per active user", "Engagement rate"],
    default=["Event count per active user", "Engaged sessions per active user"],
)

if metric_choice:
    comp = filtered_df.groupby(["Country", "Region"], as_index=False)[metric_choice].mean()
    fig_engage = px.bar(
        comp,
        x="Country",
        y=metric_choice,
        barmode="group",
        title="Engagement Metrics by Country (Averages)",
    )
    st.plotly_chart(fig_engage, use_container_width=True)

# ---------------------------
# Engagement Time
# ---------------------------
st.subheader("Average Engagement Time per Active User")

time_df = (
    filtered_df.groupby(["Country", "Region"], as_index=False)[
        "Average engagement time per active user (sec)"
    ]
    .mean()
    .sort_values("Average engagement time per active user (sec)", ascending=False)
)

fig_time = px.line(
    time_df,
    x="Country",
    y="Average engagement time per active user (sec)",
    markers=True,
    title="Average Engagement Time by Country",
)
st.plotly_chart(fig_time, use_container_width=True)

# ---------------------------
# Returning Users Ratio
# ---------------------------
st.subheader("Returning Users Ratio")

ratio_df = filtered_df.groupby(["Country", "Region"], as_index=False)[["Returning users", "Total users"]].sum()
ratio_df["Returning ratio"] = ratio_df["Returning users"] / ratio_df["Total users"]

fig_return = px.bar(
    ratio_df.sort_values("Returning ratio", ascending=False),
    x="Country",
    y="Returning ratio",
    color="Region",
    title="Returning Users / Total Users",
)
st.plotly_chart(fig_return, use_container_width=True)

# ---------------------------
# Regional Performance
# ---------------------------
st.subheader("Regional Performance Summary")

region_summary = (
    filtered_df.groupby("Region", as_index=False)
    .agg(
        {
            "Total users": "sum",
            "Active users": "sum",
            "Returning users": "sum",
            "Engaged sessions": "sum",
            "Average engagement time per active user (sec)": "mean",
            "Engagement rate": "mean",
        }
    )
)

st.dataframe(region_summary, use_container_width=True)

fig_region = px.bar(region_summary, x="Region", y="Total users", title="Users by Region")
st.plotly_chart(fig_region, use_container_width=True)

# ---------------------------
# Correlation Heatmap
# ---------------------------
st.subheader("Correlation Analysis (Filtered Data)")

num_df = filtered_df[NUMERIC_COLS].copy()
corr = num_df.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
st.pyplot(fig)

# ---------------------------
# Rankings
# ---------------------------
st.subheader("Country Rankings (Composite Engagement Score)")

rank_df = filtered_df.groupby(["Country", "Region"], as_index=False).agg(
    {
        "Engaged sessions per active user": "mean",
        "Average engagement time per active user (sec)": "mean",
        "Engagement rate": "mean",
        "Event count per active user": "mean",
        "Total users": "sum",
    }
)


def minmax(s: pd.Series):
    s = s.astype(float)
    if s.max() == s.min():
        return 0
    return (s - s.min()) / (s.max() - s.min())


rank_df["score_sessions"] = minmax(rank_df["Engaged sessions per active user"])
rank_df["score_time"] = minmax(rank_df["Average engagement time per active user (sec)"])
rank_df["score_rate"] = minmax(rank_df["Engagement rate"])
rank_df["score_events"] = minmax(rank_df["Event count per active user"])

rank_df["Engagement Score"] = (
    0.35 * rank_df["score_sessions"]
    + 0.30 * rank_df["score_time"]
    + 0.20 * rank_df["score_rate"]
    + 0.15 * rank_df["score_events"]
)

rank_df = rank_df.sort_values("Engagement Score", ascending=False)

st.dataframe(
    rank_df[
        [
            "Country",
            "Region",
            "Engagement Score",
            "Total users",
            "Engaged sessions per active user",
            "Average engagement time per active user (sec)",
            "Engagement rate",
            "Event count per active user",
        ]
    ],
    use_container_width=True,
)

# ---------------------------
# Automated Insights
# ---------------------------
st.subheader("Automated Insights")

top = rank_df.iloc[0]
most_users = rank_df.sort_values("Total users", ascending=False).iloc[0]
longest_time = rank_df.sort_values("Average engagement time per active user (sec)", ascending=False).iloc[0]

st.success(
    f"Top Engagement (Composite): {top['Country']} ({top['Region']})\n"
    f"Highest Traffic: {most_users['Country']} ({int(most_users['Total users']):,} users)\n"
    f"Longest Average Engagement Time: {longest_time['Country']} ({longest_time['Average engagement time per active user (sec)']:.1f} sec)"
)

st.markdown("---")
st.caption("Built with Streamlit")

