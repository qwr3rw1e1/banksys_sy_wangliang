"""Data analysis interactive page."""

import plotly.express as px
import streamlit as st

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL
from src.data_loader import load_raw_data
from src.eda import (
    categorical_summary,
    correlation_matrix,
    dataset_overview,
    numeric_summary,
    subscription_rate_by_category,
    target_distribution,
)


def render() -> None:
    """Render the data analysis page."""
    st.title("📊 数据分析")
    st.markdown("银行营销数据探索性分析 — 交互式可视化")

    # ── Load data ──
    @st.cache_data
    def get_data():
        return load_raw_data()

    df = get_data()

    # ── Section: Dataset Overview ──
    st.header("1. 数据概览")
    overview = dataset_overview(df)

    cols = st.columns(5)
    cols[0].metric("总行数", f"{overview['n_rows']:,}")
    cols[1].metric("总列数", overview["n_cols"])
    cols[2].metric("缺失率", f"{overview['missing_pct']}%")
    cols[3].metric("数值特征", overview["numeric_cols"])
    cols[4].metric("类别特征", overview["categorical_cols"])

    # Target distribution
    st.subheader("目标变量分布（认购情况）")
    target_dist = target_distribution(df)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(target_dist, use_container_width=True)
    with col2:
        fig = px.pie(
            target_dist, names="class", values="count",
            title="认购 vs 未认购", hole=0.4,
            color="class",
            color_discrete_map={"yes": "#2ca02c", "no": "#d62728"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Section: Numeric Analysis ──
    st.header("2. 数值特征分析")

    num_summary = numeric_summary(df)
    st.dataframe(num_summary, use_container_width=True)

    # Histogram / Box plot selector
    st.subheader("分布对比（按认购结果分组）")
    num_feat = st.selectbox(
        "选择数值特征", NUMERIC_FEATURES,
        key="num_hist_feat",
    )
    chart_type = st.radio("图表类型", ["直方图 (Histogram)", "箱线图 (Box Plot)"], horizontal=True)

    if chart_type == "直方图 (Histogram)":
        fig = px.histogram(
            df, x=num_feat, color=TARGET_COL,
            marginal="box", barmode="overlay",
            opacity=0.7,
            color_discrete_map={"yes": "#2ca02c", "no": "#d62728"},
            title=f"{num_feat} 分布 — 按认购结果分组",
        )
    else:
        fig = px.box(
            df, x=TARGET_COL, y=num_feat,
            color=TARGET_COL,
            color_discrete_map={"yes": "#2ca02c", "no": "#d62728"},
            title=f"{num_feat} 箱线图 — 按认购结果分组",
        )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Section: Categorical Analysis ──
    st.header("3. 类别特征分析")

    cat_summary = categorical_summary(df)
    st.dataframe(cat_summary, use_container_width=True)

    st.subheader("各类别认购率")
    cat_feat = st.selectbox(
        "选择类别特征", CATEGORICAL_FEATURES,
        key="cat_bar_feat",
    )
    rate_df = subscription_rate_by_category(df, cat_feat)
    fig = px.bar(
        rate_df, x=cat_feat, y="rate",
        text="rate",
        title=f"{cat_feat} — 各类别认购率 (%)",
        labels={"rate": "认购率 (%)"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Section: Correlation ──
    st.header("4. 相关性分析")

    corr = correlation_matrix(df)
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="数值特征相关性热力图",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scatter plot
    st.subheader("散点图")
    c1, c2 = st.columns(2)
    with c1:
        scatter_x = st.selectbox("X 轴", NUMERIC_FEATURES, key="scatter_x")
    with c2:
        scatter_y = st.selectbox("Y 轴", NUMERIC_FEATURES, index=min(1, len(NUMERIC_FEATURES) - 1), key="scatter_y")

    fig = px.scatter(
        df.sample(min(5000, len(df)), random_state=42),
        x=scatter_x, y=scatter_y,
        color=TARGET_COL, opacity=0.6,
        color_discrete_map={"yes": "#2ca02c", "no": "#d62728"},
        title=f"{scatter_x} vs {scatter_y}（抽样 5,000 条）",
    )
    st.plotly_chart(fig, use_container_width=True)
