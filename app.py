"""Streamlit entry point — multi-page navigation."""

import streamlit as st

st.set_page_config(
    page_title="银行营销分析预测系统",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ──────────────────────────────────
st.sidebar.title("🏦 银行营销系统")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "导航",
    ["📊 数据分析", "🔮 在线预测"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("banksys_sy_wangliang v0.1.0")

# ── Page routing ────────────────────────────────────────
if page == "📊 数据分析":
    from app.page_analysis import render
    render()
elif page == "🔮 在线预测":
    from app.page_prediction import render
    render()
