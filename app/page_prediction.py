"""Online prediction page with point-and-click form."""

import streamlit as st

from src.config import CATEGORICAL_FEATURES, FEATURE_LABELS, NUMERIC_FEATURES
from src.data_loader import load_raw_data
from src.model_predict import get_feature_choices, load_model, predict


@st.cache_resource
def load_resources():
    """Cache model and preprocessor across sessions."""
    model = load_model()
    from src.data_loader import load_preprocessor
    preprocessor = load_preprocessor()
    return model, preprocessor


@st.cache_data
def load_training_data_for_choices():
    """Load training data to populate dropdown choices."""
    df = load_raw_data()
    return get_feature_choices(df)


def render() -> None:
    """Render the online prediction page."""
    st.title("🔮 在线预测")
    st.markdown("填写以下客户信息，预测是否会认购定期存款")

    # Check if model exists
    from src.config import MODEL_FILE
    if not MODEL_FILE.exists():
        st.error("⚠️ 模型文件未找到。请先运行 `python src/model_train.py` 训练模型。")
        st.stop()

    try:
        model, preprocessor = load_resources()
        choices_map = load_training_data_for_choices()
    except Exception as e:
        st.error(f"⚠️ 模型加载失败: {e}")
        st.stop()

    # ── Input form ──
    with st.form("prediction_form"):
        st.subheader("📋 客户特征输入")

        input_data: dict = {}

        # Numeric inputs
        st.markdown("**数值特征**")
        num_cols = st.columns(3)
        for i, feat in enumerate(NUMERIC_FEATURES):
            label = FEATURE_LABELS.get(feat, feat)
            # Use sensible defaults/ranges
            default = 0
            step = 1.0
            if feat == "age":
                min_v, max_v, default = 18, 95, 40
            elif feat == "duration":
                min_v, max_v, default, step = 0, 5000, 300, 10.0
            elif feat == "campaign":
                min_v, max_v, default = 1, 50, 2
            elif feat == "pdays":
                min_v, max_v, default = 0, 999, 999
            elif feat == "previous":
                min_v, max_v, default = 0, 30, 0
            elif feat == "emp_var_rate":
                min_v, max_v, default, step = -5.0, 5.0, 0.0, 0.1
            elif feat == "cons_price_index":
                min_v, max_v, default, step = 90.0, 100.0, 94.0, 0.01
            elif feat == "cons_conf_index":
                min_v, max_v, default, step = -60.0, -20.0, -40.0, 0.1
            elif feat == "lending_rate3m":
                min_v, max_v, default, step = 0.0, 10.0, 2.0, 0.01
            elif feat == "nr_employed":
                min_v, max_v, default, step = 4000.0, 6000.0, 5000.0, 10.0
            else:
                min_v, max_v = 0, 100000
            input_data[feat] = num_cols[i % 3].number_input(
                label, min_value=min_v, max_value=max_v,
                value=default, step=step, key=feat,
                help=f"{label} — 范围: {min_v} ~ {max_v}",
            )

        st.markdown("---")
        st.markdown("**类别特征**")

        cat_cols = st.columns(3)
        for i, feat in enumerate(CATEGORICAL_FEATURES):
            label = FEATURE_LABELS.get(feat, feat)
            choices = choices_map.get(feat, ["unknown"])
            input_data[feat] = cat_cols[i % 3].selectbox(
                label, choices, key=feat,
                help=f"{label} — 可选值: {', '.join(str(c) for c in choices)}",
            )

        st.markdown("---")
        submitted = st.form_submit_button("🔍 预测", type="primary", use_container_width=True)

    # ── Prediction result ──
    if submitted:
        # Validate
        missing = [k for k, v in input_data.items() if v is None]
        if missing:
            st.error(f"请填写以下必填字段: {', '.join(missing)}")
        else:
            with st.spinner("正在预测..."):
                pred, proba = predict(model, preprocessor, input_data)

            st.markdown("---")
            st.subheader("📊 预测结果")

            result_col1, result_col2 = st.columns([1, 2])

            with result_col1:
                if pred == 1:
                    st.success("### ✅ 会认购")
                    st.metric("认购概率", f"{proba:.2%}")
                else:
                    st.warning("### ❌ 不会认购")
                    st.metric("不认购概率", f"{1 - proba:.2%}")

            with result_col2:
                # Gauge-like bar
                import plotly.graph_objects as go
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={"text": "认购概率 (%)"},
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2ca02c" if proba >= 0.5 else "#d62728"},
                        "steps": [
                            {"range": [0, 50], "color": "rgba(214,39,40,0.2)"},
                            {"range": [50, 100], "color": "rgba(44,160,44,0.2)"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 2},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

            # Reset button (outside form)
            if st.button("🔄 重置表单"):
                st.rerun()
