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

        # Numeric inputs — each tuple: (min, max, default, step)
        numeric_config: dict[str, tuple] = {
            "age":                 (18,    95,    40,    1),
            "duration":            (0,     5000,  300,   1),
            "campaign":            (1,     50,    2,     1),
            "pdays":               (0,     999,   999,   1),
            "previous":            (0,     30,    0,     1),
            "emp_var_rate":        (-5.0,  5.0,   0.0,   0.1),
            "cons_price_index":    (90.0,  100.0, 94.0,  0.01),
            "cons_conf_index":     (-60.0, -20.0, -40.0, 0.1),
            "lending_rate3m":      (0.0,   10.0,  2.0,   0.01),
            "nr_employed":         (4000.0, 6000.0, 5000.0, 10.0),
        }

        st.markdown("**数值特征**")
        num_cols = st.columns(3)
        for i, feat in enumerate(NUMERIC_FEATURES):
            label = FEATURE_LABELS.get(feat, feat)
            min_v, max_v, default, step = numeric_config.get(
                feat, (0, 100000, 0, 1),
            )
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
