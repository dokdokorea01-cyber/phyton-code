"""
선박 운항 시뮬레이션 + 연료 소비 예측 (웹앱 버전, 한글)

실행:
    streamlit run ship_fuel_app.py

브라우저가 자동으로 열리고 결과가 웹페이지로 표시됩니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
import platform

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 한글 폰트 설정
# ----------------------------
def set_korean_font():
    system = platform.system()
    if system == "Windows":
        matplotlib.rcParams['font.family'] = 'Malgun Gothic'
    elif system == "Darwin":
        matplotlib.rcParams['font.family'] = 'AppleGothic'
    else:
        matplotlib.rcParams['font.family'] = 'NanumGothic'
    matplotlib.rcParams['axes.unicode_minus'] = False

set_korean_font()

st.set_page_config(page_title="선박 연료 소비 예측", layout="wide")

st.title("🚢 선박 운항 시뮬레이션 + 연료 소비 예측")
st.write("numpy로 생성한 가상 선박 운항 데이터를 분석하고, 회귀 모델로 연료 소비량을 예측합니다.")

# ----------------------------
# 변수 한글 이름 매핑
# ----------------------------
KOR_NAMES = {
    "distance_nm": "운항거리(해리)",
    "speed_knots": "속도(노트)",
    "cargo_load_ratio": "화물적재율",
    "wave_height_m": "파고(m)",
    "wind_speed_ms": "풍속(m/s)",
    "duration_hours": "항해시간(시간)",
    "fuel_consumption_tons": "연료소비량(톤)",
}

# ----------------------------
# 사이드바 설정
# ----------------------------
st.sidebar.header("⚙️ 설정")
seed = st.sidebar.number_input("랜덤 시드 (값을 바꾸면 다른 데이터 생성)", value=42, step=1)
n_samples = st.sidebar.slider("데이터 샘플 수", min_value=200, max_value=5000, value=2000, step=100)


# ============================================================
# 데이터 생성
# ============================================================
@st.cache_data
def generate_data(n, seed):
    np.random.seed(seed)

    distance = np.random.uniform(50, 1000, n)
    speed = np.random.uniform(10, 25, n)
    cargo_load = np.random.uniform(0.2, 1.0, n)

    wave_height = np.random.gamma(shape=2.0, scale=0.7, size=n)
    wave_height = np.clip(wave_height, 0, 6)

    wind_speed = np.random.gamma(shape=2.5, scale=2.0, size=n)
    wind_speed = np.clip(wind_speed, 0, 25)

    duration = distance / speed

    base_consumption = 0.0008 * (speed ** 3) * duration
    cargo_effect = base_consumption * (0.3 * cargo_load)
    weather_effect = base_consumption * (0.05 * wave_height + 0.02 * wind_speed)
    noise = np.random.normal(0, 0.05 * base_consumption, n)

    fuel_consumption = base_consumption + cargo_effect + weather_effect + noise
    fuel_consumption = np.clip(fuel_consumption, 0.5, None)

    df = pd.DataFrame({
        "distance_nm": distance,
        "speed_knots": speed,
        "cargo_load_ratio": cargo_load,
        "wave_height_m": wave_height,
        "wind_speed_ms": wind_speed,
        "duration_hours": duration,
        "fuel_consumption_tons": fuel_consumption,
    })

    return df


def evaluate_predictions(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))

    return {"r2": r2, "rmse": rmse, "mae": mae}


# ============================================================
# 메인 실행
# ============================================================
df = generate_data(n_samples, seed)
df_kor = df.rename(columns=KOR_NAMES)

st.header("1. 생성된 데이터")
st.dataframe(df_kor.head(10))

col1, col2 = st.columns(2)
with col1:
    st.subheader("기술 통계")
    st.dataframe(df_kor.describe())

with col2:
    st.subheader("연료 소비량과의 상관관계")
    corr_matrix = np.corrcoef(df.values.T)
    feature_names = df.columns.tolist()
    fuel_idx = feature_names.index("fuel_consumption_tons")

    corr_df = pd.DataFrame({
        "변수": [KOR_NAMES[n] for n in feature_names if n != "fuel_consumption_tons"],
        "상관계수": [corr_matrix[fuel_idx, i] for i, n in enumerate(feature_names) if n != "fuel_consumption_tons"]
    }).sort_values("상관계수", ascending=False)

    st.dataframe(corr_df, hide_index=True)

# ----------------------------
# 변수 관계 시각화
# ----------------------------
st.header("2. 변수별 연료 소비량 관계")

features = ["distance_nm", "speed_knots", "cargo_load_ratio",
             "wave_height_m", "wind_speed_ms", "duration_hours"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, feat in zip(axes.flat, features):
    ax.scatter(df[feat], df["fuel_consumption_tons"], alpha=0.3, s=10)
    ax.set_xlabel(KOR_NAMES[feat])
    ax.set_ylabel(KOR_NAMES["fuel_consumption_tons"])
    ax.set_title(f"{KOR_NAMES[feat]} vs 연료소비량")
plt.tight_layout()
st.pyplot(fig)

# ============================================================
# 모델 학습
# ============================================================
st.header("3. 모델 학습 및 예측 결과")

feature_cols = ["distance_nm", "speed_knots", "cargo_load_ratio",
                 "wave_height_m", "wind_speed_ms", "duration_hours"]
target_col = "fuel_consumption_tons"

X = df[feature_cols].values
y = df[target_col].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 선형 회귀
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
metrics_lr = evaluate_predictions(y_test, y_pred_lr)

# 랜덤포레스트
rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
metrics_rf = evaluate_predictions(y_test, y_pred_rf)

col1, col2 = st.columns(2)
with col1:
    st.subheader("📐 선형 회귀")
    st.metric("결정계수 (R²)", f"{metrics_lr['r2']:.4f}")
    st.metric("평균제곱근오차 (RMSE)", f"{metrics_lr['rmse']:.4f} 톤")
    st.metric("평균절대오차 (MAE)", f"{metrics_lr['mae']:.4f} 톤")

with col2:
    st.subheader("🌲 랜덤포레스트")
    st.metric("결정계수 (R²)", f"{metrics_rf['r2']:.4f}")
    st.metric("평균제곱근오차 (RMSE)", f"{metrics_rf['rmse']:.4f} 톤")
    st.metric("평균절대오차 (MAE)", f"{metrics_rf['mae']:.4f} 톤")

# 예측 vs 실제 그래프
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_test, y_pred_lr, alpha=0.4, s=15)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel("실제 연료소비량 (톤)")
axes[0].set_ylabel("예측 연료소비량 (톤)")
axes[0].set_title(f"선형 회귀 (R²={metrics_lr['r2']:.3f})")

axes[1].scatter(y_test, y_pred_rf, alpha=0.4, s=15, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_xlabel("실제 연료소비량 (톤)")
axes[1].set_ylabel("예측 연료소비량 (톤)")
axes[1].set_title(f"랜덤포레스트 (R²={metrics_rf['r2']:.3f})")

plt.tight_layout()
st.pyplot(fig)

# ----------------------------
# 선형회귀 계수
# ----------------------------
st.header("4. 선형회귀 계수 (변수별 영향력)")
coef_df = pd.DataFrame({
    "변수": [KOR_NAMES[c] for c in feature_cols],
    "계수": lr.coef_
}).sort_values("계수", key=abs, ascending=False)
st.dataframe(coef_df, hide_index=True)
st.caption("계수가 양수이면 해당 변수가 클수록 연료 소비량이 증가, 음수이면 감소합니다.")

# ----------------------------
# 랜덤포레스트 특성 중요도
# ----------------------------
st.header("5. 랜덤포레스트 특성 중요도")

importances = rf.feature_importances_
sorted_idx = np.argsort(importances)
sorted_names = [KOR_NAMES[feature_cols[i]] for i in sorted_idx]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(sorted_names, importances[sorted_idx], color='steelblue')
ax.set_xlabel("중요도")
ax.set_title("랜덤포레스트 특성 중요도")
plt.tight_layout()
st.pyplot(fig)

# ----------------------------
# 다운로드
# ----------------------------
st.header("6. 데이터 다운로드")
csv = df_kor.to_csv(index=False).encode('utf-8-sig')
st.download_button(
    label="📥 생성된 데이터 CSV 다운로드 (엑셀에서 한글 정상 표시)",
    data=csv,
    file_name="선박운항데이터.csv",
    mime="text/csv"
)
