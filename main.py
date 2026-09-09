import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------------------
# 기본 설정
# -----------------------------------------
st.set_page_config(
    page_title="🌡️ 기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

# -----------------------------------------
# 귀여운 CSS
# -----------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #fff7fb, #eef8ff);
    }

    .title {
        text-align: center;
        font-size: 3rem;
        font-weight: 900;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    .temperature {
        text-align: center;
        font-size: 4rem;
        font-weight: 900;
        padding: 25px;
        border-radius: 30px;
        background: rgba(255,255,255,0.85);
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin: 20px 0;
    }

    .card {
        text-align: center;
        padding: 20px;
        border-radius: 20px;
        background: rgba(255,255,255,0.8);
        box-shadow: 0 5px 20px rgba(0,0,0,0.06);
        font-size: 1.1rem;
    }

    .card b {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# 제목
# -----------------------------------------
st.markdown(
    '<div class="title">🌡️ 기온 예측기 ☀️</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    '서울의 과거 기온 데이터를 이용해서 미래의 평균기온을 예측해 봐요! 🐰✨'
    '</div>',
    unsafe_allow_html=True
)

# -----------------------------------------
# 데이터 불러오기
# -----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 결측값 제거
    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

    # 연도 만들기
    df["연도"] = df["날짜"].dt.year

    return df


# -----------------------------------------
# 연도별 평균기온 계산
# -----------------------------------------
@st.cache_data
def make_yearly_data(df):

    # 2025년까지만 사용
    df = df[df["연도"] <= 2025].copy()

    # 연도별 실제 관측일 수 계산
    observation_count = (
        df.groupby("연도")["날짜"]
        .nunique()
    )

    # 관측일이 300일 이상인 연도만 사용
    valid_years = observation_count[
        observation_count >= 300
    ].index

    df = df[
        df["연도"].isin(valid_years)
    ].copy()

    # 연도별 평균기온
    yearly = (
        df.groupby("연도")["평균기온"]
        .mean()
        .reset_index()
    )

    yearly = yearly.rename(
        columns={
            "평균기온": "연평균기온"
        }
    )

    # -----------------------------------------
    # 선형 회귀
    # y = ax + b
    # -----------------------------------------
    x = yearly["연도"].to_numpy()
    y = yearly["연평균기온"].to_numpy()

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    # 회귀 직선의 예상값
    yearly["회귀기온"] = (
        slope * yearly["연도"]
        + intercept
    )

    # 상관계수
    correlation = np.corrcoef(
        x,
        y
    )[0, 1]

    return (
        yearly,
        slope,
        intercept,
        correlation
    )


# -----------------------------------------
# 실행
# -----------------------------------------
try:
    df = load_data()

    yearly, slope, intercept, correlation = (
        make_yearly_data(df)
    )

except Exception as e:
    st.error(
        "😿 데이터를 불러오는 중 문제가 발생했어요."
    )
    st.stop()


# -----------------------------------------
# 회귀에 사용된 기간 정보
# -----------------------------------------
start_year = int(
    yearly["연도"].min()
)

end_year = int(
    yearly["연도"].max()
)

number_of_years = len(yearly)


st.success(
    f"🌱 회귀 직선은 **{number_of_years}개 연도**의 "
    f"자료로 만들었어요! "
    f"({start_year}년 ~ {end_year}년)"
)


# -----------------------------------------
# 정보 카드
# -----------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="card">
            📚 사용한 연도<br>
            <b>{number_of_years}개</b>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="card">
            🐣 시작 연도<br>
            <b>{start_year}년</b>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="card">
            🐰 끝 연도<br>
            <b>{end_year}년</b>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# -----------------------------------------
# 연도 슬라이더
# -----------------------------------------
selected_year = st.slider(
    "🔎 예측할 연도를 골라 보세요!",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# -----------------------------------------
# 선택한 연도의 예상기온
# -----------------------------------------
predicted_temperature = (
    slope * selected_year
    + intercept
)


st.markdown(
    f"""
    <div class="temperature">
        🌡️ {selected_year}년 예상 평균기온<br>
        {predicted_temperature:.2f} °C
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "💡 예상값은 남은 연도별 평균기온에 "
    "선형회귀를 적용해서 계산한 값이에요."
)


# -----------------------------------------
# Plotly 그래프
# -----------------------------------------
fig = go.Figure()


# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(
            size=8,
            opacity=0.75
        ),
        hovertemplate=(
            "%{x}년<br>"
            "연평균기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)


# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["회귀기온"],
        mode="lines",
        name="회귀 직선",
        line=dict(
            width=4
        ),
        hovertemplate=(
            "%{x}년<br>"
            "회귀기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)


# 선택한 연도의 예상값
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temperature],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(
            size=20,
            symbol="star"
        ),
        hovertemplate=(
            f"{selected_year}년 예상기온: "
            f"{predicted_temperature:.2f} °C"
            "<extra></extra>"
        )
    )
)


# 그래프 꾸미기
fig.update_layout(
    title="📈 서울 연평균기온과 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(
        l=20,
        r=20,
        t=80,
        b=20
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# -----------------------------------------
# 상관계수
# -----------------------------------------
st.markdown(
    f"### 💞 연도와 연평균기온의 상관계수: **{correlation:.4f}**"
)


if correlation > 0:
    st.info(
        "🌸 상관계수가 양수예요! "
        "연도가 높아질수록 연평균기온도 높아지는 "
        "경향이 있다는 뜻이에요."
    )

elif correlation < 0:
    st.info(
        "🍃 상관계수가 음수예요! "
        "연도가 높아질수록 연평균기온이 낮아지는 "
        "경향이 있다는 뜻이에요."
    )

else:
    st.info(
        "🌈 상관계수가 0에 가까워요! "
        "연도와 연평균기온 사이의 선형적인 관계가 "
        "거의 없다는 뜻이에요."
    )


# -----------------------------------------
# 데이터 처리 기준
# -----------------------------------------
with st.expander("🔍 데이터 처리 기준 보기"):

    st.write("""
    📌 **수업 기준 기간**

    - 2025년까지의 자료만 사용
    - 2025년 이후 자료는 제외
    - 관측일이 300일 미만인 연도는 제외
    - 남은 자료를 이용해 연도별 평균기온 계산
    - 남은 연도들의 연평균기온에 선형회귀 적용
    - 연도 슬라이더는 1900년~2100년
    """)


st.caption(
    "🐣 데이터 출처: 제공된 서울 기온 CSV · "
    "기준 기간: 2025년까지"
)
