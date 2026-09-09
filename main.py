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

    .slope-card {
        text-align: center;
        padding: 25px 15px;
        border-radius: 25px;
        background: rgba(255,255,255,0.88);
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        min-height: 170px;
    }

    .slope-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .slope-value {
        font-size: 2.4rem;
        font-weight: 900;
    }

    .slope-unit {
        font-size: 1rem;
        color: #777;
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
    '서울의 과거 기온으로 미래의 기온을 살짝 예측해 봐요! 🐰✨'
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

    # 연도 추가
    df["연도"] = df["날짜"].dt.year

    return df


# -----------------------------------------
# 연도별 데이터 만들기
# -----------------------------------------
@st.cache_data
def make_yearly_data(df):

    # 2025년까지의 자료만 사용
    df = df[df["연도"] <= 2025].copy()

    # 연도별 관측일 수 계산
    observation_count = (
        df.groupby("연도")["날짜"]
        .nunique()
    )

    # 관측일 300일 이상인 연도만 사용
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

    return yearly


# -----------------------------------------
# 선형회귀 함수
# -----------------------------------------
def calculate_regression(data):

    x = data["연도"].to_numpy(dtype=float)
    y = data["연평균기온"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    # 상관계수
    correlation = np.corrcoef(
        x,
        y
    )[0, 1]

    return slope, intercept, correlation


# -----------------------------------------
# 데이터 준비
# -----------------------------------------
try:

    df = load_data()

    yearly = make_yearly_data(df)

except Exception:

    st.error(
        "😿 데이터를 불러오는 중 문제가 발생했어요. "
        "인터넷 연결을 확인한 뒤 다시 실행해 주세요."
    )

    st.stop()


# -----------------------------------------
# 전체 기간
# -----------------------------------------
start_year = int(
    yearly["연도"].min()
)

end_year = int(
    yearly["연도"].max()
)

number_of_years = len(yearly)


# -----------------------------------------
# 전체 기간 회귀
# -----------------------------------------
full_slope, full_intercept, full_correlation = (
    calculate_regression(yearly)
)


# -----------------------------------------
# 최근 20년 데이터
#
# 마지막 사용 연도를 기준으로
# 마지막 연도 포함 20개 연도
# -----------------------------------------
recent_start_year = end_year - 19

recent_20 = yearly[
    yearly["연도"] >= recent_start_year
].copy()


# 최근 20년 회귀
recent_slope, recent_intercept, recent_correlation = (
    calculate_regression(recent_20)
)


# -----------------------------------------
# 100년당 기온 변화량
# -----------------------------------------
full_slope_100 = full_slope * 100
recent_slope_100 = recent_slope * 100


# -----------------------------------------
# 회귀 직선 값 계산
# -----------------------------------------
yearly["전체회귀기온"] = (
    full_slope * yearly["연도"]
    + full_intercept
)

recent_20["최근20년회귀기온"] = (
    recent_slope * recent_20["연도"]
    + recent_intercept
)


# -----------------------------------------
# 회귀에 사용된 기간 정보
# -----------------------------------------
st.success(
    f"🌱 전체 회귀 직선은 **{number_of_years}개 연도**의 "
    f"자료로 만들었어요! "
    f"({start_year}년 ~ {end_year}년)"
)


# -----------------------------------------
# 기본 정보 카드
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
# ⭐ 100년당 기온 변화량 비교
# -----------------------------------------
st.markdown(
    "## 🔥 100년에 기온이 얼마나 변할까요?"
)

st.caption(
    "회귀선의 기울기를 100배해서 '100년에 몇 °C 변하는가'로 나타냈어요."
)


slope_col1, slope_col2 = st.columns(2)


with slope_col1:

    if full_slope_100 >= 0:
        full_text = f"+{full_slope_100:.2f} °C"
    else:
        full_text = f"{full_slope_100:.2f} °C"

    st.markdown(
        f"""
        <div class="slope-card">
            <div class="slope-title">
                🌍 전체 기간
            </div>

            <div class="slope-value">
                {full_text}
            </div>

            <div class="slope-unit">
                100년에 변화하는 평균기온
            </div>

            <br>

            <div>
                {start_year}년 ~ {end_year}년
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with slope_col2:

    if recent_slope_100 >= 0:
        recent_text = f"+{recent_slope_100:.2f} °C"
    else:
        recent_text = f"{recent_slope_100:.2f} °C"

    st.markdown(
        f"""
        <div class="slope-card">
            <div class="slope-title">
                🕐 최근 20년
            </div>

            <div class="slope-value">
                {recent_text}
            </div>

            <div class="slope-unit">
                100년에 변화하는 평균기온
            </div>

            <br>

            <div>
                {recent_start_year}년 ~ {end_year}년
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------
# 연도 슬라이더
# -----------------------------------------
st.write("")

st.markdown(
    "## 🔎 미래 기온을 확인해 보세요!"
)

selected_year = st.slider(
    "예측할 연도를 골라 보세요!",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# -----------------------------------------
# 선택 연도 예상기온
# 전체 기간 회귀선 사용
# -----------------------------------------
predicted_temperature = (
    full_slope * selected_year
    + full_intercept
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
    "💡 예상기온은 전체 기간의 회귀 직선을 이용해 계산했어요."
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
            opacity=0.7
        ),
        hovertemplate=(
            "%{x}년<br>"
            "연평균기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)


# 전체 기간 회귀선
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["전체회귀기온"],
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(
            width=4
        ),
        hovertemplate=(
            "%{x}년<br>"
            "전체 회귀기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)


# 최근 20년 회귀선
fig.add_trace(
    go.Scatter(
        x=recent_20["연도"],
        y=recent_20["최근20년회귀기온"],
        mode="lines",
        name="최근 20년 회귀선",
        line=dict(
            width=4,
            dash="dash"
        ),
        hovertemplate=(
            "%{x}년<br>"
            "최근 20년 회귀기온: %{y:.2f} °C"
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
    title="📈 서울 연평균기온과 회귀 직선 비교",
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
st.markdown("## 💞 상관계수")


corr_col1, corr_col2 = st.columns(2)

with corr_col1:

    st.metric(
        label="🌍 전체 기간 상관계수",
        value=f"{full_correlation:.4f}"
    )


with corr_col2:

    st.metric(
        label="🕐 최근 20년 상관계수",
        value=f"{recent_correlation:.4f}"
    )


# -----------------------------------------
# 해석
# -----------------------------------------
if full_slope_100 > 0:

    st.info(
        f"🌸 전체 기간을 보면 100년마다 평균기온이 "
        f"약 **{full_slope_100:.2f}°C 상승**하는 추세예요."
    )

elif full_slope_100 < 0:

    st.info(
        f"🍃 전체 기간을 보면 100년마다 평균기온이 "
        f"약 **{abs(full_slope_100):.2f}°C 하락**하는 추세예요."
    )

else:

    st.info(
        "🌈 전체 기간에서는 연도에 따른 "
        "뚜렷한 기온 변화 추세가 나타나지 않아요."
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
    - 전체 기간의 연평균기온에 선형회귀 적용
    - 마지막 사용 연도를 기준으로 최근 20년을 따로 추출
    - 최근 20년의 연평균기온에도 선형회귀 적용
    - 회귀선의 기울기 × 100을 계산하여
      '100년에 몇 °C 변하는가'로 표시
    - 연도 슬라이더는 1900년~2100년
    """)


# -----------------------------------------
# 하단
# -----------------------------------------
st.caption(
    "🐣 데이터 출처: 제공된 서울 기온 CSV · "
    "기준 기간: 2025년까지"
)
