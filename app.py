import io
import json
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from openai import OpenAI


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기후 데이터 AI 분석 도우미",
    page_icon="🌍",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #075f80;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #536471;
        margin-bottom: 1.2rem;
    }
    .section-card {
        border: 1px solid #cfe1ef;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        background: #ffffff;
        margin-bottom: 1rem;
    }
    .small-help {
        color: #6b7280;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# CSV 읽기 함수
# -----------------------------
def read_csv_safely(uploaded_file):
    """한글 CSV에서 자주 쓰는 인코딩을 순서대로 시도합니다."""
    raw = uploaded_file.getvalue()
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]

    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            return df, enc
        except Exception as e:
            last_error = e

    raise last_error


def clean_column_names(df):
    """열 이름 앞뒤 공백과 BOM 문자를 제거합니다."""
    cleaned = df.copy()
    cleaned.columns = [str(c).strip().replace("\ufeff", "") for c in cleaned.columns]
    return cleaned


def try_convert_dates(df):
    """날짜처럼 보이는 열을 datetime 형식으로 변환합니다."""
    converted = df.copy()
    date_candidates = []

    for col in converted.columns:
        col_lower = str(col).lower()

        looks_like_date_name = any(
            key in col_lower
            for key in ["date", "날짜", "일시", "연월", "year", "년도", "월"]
        )

        if looks_like_date_name or converted[col].dtype == "object":
            try:
                parsed = pd.to_datetime(converted[col], errors="coerce")
                valid_ratio = parsed.notna().mean()

                if valid_ratio >= 0.6:
                    converted[col] = parsed
                    date_candidates.append(col)
            except Exception:
                pass

    return converted, date_candidates


def detect_columns(df):
    """숫자형, 날짜형, 문자형 열을 구분합니다."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()
    category_cols = [c for c in df.columns if c not in numeric_cols and c not in date_cols]

    return numeric_cols, date_cols, category_cols


def make_data_profile(df):
    """데이터의 기본 정보를 요약합니다."""
    numeric_cols, date_cols, category_cols = detect_columns(df)

    missing = df.isna().sum().sort_values(ascending=False)
    duplicated_rows = int(df.duplicated().sum())

    outlier_info = []

    for col in numeric_cols:
        series = df[col].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(((series < lower) | (series > upper)).sum())

        if count > 0:
            outlier_info.append(
                {
                    "column": col,
                    "outliers": count,
                }
            )

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "date_columns": date_cols,
        "category_columns": category_cols,
        "missing_top": missing[missing > 0].head(10).to_dict(),
        "duplicated_rows": duplicated_rows,
        "outlier_info": outlier_info[:10],
    }


def dataframe_summary_text(df, max_rows=8):
    """AI에게 보낼 데이터 요약 텍스트를 만듭니다."""
    profile = make_data_profile(df)

    if profile["numeric_columns"]:
        numeric_desc = df[profile["numeric_columns"]].describe().round(3).to_dict()
    else:
        numeric_desc = {}

    head = df.head(max_rows).astype(str).to_dict(orient="records")

    summary = {
        "profile": profile,
        "numeric_describe": numeric_desc,
        "head": head,
    }

    return json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def get_openai_client(api_key):
    """OpenAI 클라이언트를 생성합니다."""
    if not api_key:
        return None

    return OpenAI(api_key=api_key)


def ask_ai(api_key, model, system_prompt, user_prompt):
    """OpenAI API로 AI 응답을 생성합니다."""
    client = get_openai_client(api_key)

    if client is None:
        return "OpenAI API 키를 입력하면 AI 분석 결과가 표시됩니다."

    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        return response.output_text

    except Exception as e:
        return f"AI 호출 중 오류가 발생했습니다: {e}"


def local_recommended_questions(df):
    """데이터 열 이름을 바탕으로 추천 질문을 만듭니다."""
    numeric_cols, date_cols, category_cols = detect_columns(df)

    questions = []

    if date_cols and numeric_cols:
        questions.append(f"{date_cols[0]}에 따라 {numeric_cols[0]}은 어떤 변화 추세를 보이나요?")
        questions.append(f"최근 구간에서 {numeric_cols[0]} 값이 증가하고 있나요, 감소하고 있나요?")

    if len(numeric_cols) >= 2:
        questions.append(f"{numeric_cols[0]}와 {numeric_cols[1]} 사이에는 어떤 관계가 있나요?")
        questions.append(f"{numeric_cols[0]}가 높을 때 {numeric_cols[1]}도 함께 높아지는 경향이 있나요?")

    if category_cols and numeric_cols:
        questions.append(f"{category_cols[0]}별 {numeric_cols[0]} 평균은 어떻게 다른가요?")

    if numeric_cols:
        questions.append(f"{numeric_cols[0]}에서 이상치로 의심되는 값은 무엇인가요?")
        questions.append(f"{numeric_cols[0]}의 최댓값과 최솟값이 나타난 행은 어떤 특징이 있나요?")

    questions.extend(
        [
            "이 데이터로 설명할 수 있는 환경 문제는 무엇인가요?",
            "이 분석 결과를 바탕으로 학교나 지역사회에서 실천할 수 있는 방안은 무엇인가요?",
        ]
    )

    return questions[:8]


def draw_chart(df, chart_type, x_col=None, y_col=None, hue_col=None):
    """선택한 차트를 생성합니다."""
    fig, ax = plt.subplots(figsize=(10, 5))

    plot_df = df.copy()

    if x_col and x_col in plot_df.columns:
        plot_df = plot_df.sort_values(x_col)

    if chart_type == "선 그래프" and x_col and y_col:
        sns.lineplot(data=plot_df, x=x_col, y=y_col, ax=ax)
        ax.set_title(f"{y_col} by {x_col}")

    elif chart_type == "막대그래프" and x_col and y_col:
        temp = plot_df.groupby(x_col, dropna=False)[y_col].mean().reset_index().head(30)
        sns.barplot(data=temp, x=x_col, y=y_col, ax=ax)
        ax.set_title(f"Average {y_col} by {x_col}")
        ax.tick_params(axis="x", rotation=45)

    elif chart_type == "산점도" and x_col and y_col:
        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue=hue_col if hue_col else None,
            ax=ax,
        )
        ax.set_title(f"{x_col} vs {y_col}")

    elif chart_type == "히트맵":
        numeric_df = plot_df.select_dtypes(include="number")

        if numeric_df.shape[1] >= 2:
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax)
            ax.set_title("Correlation Heatmap")
        else:
            ax.text(
                0.5,
                0.5,
                "숫자형 열이 2개 이상 필요합니다.",
                ha="center",
                va="center",
            )

    elif chart_type == "박스플롯" and y_col:
        if x_col:
            sns.boxplot(data=plot_df, x=x_col, y=y_col, ax=ax)
            ax.tick_params(axis="x", rotation=45)
        else:
            sns.boxplot(data=plot_df, y=y_col, ax=ax)

        ax.set_title(f"Boxplot of {y_col}")

    else:
        ax.text(
            0.5,
            0.5,
            "차트를 그리기 위한 열을 선택해주세요.",
            ha="center",
            va="center",
        )

    plt.tight_layout()

    return fig


def make_markdown_report(df, question, ai_result):
    """분석 결과를 Markdown 보고서로 만듭니다."""
    profile = make_data_profile(df)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""# 기후 데이터 AI 분석 보고서

생성 시각: {now}

## 1. 데이터 개요

- 행 개수: {profile["rows"]}
- 열 개수: {profile["columns"]}
- 숫자형 열: {", ".join(profile["numeric_columns"]) if profile["numeric_columns"] else "없음"}
- 날짜형 열: {", ".join(profile["date_columns"]) if profile["date_columns"] else "없음"}
- 문자형/범주형 열: {", ".join(profile["category_columns"]) if profile["category_columns"] else "없음"}
- 중복 행 수: {profile["duplicated_rows"]}

## 2. 분석 질문

{question if question else "분석 질문이 입력되지 않았습니다."}

## 3. AI 분석 결과

{ai_result}

## 4. 추가 탐구 질문

"""

    for q in local_recommended_questions(df)[:5]:
        report += f"- {q}\n"

    return report


# -----------------------------
# 사이드바
# -----------------------------
with st.sidebar:
    st.header("API 키 설정")
    st.write("OpenAI API 키를 입력하세요")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        label_visibility="collapsed",
    )

    if api_key:
        st.success("OpenAI API 키가 입력되었습니다.")
    else:
        st.warning("OpenAI API 키를 입력해주세요.")

    st.divider()

    st.header("데이터셋 업로드")
    st.write("CSV 파일을 업로드해주세요.")

    uploaded_file = st.file_uploader(
        "CSV 파일 업로드",
        type=["csv"],
        label_visibility="collapsed",
    )

    st.caption("200MB per file · CSV")

    st.divider()

    st.header("설정 옵션")

    model = st.selectbox(
        "AI 모델 선택",
        ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"],
        index=0,
    )

    max_preview_rows = st.slider(
        "미리보기 행 수",
        min_value=5,
        max_value=30,
        value=10,
    )

    auto_clean = st.checkbox(
        "열 이름 공백 자동 정리",
        value=True,
    )

    convert_dates = st.checkbox(
        "날짜형 열 자동 인식",
        value=True,
    )

    st.divider()

    st.info("💡 이 앱은 업로드된 환경 데이터 분석을 통해 기후 변화에 대한 의사 결정을 돕기 위해 GPT를 활용합니다.")
    st.success("🏫 중고등학교 환경 교육에도 활용 가능합니다.")


# -----------------------------
# 메인 화면
# -----------------------------
st.markdown(
    '<div class="main-title">Streamlit — 기후 데이터 AI 분석 도우미</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">CSV 데이터를 업로드하면 데이터 진단, 질문 추천, 시각화, AI 해석, 보고서 생성을 한 번에 수행합니다.</div>',
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.info("왼쪽 사이드바에서 OpenAI API 키와 CSV 파일을 업로드해주세요.")

    st.markdown(
        """
        ### 이 앱에서 할 수 있는 일

        1. CSV 데이터 미리보기
        2. 데이터 자동 진단
        3. 분석 질문 자동 추천
        4. 차트 자동 생성
        5. GPT 기반 분석 결과 제공
        6. 학생용 탐구 문제 및 보고서 생성
        """
    )

    st.stop()


# -----------------------------
# CSV 불러오기
# -----------------------------
try:
    df, used_encoding = read_csv_safely(uploaded_file)

    if auto_clean:
        df = clean_column_names(df)

    if convert_dates:
        df, detected_date_cols = try_convert_dates(df)
    else:
        detected_date_cols = []

except Exception as e:
    st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()


numeric_cols, date_cols, category_cols = detect_columns(df)
profile = make_data_profile(df)


# -----------------------------
# 1. 데이터 미리보기
# -----------------------------
st.markdown("## 📊 데이터 미리보기")
st.caption(f"파일 인코딩 추정: {used_encoding}")
st.dataframe(df.head(max_preview_rows), use_container_width=True)


# -----------------------------
# 2. 데이터 자동 진단
# -----------------------------
st.markdown("## 🔍 데이터 자동 진단")

col1, col2, col3, col4 = st.columns(4)

col1.metric("행 개수", f"{profile['rows']:,}")
col2.metric("열 개수", f"{profile['columns']:,}")
col3.metric("숫자형 열", f"{len(numeric_cols):,}")
col4.metric("중복 행", f"{profile['duplicated_rows']:,}")

with st.expander("자세한 데이터 진단 결과 보기", expanded=True):
    st.write("**숫자형 열**", numeric_cols if numeric_cols else "없음")
    st.write("**날짜형 열**", date_cols if date_cols else "없음")
    st.write("**문자형/범주형 열**", category_cols if category_cols else "없음")

    if profile["missing_top"]:
        st.write("**결측치가 있는 열 상위 목록**")
        st.json(profile["missing_top"])
    else:
        st.write("결측치가 발견되지 않았습니다.")

    if profile["outlier_info"]:
        st.write("**이상치 의심 열**")
        st.dataframe(pd.DataFrame(profile["outlier_info"]), use_container_width=True)
    else:
        st.write("IQR 기준 이상치가 뚜렷하게 발견되지 않았습니다.")


# -----------------------------
# 3. 분석 질문 추천
# -----------------------------
st.markdown("## ❓ 분석 질문 추천")

recommended_questions = local_recommended_questions(df)

selected_question = st.radio(
    "추천 질문 중 하나를 선택하거나 아래에 직접 입력하세요.",
    recommended_questions,
)

custom_question = st.text_area(
    "직접 질문 입력",
    placeholder="예: 이 데이터에서 기후 변화의 장기 추세를 설명할 수 있나요?",
)

question = custom_question.strip() if custom_question.strip() else selected_question


# -----------------------------
# 4. 시각화 자동 생성
# -----------------------------
st.markdown("## 📈 시각화 자동 생성")

chart_col1, chart_col2, chart_col3, chart_col4 = st.columns(4)

chart_options = [
    "선 그래프",
    "막대그래프",
    "산점도",
    "히트맵",
    "박스플롯",
]

chart_type = chart_col1.selectbox(
    "차트 종류",
    chart_options,
)

all_cols = df.columns.tolist()

x_default = date_cols[0] if date_cols else (
    category_cols[0] if category_cols else (
        all_cols[0] if all_cols else None
    )
)

y_default = numeric_cols[0] if numeric_cols else None

x_options = [None] + all_cols
y_options = [None] + numeric_cols
hue_options = [None] + category_cols

x_index = x_options.index(x_default) if x_default in x_options else 0
y_index = y_options.index(y_default) if y_default in y_options else 0

x_col = chart_col2.selectbox(
    "X축",
    x_options,
    index=x_index,
)

y_col = chart_col3.selectbox(
    "Y축",
    y_options,
    index=y_index,
)

hue_col = chart_col4.selectbox(
    "색상 구분",
    hue_options,
    index=0,
)

fig = draw_chart(
    df,
    chart_type,
    x_col=x_col,
    y_col=y_col,
    hue_col=hue_col,
)

st.pyplot(fig)


# -----------------------------
# 5. AI 분석 결과
# -----------------------------
st.markdown("## ✨ AI 분석 결과")
st.write(f"**현재 분석 질문:** {question}")

if st.button("AI 분석 실행", type="primary"):
    with st.spinner("GPT가 데이터를 분석하는 중입니다..."):
        system_prompt = (
            "너는 중고등학교 환경 교육과 데이터 리터러시 수업을 돕는 데이터 분석 교사이다. "
            "업로드된 CSV 데이터 요약을 바탕으로 과장하지 말고, 데이터에서 확인 가능한 내용과 추론을 구분해 한국어로 설명한다. "
            "학생이 이해할 수 있도록 핵심 발견, 근거, 한계, 추가 탐구 질문을 포함한다."
        )

        user_prompt = f"""
다음은 업로드된 CSV 데이터의 요약 정보입니다.

{dataframe_summary_text(df)}

사용자의 분석 질문:
{question}

다음 형식으로 답하세요.

1. 핵심 결론
2. 데이터 근거
3. 시각화에서 확인할 점
4. 해석 시 주의할 점
5. 추가 탐구 질문 3개
"""

        ai_result = ask_ai(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        st.session_state["ai_result"] = ai_result
        st.markdown(ai_result)

else:
    if "ai_result" in st.session_state:
        st.markdown(st.session_state["ai_result"])
    else:
        st.info("버튼을 누르면 AI 분석 결과가 생성됩니다.")


# -----------------------------
# 6. 학생용 문제 생성
# -----------------------------
st.markdown("## 📝 학생용 탐구 문제 생성")

problem_type = st.selectbox(
    "문제 유형",
    [
        "자동채점형",
        "사고력 평가형",
        "객관식",
        "세특 작성용 탐구 질문",
    ],
)

if st.button("학생용 문제 만들기"):
    with st.spinner("문제를 생성하는 중입니다..."):
        system_prompt = "너는 중고등학교 정보/환경 교과 수업 자료를 만드는 교사이다."

        user_prompt = f"""
다음 데이터 요약을 바탕으로 '{problem_type}' 문제 4개를 만들어줘.

{dataframe_summary_text(df)}

조건:
- 학생이 데이터와 그래프를 근거로 답하게 할 것
- 너무 단순한 사실 확인 문제는 피할 것
- 정답 또는 평가 기준을 함께 제시할 것
- 한국어로 작성할 것
"""

        problems = ask_ai(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        st.session_state["problems"] = problems
        st.markdown(problems)

else:
    if "problems" in st.session_state:
        st.markdown(st.session_state["problems"])


# -----------------------------
# 7. 분석 보고서 다운로드
# -----------------------------
st.markdown("## 📄 분석 보고서 다운로드")

ai_result_for_report = st.session_state.get(
    "ai_result",
    "아직 AI 분석을 실행하지 않았습니다.",
)

report_md = make_markdown_report(
    df=df,
    question=question,
    ai_result=ai_result_for_report,
)

st.download_button(
    label="Markdown 보고서 다운로드",
    data=report_md.encode("utf-8-sig"),
    file_name="climate_data_ai_report.md",
    mime="text/markdown",
)

with st.expander("보고서 미리보기"):
    st.markdown(report_md)
