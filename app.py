# app.py
# SM-PED Tunnel Assessment Master by LSH_DAUM
# - 엑셀(정밀안전점검/진단 물량표) 업로드 → 망도번호 자동 인식/부여
# - 점검표(물량표) 등급코드(a-1 등) 기반 결함점수 산정
# - 1단계(합계1) → 2단계(라이닝 지수) → 3단계(주변상태 합계2) → 4단계(Fi) → 기본시설 F(가중평균)
# - 부대시설: 시설명 단위 결함지수 산정 → 평균(∑fn/N) → 가중치(w) → 전체 시설물(F×w)
# - Streamlit 즉시 반영(키 고정, 버튼 처리 후 rerun, 편집값은 반환값 기반)

import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional

# ==============================================================================
# 0. 페이지 설정
# ==============================================================================
st.set_page_config(page_title="SM-PED Tunnel 버전", layout="wide")

# ==============================================================================
# 1. 기준 데이터(상수/테이블)
# ==============================================================================

TUNNEL_CONSTANTS = {
    "재래식터널(조적식)": {"L_denom": 26, "T_denom": 33},
    "재래식터널(무근)":   {"L_denom": 27, "T_denom": 34},
    "NATM터널(무근)":     {"L_denom": 27, "T_denom": 34},
    "NATM터널(철근)":     {"L_denom": 36, "T_denom": 43},
    "TBM터널(콘크리트 세그먼트)": {"L_denom": 36, "T_denom": 43},
    "개착터널(철근)":     {"L_denom": 36, "T_denom": 43},
    "지하차도(철근)":     {"L_denom": 36, "T_denom": 42},
}

F_GRADE_BOUNDS = [
    (0.00, 0.15, "A"),
    (0.15, 0.30, "B"),
    (0.30, 0.55, "C"),
    (0.55, 0.75, "D"),
    (0.75, float("inf"), "E"),
]

AUX_WEIGHT_TABLE = [
    (0.00, 0.15, 1.00),
    (0.15, 0.30, 1.00),
    (0.30, 0.55, 1.02),
    (0.55, 0.75, 1.05),
    (0.75, float("inf"), 1.10),
]

DAMAGE_CANONICAL_MAP = {
    "균열": "균열",
    "크랙": "균열",
    "누수": "누수",
    "파손": "파손 및 손상",
    "파손및손상": "파손 및 손상",
    "파손 및 손상": "파손 및 손상",
    "박리": "박리",
    "층분리": "층분리 및 박락",
    "층분리및박락": "층분리 및 박락",
    "층분리 및 박락": "층분리 및 박락",
    "박락": "층분리 및 박락",
    "백태": "백태",
    "재료분리": "재료분리",
    "철근노출": "철근노출",
    "탄산화": "탄산화",
    "염화물": "염화물",
}

# 형식별 제외(X) 항목: 귀 조직 기준표대로 채우면 됩니다.
# 현재는 "예시값"이며, 필요 시 비워두어도 됩니다.
FORM_EXCLUDED_DAMAGE: Dict[str, List[str]] = {
    "재래식터널(조적식)": ["철근노출", "탄산화", "염화물"],  # 예시
    "재래식터널(무근)":   ["철근노출"],                   # 예시
}

LINING_DAMAGE_SET = [
    "균열", "누수", "파손 및 손상",
    "박리", "층분리 및 박락", "백태", "재료분리", "철근노출", "탄산화", "염화물"
]

# ==============================================================================
# 2. 유틸
# ==============================================================================

def normalize_text(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()

def safe_float(x) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0.0
        return float(str(x).replace(",", "").strip())
    except:
        return 0.0

def canonical_damage(dmg: str) -> str:
    d = normalize_text(dmg)
    d = re.sub(r"\s+", " ", d)
    d_nospace = d.replace(" ", "")
    if d in DAMAGE_CANONICAL_MAP:
        return DAMAGE_CANONICAL_MAP[d]
    if d_nospace in DAMAGE_CANONICAL_MAP:
        return DAMAGE_CANONICAL_MAP[d_nospace]
    if "-" in d:
        head = d.split("-", 1)[0].strip()
        if head in DAMAGE_CANONICAL_MAP:
            return DAMAGE_CANONICAL_MAP[head]
    return d

def grade_to_score(grade_raw: str) -> float:
    """
    등급코드:
    - "a-1", "c-7" 형태: '-' 뒤 숫자를 점수로 사용(엑셀과 동일)
    - "a"~"e": 기본 맵
    """
    s = normalize_text(grade_raw).lower()
    if not s or s in ["-", "nan", "none"]:
        return 0.0
    if "-" in s:
        tail = s.split("-", 1)[1].strip()
        try:
            return float(tail)
        except:
            pass
    base_map = {"a": 1.0, "b": 4.0, "c": 7.0, "d": 10.0, "e": 13.0}
    return base_map.get(s[:1], 0.0)

def worst_grade_code(codes: List[str]) -> str:
    """
    최악 등급 선택 기준:
    1) 점수(코드의 '-' 뒤 숫자 또는 a~e 기본점수)가 큰 것이 더 불리
    2) 동점이면 문자 우선순위 e>d>c>b>a
    """
    if not codes:
        return "a-0"

    def key_fn(code: str) -> Tuple[float, int]:
        c = normalize_text(code).lower()
        score = grade_to_score(c)
        letter = c[:1] if c else "a"
        order = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}.get(letter, 1)
        return (score, order)

    return max(codes, key=key_fn)

def get_F_grade(F: float) -> str:
    for lo, hi, g in F_GRADE_BOUNDS:
        if lo <= F < hi:
            return g
    return "E"

def aux_weight(aux_f: float) -> float:
    for lo, hi, w in AUX_WEIGHT_TABLE:
        if lo <= aux_f < hi:
            return w
    return 1.10

# ==============================================================================
# 3. 엑셀 로더(캐시)
# ==============================================================================

EXPECTED_COL_HINTS = {
    "member": ["부재"],
    "damage": ["손상"],
    "width": ["폭"],
    "length": ["길이"],
    "depth": ["깊이"],
    "count": ["개소", "개수"],
    "unit": ["단위"],
    "qty": ["물량"],
    "grade": ["등급"],
}

def find_header_row(df_raw: pd.DataFrame) -> int:
    for i, row in df_raw.iterrows():
        line = " ".join([normalize_text(v) for v in row.values])
        if ("부재" in line) and ("손상" in line) and (("폭" in line) or ("길이" in line)):
            return i
    return -1

def pick_col(cols: List[str], hints: List[str]) -> Optional[str]:
    for h in hints:
        for c in cols:
            if h in c:
                return c
    return None

@st.cache_data(show_spinner=False)
def load_excel_quantity_table_cached(file_bytes: bytes, file_name: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    bytes 기반으로 캐시 적용.
    반환 DF 표준 컬럼:
    - 망도번호, 부재, 손상, 폭, 길이, 깊이, 개소, 단위, 물량, 등급, 점수, 구분(기본시설/부대시설), 부대시설명
    """
    try:
        from io import BytesIO
        bio = BytesIO(file_bytes)

        df_raw = pd.read_excel(bio, header=None)
        header_row = find_header_row(df_raw)
        if header_row < 0:
            return None, "헤더(부재/손상/폭/길이)를 찾지 못했습니다."

        bio.seek(0)
        df = pd.read_excel(bio, header=header_row)
        df.columns = [normalize_text(c) for c in df.columns]

        col_member = pick_col(df.columns.tolist(), EXPECTED_COL_HINTS["member"])
        col_damage = pick_col(df.columns.tolist(), EXPECTED_COL_HINTS["damage"])
        col_width  = pick_col(df.columns.tolist(), EXPECTED_COL_HINTS["width"])
        col_length = pick_col(df.columns.tolist(), EXPECTED_COL_HINTS["length"])
        col_depth  = pick_col(df.columns.tolist(), EXPECTED_COL_HINTS["depth"])
        col_count  = pick_col(df.columns.tolist(), EXPECTED_COL_HINTS["count"])
        col_unit   = pick_col(df.columns.tolist(), EXPECTED_COL_HINTS["unit"])
        col_qty    = pick_col(df.columns.tolist(), EXPECTED_COL_HINTS["qty"])
        col_grade  = pick_col(df.columns.tolist(), EXPECTED_COL_HINTS["grade"])

        if not col_member or not col_damage:
            return None, "필수 컬럼(부재, 손상)을 찾지 못했습니다."

        processed = []
        current_span = 0
        last_valid = False

        for _, r in df.iterrows():
            mem = normalize_text(r.get(col_member))
            dmg = normalize_text(r.get(col_damage))

            # 헤더 재등장/합계/평균 필터
            line = f"{mem} {dmg} {normalize_text(r.get(col_width))} {normalize_text(r.get(col_length))}"
            if ("부재" in line and "손상" in line) or ("합계" in line) or ("평균" in line):
                last_valid = False
                continue

            if mem.lower() in ["", "nan", "none"] or dmg.lower() in ["", "nan", "none"]:
                last_valid = False
                continue

            if not last_valid:
                current_span += 1
            last_valid = True

            dmg_c = canonical_damage(dmg)

            width = safe_float(r.get(col_width))
            length = safe_float(r.get(col_length))
            depth = safe_float(r.get(col_depth)) if col_depth else 0.0
            count = safe_float(r.get(col_count)) if col_count else 1.0
            unit = normalize_text(r.get(col_unit)) if col_unit else ""
            grade_raw = normalize_text(r.get(col_grade)) if col_grade else ""

            qty = safe_float(r.get(col_qty)) if col_qty else 0.0
            if qty == 0.0:
                qty = width * length * (count if count > 0 else 1.0)

            is_aux = any(k in mem for k in ["부대시설", "피난", "연락", "갱구", "옹벽", "공동구"])
            group = "부대시설" if is_aux else "기본시설"
            aux_name = ""
            if group == "부대시설":
                if "부대시설-" in mem:
                    aux_name = mem.split("부대시설-", 1)[1].strip()
                else:
                    aux_name = mem.strip()

            processed.append({
                "망도번호": int(current_span),
                "부재": mem,
                "손상": dmg_c,
                "폭": width,
                "길이": length,
                "깊이": depth,
                "개소": count if count > 0 else 1.0,
                "단위": unit,
                "물량": qty,
                "등급": grade_raw,
                "점수": grade_to_score(grade_raw),
                "구분": group,
                "부대시설명": aux_name,
            })

        if not processed:
            return None, "유효한 점검표 행을 찾지 못했습니다."

        return pd.DataFrame(processed), None

    except Exception as e:
        return None, f"엑셀 파싱 오류: {e}"

# ==============================================================================
# 4. 계산 엔진(엑셀 순서)
# ==============================================================================

def apply_form_exclusions(df_basic: pd.DataFrame, span_type_map: Dict[int, str]) -> pd.DataFrame:
    df2 = df_basic.copy()
    df2["형식"] = df2["망도번호"].map(lambda x: span_type_map.get(int(x), "NATM터널(철근)"))

    def is_excluded(row) -> bool:
        form = row["형식"]
        dmg = row["손상"]
        excluded = FORM_EXCLUDED_DAMAGE.get(form, [])
        return dmg in excluded

    mask = df2.apply(is_excluded, axis=1)
    df2.loc[mask, "점수"] = 0.0
    return df2

def step1_lining_defect_sum(df_basic: pd.DataFrame) -> pd.DataFrame:
    df = df_basic.copy()
    df["손상유형"] = df["손상"].apply(canonical_damage)
    df = df[df["손상유형"].isin(LINING_DAMAGE_SET)]

    max_by_type = df.groupby(["망도번호", "손상유형"])["점수"].max().reset_index()
    sum1 = max_by_type.groupby("망도번호")["점수"].sum().reset_index()
    sum1.columns = ["망도번호", "결함점수 합계1"]
    return sum1

def step2_lining_index(sum1: pd.DataFrame, span_type_map: Dict[int, str]) -> pd.DataFrame:
    rows = []
    for _, r in sum1.iterrows():
        sid = int(r["망도번호"])
        form = span_type_map.get(sid, "NATM터널(철근)")
        denom = TUNNEL_CONSTANTS[form]["L_denom"]
        idx = round(float(r["결함점수 합계1"]) / (denom if denom > 0 else 1), 3)
        rows.append({
            "망도번호": sid,
            "형식": form,
            "결함점수 합계1": float(r["결함점수 합계1"]),
            "라이닝 결함지수(f)": idx,
            "라이닝 등급": get_F_grade(idx),
        })
    return pd.DataFrame(rows)

def step2_lining_damage_grades(df_basic: pd.DataFrame) -> pd.DataFrame:
    df = df_basic.copy()
    df["손상유형"] = df["손상"].apply(canonical_damage)
    df = df[df["손상유형"].isin(LINING_DAMAGE_SET)]
    df["등급코드"] = df["등급"].apply(lambda x: normalize_text(x).lower() if normalize_text(x) else "a-0")

    rep = (
        df.groupby(["망도번호", "손상유형"])["등급코드"]
          .apply(lambda s: worst_grade_code(list(s)))
          .reset_index()
    )
    rep.columns = ["망도번호", "손상유형", "대표등급(최악)"]
    return rep

def step3_surround_input(spans: List[int]) -> pd.DataFrame:
    # 엑셀 기본값을 반영하려면 1로 시작(합계2=5)
    init = []
    for sid in spans:
        init.append({
            "망도번호": sid,
            "배수상태": 1,
            "지반상태": 1,
            "갱문상태": 1,
            "공동구상태": 1,
            "특수조건": 1,
            "결함점수 합계2": 5
        })
    return pd.DataFrame(init)

def step4_tunnel_index(df_step2: pd.DataFrame, df_step3: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(df_step2, df_step3[["망도번호", "결함점수 합계2"]], on="망도번호", how="left")
    df["결함점수 합계2"] = df["결함점수 합계2"].fillna(0).astype(float)

    f_list = []
    for _, r in df.iterrows():
        form = r["형식"]
        t_denom = TUNNEL_CONSTANTS[form]["T_denom"]
        val = round((float(r["결함점수 합계1"]) + float(r["결함점수 합계2"])) / (t_denom if t_denom > 0 else 1), 3)
        f_list.append(val)

    df["터널 결함지수(Fi)"] = f_list
    df["터널 등급"] = df["터널 결함지수(Fi)"].apply(get_F_grade)
    return df

def compute_basic_F_weighted(df_step4: pd.DataFrame, span_lengths: Dict[int, float]) -> Tuple[float, pd.DataFrame]:
    df = df_step4.copy()
    df["연장"] = df["망도번호"].map(lambda x: float(span_lengths.get(int(x), 0.0)))
    total_len = df["연장"].sum()

    df["연장비"] = df["연장"].apply(lambda x: (x / total_len) if total_len > 0 else 0.0)
    df["Fi×연장비"] = df["터널 결함지수(Fi)"] * df["연장비"]
    F_basic = float(df["Fi×연장비"].sum())

    return round(F_basic, 3), df

def compute_auxiliary(df_aux: pd.DataFrame) -> Tuple[float, float, float, pd.DataFrame]:
    """
    부대시설:
    - 시설명(부대시설명) 단위로 손상유형별 max → 합계3
    - 부대시설 결함지수 f = 합계3 / denom
    - 평균(∑fn/N)으로 w 산정
    """
    if df_aux.empty:
        return 0.0, 0.0, 1.0, pd.DataFrame()

    df = df_aux.copy()
    df["손상유형"] = df["손상"].apply(canonical_damage)
    df["부대시설명"] = df["부대시설명"].apply(lambda x: x if x else "부대시설(미지정)")
    df["등급코드"] = df["등급"].apply(lambda x: normalize_text(x).lower() if normalize_text(x) else "a-0")

    max_by_type = df.groupby(["부대시설명", "손상유형"])["점수"].max().reset_index()
    sum3 = max_by_type.groupby("부대시설명")["점수"].sum().reset_index()
    sum3.columns = ["부대시설명", "결함점수 합계3"]

    # 기본값: 27 (엑셀 캡처의 부대시설(무근)=27 사례 반영)
    # 필요 시 시설별(철근/무근) 분기 로직을 추가하세요.
    denom = 27.0
    sum3["부대시설 결함지수(f)"] = (sum3["결함점수 합계3"] / denom).round(3)
    sum3["부대시설 등급"] = sum3["부대시설 결함지수(f)"].apply(get_F_grade)

    fn_sum = float(sum3["부대시설 결함지수(f)"].sum())
    N = int(len(sum3))
    avg = round(fn_sum / N, 3) if N > 0 else 0.0
    w = aux_weight(avg)

    return round(fn_sum, 3), avg, w, sum3

# ==============================================================================
# 5. UI
# ==============================================================================

st.title("SM-PED Tunnel Assessment Master by LSH_DAUM")

with st.sidebar:
    st.header("입력")
    uploaded = st.file_uploader("점검표(물량표) 엑셀 업로드", type=["xlsx", "xls"])

    st.divider()
    st.header("세션")
    if st.button("세션 초기화", key="btn_reset_session"):
        st.session_state.clear()
        st.rerun()

if not uploaded:
    st.info("좌측에서 엑셀 파일을 업로드하세요.")
    st.stop()

file_bytes = uploaded.getvalue()
df_loaded, err = load_excel_quantity_table_cached(file_bytes, uploaded.name)
if df_loaded is None:
    st.error(err)
    st.stop()

df_all = df_loaded.copy()
df_basic = df_all[df_all["구분"] == "기본시설"].copy()
df_aux = df_all[df_all["구분"] == "부대시설"].copy()

basic_spans = sorted(df_basic["망도번호"].unique().tolist())
if not basic_spans:
    st.error("기본시설(라이닝) 데이터가 없습니다. 엑셀의 점검표 블럭을 확인하세요.")
    st.stop()

# 스팬 설정: 엑셀 업로드가 바뀌면 재생성
cfg_sig = f"{uploaded.name}:{len(df_all)}:{max(basic_spans)}"
if "cfg_sig" not in st.session_state or st.session_state["cfg_sig"] != cfg_sig:
    st.session_state["cfg_sig"] = cfg_sig

    # 망도별 기본 설정(연장은 사용자가 입력)
    st.session_state["span_cfg"] = pd.DataFrame([{
        "망도번호": int(s),
        "형식": "NATM터널(철근)",
        "시점": 0.0,
        "종점": 0.0,
        "연장": 0.0
    } for s in basic_spans])

    # 주변상태 기본값 1로 초기화(엑셀 기본 반영)
    st.session_state["step3"] = step3_surround_input([int(x) for x in basic_spans])

t1, t2, t3, t4 = st.tabs(["점검표", "기본시설(라이닝)", "부대시설", "종합결과"])

with t1:
    st.subheader("원자료(EXCEL) 인식 결과")
    st.dataframe(df_all, use_container_width=True, hide_index=True)

with t2:
    st.subheader("기본시설(라이닝) 상태평가")

    st.markdown("#### 0) 망도별 형식/시점/종점 설정(필수)")
    edited_cfg = st.data_editor(
        st.session_state["span_cfg"],
        column_config={
            "망도번호": st.column_config.NumberColumn(disabled=True),
            "형식": st.column_config.SelectboxColumn(options=list(TUNNEL_CONSTANTS.keys()), required=True),
            "시점": st.column_config.NumberColumn(required=True),
            "종점": st.column_config.NumberColumn(required=True),
            "연장": st.column_config.NumberColumn(disabled=True),
        },
        hide_index=True,
        use_container_width=True,
        key="editor_span_cfg",
    )
    edited_cfg["연장"] = (edited_cfg["종점"] - edited_cfg["시점"]).apply(lambda x: max(0.0, float(x)))
    st.session_state["span_cfg"] = edited_cfg

    span_type_map = edited_cfg.set_index("망도번호")["형식"].to_dict()
    span_len_map = edited_cfg.set_index("망도번호")["연장"].to_dict()
    st.info(f"총 연장: {edited_cfg['연장'].sum():.1f} m / 망도 개수: {len(edited_cfg)}개")

    st.markdown("#### 3) 주변상태 입력(결함점수 합계2)")

    # 버튼은 data_editor 위에 배치 + rerun(즉시 반영)
    if st.button("주변상태 기본값(1) 일괄 적용", key="btn_step3_apply_default1"):
        s3 = st.session_state["step3"].copy()
        for c in ["배수상태", "지반상태", "갱문상태", "공동구상태", "특수조건"]:
            s3[c] = 1
        s3["결함점수 합계2"] = s3[["배수상태", "지반상태", "갱문상태", "공동구상태", "특수조건"]].sum(axis=1)
        st.session_state["step3"] = s3
        st.rerun()

    step3_edit = st.data_editor(
        st.session_state["step3"],
        column_config={
            "망도번호": st.column_config.NumberColumn(disabled=True),
            "배수상태": st.column_config.NumberColumn(min_value=0, max_value=4, step=1),
            "지반상태": st.column_config.NumberColumn(min_value=0, max_value=4, step=1),
            "갱문상태": st.column_config.NumberColumn(min_value=0, max_value=4, step=1),
            "공동구상태": st.column_config.NumberColumn(min_value=0, max_value=4, step=1),
            "특수조건": st.column_config.NumberColumn(min_value=0, max_value=4, step=1),
            "결함점수 합계2": st.column_config.NumberColumn(disabled=True),
        },
        hide_index=True,
        use_container_width=True,
        key="editor_step3",
    )
    step3_edit["결함점수 합계2"] = step3_edit[["배수상태", "지반상태", "갱문상태", "공동구상태", "특수조건"]].sum(axis=1)
    st.session_state["step3"] = step3_edit

    # 1단계~4단계 계산(편집된 값 기반)
    st.markdown("#### 1) 1단계: 결함점수 합계1")
    df_basic_eff = apply_form_exclusions(df_basic, span_type_map)
    df_sum1 = step1_lining_defect_sum(df_basic_eff)
    st.dataframe(df_sum1, use_container_width=True, hide_index=True)

    st.markdown("#### 2) 2단계: 라이닝 결함지수(f)")
    df_step2 = step2_lining_index(df_sum1, span_type_map)
    st.dataframe(df_step2, use_container_width=True, hide_index=True)

    st.markdown("#### 2-보조) 손상유형별 대표 등급코드(최악)")
    df_grade_rep = step2_lining_damage_grades(df_basic_eff)
    st.dataframe(df_grade_rep, use_container_width=True, hide_index=True)

    st.markdown("#### 4) 4단계: 구간별 터널 결함지수(Fi)")
    df_step4 = step4_tunnel_index(df_step2, step3_edit)
    st.dataframe(df_step4, use_container_width=True, hide_index=True)

    # 종합 탭에서 사용
    st.session_state["df_step4"] = df_step4
    st.session_state["span_len_map"] = span_len_map

with t3:
    st.subheader("부대시설 상태평가")

    if df_aux.empty:
        st.info("부대시설 데이터가 없습니다. 가중치(w)=1.0으로 처리합니다.")
        st.session_state["aux_w"] = 1.0
        st.session_state["aux_avg"] = 0.0
        st.session_state["aux_table"] = pd.DataFrame()
    else:
        st.markdown("#### 원자료(부대시설)")
        st.dataframe(df_aux, use_container_width=True, hide_index=True)

        fn_sum, avg_aux, w, aux_table = compute_auxiliary(df_aux)
        st.session_state["aux_w"] = w
        st.session_state["aux_avg"] = avg_aux
        st.session_state["aux_table"] = aux_table

        c1, c2, c3 = st.columns(3)
        c1.metric("부대시설 결함지수 합계(∑fn)", f"{fn_sum:.3f}")
        c2.metric("부대시설 평균 결함지수(∑fn/N)", f"{avg_aux:.3f}")
        c3.metric("부대시설 가중치(w)", f"{w:.2f}")

        st.markdown("#### 부대시설(시설명 단위) 결함지수 산정")
        st.dataframe(aux_table, use_container_width=True, hide_index=True)

with t4:
    st.subheader("종합 상태평가 결과(엑셀 '터널 전체 시설물 상태평가 결과' 대응)")

    if "df_step4" not in st.session_state:
        st.warning("기본시설 탭에서 4단계 계산이 필요합니다.")
        st.stop()

    df_step4 = st.session_state["df_step4"].copy()
    span_len_map = st.session_state.get("span_len_map", {})

    F_basic, df_weighted = compute_basic_F_weighted(df_step4, span_len_map)

    w = float(st.session_state.get("aux_w", 1.0))
    F_total = round(F_basic * w, 3)

    grade_basic = get_F_grade(F_basic)
    grade_total = get_F_grade(F_total)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("기본시설 결함지수(F)", f"{F_basic:.3f}")
    c2.metric("기본시설 등급", grade_basic)
    c3.metric("부대시설 가중치(w)", f"{w:.2f}")
    c4.metric("전체 시설물 결함지수(F×w)", f"{F_total:.3f}")

    st.success(f"최종 종합 등급: {grade_total} (F={F_total:.3f})")

    st.markdown("#### 형식/구간별 연장비 및 가중 합산 내역(엑셀 4단계 대응)")
    show_cols = [
        "망도번호", "형식", "결함점수 합계1", "결함점수 합계2",
        "터널 결함지수(Fi)", "연장", "연장비", "Fi×연장비", "터널 등급"
    ]
    st.dataframe(df_weighted[show_cols], use_container_width=True, hide_index=True)

    with st.expander("참고: 형식별 제외(X) 처리 설정", expanded=False):
        st.write("현재 설정된 제외 항목은 점수=0으로 처리됩니다. (필요 시 귀 기준표로 업데이트)")
        st.json(FORM_EXCLUDED_DAMAGE)

st.caption("참고: 스트리밍 모드이며, delay가 발생할 수 있습니다. 수치 기입하거나 형식 바꿀때 2~3번 반복 부탁드립니다.")