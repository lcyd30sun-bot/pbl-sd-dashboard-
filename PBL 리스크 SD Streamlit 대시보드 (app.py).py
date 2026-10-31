# -*- coding: utf-8 -*-
"""
공군 중거리통제레이더 PBL 성과지표(NMCS + DRT) 장기 군수 리스크 시스템 다이내믹스 대시보드
실행:  pip install streamlit numpy pandas matplotlib
       streamlit run app.py
- 외부 SD 라이브러리 없이 월 단위 Euler 수치적분(dt=1개월)으로 로직을 투명하게 구현
- 모든 계수는 '가정값'이며 실제 적용 시 OPUS 10 산출값/계약조건/실적으로 교체해야 함
"""
import itertools
import math

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ------------------------- 한글 폰트 설정 -------------------------
for _f in ["NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
    if any(_f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

# =====================================================================
# 1. 모형 고정 상수 (가정값)
# =====================================================================
N_RADAR = 20          # 레이더 운용 대수
DRT_BASE = 24.0       # 기본 DRT(시간)
CAP_BASE = 8.0        # 업체 월간 기본 출고능력 상한(건/월)
INV_REF = 50.0        # 재고 기준치
RESTOCK_BASE = 4.5    # 월간 기본 재고 보충량
INV_MAX = 60.0        # 재고 목표 상한
PEN_THRESHOLD = 300.0 # 패널티 피로 임계점
PEN_SCALE = 300.0     # 임계점 초과 후 재투자율이 절반이 되는 추가 누적량
PEN_DECAY = 0.02      # 패널티 월 자연감쇠율
PENALTY_PER_HOUR = 0.5  # 목표 초과 1시간당 월 패널티 점수
N_MC = 50             # 몬테카를로 반복 횟수(기본)


# =====================================================================
# 2. 인과관계식 (Stock & Flow 보조 함수)
# =====================================================================
def calc_drt(backlog, inventory):
    """DRT(시간): 대기선 증가·재고 고갈 시 비선형 증가 (최대 720시간)"""
    backlog_pressure = (backlog / N_RADAR) ** 2 * 12.0
    stock_ratio = max(inventory, 0.0) / INV_REF
    shortage = 1.0 / (stock_ratio + 0.05) - 1.0 / 1.05
    shortage_pressure = 6.0 * max(shortage, 0.0) ** 1.3
    return min(DRT_BASE + backlog_pressure + shortage_pressure, 720.0)


def calc_capacity(inventory, drt):
    """업체 월간 출고능력: 재고에 비례, DRT가 길수록 감소(하한 60%)"""
    stock_factor = min(max(inventory, 0.0) / INV_REF, 1.0)
    drt_factor = DRT_BASE / max(drt, DRT_BASE)
    return CAP_BASE * stock_factor * max(drt_factor, 0.6)


def calc_restock(penalty, inventory):
    """재고 보충 흐름: 패널티 누적이 임계점을 넘으면 재투자율 감소(음의 피드백)"""
    excess = max(penalty - PEN_THRESHOLD, 0.0)
    reinvest = 1.0 / (1.0 + excess / PEN_SCALE)
    room = max(1.0 - inventory / INV_MAX, 0.0)
    return RESTOCK_BASE * reinvest * min(room * 3.0, 1.0), reinvest


# =====================================================================
# 3. 시뮬레이션 본체 (Euler, dt = 1개월)
# =====================================================================
def run(seed=42, penalty_feedback=True, drt_target=24.0, inv0=50.0, fail_mult=1.0,
        drt_fixed=None, restock_mult=1.0, months=60, fail_mean=5.0):
    """
    반환: 월별 상태 dict
    - drt_fixed   : [극단테스트] DRT 강제 고정(0이면 즉시 인도, 재고만 제약)
    - restock_mult: [극단테스트] 0이면 보급망 마비
    """
    rng = np.random.default_rng(seed)
    inv, bl, pen = float(inv0), 0.0, 0.0
    keys = ["inv", "bl", "drt", "pen", "nmcs", "reinv", "fail", "ful"]
    H = {k: [] for k in keys}
    for _ in range(months):
        fail = rng.exponential(fail_mean * fail_mult)                 # 고장발생률(지수분포 난수)
        drt = calc_drt(bl, inv) if drt_fixed is None else float(drt_fixed)
        cap = float("inf") if drt_fixed == 0 else calc_capacity(inv, drt)
        ful = min(bl + fail, cap, max(inv, 0.0))                      # 부품인도율 = Min(대기선, 출고능력, 재고)
        restock, reinv = calc_restock(pen if penalty_feedback else 0.0, inv)
        restock *= restock_mult
        over = max(drt - drt_target, 0.0)                             # 목표 초과 시간
        pen = pen * (1 - PEN_DECAY) + PENALTY_PER_HOUR * over         # 패널티 누적
        nmcs = bl / N_RADAR * 100.0                                   # NMCS(%) = 대기선/운용대수
        for k, v in zip(keys, [inv, bl, drt, pen, nmcs, reinv, fail, ful]):
            H[k].append(v)
        bl = bl + fail - ful                                          # 대기선 변동 = 고장발생 - 인도
        inv = inv - ful + restock                                     # 재고 변동 = 보충 - 출고
    return {k: np.array(v) for k, v in H.items()}


def cum_nmcs_days(h):
    """누적 NMCS 일수(레이더-일) = Σ min(대기선, 운용대수) x 30일"""
    return float(np.sum(np.minimum(h["bl"], N_RADAR)) * 30.0)


def monthly_nmcs_days(h):
    """월별 NMCS 일수(레이더-일)"""
    return np.minimum(h["bl"], N_RADAR) * 30.0


# =====================================================================
# 4. 분석 모듈: 몬테카를로 / 민감도 / Shapley 기여도
# =====================================================================
@st.cache_data(show_spinner=False)
def mc_paths(n_mc, **kw):
    """n_mc개 시드의 시뮬레이션 결과를 배열로 반환"""
    runs = [run(seed=s, **kw) for s in range(n_mc)]
    return {k: np.array([r[k] for r in runs]) for k in runs[0]}


@st.cache_data(show_spinner=False)
def sensitivity_table(n_mc, fail_mult):
    """목표 DRT(24/48/72) x 초기 재고(40/50/60) 9개 조건 비교표"""
    rows = []
    for tgt, inv0 in itertools.product([24, 48, 72], [40, 50, 60]):
        runs = [run(seed=s, drt_target=tgt, inv0=inv0, fail_mult=fail_mult) for s in range(n_mc)]
        v = np.array([cum_nmcs_days(r) for r in runs])
        rows.append({"목표DRT(h)": tgt, "초기재고": inv0,
                     "누적NMCS(레이더-일) 평균": round(v.mean()), "표준편차": round(v.std()),
                     "60개월차 평균 패널티": round(np.mean([r["pen"][-1] for r in runs]), 1),
                     "60개월차 평균 재투자율": round(np.mean([r["reinv"][-1] for r in runs]), 3)})
    return pd.DataFrame(rows)


FACTORS = {
    "외부 고장률 충격(+20%)": {"fail_mult": 1.2},
    "업체 재투자 위축(피드백)": {"penalty_feedback": True},
    "초기 재고 부족(50→40)": {"inv0": 40},
}
BASE_KW = {"fail_mult": 1.0, "penalty_feedback": False, "inv0": 50, "drt_target": 24}


@st.cache_data(show_spinner=False)
def shapley(n_mc):
    """요인 on/off 전수 조합으로 Shapley 기여도 계산 (합 = 총 악화량)"""
    names = list(FACTORS)
    n = len(names)
    cache = {}
    for r in range(n + 1):
        for c in itertools.combinations(names, r):
            kw = dict(BASE_KW)
            for nm in c:
                kw.update(FACTORS[nm])
            cache[frozenset(c)] = float(np.mean([cum_nmcs_days(run(seed=s, **kw)) for s in range(n_mc)]))
    phi = {}
    for f in names:
        others = [x for x in names if x != f]
        val = 0.0
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                val += w * (cache[frozenset(S) | {f}] - cache[frozenset(S)])
        phi[f] = val
    return phi, cache[frozenset(names)] - cache[frozenset()]


# =====================================================================
# 5. VV&A 검증 모듈 (백테스팅 + 극단조건 테스트)
# =====================================================================
MAPE_LIMIT = 5.0
# 가상의 과거 12개월 실제 월별 NMCS 일수(레이더-일). 실제 공군 실적으로 반드시 교체할 것.
VIRTUAL_ACTUAL = [0, 39, 37, 70, 59, 80, 113, 137, 157, 163, 177, 176]


@st.cache_data(show_spinner=False)
def backtest(actual, n_mc):
    """예측(몬테카를로 평균 경로) 대비 실적 MAPE 계산. 실적 0인 달은 제외"""
    actual = np.asarray(actual, dtype=float)
    pred = np.mean([monthly_nmcs_days(run(seed=s, months=len(actual))) for s in range(n_mc)], axis=0)
    mask = actual > 0
    mape = float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)
    return pred, mape, int((~mask).sum())


def is_stable(h):
    """수치 안정성: NaN/무한대 없음, 재고·대기선 음수 없음"""
    vals = np.concatenate([h[k] for k in ("inv", "bl", "drt", "nmcs", "pen")])
    return bool(np.all(np.isfinite(vals)) and h["inv"].min() >= -1e-9 and h["bl"].min() >= -1e-9)


@st.cache_data(show_spinner=False)
def extreme_tests(n_seeds=20):
    """극단조건 3종 테스트. 결과 dict 반환"""
    seeds = range(n_seeds)
    r1 = [run(seed=s, drt_fixed=0.0, inv0=500.0) for s in seeds]      # 완벽한 조달(재고 충분)
    r1b = [run(seed=s, drt_fixed=0.0) for s in seeds]                  # 참고: 기본 재고
    r2 = [run(seed=s, restock_mult=0.0) for s in seeds]                # 보급망 마비
    r3 = [run(seed=s, fail_mult=10.0) for s in seeds]                  # 고장 10배 스트레스
    max_nmcs1 = max(h["nmcs"].max() for h in r1)
    avail_end = [1 - min(h["nmcs"][-1] / 100, 1) for h in r2]
    out = {
        "S1_pass": all(is_stable(h) for h in r1) and max_nmcs1 < 1e-9,
        "S1_max_nmcs": max_nmcs1,
        "S1b_max_nmcs": max(h["nmcs"].max() for h in r1b),
        "S2_pass": all(is_stable(h) for h in r2) and max(avail_end) < 1e-9 and all(h["inv"][-1] < 1.0 for h in r2),
        "S2_max_avail": max(avail_end) * 100,
        "S2_min_inv": min(h["inv"].min() for h in r2),
        "S2_end_inv": max(h["inv"][-1] for h in r2),
        "S3_pass": all(is_stable(h) for h in r3),
        "r1": r1[0], "r2": r2[0],
    }
    out["all_pass"] = out["S1_pass"] and out["S2_pass"] and out["S3_pass"]
    return out


# =====================================================================
# 6. Streamlit UI
# =====================================================================
st.set_page_config(page_title="PBL 장기 군수 리스크 SD 대시보드", layout="wide")
st.title("중거리통제레이더 PBL 성과지표 전환(NMCS + DRT) 장기 리스크 대시보드")
st.caption("가정 파라미터 기반 상대 비교 모형입니다. 절대값은 실제 계약·정비 데이터로 보정이 필요합니다.")

with st.sidebar:
    st.header("시뮬레이션 입력")
    inv0_in = st.slider("초기 OPUS 재고(단위)", 20, 100, 50, 5)
    tgt_in = st.select_slider("목표 DRT(시간)", options=[24, 48, 72], value=24)
    fail_in = st.slider("고장률 충격 배율", 0.5, 2.0, 1.0, 0.1)
    fb_in = st.checkbox("패널티→재투자 위축 피드백 적용", value=True)
    seed_in = st.number_input("난수 시드(단일 경로)", 0, 9999, 42)
    n_mc_in = st.slider("몬테카를로 반복 횟수", 20, 200, N_MC, 10)

tab1, tab2, tab3 = st.tabs(["1. 장기 리스크 시뮬레이션", "2. 지표 민감도 분석", "3. 국방 VV&A 검증 가이드"])

# ---------------------------- 탭 1 ----------------------------
with tab1:
    kw = dict(penalty_feedback=fb_in, drt_target=tgt_in, inv0=inv0_in, fail_mult=fail_in)
    base = run(seed=int(seed_in), **kw)
    mc = mc_paths(n_mc_in, **kw)
    m = np.arange(1, 61)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("60개월차 업체 재고", f"{base['inv'][-1]:.1f}")
    c2.metric("60개월차 DRT(시간)", f"{base['drt'][-1]:.1f}")
    c3.metric("60개월차 NMCS(%)", f"{base['nmcs'][-1]:.1f}")
    c4.metric(f"MC {n_mc_in}회 평균 NMCS(%)", f"{mc['nmcs'][:, -1].mean():.1f}")

    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    ax[0].plot(m, base["inv"], "b-", label="단일 경로")
    ax[0].plot(m, mc["inv"].mean(axis=0), "b--", label="MC 평균")
    ax[0].set_ylabel("재고(단위)"); ax[0].set_title("업체 수리부속 재고량"); ax[0].legend()
    ax[1].plot(m, base["drt"], "r-", label="DRT")
    ax[1].axhline(tgt_in, color="k", ls="--", label=f"목표 {tgt_in}h")
    ax[1].set_ylabel("DRT(시간)"); ax[1].set_title("DRT(인도응답지연시간)"); ax[1].legend()
    ax[2].plot(m, base["nmcs"], "g-", label="단일 경로")
    ax[2].fill_between(m, np.percentile(mc["nmcs"], 5, axis=0), np.percentile(mc["nmcs"], 95, axis=0),
                       color="g", alpha=.15, label="MC 90% 구간")
    ax[2].set_ylabel("NMCS(%)"); ax[2].set_title("NMCS 비율 = 대기선/운용대수"); ax[2].legend()
    ax[3].plot(m, base["pen"], "m-", label="패널티 누적")
    ax[3].axhline(PEN_THRESHOLD, color="k", ls="--", label="피로 임계점")
    ax3 = ax[3].twinx(); ax3.plot(m, base["reinv"], "c-"); ax3.set_ylabel("재투자율(0~1)")
    ax[3].set_ylabel("패널티"); ax[3].set_xlabel("월"); ax[3].set_title("패널티 누적과 재투자율(음의 피드백)")
    ax[3].legend(loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)
    st.info("NMCS는 요청 정의(대기선/운용대수)를 그대로 쓰므로 100%를 넘을 수 있습니다. 보고용이라면 상한(cap) 적용을 검토하세요.")

# ---------------------------- 탭 2 ----------------------------
with tab2:
    st.subheader("목표 DRT x 초기 재고 민감도 (5년 누적 NMCS, 레이더-일)")
    df = sensitivity_table(n_mc_in, fail_in)
    st.dataframe(df, hide_index=True)
    pv = df.pivot(index="목표DRT(h)", columns="초기재고", values="누적NMCS(레이더-일) 평균")
    st.write("피벗(행: 목표 DRT, 열: 초기 재고)")
    st.dataframe(pv)

    st.subheader("가동률 붕괴 요인별 기여도 (Shapley)")
    phi, total = shapley(min(n_mc_in, 50))
    pos_sum = max(sum(max(v, 0) for v in phi.values()), 1e-9)
    names = list(phi)
    share = [max(phi[k], 0) / pos_sum * 100 for k in names]
    fig2, ax2 = plt.subplots(figsize=(8, 3.8))
    bars = ax2.barh(names, share, color=["#d9534f", "#f0ad4e", "#5bc0de"])
    for b, k, sv in zip(bars, names, share):
        ax2.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{sv:.1f}% ({phi[k]:+,.0f} 레이더-일)", va="center")
    ax2.set_xlim(0, 140); ax2.set_xlabel("NMCS 악화 기여 비중(%)")
    fig2.tight_layout()
    st.pyplot(fig2)

    # 자동 결론문
    mean_by_tgt = pv.mean(axis=1)
    equiv = mean_by_tgt[mean_by_tgt <= mean_by_tgt.min() * 1.01].index
    opp = df.groupby("목표DRT(h)")["60개월차 평균 재투자율"].mean()
    best = int(opp[equiv].idxmax())
    gap = (mean_by_tgt.max() - mean_by_tgt.min()) / mean_by_tgt.min() * 100
    top = max(phi, key=phi.get)
    st.subheader("자동 생성 종합 결론")
    st.write(f"본 시뮬레이션 결과, 목표 DRT를 {best}시간으로 설정한 경우 5년 누적 NMCS가 평균 {mean_by_tgt[best]:,.0f} 레이더-일로 "
             f"동등 후보 중 업체 재투자율({opp[best]:.2f})이 가장 높아 기회주의적 행동 리스크가 가장 작게 나타났습니다.")
    st.write(f"가동률 붕괴의 최대 원인은 '{top}'(기여 {max(phi[top], 0) / pos_sum * 100:.0f}%)입니다.")
    if gap < 1.0:
        st.warning(f"목표 DRT 간 NMCS 차이는 최대 {gap:.1f}%로 미미합니다. DRT 목표 조정만으로 가동률이 좌우되지 않으며, "
                   "선택 근거는 패널티 부담·재투자율 차이입니다. 재고·보충 확보가 더 큰 영향 요인입니다.")
    else:
        st.info(f"목표 DRT 간 NMCS 차이는 최대 {gap:.1f}%입니다.")

# ---------------------------- 탭 3 ----------------------------
with tab3:
    st.subheader("VV&A 검증 1: 과거 12개월 백테스팅 (MAPE)")
    st.caption("아래 12개월 실적은 가상 데이터입니다. 표를 직접 수정해 실제 공군 월별 NMCS 일수(레이더-일)로 바꿔 검증하세요.")
    edited = st.data_editor(pd.DataFrame({"월": range(1, 13), "실제 NMCS 일수": VIRTUAL_ACTUAL}),
                            hide_index=True, disabled=["월"])
    actual = tuple(float(x) for x in edited["실제 NMCS 일수"])
    pred, mape, n_zero = backtest(actual, n_mc_in)

    fig3, ax3 = plt.subplots(1, 2, figsize=(12, 4))
    mm = np.arange(1, 13)
    ax3[0].plot(mm, actual, "ko-", label="실제(가상) 실적")
    ax3[0].plot(mm, pred, "r^--", label="시뮬레이션 예측(MC 평균)")
    ax3[0].set_xlabel("월"); ax3[0].set_ylabel("NMCS 일수(레이더-일)"); ax3[0].set_title("실적 vs 예측"); ax3[0].legend()
    a = np.array(actual)
    ape = np.where(a > 0, np.abs((a - pred) / np.where(a > 0, a, 1)) * 100, np.nan)
    ax3[1].bar(mm, ape, color=["#5cb85c" if (not np.isnan(x) and x <= MAPE_LIMIT) else "#d9534f" for x in ape])
    ax3[1].axhline(MAPE_LIMIT, color="k", ls="--", label=f"기준 {MAPE_LIMIT}%")
    ax3[1].set_xlabel("월"); ax3[1].set_ylabel("월별 오차율(%)"); ax3[1].set_title(f"오차율 (MAPE = {mape:.2f}%)"); ax3[1].legend()
    fig3.tight_layout()
    st.pyplot(fig3)
    st.metric("MAPE", f"{mape:.2f}%", help=f"실적 0인 {n_zero}개월은 0으로 나눌 수 없어 제외")
    if mape <= MAPE_LIMIT:
        st.success(f"검증 통과: 신뢰성 확보 (MAPE {mape:.2f}% ≤ {MAPE_LIMIT}%)")
    else:
        st.error(f"검증 실패: 모형 튜닝 필요 (MAPE {mape:.2f}% > {MAPE_LIMIT}%)")
    st.warning("기본 제공 실적은 모형 예측 경로를 참고해 만든 가상값이므로 통과는 코드 동작 확인용입니다. 실제 실적 입력 시 결과가 달라질 수 있습니다.")

    st.divider()
    st.subheader("VV&A 검증 2: 극단적 조건 테스트")
    ex = extreme_tests()
    e1, e2, e3 = st.columns(3)
    e1.metric("시나리오1 DRT=0h: 최대 NMCS", f"{ex['S1_max_nmcs']:.2f}%")
    e2.metric("시나리오2 재보충=0: 최종 가동률", f"{ex['S2_max_avail']:.1f}%")
    e3.metric("시나리오2 최소 재고", f"{ex['S2_min_inv']:.2f}")

    fig4, ax4 = plt.subplots(1, 2, figsize=(12, 3.8))
    ax4[0].plot(np.arange(1, 61), ex["r1"]["nmcs"], "g-")
    ax4[0].set_title("시나리오1: DRT=0h(재고 충분) NMCS"); ax4[0].set_xlabel("월"); ax4[0].set_ylabel("NMCS(%)")
    ax4[1].plot(np.arange(1, 61), ex["r2"]["inv"], "b-", label="재고")
    ax4[1].plot(np.arange(1, 61), ex["r2"]["nmcs"] / 10, "r-", label="NMCS(%)/10")
    ax4[1].axhline(0, color="k", lw=.8)
    ax4[1].set_title("시나리오2: 재보충=0 (재고 0 수렴, 음수 없음)"); ax4[1].set_xlabel("월"); ax4[1].legend()
    fig4.tight_layout()
    st.pyplot(fig4)

    if ex["S1_pass"]:
        st.success("검증 통과 (시나리오1, 완벽한 조달): DRT=0h 시 NMCS가 0%로 수렴하며 수치 이상 없음")
    else:
        st.error("검증 실패 (시나리오1): NMCS가 0%로 수렴하지 않음")
    if ex["S2_pass"]:
        st.success("검증 통과 (시나리오2, 보급망 마비): 재고가 음수 없이 0으로 수렴하고 가동률 0%로 붕괴")
    else:
        st.error("검증 실패 (시나리오2): 음수 재고/발산 또는 가동률 비붕괴")
    if ex["S3_pass"]:
        st.success("검증 통과 (추가 스트레스, 고장 10배): 음수·무한대 발산 없음")
    else:
        st.error("검증 실패 (추가 스트레스): 수치 불안정")
    if ex["all_pass"]:
        st.success("검증 통과: 극단조건 전체에서 시스템이 수학적으로 안정적입니다.")
    else:
        st.error("검증 실패: 모형 수정이 필요합니다.")
    st.info(f"참고: DRT=0h여도 기본 재고/보충 조건에서는 재고 고갈로 최대 NMCS {ex['S1b_max_nmcs']:.0f}%가 발생합니다. 즉 DRT보다 재고·보충이 병목입니다. "
            f"(시나리오2는 출고능력이 재고에 비례해 재고가 0에 지수적으로 수렴하므로 1단위 미만이면 고갈로 판정, 최종 재고 최대 {ex['S2_end_inv']:.2f})")

    st.divider()
    st.subheader("VV&A 가이드")
    st.markdown("""
- **검증(Verification)**: 모형이 의도대로 구현되었는지 확인 (수치 안정성, 극단조건, 보존법칙).
- **확인(Validation)**: 모형이 실제 체계를 충분히 대변하는지 확인 (과거 실적 백테스팅, MAPE 5% 이내).
- **인정(Accreditation)**: 위 결과를 근거로 기품원 등 승인 기관이 용도별 사용을 인정.
- 제출 전 필수: 가상 실적을 실제 월별 NMCS 일수로 교체, 계수(출고능력·보충량·임계점)를 OPUS 10 및 계약 자료로 보정, 별도 기간 데이터로 교차검증.
""")
