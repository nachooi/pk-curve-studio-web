"""PK Curve Studio — scaffold-CV out-of-fold PK profile viewer.

The public-facing app is intentionally a result viewer, not a new automatic
predictor. For a molecule that matches the rat PK dataset, it displays the
saved out-of-fold curve from the final scaffold-CV analysis together with the
observed concentration points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend import Compound, canonical_smiles, get_oof_repository, pubchem_lookup


st.set_page_config(page_title="PK Curve Studio", page_icon="◒", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      .stApp { background: #f5f7fa; color: #162338; }
      [data-testid="stHeader"] { background: rgba(245,247,250,.92); }
      .block-container { max-width: 1500px; padding-top: 2.2rem; padding-bottom: 2.5rem; }
      h1, h2, h3 { color: #082d69 !important; letter-spacing: -.035em; }
      .brand-kicker { color:#3685c7; font-weight:700; letter-spacing:.12em; font-size:.73rem; text-transform:uppercase; }
      .subtle { color:#66768a; font-size:.92rem; }
      .section-label { color:#082d69; font-weight:800; font-size:1rem; padding-bottom:.35rem; border-bottom:2px solid #d9e5f1; margin-bottom:.75rem; }
      .compound-card { background:#fff; border:1px solid #dce6f0; border-radius:14px; padding:1rem; min-height:158px; }
      .mode-tag { display:inline-block; border-radius:20px; padding:.25rem .60rem; font-size:.74rem; font-weight:700; background:#e8f3fb; color:#176a9d; }
      .method-note { background:#edf7f3; border-left:3px solid #2ba497; border-radius:4px; padding:.72rem .85rem; color:#405a56; font-size:.87rem; line-height:1.5; }
      .no-match { background:#fff8ed; border-left:3px solid #eea73c; border-radius:4px; padding:.72rem .85rem; color:#6d5429; font-size:.87rem; line-height:1.5; }
      [data-testid="stMetric"] { background:#fff; border:1px solid #dae5f0; border-radius:12px; padding:.55rem .8rem; }
      [data-testid="stMetricLabel"] { color:#647488; font-size:.73rem; }
      [data-testid="stMetricValue"] { color:#082d69; font-weight:750; font-size:1.22rem; }
      .stButton > button[kind="primary"] { background:#082d69; border-color:#082d69; border-radius:8px; font-weight:700; }
      div[data-testid="stExpander"] { background:#fff; border:1px solid #dce6f0; border-radius:12px; }
      .footer-note { color:#8190a1; font-size:.76rem; margin-top:.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def lookup_compound(name: str) -> Compound:
    return pubchem_lookup(name)


def init_state() -> None:
    st.session_state.setdefault("compound", None)
    st.session_state.setdefault("selected_profile", None)


def number(value: float, digits: int = 3) -> str:
    return f"{float(value):,.{digits}f}"


def profile_label(row: pd.Series) -> str:
    return f"{row['drug']} · {row['route']} · {float(row['dose_ugkg']):,.0f} µg/kg · n={int(row['n_points'])}"


def concentration_figure(payload: dict, log_scale: bool) -> go.Figure:
    points, curve = payload["points"], payload["curve"]
    observed = np.clip(points["obs"].to_numpy(float), 1e-8, None)
    predicted = np.clip(curve["pred"].to_numpy(float), 1e-8, None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["time_h"], y=predicted, mode="lines", name="Predicted (OOF)",
        line=dict(color="#f26c23", width=3),
        hovertemplate="Predicted<br>%{x:.3g} h · %{y:.4g} µg/L<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=points["time_h"], y=observed, mode="markers", name="Observed",
        marker=dict(color="#0b4c6d", size=8, line=dict(color="#ffffff", width=1)),
        hovertemplate="Observed<br>%{x:.3g} h · %{y:.4g} µg/L<extra></extra>",
    ))
    fig.update_layout(
        height=505, margin=dict(l=24, r=18, t=43, b=32),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        title=dict(text="Observed vs scaffold-held-out prediction", font=dict(size=17, color="#082d69"), x=0.02),
        legend=dict(orientation="h", y=1.10, x=.02),
        font=dict(family="Arial, 'Noto Sans KR', sans-serif", color="#263548"),
        hovermode="closest",
    )
    fig.update_xaxes(title="Time (h)", showgrid=True, gridcolor="#edf1f5", zeroline=False)
    fig.update_yaxes(
        title="Concentration (µg/L)", type="log" if log_scale else "linear",
        showgrid=True, gridcolor="#edf1f5", zeroline=False,
    )
    return fig


def structure_card(compound: Compound) -> None:
    left, right = st.columns([.9, 1.5], vertical_alignment="center")
    with left:
        if compound.image_url:
            st.image(compound.image_url, width=100)
        else:
            st.markdown("### SMILES")
    with right:
        st.markdown(f"**{compound.title}**  ")
        source = f"CID {compound.cid}" if compound.cid is not None else "Dataset structure"
        st.caption(f"{source} · {compound.formula} · {compound.molecular_weight:.2f} g/mol")
        st.link_button("PubChem에서 구조 보기 ↗", compound.pubchem_url, use_container_width=True)
    st.caption(f"SMILES  {compound.smiles}")


def download_table(payload: dict) -> bytes:
    points = payload["points"][["time_h", "obs", "pred"]].copy()
    points["record_type"] = "observed_point"
    points = points.rename(columns={"obs": "observed_ug_L", "pred": "predicted_ug_L"})
    curve = payload["curve"][["time_h", "pred"]].copy()
    curve["record_type"] = "dense_predicted_curve"
    curve["observed_ug_L"] = np.nan
    curve = curve.rename(columns={"pred": "predicted_ug_L"})
    return pd.concat([points, curve], ignore_index=True).to_csv(index=False).encode("utf-8")


def main() -> None:
    init_state()
    repository = get_oof_repository()

    st.markdown("<div class='brand-kicker'>Rat pharmacokinetic workbench</div>", unsafe_allow_html=True)
    st.title("PK Curve Studio")
    st.markdown(
        "<div class='subtle'>약물명을 입력해 SMILES 구조를 확인한 뒤, 데이터셋에 일치하는 약물의 scaffold-CV out-of-fold C–t 결과를 확인합니다.</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    controls, display, summary = st.columns([1.05, 1.8, 1.05], gap="large")

    with controls:
        st.markdown("<div class='section-label'>Drug & structure</div>", unsafe_allow_html=True)
        drug_name = st.text_input("약물명", placeholder="예: Acetaminophen")
        if st.button("구조 확인", type="primary", use_container_width=True):
            if not drug_name.strip():
                st.warning("약물명을 입력해 주세요.")
            else:
                dataset_compound = repository.compound_from_dataset_name(drug_name)
                if dataset_compound is not None:
                    st.session_state.compound = dataset_compound
                    st.session_state.selected_profile = None
                    st.info("데이터셋에 기록된 SMILES 구조를 불러왔습니다. 구조 카드를 통해 PubChem에서도 확인할 수 있습니다.")
                else:
                    try:
                        with st.spinner("PubChem 구조를 확인하고 있습니다..."):
                            st.session_state.compound = lookup_compound(drug_name.strip())
                        st.session_state.selected_profile = None
                    except Exception:
                        st.error("구조를 찾지 못했습니다. 약물명 표기 또는 PubChem 등록명을 확인해 주세요.")

        compound = st.session_state.compound
        if compound:
            st.markdown("<div class='compound-card'>", unsafe_allow_html=True)
            structure_card(compound)
            st.markdown("</div>", unsafe_allow_html=True)
            with st.expander("SMILES 직접 수정"):
                edited = st.text_area("SMILES", value=compound.smiles, label_visibility="collapsed", height=70)
                if st.button("수정한 구조로 데이터셋 확인"):
                    try:
                        st.session_state.compound = Compound(
                            cid=compound.cid, title=compound.title, smiles=canonical_smiles(edited),
                            formula=compound.formula, molecular_weight=compound.molecular_weight,
                        )
                        st.session_state.selected_profile = None
                        st.rerun()
                    except ValueError:
                        st.error("유효한 SMILES가 아닙니다.")

            profiles = repository.profiles_for_smiles(compound.smiles)
            st.markdown("<br>", unsafe_allow_html=True)
            if profiles.empty:
                st.markdown(
                    "<div class='no-match'><b>데이터셋 일치 없음</b><br>이 구조는 공개용 OOF 결과 데이터에 없습니다. 이 사이트는 새로운 약물의 자동 예측을 만들지 않고, 사전에 계산된 scaffold-CV 검증 결과만 표시합니다.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<div class='section-label'>Dataset profile</div>", unsafe_allow_html=True)
                profile_ids = profiles["profile_id"].tolist()
                profile_map = profiles.set_index("profile_id")
                default_index = profile_ids.index(st.session_state.selected_profile) if st.session_state.selected_profile in profile_ids else 0
                selected = st.selectbox(
                    "Route · dose · profile", profile_ids, index=default_index,
                    format_func=lambda item: profile_label(profile_map.loc[item]),
                )
                st.session_state.selected_profile = selected
                selected_row = profile_map.loc[selected]
                st.markdown(
                    "<div class='method-note'><b>Scaffold-held-out OOF result</b><br>"
                    f"Fold {int(selected_row['fold'])} 모델은 이 약물과 동일한 scaffold를 학습에 포함하지 않은 상태에서 예측했습니다.</div>",
                    unsafe_allow_html=True,
                )

    payload: dict | None = None
    if compound is not None and st.session_state.selected_profile:
        try:
            payload = repository.profile_payload(st.session_state.selected_profile)
        except (KeyError, ValueError):
            st.session_state.selected_profile = None

    with display:
        st.markdown("<div class='section-label'>Concentration–time profile</div>", unsafe_allow_html=True)
        log_scale = st.toggle("Log-scale", value=True)
        if payload:
            st.plotly_chart(concentration_figure(payload, log_scale), use_container_width=True, config={"displaylogo": False})
            st.download_button(
                "Observed + predicted CSV 다운로드", download_table(payload),
                file_name="pk_curve_studio_oof_profile.csv", mime="text/csv", use_container_width=True,
            )
        else:
            st.info("약물명을 입력해 구조를 확인하면, 데이터셋과 일치하는 OOF 검증 결과가 표시됩니다.")

    with summary:
        st.markdown("<div class='section-label'>Validation summary</div>", unsafe_allow_html=True)
        if payload:
            metrics = payload["metrics"]
            st.markdown(f"<span class='mode-tag'>{metrics['route']} · Fold {int(metrics['fold'])} OOF</span>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("AAFE", number(metrics["AAFE"], 3))
            m2.metric("n", int(metrics["n_points"]))
            m3, m4 = st.columns(2)
            m3.metric("2-fold 이내", f"{float(metrics['within_2fold_pct']):.1f} %")
            m4.metric("3-fold 이내", f"{float(metrics['within_3fold_pct']):.1f} %")
            st.metric("R²-log", f"{float(metrics['R2_log']):.3f}")
            st.markdown("#### Profile conditions")
            st.write(f"**Drug**  {metrics['drug']}")
            st.write(f"**Dose**  {float(metrics['dose_ugkg']):,.0f} µg/kg")
            if metrics["route"] == "PO":
                st.write(f"**F**  {float(metrics['F_fraction']) * 100:.1f} %")
            with st.expander("검증 방법", expanded=False):
                st.write(
                    "표시된 주황색 선은 선택한 profile의 out-of-fold 예측입니다. "
                    "즉, 해당 약물의 scaffold를 포함하지 않은 학습 fold가 만든 결과이며, "
                    "파란 점은 문헌에서 수집한 실제 농도–시간 데이터입니다."
                )
        else:
            st.caption("OOF profile을 선택하면 약물별 AAFE, 2-fold, 3-fold 및 R²-log가 표시됩니다.")

    st.markdown(
        "<div class='footer-note'>Rat-only · final 5-fold scaffold-CV result viewer · observed values are literature-curated concentration points</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
