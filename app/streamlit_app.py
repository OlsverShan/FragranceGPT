"""
FragranceGPT — Perfume Recommendation & Rating Prediction Demo.
Run: streamlit run app/streamlit_app.py
"""
import sys
import os
import json
import hashlib
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.i18n import (t, set_lang, get_lang, LANGUAGES, stars_html_i18n as stars_html,
                      quality_tier_i18n as quality_tier, polarization_i18n,
                      accord_names_bilingual, accord_name, note_name, note_names_translated)
from app.async_runner import (start_persona_ws, start_persona_rest, start_rest_task,
                               start_direct_task, is_running, get_result, get_partials,
                               get_progress, reset_task, poll_loop)
from app.history_manager import save_entry, load_history, clear_history, delete_entry

# ── API key (from project settings) ───────────────────────────
if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "sk-5e871d95b53d4610a967488ec143fae9"

# ── History auto-save dedup ─────────────────────────────────────
_SAVED: set[str] = set()

def _try_save(entry_type, input_summary, result_summary, data):
    key = f"{entry_type}:{json.dumps(data, sort_keys=True, default=str)}"
    if key not in _SAVED:
        _SAVED.add(key)
        save_entry(entry_type, input_summary, result_summary, data, get_lang())

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="FragranceGPT",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Compact UI ─────────────────────────────────────────────────
st.markdown("""
<style>
div.stMain .block-container { padding: 1.5rem 1.5rem 1rem 1.5rem; }
div.stExpander div.stMarkdown { margin-bottom: 0.15rem; }
div.stMetric label { font-size: 0.75rem; }
div.stContainer { padding-top: 0.3rem; padding-bottom: 0.3rem; }
button[data-baseweb="tab"] { padding-left: 0.8rem; padding-right: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Language init ────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# ── Cached resources ─────────────────────────────────────────

@st.cache_resource
def get_vector_store():
    from src.rag import FragranceVectorStore
    store = FragranceVectorStore()
    return store


@st.cache_data
def get_data():
    from utils import load_data, preprocess
    df = load_data()
    df = preprocess(df)
    return df


@st.cache_resource
def get_rating_predictor():
    model_path = "models/rating_predictor"
    if os.path.exists(os.path.join(model_path, "model.json")):
        from src.rating_predictor import RatingPredictor
        return RatingPredictor.load(model_path)
    return None


@st.cache_resource
def get_multi_persona_rater():
    from src.multi_persona_rater import MultiPersonaRater
    return MultiPersonaRater()


@st.cache_data
def get_analytics(df):
    from src.analytics import MarketAnalytics
    return MarketAnalytics(df)


@st.cache_data
def get_all_accords(df):
    accords = set()
    for a_list in df['accords']:
        accords.update(a_list)
    return sorted(accords)


@st.cache_data
def get_all_notes(df):
    notes = set()
    for col in ['Top', 'Middle', 'Base']:
        for val in df[col]:
            if isinstance(val, str):
                for n in val.split(','):
                    n = n.strip()
                    if n and len(n) > 1:
                        notes.add(n)
    return sorted(notes)


# ── Load everything ──────────────────────────────────────────
df = get_data()
store = get_vector_store()
all_accords = get_all_accords(df)
all_notes = get_all_notes(df)
predictor = get_rating_predictor()
multi_rater = get_multi_persona_rater()


# ── Helpers ──────────────────────────────────────────────────

def persona_display_name(p):
    """Return persona name in current language."""
    if get_lang() == "zh":
        return p.get("name_zh", p.get("name", "?"))
    return p.get("name", "?")


def note_badges(notes, highlight=None):
    """Render notes as colored badges with translation. Matching ones get green highlight."""
    html = ""
    for n in notes:
        n_clean = n.strip().lower()
        display_name = note_name(n.strip())
        if highlight and n_clean in [h.lower().strip() for h in highlight]:
            html += f'<span style="background:#4CAF50;color:white;padding:2px 8px;border-radius:12px;margin:2px;display:inline-block;font-size:0.85em">{display_name}</span> '
        else:
            html += f'<span style="background:#E0E0E0;color:#333;padding:2px 8px;border-radius:12px;margin:2px;display:inline-block;font-size:0.85em">{display_name}</span> '
    return html


# ── Header: Title + Language selector ────────────────────────
col_title, col_lang = st.columns([6, 1])
with col_title:
    st.title(t("title"))
    st.caption(t("caption"))
with col_lang:
    st.selectbox(
        t("lang_label"),
        options=list(LANGUAGES.keys()),
        format_func=lambda k: LANGUAGES[k],
        key="lang_selector",
        on_change=lambda: set_lang(st.session_state.lang_selector),
        label_visibility="collapsed",
    )
    # Sync initial value
    if st.session_state.lang_selector != st.session_state.lang:
        st.session_state.lang = st.session_state.lang_selector

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    t("tab_recommend"),
    t("tab_rating"),
    t("tab_analytics"),
    t("tab_history"),
])

# ════════════════════════════════════════════════════════════
# TAB 1: Perfume Recommendation
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader(t("rec_subtitle"))

    # ── Scent Concierge: Sales Expert Agent ──
    sales_expanded = is_running("sales_expert") or get_result("sales_expert") is not None
    with st.expander(f"### {t('sales_title')}", expanded=sales_expanded):
        st.caption(t("sales_subtitle"))
        sc1, sc2 = st.columns(2)
        with sc1:
            user_desc = st.text_area(
                t("sales_desc_label"),
                placeholder=t("sales_desc_ph"),
                height=100,
                key="sales_desc",
            )
        with sc2:
            user_scenario = st.text_input(
                t("sales_scenario_label"),
                placeholder=t("sales_scenario_ph"),
                key="sales_scenario",
            )
            user_target = st.text_input(
                t("sales_target_label"),
                placeholder=t("sales_target_ph"),
                key="sales_target",
            )

        sales_task_key = "sales_expert"
        if st.button(t("sales_btn"), type="primary", key="sales_btn_key",
                      disabled=not user_desc.strip()):
            reset_task(sales_task_key)
            start_rest_task(sales_task_key, "/api/sales-expert", {
                "description": user_desc.strip(),
                "scenario": user_scenario.strip(),
                "target": user_target.strip(),
                "lang": get_lang(),
            })
            st.rerun()

        # Show results when done
        if is_running(sales_task_key):
            st.status(t("sales_analyzing"), expanded=True)
        elif (sales_result := get_result(sales_task_key)) is not None:
            if "error" in sales_result:
                st.error(t("sales_error").format(sales_result["error"]))
            else:
                # ── Phase 1: LLM Analysis ──
                st.markdown(f"#### {t('sales_needs_title')}")
                st.info(sales_result.get("needs_analysis", ""))

                rec_accords = sales_result.get("recommended_accords", [])
                if rec_accords:
                    st.markdown(f"#### {t('sales_accords_title')}")
                    acc_cols = st.columns(min(len(rec_accords), 3))
                    for i, item in enumerate(rec_accords):
                        with acc_cols[i % 3]:
                            acc_en = item.get("accord", "")
                            acc_display = accord_name(acc_en) if get_lang() == "zh" else acc_en
                            st.markdown(f"**{acc_display}**")
                            st.caption(item.get("reason", ""))

                blend = sales_result.get("ai_blend_direction", "")
                if blend:
                    st.markdown(f"#### {t('sales_blend_title')}")
                    st.markdown(f"> {blend}")

                # Auto-save: sales expert analysis
                _acc_names = [a.get("accord","") for a in rec_accords if a.get("accord")]
                _try_save("sales_expert",
                    f"{user_desc[:80]}{' | ' + user_scenario[:40] if user_scenario else ''}",
                    f"Accords: {', '.join(_acc_names)} | {blend[:60]}",
                    {"description": user_desc[:200], "scenario": user_scenario or "",
                     "target": user_target or "",
                     "needs_analysis": sales_result.get("needs_analysis", "")[:300],
                     "recommended_accords": _acc_names,
                     "ai_blend": blend[:200]})

                # ── Phase 2: Data Pipeline Results (RAG + DB recommendations) ──
                data_pipe = sales_result.get("data_pipeline") or {}
                has_predicted = data_pipe.get("predicted_top_notes") or data_pipe.get("predicted_mid_notes") or data_pipe.get("predicted_base_notes")
                has_db_results = bool(data_pipe.get("recommendations"))

                # Initialize vars that may be set inside has_predicted block
                pred_top, pred_mid, pred_base = [], [], []
                accords_list = []

                if has_predicted or has_db_results:
                    st.markdown("---")
                    st.subheader(t("rec_predicted_formula"))

                if has_predicted:
                    # Predicted notes from RAG
                    pred_top = data_pipe.get("predicted_top_notes", [])
                    pred_mid = data_pipe.get("predicted_mid_notes", [])
                    pred_base = data_pipe.get("predicted_base_notes", [])
                    col_t, col_m, col_b = st.columns(3)
                    with col_t:
                        st.markdown(f"**{t('rec_top_notes')}**")
                        st.markdown(note_badges(pred_top), unsafe_allow_html=True)
                    with col_m:
                        st.markdown(f"**{t('rec_middle_notes')}**")
                        st.markdown(note_badges(pred_mid), unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f"**{t('rec_base_notes')}**")
                        st.markdown(note_badges(pred_base), unsafe_allow_html=True)

                    # XGBoost rating
                    accords_list = [item.get("accord", "") for item in rec_accords if item.get("accord")]
                    if predictor is not None and accords_list:
                        try:
                            pred_rating = predictor.predict(
                                pred_top, pred_mid, pred_base, accords=accords_list
                            )
                            st.caption(f"{t('rec_xgb_rating')}: {stars_html(pred_rating)}",
                                       unsafe_allow_html=True)
                        except Exception:
                            pass

                    # 8-Persona expert panel (quiet mode — REST, aggregate only)
                    with st.expander(t("rec_expert_panel"), expanded=False):
                        task_key_sales = "persona_sales"
                        if st.button(t("rec_rate_experts"), key="rate_sales"):
                            reset_task(task_key_sales)
                            start_persona_rest(task_key_sales, accords_list,
                                              pred_top, pred_mid, pred_base, lang=get_lang())
                            st.rerun()

                        if is_running(task_key_sales):
                            st.status(t("rec_evaluating"), expanded=True)
                        elif (persona_result := get_result(task_key_sales)) is not None:
                            if "error" in persona_result:
                                st.error(f"{t('error')}: {persona_result['error']}")
                            elif persona_result.get("overall_weighted"):
                                st.metric(t("rec_weighted_score"),
                                          f"{persona_result['overall_weighted']:.2f} / 5")
                                pol_label = polarization_i18n(persona_result.get('polarization', ''))
                                st.caption(f"{t('rec_polarization')}: {pol_label} "
                                           f"({t('rec_range')} {persona_result.get('score_range', 0):.2f})")
                                st.caption(f"Best: {persona_result.get('best','')}  |  "
                                           f"Worst: {persona_result.get('worst','')}")

                                # Show individual persona cards
                                partials = persona_result.get("personas", {})
                                persona_items = list(partials.items())
                                if persona_items:
                                    for row_start in range(0, len(persona_items), 3):
                                        cols = st.columns(min(3, len(persona_items) - row_start))
                                        for col_idx, (key, p) in enumerate(persona_items[row_start:row_start+3]):
                                            with cols[col_idx]:
                                                st.markdown(f"**{persona_display_name(p)}**")
                                                if 'error' not in p:
                                                    st.markdown(stars_html(p['overall']), unsafe_allow_html=True)
                                                    if p.get('comment'):
                                                        st.caption(f"_{p['comment']}_")
                                                else:
                                                    st.caption(f"{t('error')}: {p['error']}")

                                # Auto-save: persona rating in sales expert
                                _try_save("persona_rating",
                                          f"Sales Expert: {user_desc[:60]}",
                                          f"Weighted {persona_result.get('overall_weighted',0):.2f}/5 | {persona_result.get('polarization','')}",
                                          {"source": "sales_expert", "accords": accords_list,
                                           "top_notes": pred_top, "mid_notes": pred_mid, "base_notes": pred_base,
                                           "overall_weighted": persona_result.get("overall_weighted"),
                                           "polarization": persona_result.get("polarization"),
                                           "score_range": persona_result.get("score_range")})

                    # ── Top 5 from Database (interactive cards) ──
                    st.markdown("---")
                    st.subheader(f"🏆 {t('rec_top5_title')} (DB)")

                    db_recs = data_pipe["recommendations"]
                    for i, rec in enumerate(db_recs):
                        name = rec.get('name', '').replace('-', ' ').title()
                        brand = rec.get('brand', '').replace('-', ' ').title()
                        with st.container():
                            c1, c2, c3, c4, c5 = st.columns([1.5, 1.0, 1.0, 0.8, 0.6])
                            with c1:
                                st.markdown(f"**{i+1}. {name}**  —  *{brand}*")
                            with c2:
                                st.markdown(stars_html(rec.get('rating_value', 0)), unsafe_allow_html=True)
                            with c3:
                                st.metric(t("rec_match_score"), f"{rec.get('composite_score', 0):.2f}",
                                          help=t("rec_match_help"))
                            with c4:
                                st.metric(t("rec_note_overlap"), f"{rec.get('content_overlap', 0):.0%}")
                            with c5:
                                buy_key = f"buy_sales_{hashlib.md5(rec['name'].encode()).hexdigest()[:8]}"
                                st.markdown("<br>", unsafe_allow_html=True)
                                buy_btn = st.button(t("purchase_btn"), key=buy_key,
                                                   type="secondary", use_container_width=True)

                            # Notes by layer
                            tc, mc, bc = st.columns(3)
                            matching = rec.get('matching_accords', [])
                            with tc:
                                top_list = [x.strip() for x in rec.get('top_notes', '').split(',') if x.strip()]
                                st.caption(f"**{t('rec_top_notes')}**")
                                st.markdown(note_badges(top_list, matching), unsafe_allow_html=True)
                            with mc:
                                mid_list = [x.strip() for x in rec.get('middle_notes', '').split(',') if x.strip()]
                                st.caption(f"**{t('rec_middle_notes')}**")
                                st.markdown(note_badges(mid_list, matching), unsafe_allow_html=True)
                            with bc:
                                base_list = [x.strip() for x in rec.get('base_notes', '').split(',') if x.strip()]
                                st.caption(f"**{t('rec_base_notes')}**")
                                st.markdown(note_badges(base_list, matching), unsafe_allow_html=True)

                            # Purchase links (async)
                            purchase_task_key = f"purchase_sales_{hashlib.md5(rec['name'].encode()).hexdigest()[:8]}"
                            if buy_btn:
                                reset_task(purchase_task_key)
                                start_direct_task(purchase_task_key, "/api/purchase/direct",
                                               {"name": rec['name'].replace('-', ' '), "brand": brand})
                                st.rerun()

                            if is_running(purchase_task_key):
                                st.caption(f"🔍 {t('purchase_loading')}")
                            elif (purchase_result := get_result(purchase_task_key)) is not None:
                                if "error" in purchase_result:
                                    st.error(t("purchase_error").format(purchase_result["error"]))
                                elif purchase_result.get("retailers") or purchase_result.get("similar_perfumes"):
                                    retailers = purchase_result.get("retailers", [])
                                    similar = purchase_result.get("similar_perfumes", [])
                                    with st.expander(
                                        t("purchase_title").format(f"{name} by {brand}"),
                                        expanded=True,
                                    ):
                                        price_tier = purchase_result.get("price_tier", "?").upper()
                                        price_est = purchase_result.get("price_estimate", "?")
                                        st.caption(f"💰 {t('purchase_price')}: **{price_tier}** — {price_est}")
                                        if retailers:
                                            st.markdown(f"**{t('purchase_retailer')}:**")
                                            for r in retailers:
                                                rtype = r.get("type", "")
                                                rname = r.get("name", "?")
                                                rurl = r.get("url", "#")
                                                rnotes = r.get("notes", "")
                                                st.markdown(
                                                    f"- [{rname}]({rurl}) `({rtype})` — {rnotes}"
                                                )
                                        if similar:
                                            st.markdown(f"**{t('purchase_similar')}:**")
                                            for s in similar:
                                                sname = s.get("name", "?")
                                                sbrand = s.get("brand", "?")
                                                swhy = s.get("why_similar", "")
                                                st.caption(f"- **{sname}** ({sbrand}): _{swhy}_")
                                        st.caption(t("purchase_disclaimer"))
                                    # Auto-save: purchase search
                                    _try_save("purchase_search",
                                        f"{name} by {brand}",
                                        f"{purchase_result.get('price_tier','?')} — {len(retailers)} retailers",
                                        {"perfume_name": name, "brand": brand,
                                         "price_tier": purchase_result.get("price_tier",""),
                                         "retailer_count": len(retailers),
                                         "similar_count": len(similar)})
                            st.markdown("---")

                # Perfume recommendations come from the data pipeline (RAG + DB) in Phase 2.
                # The LLM now focuses solely on needs analysis + accord recommendation.

    # ── Accord Selection ──

    col1, col2 = st.columns([2, 1])
    with col1:
        selected = st.multiselect(
            t("rec_pick_accords"),
            options=all_accords,
            format_func=lambda a: accord_names_bilingual([a])[0],
            max_selections=5,
            placeholder=t("rec_placeholder"),
            help=t("rec_help"),
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        go_btn = st.button(t("rec_button"), type="primary", use_container_width=True)

    if go_btn and len(selected) >= 1:
        with st.spinner(t("loading")):

            # Step 1: Predict notes via RAG + LLM
            from src.recommender import PerfumeRecommender
            recommender = PerfumeRecommender(store, df)

            predicted_notes = recommender.predict_notes(selected)
            if predicted_notes is None:
                st.warning(t("rec_no_api"))
                top_notes, mid_notes, base_notes = [], [], []
            else:
                top_notes, mid_notes, base_notes = predicted_notes

            # Step 2: Get recommendations (with predicted notes for content scoring)
            all_pred_notes = []
            if predicted_notes:
                all_pred_notes = top_notes + mid_notes + base_notes
            recommendations = recommender.recommend(selected, predicted_notes=predicted_notes, top_k=5)

            # Persist in session_state so purchase buttons survive reruns
            st.session_state.recommendations = recommendations
            st.session_state.predicted_notes = predicted_notes
            st.session_state.selected_accords = selected

    elif go_btn:
        st.warning(t("rec_no_selection"))

    # ── Display results from session_state (survives button-click reruns) ──
    if st.session_state.get("recommendations"):
        recommendations = st.session_state.recommendations
        predicted_notes = st.session_state.predicted_notes
        selected = st.session_state.selected_accords

        if predicted_notes:
            top_notes, mid_notes, base_notes = predicted_notes
            st.markdown("---")
            st.subheader(t("rec_predicted_formula"))
            col_t, col_m, col_b = st.columns(3)
            with col_t:
                st.markdown(f"**{t('rec_top_notes')}**")
                st.markdown(note_badges(top_notes), unsafe_allow_html=True)
            with col_m:
                st.markdown(f"**{t('rec_middle_notes')}**")
                st.markdown(note_badges(mid_notes), unsafe_allow_html=True)
            with col_b:
                st.markdown(f"**{t('rec_base_notes')}**")
                st.markdown(note_badges(base_notes), unsafe_allow_html=True)

            # Quick rating + multi-persona expert review
            pred_rating = None
            if predictor is not None:
                try:
                    pred_rating = predictor.predict(
                        top_notes, mid_notes, base_notes,
                        accords=selected,
                    )
                    st.caption(f"{t('rec_xgb_rating')}: {stars_html(pred_rating)}",
                               unsafe_allow_html=True)
                except Exception:
                    pass

            # Auto-save: accord selection
            xgb_val = float(pred_rating) if pred_rating is not None else None
            _try_save("accord_selection",
                f"Accords: {', '.join(selected)}",
                f"Predicted notes | XGBoost: {xgb_val:.2f}" if xgb_val is not None else "Predicted notes",
                {"accords": selected,
                 "predicted_top": top_notes, "predicted_mid": mid_notes, "predicted_base": base_notes,
                 "xgb_rating": xgb_val})

            with st.expander(t("rec_expert_panel"), expanded=False):
                task_key = "persona_tab1"
                if st.button(t("rec_rate_experts"), key="rate_tab1"):
                    reset_task(task_key)
                    start_persona_ws(task_key, selected, top_notes, mid_notes, base_notes,
                                    lang=get_lang())
                    st.rerun()

                # ── Non-blocking display ──
                if is_running(task_key):
                    proj = get_progress(task_key)
                    status_text = t("rec_evaluating") + f" ({proj.get('completed', 0)}/8)"
                    st.status(status_text, expanded=True)
                    partials = get_partials(task_key)
                    if partials:
                        persona_items = list(partials.items())
                        for row_start in range(0, len(persona_items), 3):
                            cols = st.columns(min(3, len(persona_items) - row_start))
                            for col_idx, (key, p) in enumerate(persona_items[row_start:row_start+3]):
                                with cols[col_idx]:
                                    st.markdown(f"**{persona_display_name(p)}**")
                                    if 'error' not in p:
                                        st.markdown(stars_html(p['overall']), unsafe_allow_html=True)
                                        st.caption(f"_{p.get('comment', '')}_")
                                    else:
                                        st.caption(f"{t('error')}: {p['error']}")
                elif (result := get_result(task_key)) is not None:
                    if "error" in result:
                        st.error(f"{t('error')}: {result['error']}")
                    elif result.get("overall_weighted"):
                        st.metric(t("rec_weighted_score"),
                                  f"{result['overall_weighted']:.2f} / 5")
                        pol_label = polarization_i18n(result.get('polarization', ''))
                        st.caption(f"{t('rec_polarization')}: {pol_label} "
                                   f"({t('rec_range')} {result.get('score_range', 0):.2f})")
                        partials = result.get("personas", get_partials(task_key))
                    else:
                        partials = get_partials(task_key)
                    persona_items = list(partials.items()) if partials else []
                    if persona_items:
                        for row_start in range(0, len(persona_items), 3):
                            cols = st.columns(min(3, len(persona_items) - row_start))
                            for col_idx, (key, p) in enumerate(persona_items[row_start:row_start+3]):
                                with cols[col_idx]:
                                    st.markdown(f"**{persona_display_name(p)}**")
                                    if 'error' not in p:
                                        st.markdown(stars_html(p['overall']), unsafe_allow_html=True)
                                        st.caption(f"_{p.get('comment', '')}_")
                                    else:
                                        st.caption(f"{t('error')}: {p['error']}")

                # Auto-save: persona rating in accord selection
                if result.get("overall_weighted"):
                    _try_save("persona_rating",
                        f"Accords: {', '.join(selected)}",
                        f"Weighted {result.get('overall_weighted',0):.2f}/5 | {result.get('polarization','')}",
                        {"source": "tab1_accord", "accords": selected,
                         "top_notes": top_notes, "mid_notes": mid_notes, "base_notes": base_notes,
                         "overall_weighted": result.get("overall_weighted"),
                         "polarization": result.get("polarization"),
                         "score_range": result.get("score_range")})

        # ── Recommendations Display ──
        st.markdown("---")
        st.subheader(t("rec_top5_title"))

        for i, rec in enumerate(recommendations):
            name = rec['name'].replace('-', ' ').title()
            brand = rec['brand'].replace('-', ' ').title()
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([1.5, 1.0, 1.0, 0.8, 0.6])
                with c1:
                    st.markdown(f"**{i+1}. {name}**  —  *{brand}*")
                with c2:
                    st.markdown(stars_html(rec['rating_value']), unsafe_allow_html=True)
                with c3:
                    st.metric(t("rec_match_score"), f"{rec.get('composite_score', 0):.2f}",
                              help=t("rec_match_help"))
                with c4:
                    st.metric(t("rec_note_overlap"), f"{rec.get('content_overlap', 0):.0%}")
                with c5:
                    buy_key = f"buy_{hashlib.md5(rec['name'].encode()).hexdigest()[:8]}"
                    st.markdown("<br>", unsafe_allow_html=True)
                    buy_btn = st.button(t("purchase_btn"), key=buy_key,
                                       type="secondary", use_container_width=True)

                # Notes by layer
                tc, mc, bc = st.columns(3)
                matching = rec.get('matching_accords', [])
                with tc:
                    top_list = [x.strip() for x in rec['top_notes'].split(',') if x.strip()]
                    st.caption(f"**{t('rec_top_notes')}**")
                    st.markdown(note_badges(top_list, matching), unsafe_allow_html=True)
                with mc:
                    mid_list = [x.strip() for x in rec['middle_notes'].split(',') if x.strip()]
                    st.caption(f"**{t('rec_middle_notes')}**")
                    st.markdown(note_badges(mid_list, matching), unsafe_allow_html=True)
                with bc:
                    base_list = [x.strip() for x in rec['base_notes'].split(',') if x.strip()]
                    st.caption(f"**{t('rec_base_notes')}**")
                    st.markdown(note_badges(base_list, matching), unsafe_allow_html=True)

                # ── Purchase Links (async, non-blocking) ──
                purchase_task_key = f"purchase_{hashlib.md5(rec['name'].encode()).hexdigest()[:8]}"
                if buy_btn:
                    reset_task(purchase_task_key)
                    start_direct_task(purchase_task_key, "/api/purchase/direct",
                                   {"name": rec['name'].replace('-', ' '), "brand": brand})
                    st.rerun()

                if is_running(purchase_task_key):
                    st.caption(f"🔍 {t('purchase_loading')}")
                elif (purchase_result := get_result(purchase_task_key)) is not None:
                    if "error" in purchase_result:
                        st.error(t("purchase_error").format(purchase_result["error"]))
                    elif purchase_result.get("retailers") or purchase_result.get("similar_perfumes"):
                        retailers = purchase_result.get("retailers", [])
                        similar = purchase_result.get("similar_perfumes", [])
                        with st.expander(
                            t("purchase_title").format(f"{name} by {brand}"),
                            expanded=True,
                        ):
                            price_tier = purchase_result.get("price_tier", "?").upper()
                            price_est = purchase_result.get("price_estimate", "?")
                            st.caption(f"💰 {t('purchase_price')}: **{price_tier}** — {price_est}")
                            if retailers:
                                st.markdown(f"**{t('purchase_retailer')}:**")
                                for r in retailers:
                                    rtype = r.get("type", "")
                                    rname = r.get("name", "?")
                                    rurl = r.get("url", "#")
                                    rnotes = r.get("notes", "")
                                    st.markdown(
                                        f"🔗 [{rname}]({rurl}) _{rtype}_ — {rnotes}"
                                    )
                            else:
                                st.caption(t("purchase_no_links"))
                            if similar:
                                st.markdown(f"**{t('purchase_similar')}:**")
                                for sp in similar:
                                    st.markdown(
                                        f"• **{sp.get('name', '?')}** _{sp.get('brand', '?')}_ — "
                                        f"{sp.get('why_similar', '')}"
                                    )
                            st.caption(f"⚠️ {t('purchase_disclaimer')}")
                            # Auto-save: purchase search (accord selection)
                            _try_save("purchase_search",
                                f"{name} by {brand}",
                                f"{purchase_result.get('price_tier','?')} — {len(retailers)} retailers",
                                {"perfume_name": name, "brand": brand,
                                 "price_tier": purchase_result.get("price_tier",""),
                                 "retailer_count": len(retailers),
                                 "similar_count": len(similar)})
                st.markdown("---")

# ════════════════════════════════════════════════════════════
# TAB 2: Rating Predictor (Dual Engine)
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader(t("rate_subtitle"))

    if predictor is None:
        st.info(t("rate_no_model"))

    st.markdown(t("rate_enter_notes"))
    col_t2, col_m2, col_b2 = st.columns(3)
    with col_t2:
        top_pick = st.multiselect(
            t("rate_top_input"), options=all_notes,
            format_func=lambda n: note_names_translated([n])[0],
            max_selections=8, key="top_t2",
            placeholder=t("rec_placeholder"))
        top_input = st.text_input(
            "Or type custom notes", placeholder=t("rate_top_ph"),
            key="top_free")
        top_list = top_pick + [n.strip() for n in top_input.split(',') if n.strip()]
    with col_m2:
        mid_pick = st.multiselect(
            t("rate_mid_input"), options=all_notes,
            format_func=lambda n: note_names_translated([n])[0],
            max_selections=8, key="mid_t2",
            placeholder=t("rec_placeholder"))
        mid_input = st.text_input(
            "Or type custom notes", placeholder=t("rate_mid_ph"),
            key="mid_free")
        mid_list = mid_pick + [n.strip() for n in mid_input.split(',') if n.strip()]
    with col_b2:
        base_pick = st.multiselect(
            t("rate_base_input"), options=all_notes,
            format_func=lambda n: note_names_translated([n])[0],
            max_selections=8, key="base_t2",
            placeholder=t("rec_placeholder"))
        base_input = st.text_input(
            "Or type custom notes", placeholder=t("rate_base_ph"),
            key="base_free")
        base_list = base_pick + [n.strip() for n in base_input.split(',') if n.strip()]

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        xgb_btn = st.button(t("rate_xgb_btn"), type="primary", use_container_width=True)
    with col_btn2:
        llm_btn = st.button(t("rate_llm_btn"), use_container_width=True)
    with col_btn3:
        fuse_btn = st.button(t("fuse_btn"), type="primary", use_container_width=True)

    if not (top_list or mid_list or base_list):
        if xgb_btn or llm_btn or fuse_btn:
            st.warning(t("rate_no_notes"))
    else:
        if xgb_btn and predictor is not None:
            pred = predictor.predict(top_list, mid_list, base_list)
            st.markdown("---")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"### {t('rate_xgb_title')}")
                st.markdown(stars_html(pred), unsafe_allow_html=True)
            with c2:
                tier_name, tier_color, tier_icon = quality_tier(pred)
                st.metric(t("rate_quality_tier"), f"{tier_icon} {tier_name}")
                st.caption(t("rate_xgb_note1"))
                st.caption(t("rate_xgb_note2"))

            # Auto-save: Tab 2 XGBoost
            _try_save("accord_selection",
                f"Notes: {', '.join(top_list[:5])} | {', '.join(mid_list[:3])} | {', '.join(base_list[:3])}",
                f"XGBoost: {pred:.2f}/5 | {quality_tier(pred)[0]}",
                {"accords": [], "predicted_top": top_list[:5], "predicted_mid": mid_list[:5],
                 "predicted_base": base_list[:5], "xgb_rating": float(pred)})

        task_key_t2 = "persona_tab2"
        if llm_btn:
            reset_task(task_key_t2)
            start_persona_ws(task_key_t2, None, top_list, mid_list, base_list,
                            lang=get_lang())
            st.rerun()

        # ── Non-blocking persona display for Tab 2 ──
        if is_running(task_key_t2):
            st.markdown("---")
            proj = get_progress(task_key_t2)
            st.status(t("rec_evaluating") + f" ({proj.get('completed', 0)}/8)", expanded=True)
            partials = get_partials(task_key_t2)
            if partials:
                persona_items = list(partials.items())
                for row_start in range(0, len(persona_items), 3):
                    cols = st.columns(min(3, len(persona_items) - row_start))
                    for col_idx, (key, p) in enumerate(persona_items[row_start:row_start+3]):
                        with cols[col_idx]:
                            st.markdown(f"**{persona_display_name(p)}**")
                            if 'error' not in p:
                                st.markdown(stars_html(p['overall']), unsafe_allow_html=True)
                                st.caption(f"_{p.get('comment', '')}_")
                            else:
                                st.caption(f"{t('error')}: {p['error']}")
        elif (result_t2 := get_result(task_key_t2)) is not None:
            st.markdown("---")
            if "error" in result_t2:
                st.error(f"{t('error')}: {result_t2['error']}")
            elif result_t2.get("overall_weighted"):
                st.markdown(f"### {t('rate_llm_weighted')}: {result_t2['overall_weighted']:.2f} / 5")
                pol_label = polarization_i18n(result_t2.get('polarization', ''))
                st.caption(f"{t('rec_polarization')}: {pol_label} "
                           f"({t('rec_range')} {result_t2.get('score_range', 0):.2f}) — "
                           + t('rate_result_note').format(
                               result_t2.get('best', '?'), result_t2.get('worst', '?')))
                partials = result_t2.get("personas", get_partials(task_key_t2))
            else:
                partials = get_partials(task_key_t2)
            persona_items = list(partials.items()) if partials else []
            if persona_items:
                for row_start in range(0, len(persona_items), 3):
                    cols = st.columns(min(3, len(persona_items) - row_start))
                    for col_idx, (key, p) in enumerate(persona_items[row_start:row_start+3]):
                        with cols[col_idx]:
                            st.markdown(f"**{persona_display_name(p)}**")
                            if 'error' not in p:
                                st.markdown(stars_html(p['overall']), unsafe_allow_html=True)
                                st.caption(f"_{p.get('comment', '')}_")
                            else:
                                st.caption(f"{t('error')}: {p['error']}")

            # Auto-save: persona rating in Tab 2
            if result_t2.get("overall_weighted"):
                _try_save("persona_rating",
                    f"Tab 2: {', '.join(top_list[:5])} | {', '.join(mid_list[:3])}",
                    f"Weighted {result_t2.get('overall_weighted',0):.2f}/5 | {result_t2.get('polarization','')}",
                    {"source": "tab2", "accords": [],
                     "top_notes": top_list[:5], "mid_notes": mid_list[:5], "base_notes": base_list[:5],
                     "overall_weighted": result_t2.get("overall_weighted"),
                     "polarization": result_t2.get("polarization"),
                     "score_range": result_t2.get("score_range")})

        if fuse_btn:
            st.markdown("---")
            from src.fused_rater import FusedRater
            fr = FusedRater(predictor, multi_rater) if predictor else None

            if fr is None:
                st.info(t("fuse_persona_only"))
                with st.spinner(t("rec_evaluating")):
                    result = multi_rater.rate(None, top_list, mid_list, base_list, lang=get_lang())
                if result:
                    st.markdown(f"### 8-Persona: {result['overall_weighted']:.2f} / 5")
            else:
                with st.spinner(t("rec_evaluating")):
                    fused = fr.rate(top_list, mid_list, base_list, lang=get_lang())
                if fused:
                    # ── Header: Fused Score ──
                    st.markdown(f"## {t('fuse_title')}: "
                                f"{stars_html(fused['fused_score'])}",
                                unsafe_allow_html=True)
                    st.caption(fused["reasoning"])

                    st.markdown("---")
                    st.markdown(f"### {t('fuse_breakdown')}")

                    # ── Score cards ──
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric(t("fuse_xgb_base"),
                                  f"{fused['xgb_score']:.2f} / 5")
                    with c2:
                        ps = fused.get("persona_score")
                        st.metric(t("fuse_persona_raw"),
                                  f"{ps:.2f} / 5" if ps else "N/A")
                    with c3:
                        cp = fused.get("calibrated_persona")
                        st.metric(t("fuse_persona_cal"),
                                  f"{cp:.2f} / 5" if cp else "N/A",
                                  help="Persona + 1.31 bias correction")
                    with c4:
                        adj = fused["adjustment"]
                        adj_str = f"{adj:+.2f}" if adj else "0.00"
                        st.metric(t("fuse_adjustment"), adj_str,
                                  delta=f"{adj:+.2f}" if abs(adj) > 0.05 else None)

                    # ── Fusion details ──
                    c_meta1, c_meta2 = st.columns(2)
                    with c_meta1:
                        st.metric(t("fuse_confidence"),
                                  f"{fused['confidence'].upper()}  "
                                  f"(α={fused['alpha_used']:.2f})")
                    with c_meta2:
                        rng = fused.get("score_range", "?")
                        st.metric(t("fuse_agreement"),
                                  f"{fused.get('confidence_label', '')}  "
                                  f"(range: {rng})")

                    # ── Persona panel (collapsed) ──
                    with st.expander(t("rec_expert_panel"), expanded=False):
                        p_details = fused.get("persona_details", {})
                        if p_details:
                            persona_items = list(p_details.items())
                            for row_start in range(0, len(persona_items), 3):
                                cols = st.columns(min(3, len(persona_items) - row_start))
                                for col_idx, (key, p) in enumerate(persona_items[row_start:row_start+3]):
                                    with cols[col_idx]:
                                        st.markdown(f"**{persona_display_name(p) if p.get('name') else key}**")
                                        if 'error' not in p:
                                            st.markdown(stars_html(p['overall']),
                                                        unsafe_allow_html=True)
                                            st.caption(f"_{p.get('comment', '')}_")
                                        else:
                                            st.caption(f"{t('error')}: {p['error']}")

                        # Auto-save: fused rating
                        _try_save("fused_rating",
                            f"Notes: {', '.join(top_list[:5])} | {', '.join(mid_list[:3])} | {', '.join(base_list[:3])}",
                            f"Fused {fused.get('fused_score',0):.2f}/5 | {fused.get('confidence','')}",
                            {"top_notes": top_list[:5], "mid_notes": mid_list[:5], "base_notes": base_list[:5],
                             "fused_score": fused.get("fused_score"), "xgb_score": fused.get("xgb_score"),
                             "persona_score": fused.get("persona_score"),
                             "confidence": fused.get("confidence"), "agreement": fused.get("confidence_label")})

# ════════════════════════════════════════════════════════════
# TAB 3: Market Analytics
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader(t("analytics_subtitle"))
    analytics = get_analytics(df)

    tab3a, tab3b, tab3c = st.tabs([
        t("analytics_accord_rank"),
        t("analytics_brand_rank"),
        t("analytics_rating_dist"),
    ])

    with tab3a:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(f"#### {t('ana_top_combos')}")
            combos = analytics.top_accord_combos(n=10)
            if combos:
                fig = px.bar(
                    combos, x='avg_score', y='combo', orientation='h',
                    title=t("ana_combo_chart"),
                    labels={'avg_score': t('ana_avg_score'), 'combo': ''},
                    color='avg_score', color_continuous_scale='Viridis',
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown(f"#### {t('ana_accord_corr')}")
            corr = analytics.accord_correlation(n=15)
            if corr:
                fig = px.bar(
                    corr,
                    x='vs_global', y='accord', orientation='h',
                    title=t("ana_corr_chart"),
                    labels={'vs_global': t('ana_deviation'), 'accord': ''},
                    color='vs_global',
                    color_continuous_scale=['red', 'lightgrey', 'green'],
                    color_continuous_midpoint=0,
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"#### {t('ana_common_notes')}")
        notes = analytics.most_common_notes_high_rated(n=20)
        if notes:
            fig = px.bar(
                notes, x='pct', y='note', orientation='h',
                title=t("ana_note_freq"),
                labels={'pct': t('ana_pct'), 'note': ''},
                color='pct', color_continuous_scale='Blues',
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450)
            st.plotly_chart(fig, use_container_width=True)

    with tab3b:
        st.markdown(f"#### {t('ana_top_brands')}")
        brands = analytics.brand_rankings(n=15)
        if brands:
            df_brands = pd.DataFrame(brands)
            df_brands.index = range(1, len(df_brands) + 1)
            st.dataframe(
                df_brands.rename(columns={
                    'brand': t('ana_brand_col'),
                    'avg_score': t('ana_avg_score_col'),
                    'count': t('ana_perfumes_col'),
                }),
                use_container_width=True,
                column_config={
                    t('ana_avg_score_col'): st.column_config.NumberColumn(format="%.3f"),
                },
            )

    with tab3c:
        st.markdown(f"#### {t('ana_rating_dist_chart')}")
        labels, counts = analytics.rating_distribution()
        fig = px.bar(
            x=labels, y=counts,
            title=t("ana_rating_dist_chart"),
            labels={'x': t('ana_rating_range_x'), 'y': t('ana_count_y')},
            color=counts, color_continuous_scale='Oranges',
        )
        st.plotly_chart(fig, use_container_width=True)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric(t("ana_total_perfumes"), f"{len(df):,}")
            st.metric(t("ana_high_conf"), f"{len(analytics.df_high):,}")
        with col_s2:
            st.metric(t("ana_global_mean"), f"{analytics.df_high['bayesian_score'].mean():.3f}")
            st.metric(t("ana_rating_range"),
                      f"{analytics.df_high['Rating Value'].min():.2f} – "
                      f"{analytics.df_high['Rating Value'].max():.2f}")

# ═══════════════════════════════════════════════════════════════
# TAB 4: History
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.subheader(t("history_title"))

    history_entries = load_history()
    if not history_entries:
        st.info(t("history_empty"))
    else:
        # Filter by type
        all_types = sorted(set(e.get("type", "?") for e in history_entries))
        type_labels = {
            "sales_expert": t("history_type_sales"),
            "accord_selection": t("history_type_accord"),
            "persona_rating": t("history_type_persona"),
            "fused_rating": t("history_type_fused"),
            "purchase_search": t("history_type_purchase"),
        }
        type_options = [type_labels.get(tp, tp) for tp in all_types]
        selected_labels = st.multiselect(
            t("history_filter"), type_options,
            placeholder=t("history_filter"),
            key="history_filter",
        )
        selected_types = [all_types[type_options.index(lbl)] for lbl in selected_labels] if selected_labels else []

        filtered = history_entries
        if selected_types:
            filtered = [e for e in history_entries if e.get("type") in selected_types]

        # Clear all button
        ctl_col, _ = st.columns([1, 3])
        with ctl_col:
            if st.button(t("history_clear"), key="clear_history_btn", type="secondary"):
                clear_history()
                st.rerun()

        st.caption(f"{len(filtered)} entries")

        for entry in filtered:
            eid = entry.get("id", "")
            etype = entry.get("type", "?")
            ts = entry.get("timestamp", "")[:16].replace("T", " ")
            inp = entry.get("input_summary", "")
            res = entry.get("result_summary", "")
            etype_label = type_labels.get(etype, etype)

            with st.expander(f"**{etype_label}** — {ts} — {inp[:60]}", expanded=False, key=f"hist_exp_{eid}"):
                st.caption(f"**{t('history_input')}:** {inp}")
                st.caption(f"**{t('history_result')}:** {res}")
                c1, c2, c3 = st.columns([1, 1, 3])
                with c1:
                    if st.button(t("history_rerun"), key=f"rerun_{eid}"):
                        edata = entry.get("data", {})
                        if etype == "sales_expert":
                            st.session_state.sales_desc = edata.get("description", "")
                            st.session_state.sales_scenario = edata.get("scenario", "")
                            st.session_state.sales_target = edata.get("target", "")
                        elif etype in ("accord_selection", "persona_rating"):
                            if edata.get("accords"):
                                st.session_state[f"accords_{get_lang()}"] = edata["accords"]
                            if edata.get("top_notes"):
                                st.session_state.top_free = ", ".join(edata["top_notes"][:5])
                            if edata.get("mid_notes"):
                                st.session_state.mid_free = ", ".join(edata["mid_notes"][:5])
                            if edata.get("base_notes"):
                                st.session_state.base_free = ", ".join(edata["base_notes"][:5])
                        st.rerun()
                with c2:
                    if st.button(t("history_delete"), key=f"del_{eid}"):
                        delete_entry(eid)
                        st.rerun()

# ═══════════════════════════════════════════════════════════════
# Auto-poll: rerun page while any async task is running
# ═══════════════════════════════════════════════════════════════
poll_loop()
