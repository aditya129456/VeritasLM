"""
Sidebar Component: 6-Item Navigation System & Developer Telemetry.
The sidebar strictly contains exactly these six items in this exact order:
1. Detector
2. Benchmark Mode
3. System Health
4. History
5. About
6. Profile (at the very bottom, visually separated)
"""

from typing import List, Tuple
import streamlit as st

from app.config import (
    DEFAULT_DETECTOR_MODEL_TYPE,
    DETECTOR_MODEL_TYPES,
    MODEL_OPTIONS,
    get_detector_mode_specs,
)
from src.utils.model_registry import has_intermediate_models

NAV_ITEMS: List[Tuple[str, str, str]] = [
    ("Detector", "detector", ":material/radar:"),
    ("Benchmark Mode", "benchmark_mode", ":material/leaderboard:"),
    ("System Health", "system_health", ":material/monitor_heart:"),
    ("History", "history", ":material/history:"),
    ("About", "about", ":material/info:"),
    ("Profile", "profile", ":material/account_circle:"),
]


def render_sidebar():
    """
    Render reconstructed 6-item navigation sidebar with clean typography,
    consistent icons, and bottom-anchored profile navigation.
    """
    if "active_nav" not in st.session_state:
        st.session_state["active_nav"] = "Detector"

    current_nav = st.session_state["active_nav"]

    with st.sidebar:
        # Brand Identity Header with Live Status Badge (Clicking toggles navigation bar)
        st.markdown(
            """
            <div class="sidebar-brand-header" id="sidebarBrandHeader" title="Click logo to toggle navigation bar">
                <div class="sidebar-brand-logo"></div>
                <div class="sidebar-brand-text">
                    <div class="sidebar-brand-title">VeritasLM</div>
                    <div class="sidebar-brand-subtitle">HALLUCINATION DEFENSE</div>
                </div>
                <div class="sidebar-status-pill">
                    <span class="sidebar-pulse-dot"></span> LIVE
                </div>
            </div>
            <div class="sidebar-section-label">NAVIGATION</div>
            """,
            unsafe_allow_html=True,
        )

        # Micro-client script: Bind click on brand header to trigger collapse/expand smoothly
        if hasattr(st, "html"):
            st.html(
                """
                <script>
                (function() {
                    try {
                        function setupBrandToggle() {
                            const brandHeader = document.getElementById('sidebarBrandHeader');
                            if (brandHeader && !brandHeader.dataset.collapseBound) {
                                brandHeader.dataset.collapseBound = 'true';
                                brandHeader.addEventListener('click', function(e) {
                                    e.preventDefault();
                                    const collapseBtn = document.querySelector('[data-testid="stSidebarCollapseButton"] button') ||
                                                        document.querySelector('[data-testid="stSidebarCollapseButton"]');
                                    if (collapseBtn) {
                                        collapseBtn.click();
                                    }
                                });
                            }
                        }
                        setupBrandToggle();
                        if (!window._veritaslmSidebarObserver) {
                            window._veritaslmSidebarObserver = new MutationObserver(setupBrandToggle);
                            window._veritaslmSidebarObserver.observe(document.body, { childList: true, subtree: true });
                        }
                    } catch (e) {}
                })();
                </script>
                """,
                unsafe_allow_javascript=True,
            )

        # Main Navigation Group (Detector, Benchmark Mode, System Health, History, About)
        for label, slug, icon in NAV_ITEMS[:-1]:
            is_active = (current_nav == label)
            if st.button(
                label,
                key=f"sidebar_nav_{slug}",
                icon=icon,
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if st.session_state.get("active_nav") != label:
                    st.session_state["active_nav"] = label
                    st.rerun()

        # Detector Model Type Selector (Below option menu, above Account section)
        if "detector_model_type" not in st.session_state:
            st.session_state["detector_model_type"] = DEFAULT_DETECTOR_MODEL_TYPE

        active_detector_type = st.session_state["detector_model_type"]
        selected_model_name = st.session_state.get("selected_model_name", "Qwen 3.5 (0.8B)")
        active_model_id = MODEL_OPTIONS.get(selected_model_name, "Qwen/Qwen3.5-0.8B")
        intermediate_ready = has_intermediate_models(active_model_id)

        st.markdown(
            """
            <div class="sidebar-detector-section">
                <div class="sidebar-section-label" style="margin-bottom: 4px;">DETECTOR MODEL</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for mode_item in DETECTOR_MODEL_TYPES:
            mode_slug = mode_item["slug"]
            is_mode_active = (active_detector_type == mode_slug)
            is_disabled = (mode_slug == "intermediate" and not intermediate_ready)
            if st.button(
                mode_item["label"],
                key=f"sidebar_detector_{mode_slug}",
                icon=mode_item["icon"],
                use_container_width=True,
                type="primary" if is_mode_active else "secondary",
                disabled=is_disabled,
                help=(
                    "Intermediate probing artifacts are not available for the selected LLM."
                    if is_disabled
                    else mode_item["help"]
                ),
            ):
                if st.session_state.get("detector_model_type") != mode_slug:
                    st.session_state["detector_model_type"] = mode_slug
                    st.session_state["inference_result"] = None
                    st.session_state["bm_results"] = None
                    st.rerun()

        active_mode_specs = get_detector_mode_specs(active_detector_type)
        st.markdown(
            f"""
            <div class="sidebar-detector-meta">
                {active_mode_specs["meta"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Visual Separator & Flexible Spacer to push Appearance and Profile to the bottom
        st.markdown(
            """
            <div class="sidebar-nav-spacer"></div>
            """,
            unsafe_allow_html=True,
        )



        # Profile / Operator Section
        st.markdown(
            """
            <div class="sidebar-profile-section">
                <div class="sidebar-section-label" style="margin-bottom: 3px;">ACCOUNT</div>
                <div class="sidebar-operator-card">
                    <div class="sidebar-operator-avatar">OP</div>
                    <div class="sidebar-operator-info">
                        <div class="sidebar-operator-name">Operator</div>
                        <div class="sidebar-operator-role">Neural Defense Lab</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Profile Navigation Item (At the bottom, visually anchored)
        label, slug, icon = NAV_ITEMS[-1]
        is_active = (current_nav == label)
        if st.button(
            label,
            key=f"sidebar_nav_{slug}",
            icon=icon,
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            if st.session_state.get("active_nav") != label:
                st.session_state["active_nav"] = label
                st.rerun()

        # Version stamp
        st.markdown(
            """
            <div class="sidebar-version-badge">
                VeritasLM · v2.4.0 Applied ML
            </div>
            """,
            unsafe_allow_html=True,
        )
