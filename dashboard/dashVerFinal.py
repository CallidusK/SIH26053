"""
Adaptive Variable-Resolution 2.5D LiDAR Mapping — Dashboard (Dept 4)

Layout: a landing page gates entry into the dashboard. Inside the dashboard,
the left rail shows small preview cards (one per view, Task-Manager style).
Clicking a card swaps the big center panel to that view's full detail.
With nothing selected, the center panel shows the combined adaptive grid
overview instead.

Everything is sized to fit inside one viewport — no page scrolling. If your
browser window is unusually short, the sidebar can still scroll internally,
but the main panel never should.

Runs entirely on MOCK data (see mock_data.py) shaped exactly like what
Depts 2/3 will hand off: dict[(ring_id, row, col) -> Cell]. Swap
`load_grid()` for the real thing later — nothing else needs to change.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st

from Sihresources00.dashboard.mockData import mock_grid, mock_uniform_grid, SEMANTIC_CLASS_NAMES, RES_TABLE

st.set_page_config(page_title="Adaptive 2.5D LiDAR Map", layout="wide")

#the css
st.markdown(
    """
    <style>
        :root {
            --ink: #e8eee9;
            --muted: #9aa9a2;
            --canvas: #111817;
            --panel: #182321;
            --panel-raised: #20302b;
            --line: #33453f;
            --copper: #e08f5b;
            --mint: #78c7a5;
        }
        html, body, [data-testid="stAppViewContainer"] {
            overflow: hidden;
            background: var(--canvas);
            color: var(--ink);
        }
        [data-testid="stAppViewContainer"] {
            background-image: linear-gradient(135deg, rgba(120, 199, 165, 0.06), transparent 42%),
                repeating-linear-gradient(90deg, rgba(232, 238, 233, 0.018) 0 1px, transparent 1px 72px);
        }
        header[data-testid="stHeader"] { background: transparent; }
        .block-container {
            padding-top: 0.8rem !important;
            padding-bottom: 0.3rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 0.6rem !important;
            padding-bottom: 0.3rem !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] { padding: 0.15rem !important; }
        section[data-testid="stSidebar"] {
            background: #0d1312;
            border-right: 1px solid var(--line);
        }
        section[data-testid="stSidebar"] .block-container { background: transparent; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.15rem !important;
            border-color: var(--line) !important;
            background: rgba(24, 35, 33, 0.72);
        }
        div[data-testid="stMetric"] {
            padding: 0.35rem 0.55rem !important;
            background: var(--panel);
            border-left: 2px solid var(--copper);
        }
        div[data-testid="stMetricLabel"] { color: var(--muted); }
        div[data-testid="stMetricValue"] { color: var(--ink); }
        h1 { font-size: 1.5rem !important; margin: 0 0 0.1rem 0 !important; color: var(--ink); }
        h2, h3 { color: var(--ink); }
        hr { margin: 0.4rem 0 !important; border-color: var(--line); }
        .stCaption, [data-testid="stCaptionContainer"] { margin: 0 !important; color: var(--muted); }
        div.stButton > button {
            padding: 0.15rem 0.6rem !important;
            border: 1px solid var(--line);
            background: var(--panel);
            color: var(--ink);
        }
        div.stButton > button:hover { border-color: var(--copper); color: var(--copper); }
        div.stButton > button[kind="primary"] {
            background: var(--copper);
            border-color: var(--copper);
            color: #16110e;
        }
        [data-testid="stToggle"] label, [data-testid="stSlider"] label { color: var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)

# constants (defines the views) 
TRAV_COLOR = {"safe": "#78c7a5", "caution": "#e0b15b", "non_drivable": "#df6f62"}
SEMANTIC_COLORS = [
    "#e08f5b", "#78c7a5", "#6fa8dc", "#d88383", "#c5a66a", "#b58bd9",
    "#e6c46a", "#79b8b0", "#9fca72", "#d58dba", "#8ea8c7", "#c3cbc4",
]
PLOTLY_TEMPLATE = "plotly_dark"
VIEWS = ["Semantic", "Elevation", "Traversability", "Resolution", "Performance"]
VIEW_DESC = {
    "Semantic": "A map from above where every little square is colored by what it actually is — road, sidewalk, car, person, tree, wall.",
    "Elevation": "Top-down map where color means height. Dark = low ground, bright = high ground. ",
    "Traversability": "Drivability rating per cell — safe / caution / non_drivable.",
    "Resolution": "Dense fine detail near the center, chunkier squares as you go outward. It's proof the system is smart about where it spends its 'attention'.",
    "Performance": "Runtime stats: FPS, memory, active cells, latency.",
}

# ----------------------------------------------------------------------------
# Landing page — gates entry into the dashboard
# ----------------------------------------------------------------------------
if "entered" not in st.session_state:
    st.session_state.entered = False
if "selected_view" not in st.session_state:
    st.session_state.selected_view = None  # None = overview

# landing page
def render_landing():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<div style='height:12vh'></div>", unsafe_allow_html=True)
        st.title("Adaptive 2.5D LiDAR Mapping")
        st.markdown(
            "A distance-banded, variable-resolution occupancy grid with "
            "fine resolution near the sensor, coarser far away."
        )
        st.markdown(
            " **Semantic**: What is this? &nbsp;&nbsp;&nbsp;\n"
            "\n**Elevation**: How bumpy is it? &nbsp;&nbsp;&nbsp;\n"
            "\n**Traversability**: Can I drive here? \n"
            "\n **Resolution**: Where we are looking; closely vs loosely? &nbsp;&nbsp;&nbsp;\n"
            "\n**Performance**: Shows the performance metrics \n"
        )
        st.caption("Built to cut memory and compute versus a fixed-resolution baseline.")
        if st.button("Enter And View Grids", type="primary", use_container_width=True):
            st.session_state.entered = True
            st.rerun()


if not st.session_state.entered:
    render_landing()
    st.stop()


# Data loading (cached so mock data doesn't regenerate on every click)

@st.cache_data
def load_grid(frame: int = 0):
    return mock_grid(n_cells=1500, seed=frame)


@st.cache_data
def load_uniform_grids(seed: int = 0):
    return {
        "5cm": mock_uniform_grid(res_label="5cm", n_cells=6000, seed=seed),
        "50cm": mock_uniform_grid(res_label="50cm", n_cells=400, seed=seed),
    }


def flatten_grid(g):
    records = []
    for (ring, row, col), cell in g.items():
        records.append(
            {
                "ring": ring,
                "row": row,
                "col": col,
                "resolution": RES_TABLE[ring],
                "x": col * RES_TABLE[ring],
                "y": row * RES_TABLE[ring],
                "elevation_mean": cell.elevation_mean,
                "semantic_class": cell.semantic_class,
                "semantic_name": SEMANTIC_CLASS_NAMES[cell.semantic_class],
                "occupancy_prob": cell.occupancy_prob,
                "dynamic_prob": cell.dynamic_prob,
                "traversability": cell.traversability,
                "point_count": cell.point_count,
            }
        )
    return pd.DataFrame.from_records(records)


frame = st.sidebar.slider("Replay frame", min_value=0, max_value=30, value=0, step=1)
grid = load_grid(frame)
uniform_grids = load_uniform_grids()
df = flatten_grid(grid)


def dense_elevation(df):
    """Sparse dict -> dense 2D array for imshow, shared by mini + full elevation."""
    if df.empty:
        return None
    row_min, row_max = df["row"].min(), df["row"].max()
    col_min, col_max = df["col"].min(), df["col"].max()
    dense = np.full((row_max - row_min + 1, col_max - col_min + 1), np.nan)
    for _, r in df.iterrows():
        dense[int(r["row"] - row_min), int(r["col"] - col_min)] = r["elevation_mean"]
    return dense


# ----------------------------------------------------------------------------
# Mini preview charts — small, axis-free, for the left rail cards
# ----------------------------------------------------------------------------
MINI_H = 68


def mini_layout(fig):
    fig.update_layout(
        height=MINI_H, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_traces(selector=dict(name="Vehicle / Sensor"), marker=dict(size=7))
    return fig


def add_origin_marker(fig):
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode="markers",
        name="Vehicle / Sensor",
        marker=dict(size=13, color="#e08f5b", symbol="diamond", line=dict(color="#f4eadf", width=2)),
        hovertemplate="vehicle / sensor origin (0, 0)<extra></extra>",
    ))
    return fig


def mini_semantic(sample):
    fig = px.scatter(sample, x="col", y="row", color="semantic_name", color_discrete_sequence=SEMANTIC_COLORS)
    add_origin_marker(fig)
    st.plotly_chart(mini_layout(fig), use_container_width=True, config={"displayModeBar": False})
    st.caption(f"{df['semantic_name'].nunique()} classes present")


def mini_elevation(_sample):
    dense = dense_elevation(df)
    fig, ax = plt.subplots(figsize=(2.2, 0.68))
    if dense is not None:
        ax.imshow(dense, cmap="magma", origin="lower")
        row_min, col_min = df["row"].min(), df["col"].min()
        ax.scatter(-col_min, -row_min, s=55, c="#e08f5b", marker="D", edgecolors="#f4eadf", linewidths=1, zorder=3)
    ax.axis("off")
    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(f"{df['elevation_mean'].min():.1f}m to {df['elevation_mean'].max():.1f}m")


def mini_traversability(sample):
    fig = px.scatter(sample, x="col", y="row", color="traversability", color_discrete_map=TRAV_COLOR)
    add_origin_marker(fig)
    st.plotly_chart(mini_layout(fig), use_container_width=True, config={"displayModeBar": False})
    pct_safe = (df["traversability"] == "safe").mean() * 100
    st.caption(f"{pct_safe:.0f}% safe cells")


def mini_resolution(sample):
    fig = px.scatter(sample, x="col", y="row", color=sample["ring"].astype(str), color_discrete_sequence=SEMANTIC_COLORS)
    add_origin_marker(fig)
    st.plotly_chart(mini_layout(fig), use_container_width=True, config={"displayModeBar": False})
    st.caption(f"{df['ring'].nunique()} active rings")


def mini_performance(_sample):
    rng = np.random.default_rng(0)
    history = rng.normal(loc=9.4, scale=0.4, size=20)
    fig = px.line(y=history)
    fig.update_traces(line=dict(color="#78c7a5", width=1.5))
    st.plotly_chart(mini_layout(fig), use_container_width=True, config={"displayModeBar": False})
    st.caption(f"{len(grid):,} active cells")


MINI_RENDERERS = {
    "Semantic": mini_semantic,
    "Elevation": mini_elevation,
    "Traversability": mini_traversability,
    "Resolution": mini_resolution,
    "Performance": mini_performance,
}

# ----------------------------------------------------------------------------
# Full detail charts — shown in the center panel when a view is selected
# All heights are fixed and modest on purpose so the whole page fits in one
# viewport without scrolling. Shrink FULL_H further if it still overflows on
# your screen.
# ----------------------------------------------------------------------------
FULL_H = 430


def render_semantic(df):
    fig = px.scatter(df, x="col", y="row", color="semantic_name", opacity=0.85, color_discrete_sequence=SEMANTIC_COLORS)
    add_origin_marker(fig)
    fig.update_traces(marker=dict(size=6))
    fig.update_traces(selector=dict(name="Vehicle / Sensor"), marker=dict(size=13))
    fig.update_layout(height=FULL_H, legend_title_text="Class", margin=dict(t=10, b=10), template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)


def render_elevation(df):
    dense = dense_elevation(df)
    if dense is None:
        st.info("No cells to display.")
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(dense, cmap="magma", origin="lower")
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    row_min, col_min = df["row"].min(), df["col"].min()
    ax.scatter(-col_min, -row_min, s=110, c="#e08f5b", marker="D", edgecolors="#f4eadf", linewidths=1.5, zorder=3)
    fig.colorbar(im, ax=ax, label="meters").ax.tick_params(colors="#9aa9a2")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_traversability(df):
    fig = px.scatter(
        df, x="col", y="row", color="traversability",
        color_discrete_map=TRAV_COLOR,
        category_orders={"traversability": ["safe", "caution", "non_drivable"]},
        opacity=0.85,
    )
    add_origin_marker(fig)
    fig.update_traces(marker=dict(size=6))
    fig.update_traces(selector=dict(name="Vehicle / Sensor"), marker=dict(size=13))
    fig.update_layout(height=FULL_H, margin=dict(t=10, b=10), template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)


def render_resolution(df):
    fig = px.scatter(df, x="col", y="row", color=df["ring"].astype(str), opacity=0.85, labels={"color": "Ring"})
    add_origin_marker(fig)
    fig.update_traces(marker=dict(size=6))
    fig.update_traces(selector=dict(name="Vehicle / Sensor"), marker=dict(size=13))
    fig.update_layout(height=FULL_H, margin=dict(t=10, b=10), template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)


def render_performance(g):
    n_cells = len(g)
    fps, mem_mb, latency_ms = 9.4, n_cells * 0.0021, 38 + n_cells * 0.004
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("FPS", f"{fps:.1f}")
    col2.metric("Active cells", f"{n_cells:,}")
    col3.metric("Memory", f"{mem_mb:.1f} MB")
    col4.metric("Latency", f"{latency_ms:.0f} ms")
    st.caption("Placeholder mock metrics — wire to real timing/memory measurements once Dept 3 is live.")


def render_overview(df):
    """Render each adaptive cell as a physical quadrilateral with elevation."""
    if df.empty:
        st.info("No cells to display.")
        return

    fig = go.Figure()
    semantic_colors = SEMANTIC_COLORS
    color_by_class = {
        name: semantic_colors[index % len(semantic_colors)]
        for index, name in enumerate(SEMANTIC_CLASS_NAMES)
    }
    outline_x = []
    outline_y = []
    outline_z = []

    for semantic_name, class_df in df.groupby("semantic_name", sort=False):
        x_values = []
        y_values = []
        z_values = []
        i_values = []
        j_values = []
        k_values = []
        customdata = []

        for _, cell in class_df.iterrows():
            x0, y0 = cell["x"], cell["y"]
            size, z = cell["resolution"], cell["elevation_mean"]
            start = len(x_values)
            x_values.extend([x0, x0 + size, x0 + size, x0])
            y_values.extend([y0, y0, y0 + size, y0 + size])
            z_values.extend([z, z, z, z])
            outline_x.extend([x0, x0 + size, x0 + size, x0, x0, None])
            outline_y.extend([y0, y0, y0 + size, y0 + size, y0, None])
            outline_z.extend([z + 0.004] * 5 + [None])
            i_values.extend([start, start])
            j_values.extend([start + 1, start + 2])
            k_values.extend([start + 2, start + 3])
            customdata.extend([[cell["ring"], size, z]] * 4)

        face_customdata = [customdata[index] for index in i_values]
        fig.add_trace(go.Mesh3d(
            x=x_values,
            y=y_values,
            z=z_values,
            i=i_values,
            j=j_values,
            k=k_values,
            name=semantic_name,
            color=color_by_class[semantic_name],
            opacity=0.82,
            flatshading=True,
            hovertemplate=(
                "class=%{fullData.name}<br>ring=%{customdata[0]}<br>"
                "resolution=%{customdata[1]:.2f} m<br>elevation=%{customdata[2]:.2f} m"
                "<extra></extra>"
            ),
            customdata=face_customdata,
            showscale=False,
        ))

    fig.add_trace(go.Scatter3d(
        x=outline_x,
        y=outline_y,
        z=outline_z,
        mode="lines",
        name="Cell boundaries",
        line=dict(color="rgba(25, 35, 45, 0.72)", width=2),
        hoverinfo="skip",
        showlegend=False,
        connectgaps=False,
    ))

    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode="markers",
        name="Vehicle / Sensor",
        marker=dict(size=12, color="#e08f5b", symbol="diamond", line=dict(color="#f4eadf", width=2)),
        hovertemplate="vehicle / sensor origin (0, 0, 0 m)<extra></extra>",
    ))

    fig.update_layout(
        height=FULL_H,
        margin=dict(t=10, b=10, l=0, r=0),
        legend_title_text="Class",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="elevation (m)",
            aspectmode="auto",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.15)),
            bgcolor="#182321",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Each quadrilateral is a physical ring-sized cell; elevation supplies its height. "
        "Click any view on the left for a focused breakdown."
    )


FULL_RENDERERS = {
    "Semantic": lambda: render_semantic(df),
    "Elevation": lambda: render_elevation(df),
    "Traversability": lambda: render_traversability(df),
    "Resolution": lambda: render_resolution(df),
    "Performance": lambda: render_performance(grid),
}


def render_comparison():
    df_5cm = flatten_grid(uniform_grids["5cm"])
    df_50cm = flatten_grid(uniform_grids["50cm"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Uniform 5cm", f"{len(df_5cm):,}")
    c2.metric("Adaptive", f"{len(df):,}")
    c3.metric("Uniform 50cm", f"{len(df_50cm):,}")
    bench_df = pd.DataFrame({
        "grid": ["uniform-5cm", "adaptive", "uniform-50cm"],
        "cell_count": [len(df_5cm), len(df), len(df_50cm)],
    })
    fig = px.bar(bench_df, x="grid", y="cell_count")
    fig.update_layout(height=200, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# Left rail — mini preview cards (Task-Manager style)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("**Views**")
    sample = df.sample(min(300, len(df)), random_state=1) if not df.empty else df

    for name in VIEWS:
        is_selected = st.session_state.selected_view == name
        with st.container(border=True):
            MINI_RENDERERS[name](sample)
            btn_label = f"●  {name}" if is_selected else name
            if st.button(
                btn_label, key=f"card_{name}", use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state.selected_view = None if is_selected else name
                st.rerun()

    compare = st.toggle("Uniform vs Adaptive", value=False)
    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("Reshuffle", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with bc2:
        if st.button("← Landing", use_container_width=True):
            st.session_state.entered = False
            st.rerun()

# ----------------------------------------------------------------------------
# Center panel — overview by default, or the selected view's full detail
# ----------------------------------------------------------------------------
selected = st.session_state.selected_view
st.title(selected if selected else "Overview")
caption = VIEW_DESC.get(selected) if selected else None
st.caption(caption or "Combined adaptive grid — physical cell tiles colored by class and raised by elevation.")

if compare:
    render_comparison()
else:
    if selected is None:
        render_overview(df)
    else:
        FULL_RENDERERS[selected]()
