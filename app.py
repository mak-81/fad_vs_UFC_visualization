import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Annulus

st.set_page_config(page_title="Through-Wall Crack Assessment", layout="wide")
st.title("Through-Wall Crack Assessment in Pipe")

# --- Mathematical Functions ---
def FAL(Lr, YS, UTS):
    y = np.zeros_like(Lr, dtype=float)
    cutoff = (YS + UTS) / (2 * YS)
    mask = Lr <= cutoff
    y[mask] = (1 - 0.14 * Lr[mask]**2) * (0.3 + 0.7 * np.exp(-0.65 * Lr[mask]**6))
    return y

def UFC(K_hat, A2, A1, p, B1, B2, h1, h2):
    return (
        A1
        + ((A2 - A1) * p / (1 + 10 ** ((B1 - K_hat) * h1)))
        + ((A2 - A1) * (1 - p) / (1 + 10 ** ((B2 - K_hat) * h2)))
    )

def sigma_using_Folias_factor(c, t, R, P):
    return (P * R / t) * (1 + 1.255 * c**2 / (t * R) - 0.0135 * c**4 / ((t * R) ** 2)) ** 0.5

# --- Load Data ---
@st.cache_data
def load_fitting_parameters(filepath="UFC_fitting_parameters.xlsx"):
    df = pd.read_excel(filepath, sheet_name=0)
    return df.set_index(["delta", "Type"]).to_dict(orient="index")

try:
    double_index_dict = load_fitting_parameters()
except FileNotFoundError:
    st.error(
        "Could not find `UFC_fitting_parameters.xlsx`. "
        "Please place it in the same directory as `app.py`."
    )
    st.stop()

# --- Sidebar Controls & Reset Logic ---
INIT_YS = 500.0
INIT_P = 10.0
INIT_C = 10.0
INIT_KMAT = 100.0

for key, default in [("ys", INIT_YS), ("p", INIT_P), ("c", INIT_C), ("kmat", INIT_KMAT)]:
    if key not in st.session_state:
        st.session_state[key] = default

def reset_sliders():
    st.session_state.ys = INIT_YS
    st.session_state.p = INIT_P
    st.session_state.c = INIT_C
    st.session_state.kmat = INIT_KMAT

st.sidebar.header("Parameters")
YS = st.sidebar.slider("σ_y (MPa)", min_value=300.0, max_value=1000.0, step=1.0, key="ys")
P = st.sidebar.slider("P (MPa)", min_value=1.0, max_value=100.0, step=0.5, key="p")
c = st.sidebar.slider("c (mm)", min_value=0.05, max_value=50.0, step=0.05, key="c")
Kmat = st.sidebar.slider("K_mat (MPa·m^1/2)", min_value=10.0, max_value=500.0, step=1.0, key="kmat")

st.sidebar.button("Reset to Defaults", on_click=reset_sliders)

# --- Constants & Calculations ---
R = 250  # pipeline radius (mm)
t = 10   # pipeline thickness (mm)
lr = np.arange(0, 1.21, 0.001)
K_hat = np.arange(0, 10, 0.01)

sigma_folias = sigma_using_Folias_factor(c, t, R, P)
Lr_assess = sigma_folias / YS
Kr_assess = sigma_folias * ((np.pi * c / 1000) ** 0.5) / Kmat
FAL_at_Lr = FAL(np.array([Lr_assess]), YS, YS * 1.2)[0]
is_safe_fad = Kr_assess <= FAL_at_Lr

lambda_val = P * R / (t * YS)
k_hat_assess = (Kmat * (P * R / t)) / (
    (YS * sigma_folias) * ((np.pi * c / 1000) ** 0.5)
)

lb_params = double_index_dict[(0, "Lower Bound")]
ufc_limit = min(
    1.2,
    UFC(
        k_hat_assess,
        lb_params["A2"],
        lb_params["A1"],
        lb_params["p"],
        lb_params["B1"],
        lb_params["B2"],
        lb_params["h1"],
        lb_params["h2"],
    ),
)
is_safe_ufc = lambda_val <= ufc_limit

# --- Create Figure ---
fig = plt.figure(figsize=(8, 5))
fig.suptitle(f"Through-wall crack in pipe (t={t} mm, R={R} mm)", fontsize=14)

ax = plt.subplot2grid((3, 4), (1, 0), rowspan=2, colspan=2)
ax3 = plt.subplot2grid((3, 4), (1, 2), rowspan=2, colspan=2)
ax1 = plt.subplot2grid((3, 4), (0, 3))
ax2 = plt.subplot2grid((3, 4), (0, 0), colspan=3)

# 1. FAD Plot (Left)
ax.plot(lr, FAL(lr, YS, YS * 1.2), label="FAL")
ax.plot(
    [0, Lr_assess],
    [0, Kr_assess],
    color="red",
    marker="X",
    label="Assessment line",
)
ax.set_xlabel(r"$L_r$", fontsize=12)
ax.set_ylabel(r"$K_r$", fontsize=12)
ax.set_xlim([0, 1.2])
ax.set_ylim([0, 1.1])
ax.legend()

if is_safe_fad:
    ax.text(0.5, 0.5, "safe", color="green", fontsize=14, fontweight="bold", ha="center", va="center")
else:
    ax.text(1.0, 1.0, "fail", color="red", fontsize=14, fontweight="bold", ha="center", va="center")

# 2. UFC Plot (Right)
for delta, curve in ((0, "Mean"), (0, "Lower Bound")):
    params = double_index_dict[(delta, curve)]
    y_ufc = np.minimum(
        1.2 * np.ones_like(K_hat),
        UFC(
            K_hat,
            params["A2"],
            params["A1"],
            params["p"],
            params["B1"],
            params["B2"],
            params["h1"],
            params["h2"],
        ),
    )
    ax3.plot(K_hat, y_ufc, label=f"δ={delta}, {curve}")

ax3.plot(
    [k_hat_assess, k_hat_assess],
    [0, lambda_val],
    color="red",
    marker="X",
    label="Assessment line",
)

ax3.set_xlabel(r"$\hat{K}$", fontsize=12)
ax3.set_ylabel(r"$\lambda$", fontsize=12)
ax3.set_xlim([0, 10])
ax3.set_ylim([0, 1.3])
ax3.legend()

if is_safe_ufc:
    ax3.text(2.0, 0.2, "safe", color="green", fontsize=14, fontweight="bold", ha="center", va="center")
else:
    ax3.text(1.0, 1.0, "fail", color="red", fontsize=14, fontweight="bold", ha="center", va="center")

# 3. Pipe Schematic (Top Left/Center)
rect1 = Rectangle((0, 0), 1, 0.05, facecolor="grey", edgecolor="black")
rect2 = Rectangle((0, 0.95), 1, 0.05, facecolor="grey", edgecolor="black")
rect3 = Rectangle((0.5 - c / 400, 0.95), c / 200, 0.05, facecolor="white", edgecolor="black")

ax2.add_patch(rect1)
ax2.add_patch(rect2)
ax2.add_patch(rect3)
ax2.annotate("", xy=(0, 0.5), xycoords="data", xytext=(1, 0.5), arrowprops=dict(arrowstyle="-", color="grey", ls="-."))
ax2.text(0.45, 0.75, f"2c={2*c:.1f} mm", color="black", fontsize=11)
ax2.text(0.5, 0.5, f"P={P:.1f} MPa", color="black", ha="center", va="center", fontsize=10 + int(P / 10))
ax2.text(0.03, 0.1, rf"$\sigma_{{y}}={YS:.0f}$ MPa", color="black", fontsize=11)
ax2.text(0.00, 0.07, "↙", fontsize=12)
ax2.text(0.47, 0.07, "↙", fontsize=12)
ax2.text(0.5, 0.1, rf"$K_{{mat}}={Kmat:.0f}$ MPa·m$^{{1/2}}$", color="black", fontsize=11)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_frame_on(False)

# 4. Cross-section Schematic (Top Right)
annulus = Annulus((0.5, 0.5), 0.49, 0.05, facecolor="grey", edgecolor="black")
ax1.add_patch(annulus)
ax1.annotate("", xy=(0, 0.5), xycoords="axes fraction", xytext=(1, 0.5), arrowprops=dict(arrowstyle="-", color="grey", ls="-."))
ax1.annotate("", xy=(0.5, 0), xycoords="axes fraction", xytext=(0.5, 1), arrowprops=dict(arrowstyle="-", color="grey", ls="-."))
ax1.text(0.5, 0.5, "P", ha="center", va="center", color="black", fontsize=16 + int(P / 10))
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_frame_on(False)
ax1.set_aspect(1)

plt.tight_layout()
st.pyplot(fig)
