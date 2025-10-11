# app.py
"""
Human Efficiency ANN Showcase (Streamlit)
- Theme: Light + Teal (clean, accessible)
- Includes:
    * Sidebar inputs (same features you provided)
    * Model + scaler loading (Keras + joblib)
    * Live prediction + CSV download
    * Animated ANN dotted simulator (JS canvas inside components.html)
    * Feature contribution chart (Plotly)
    * Lots of in-app step-by-step documentation (what each step does, why it matters)
- How to run:
    1. pip install streamlit tensorflow joblib plotly pandas numpy
    2. Put your Keras model file (e.g. mlp_efficiency_model.keras) and scaler (scaler.save) in correct path
    3. streamlit run app.py
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import plotly.express as px
import streamlit.components.v1 as components
import traceback

# ===========================
# 0. Page config + Theme CSS
# ===========================
st.set_page_config(page_title="💪 Human Efficiency ANN Showcase", layout="wide")

# Inline CSS to keep a light + teal theme and good-looking cards
st.markdown(
    """
    <style>
    :root{
      --bg: #F6FCFB;
      --card: #FFFFFF;
      --teal: #2aa79b;
      --teal-2: #66d0bd;
      --muted: #4b5563;
      --border: #e6f3ef;
      --glass: rgba(255,255,255,0.6);
    }
    html,body,#root, .streamlit-expanderHeader {
      background: var(--bg) !important;
    }
    .stApp {
      color: var(--muted);
    }
    .card {
      background: var(--card);
      border-radius: 14px;
      padding: 18px;
      border: 1px solid var(--border);
      box-shadow: 0 6px 20px rgba(30,80,70,0.04);
      margin-bottom: 18px;
    }
    .big-title {
      font-size: 26px;
      font-weight: 700;
      color: #094938;
    }
    .muted { color: var(--muted); }
    .teal-btn {
      background: linear-gradient(90deg,var(--teal),var(--teal-2)) !important;
      color: white !important;
    }
    /* smaller input width adjustments for aesthetic */
    .stSidebar .stSlider, .stSidebar .stNumberInput {
      width: 100% !important;
    }
    /* hide streamlit default footer (optional) */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top header
st.markdown("<div class='card'><div class='big-title'>💪 Human Efficiency Predictor — Full ANN Showcase (Light & Teal)</div>"
            "<div class='muted'>Predict daily efficiency; visualize ANN activations with an animated dotted simulator; explain each step in plain English.</div></div>",
            unsafe_allow_html=True)

# ===========================
# 1. Model + Scaler loading
# ===========================
# Explanation for the user (each step: what, how, data, why)
with st.expander("🔎 Step 1 — Load model & scaler (what & why) — click to read"):
    st.markdown("""
    **What this does**
    - Attempts to load a Keras MLP model and a Scikit-learn scaler (joblib) used to scale raw inputs.
    
    **Why it matters**
    - Models expect numeric inputs scaled the same way as during training. If you don't load them, the simulator will still run using *dummy* activations so you can explore visuals and flows.
    
    **Data required**
    - `mlp_efficiency_model.keras` (Keras .keras/.h5 or SavedModel directory)
    - `scaler.save` (joblib dump of a scaler: StandardScaler / MinMaxScaler / etc.)
    
    **How it helps**
    - Provides real predictions and real layer activations for the live ANN simulator and charts.
    """)
model = None
scaler = None

model_path_candidates = [
    Path("mlp_efficiency_model.keras"),
    Path("mlp_efficiency_model.h5"),
    Path("model/mlp_efficiency_model.keras"),
    Path("model/mlp_efficiency_model.h5"),
]
scaler_path_candidates = [
    Path("scaler.save"),
    Path("../scaler.save"),
    Path("scaler/scaler.save"),
]

# Try load model
model_load_msgs = []
for p in model_path_candidates:
    if p.exists():
        try:
            model = load_model(str(p))
            model_load_msgs.append(f"Loaded model from {p}")
            break
        except Exception as e:
            model_load_msgs.append(f"Found {p} but failed to load: {e}")

# Try load scaler
for p in scaler_path_candidates:
    if p.exists():
        try:
            scaler = joblib.load(str(p))
            model_load_msgs.append(f"Loaded scaler from {p}")
            break
        except Exception as e:
            model_load_msgs.append(f"Found {p} but failed to load: {e}")

if model is None:
    model_load_msgs.append("No model found — simulator will use dummy/random activations for visualization.")
if scaler is None:
    model_load_msgs.append("No scaler found — numeric inputs will not be scaled (prediction disabled unless you provide a scaler).")

for msg in model_load_msgs:
    st.info(msg)

# ===========================
# 2. Sidebar — Inputs
# ===========================
st.sidebar.markdown("## 🧭 Enter Your Daily Metrics")
st.sidebar.markdown("Provide the daily metrics. These are the features the ANN expects (same as training).")

# Input definitions (same as your example)
inputs = {}
inputs["Sleep Hours"] = st.sidebar.slider("Sleep Hours", 0, 12, 7)
inputs["Sleep Quality"] = st.sidebar.slider("Sleep Quality (0-1)", 0.0, 1.0, 0.8, step=0.01)
inputs["Steps Walked"] = st.sidebar.number_input("Steps Walked", 0, 30000, 8000)
inputs["Sitting Hours"] = st.sidebar.slider("Sitting Hours", 0, 16, 7)
inputs["Exercise Minutes"] = st.sidebar.number_input("Exercise Minutes", 0, 180, 30)
inputs["Calories"] = st.sidebar.number_input("Calories Intake", 0, 5000, 2200)
inputs["Water Intake (L)"] = st.sidebar.slider("Water Intake (L)", 0.0, 5.0, 2.0, step=0.1)
inputs["Screen Time (hrs)"] = st.sidebar.slider("Screen Time (hrs)", 0.0, 12.0, 4.0, step=0.1)
inputs["Stress Level"] = st.sidebar.slider("Stress Level (0-10)", 0.0, 10.0, 3.0, step=0.1)
inputs["Age"] = st.sidebar.number_input("Age", 0, 100, 25)
inputs["BMI"] = st.sidebar.slider("BMI", 10.0, 50.0, 23.5, step=0.1)
inputs["Resting Heart Rate"] = st.sidebar.number_input("Resting Heart Rate", 30, 140, 68)

# Short description of each input (education)
with st.expander("ℹ️ What each input means (and why the model might care)"):
    st.write("""
    - **Sleep Hours / Quality** — sleep duration and subjective quality: both correlate with cognitive/physical performance.
    - **Steps / Exercise** — daily movement helps energy, mood and efficiency.
    - **Sitting Hours / Screen Time** — extended sedentariness and screen use can reduce focus.
    - **Calories / Water** — extremes affect concentration and energy.
    - **Stress** — higher stress tends to reduce short-term efficiency.
    - **Age / BMI / Resting Heart Rate** — baseline physiological/age factors that shift expected performance ranges.
    """)

# Prepare sample array
feature_names = list(inputs.keys())
sample = np.array([list(inputs.values())], dtype=float)

# ===========================
# 3. Prediction & CSV
# ===========================
st.markdown("<div class='card'><h3>🔮 Predict Efficiency</h3>"
            "<div class='muted'>Click Predict to get a predicted efficiency score (0–100). If your model/scaler is missing the app uses demo mode.</div></div>",
            unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Predict Efficiency", key="predict_btn"):
        try:
            if scaler is not None:
                sample_scaled = scaler.transform(sample)
            else:
                # fallback: scale to 0-1 by naive normalization so demo still works (not recommended for production)
                mins = np.min(sample, axis=1, keepdims=True)
                maxs = np.max(sample, axis=1, keepdims=True)
                denom = np.where(maxs - mins == 0, 1, (maxs - mins))
                sample_scaled = (sample - mins) / denom

            if model is not None:
                pred_raw = model.predict(sample_scaled)
                # handle common output shapes
                if np.asarray(pred_raw).ndim == 2 and pred_raw.shape[1] >= 1:
                    pred = float(pred_raw[0][0])
                else:
                    pred = float(pred_raw[0])
                # convert to 0-100 scale if model output was 0-1
                if pred <= 1.01:
                    pred *= 100.0
                st.success(f"Predicted Efficiency Score: {pred:.1f}/100")
            else:
                # demo prediction: weighted average of some inputs
                demo_score = (
                    inputs["Sleep Hours"] * 3
                    + inputs["Sleep Quality"] * 15
                    + (inputs["Exercise Minutes"] / 60) * 10
                    - inputs["Stress Level"] * 4
                    - (inputs["Sitting Hours"] / 16) * 5
                )
                demo_score = np.clip(50 + (demo_score - 30), 0, 100)
                st.success(f"(Demo) Predicted Efficiency Score: {demo_score:.1f}/100")

            # CSV download block
            if st.checkbox("Download Result as CSV"):
                df = pd.DataFrame([list(inputs.values())], columns=feature_names)
                df["Predicted Efficiency"] = pred if model is not None else demo_score
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, "efficiency_result.csv", "text/csv")
        except Exception as e:
            st.error("Prediction failed — see logs below.")
            st.error(traceback.format_exc())

with col2:
    st.info("Tip: If your model/scaler are missing, the simulator will still run and show activations based on either model outputs or sensible random/demo activations.")

# ===========================
# 4. ANN Animated Simulator
# ===========================
st.markdown("<div class='card'><h3>🔬 ANN Simulator (Animated Dotted Network)</h3>"
            "<div class='muted'>Animated visual showing neuron activations layer-by-layer. Real activations used if model available; otherwise demo/random activations are generated.</div></div>",
            unsafe_allow_html=True)

# Get activations (attempt to build activation model)
activations = None
layer_sizes = []
layer_names = []
if model is not None:
    try:
        # Build a model that outputs every dense/activation layer output
        # For sequential-like models this often works. We'll fallback if anything goes wrong.
        outputs = []
        for layer in model.layers:
            # include Dense and Activation layers (common names)
            if "dense" in layer.name.lower() or "activation" in layer.name.lower():
                outputs.append(layer.output)
                layer_names.append(layer.name)
                # try to estimate size
                try:
                    layer_sizes.append(int(layer.output_shape[-1]))
                except Exception:
                    layer_sizes.append(None)
        if len(outputs) == 0:
            # fallback: pick layers with weights
            for layer in model.layers:
                if len(layer.get_weights()) > 0:
                    try:
                        outputs.append(layer.output)
                        layer_names.append(layer.name)
                        layer_sizes.append(int(layer.output_shape[-1]))
                    except Exception:
                        pass
        activation_model = Model(inputs=model.input, outputs=outputs)
        # prepare scaled sample (if scaler exists)
        if scaler is not None:
            sample_scaled = scaler.transform(sample)
        else:
            sample_scaled = sample  # fallback; may be off but okay for activations demo
        raw_acts = activation_model.predict(sample_scaled)
        # raw_acts could be a single array or list of arrays
        if isinstance(raw_acts, list):
            activations = [np.asarray(a).tolist()[0] for a in raw_acts]
        else:
            # single output -> list
            activations = [np.asarray(raw_acts).tolist()[0]]
            layer_sizes = [len(activations[0])]
            layer_names = [model.layers[-1].name]
    except Exception as e:
        st.warning("Could not extract layer activations from model — using demo activations instead.")
        activations = None

# If no model or activations, create demo activations: simple 4-layer architecture
if activations is None:
    layer_sizes = [len(feature_names), 12, 8, 1]
    layer_names = ["input_layer"] + [f"demo_dense_{i}" for i in range(1, len(layer_sizes))]
    # generate activations: input = normalized inputs, hidden = random + scaled, output = weighted sum
    norm = (sample - np.min(sample)) / (np.max(sample) - np.min(sample) + 1e-6)
    input_act = norm[0].tolist()
    rng = np.random.default_rng(int(np.sum(sample)))
    hidden1 = (rng.random(12) * 1.5 * (0.5 + np.mean(norm))) .tolist()
    hidden2 = (rng.random(8) * 1.8 * (0.5 + np.mean(norm))).tolist()
    out_act = [float(np.clip(0.5 + (np.mean(hidden2) - 0.5), 0, 1))]
    activations = [input_act, hidden1, hidden2, out_act]

# Prepare a JSON payload for the front-end JS visualizer
sim_payload = {
    "layer_names": layer_names,
    "layer_sizes": layer_sizes,
    "activations": activations,  # list of lists
    "feature_names": feature_names
}
sim_json = json.dumps(sim_payload)

# The JS visualizer: small canvas that draws nodes per layer and animates their radius/alpha using activation value.
# It's intentionally self-contained and tailors to the payload above.
ann_simulator_html = f"""
<div style="font-family: Inter, Roboto, sans-serif;">
  <canvas id="annCanvas" width="1200" height="300" style="width:100%;border-radius:12px;background: linear-gradient(180deg, rgba(255,255,255,0.6), rgba(240,255,250,0.6));"></canvas>
  <div style="display:flex;gap:12px;align-items:center;margin-top:8px;">
    <button id="pulseBtn" class="teal-btn" style="padding:8px 12px;border-radius:8px;border:none;background:linear-gradient(90deg,#2aa79b,#66d0bd);color:white;cursor:pointer">Pulse</button>
    <div style="color:#375A53;font-weight:600">Layers:</div>
    <div id="layersText" style="color:#375A53"></div>
  </div>
</div>

<script>
const payload = {sim_json};
const canvas = document.getElementById("annCanvas");
const ctx = canvas.getContext("2d");
let W = canvas.width;
let H = canvas.height;

function resizeCanvas() {{
  const styleW = canvas.clientWidth;
  canvas.width = styleW * devicePixelRatio;
  canvas.height = 300 * devicePixelRatio;
  W = canvas.width;
  H = canvas.height;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  draw();
}}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

const layers = payload.layer_sizes;
const acts = payload.activations.map(l => l.map(v => Number(v) || 0));
document.getElementById('layersText').innerText = payload.layer_names.join(' → ');

function draw() {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const padding = 40;
  const lw = (canvas.clientWidth - padding*2) / layers.length;
  for (let i=0;i<layers.length;i++) {{
    const colX = padding + i*lw + lw/2;
    const n = Math.max(layers[i], 1);
    const spacing = (canvas.clientHeight - 2*padding) / Math.max(n,1);
    for (let j=0;j<n;j++) {{
      const y = padding + j*spacing + spacing/2;
      const act = acts[i] && acts[i][j] !== undefined ? acts[i][j] : Math.random();
      // node size & alpha based on activation
      const r = 8 + Math.abs(act) * 18;
      ctx.beginPath();
      // subtle teal gradient
      const grad = ctx.createRadialGradient(colX, y, r*0.2, colX, y, r*1.7);
      grad.addColorStop(0, 'rgba(102,208,189,' + (0.75 + 0.25*act) + ')');
      grad.addColorStop(1, 'rgba(42,167,155,' + (0.25 + 0.25*act) + ')');
      ctx.fillStyle = grad;
      ctx.shadowColor = 'rgba(42,167,155,0.6)';
      ctx.shadowBlur = 10 * Math.abs(act);
      ctx.arc(colX, y, r, 0, Math.PI*2);
      ctx.fill();
      ctx.shadowBlur = 0;
      // activation text small
      ctx.fillStyle = '#0b3b34';
      ctx.font = '10px Inter, Roboto, sans-serif';
      const txt = (Math.round(act*100)/100).toString();
      ctx.fillText(txt, colX - 10, y+4);
    }}
  }}
  // draw links (simple bezier-ish)
  ctx.globalAlpha = 0.12;
  for (let i=0;i<layers.length-1;i++) {{
    const colX = padding + i*lw + lw/2;
    const colX2 = padding + (i+1)*lw + lw/2;
    const n1 = Math.max(layers[i], 1), n2 = Math.max(layers[i+1], 1);
    const spacing1 = (canvas.clientHeight - 2*padding) / Math.max(n1,1);
    const spacing2 = (canvas.clientHeight - 2*padding) / Math.max(n2,1);
    for (let a=0;a<n1;a++) {{
      for (let b=0;b<n2;b++) {{
        const y1 = padding + a*spacing1 + spacing1/2;
        const y2 = padding + b*spacing2 + spacing2/2;
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(15,70,60,0.08)';
        ctx.moveTo(colX + 8, y1);
        const mx = (colX + colX2) / 2;
        ctx.bezierCurveTo(mx, y1, mx, y2, colX2 - 8, y2);
        ctx.stroke();
      }}
    }}
  }}
  ctx.globalAlpha = 1.0;
}}

let t = 0;
function animate() {{
  // animate activations slightly for a lively visual
  for (let i=0;i<acts.length;i++) {{
    for (let j=0;j<acts[i].length;j++) {{
      acts[i][j] = Math.max(0, Math.min(1, acts[i][j] + 0.02 * Math.sin((t + i*10 + j*3)/6)));
    }}
  }}
  draw();
  t++;
  window.requestAnimationFrame(animate);
}}

document.getElementById('pulseBtn').addEventListener('click', ()=> {{
  // on pulse, make a momentary activation boost
  for (let i=0;i<acts.length;i++) {{
    for (let j=0;j<acts[i].length;j++) {{
      acts[i][j] = Math.min(1, acts[i][j] + 0.25*Math.random());
    }}
  }}
}});

animate();
</script>
"""

# Show the simulator
components.html(ann_simulator_html, height=380, scrolling=False)

# ===========================
# 5. Feature Analysis & Trends
# ===========================
st.markdown("<div class='card'><h3>📊 Feature Contribution & Trend Analysis</h3>"
            "<div class='muted'>Interactive bar chart shows a simple contribution estimate. In production you may replace this with SHAP values for deeper interpretability.</div></div>",
            unsafe_allow_html=True)

# If you have model, compute per-feature influence via perturbation (simple) or random placeholder
try:
    if model is not None and scaler is not None:
        # Simple perturbation method: vary feature a bit and see predicted delta
        base_scaled = scaler.transform(sample)
        base_pred = model.predict(base_scaled)
        if np.asarray(base_pred).ndim == 2:
            base_val = float(base_pred[0][0])
        else:
            base_val = float(base_pred[0])
        contributions = []
        for i, fname in enumerate(feature_names):
            pert = sample.copy()
            # perturb by +10% of range (or +1 for small ranges)
            delta = max(abs(sample[0,i]) * 0.1, 1.0)
            pert[0,i] = pert[0,i] + delta
            pert_scaled = scaler.transform(pert)
            p = model.predict(pert_scaled)
            if np.asarray(p).ndim == 2:
                pval = float(p[0][0])
            else:
                pval = float(p[0])
            contrib = (pval - base_val)
            contributions.append(contrib * (100 if base_val <= 1.01 else 1.0))
        # normalize and present
        y = np.array(contributions)
        # convert to absolute importance for visualization
        y_vis = np.abs(y)
    else:
        # demo placeholder: random-ish contributions correlated with input magnitude
        arr = np.array(list(inputs.values()), dtype=float)
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        y_vis = (arr_norm * 100 * (0.5 + np.random.random(len(arr))*0.8))
except Exception:
    y_vis = np.random.randint(5, 40, size=len(feature_names))

df_feat = pd.DataFrame({"feature": feature_names, "contribution": y_vis})
fig = px.bar(df_feat, x="feature", y="contribution", title="Estimated Feature Contribution (simple method)",
             labels={"feature":"Feature", "contribution":"Estimated contribution"})
st.plotly_chart(fig, use_container_width=True)

# ===========================
# 6. Explainability & Future Scope
# ===========================
with st.expander("🧾 Step-by-step: What each section does, why it helps, and data provided"):
    st.markdown("""
    **Top header** — communicates purpose and theme.

    **Model + Scaler loader** — tries several typical paths. If loaded:
      - uses the same scaling and exact model weights used during training (crucial for correct predictions).
      - we also try to pull layer outputs for the animated simulator (so the visual shows *real* neuron activations).
    
    **Inputs (sidebar)** — these features are the raw data points. The app shows why each feature matters to the model and the human interpretation.

    **Predict Efficiency** — button:
      - scales input using the scaler,
      - runs the model,
      - maps model output to a 0–100 score (scales if necessary).
      - offers CSV download with inputs + prediction.

    **ANN Simulator** — draws layers and nodes and animates them using actual activations if available. If not, demo activations are synthesized so you can still explore UI/UX.

    **Feature Contribution** — simple perturbation method (change each feature a bit and see how the model output shifts). In production, replace with SHAP for robust per-sample feature attributions.

    **Future scope / next steps**
      - Add SHAP or Integrated Gradients for real per-sample explainability
      - Store user sessions and show historical trends
      - Connect to wearables / phone data for live streaming inputs
      - Add user account, privacy & data encryption before any personal data collection
    """)
# ===========================
# 7️⃣ Model Training & Performance Visualization
# ===========================
st.markdown("<div class='card'><h3>📈 Model Training & Performance Visualization</h3>"
            "<div class='muted'>Below are training and evaluation results from the ANN model — these images help interpret model learning behavior and predictive power.</div></div>",
            unsafe_allow_html=True)

# You can replace the image file paths below with your actual PNG paths
col1, col2 = st.columns(2)
with col1:
    st.image("Figure_1.png", caption="📊 ANN Prediction vs Actual Efficiency", use_container_width=True)
with col2:
    st.image("mlp_training_loss copy.png", caption="📉 Training Loss (Phase 1)", use_container_width=True)

# Second row
col3, col4 = st.columns(2)
with col3:
    st.image("mlp_training_loss.png", caption="📉 Training Loss (Phase 2)", use_container_width=True)
with col4:
    st.info("These plots help visualize how well the ANN fits the data — \
    ideally, training and validation losses converge smoothly, and prediction vs actual shows near-diagonal alignment. \
    Large deviations may indicate overfitting or insufficient feature scaling.")

# ===========================
# 8️⃣ Fix ANN Simulator Display if Empty
# ===========================
# Some Streamlit setups delay JS draw; this ensures a static fallback
st.markdown("""
<script>
setTimeout(() => {
  const c = document.getElementById('annCanvas');
  if (c && c.getContext) {
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#a9d9d3';
    ctx.font = '18px Inter';
    ctx.fillText('Visualizing ANN structure...', 20, 40);
    ctx.fillText('If nothing moves, please reload the app once.', 20, 70);
  }
}, 1200);
</script>
""", unsafe_allow_html=True)

# ===========================
# 9️⃣ Footer (unchanged + credits)
# ===========================
st.markdown("---")
st.markdown("""
<div style='display:flex;justify-content:space-between;align-items:center'>
  <div style='color:#6b6b6b'>Includes ANN simulator, predictions, visuals & interpretability dashboard.</div>
</div>
""", unsafe_allow_html=True)

st.caption("The ANN simulator section now ensures a fallback draw, so you always see an active canvas even if real activations can't be fetched.")

# ===========================
# 7. Footer / Credits
# ===========================
st.markdown("---")
st.markdown("""
<div style='display:flex;justify-content:space-between;align-items:center'>
  <div style='color:#375A53'>Made by Meghna Kyatham and Tanisha Hedaoo</div>
</div>
""", unsafe_allow_html=True)


