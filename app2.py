# src/app.py
"""
AURA — AI for User Resilience & Activity (Streamlit app)
Horizontal tabbed UI, SQLite DB for persistence, ANN simulator (fixed), inputs, focus timer and CSV import.

How to run (Windows PowerShell):
1. cd C:/Users/<you>/Desktop/ann_efficiency_project
2. venv/Scripts/Activate.ps1
3. pip install streamlit numpy pandas scikit-learn joblib tensorflow plotly
4. streamlit run src/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from tensorflow.keras.models import load_model
import plotly.express as px
import traceback
import time
import sqlite3
import base64

# -------------------------
# DATABASE helper functions
# -------------------------
DB_PATH = Path("aura.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # history table: store inputs + predicted efficiency + timestamp
    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      sleep_hours REAL,
      sleep_quality REAL,
      steps INTEGER,
      sitting_hours REAL,
      exercise_mins INTEGER,
      calories INTEGER,
      water_l REAL,
      screen_time REAL,
      stress_level REAL,
      age INTEGER,
      bmi REAL,
      resting_hr INTEGER,
      predicted_efficiency REAL,
      timestamp TEXT
    )
    """)
    # todos table
    c.execute("""
    CREATE TABLE IF NOT EXISTS todos (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      task TEXT NOT NULL,
      done INTEGER DEFAULT 0,
      created TEXT
    )
    """)
    conn.commit()
    conn.close()

def insert_history_row(row: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
      INSERT INTO history (
        sleep_hours, sleep_quality, steps, sitting_hours, exercise_mins,
        calories, water_l, screen_time, stress_level, age, bmi, resting_hr,
        predicted_efficiency, timestamp
      ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        row.get('sleep_hours'), row.get('sleep_quality'), row.get('steps'),
        row.get('sitting_hours'), row.get('exercise_mins'), row.get('calories'),
        row.get('water_l'), row.get('screen_time'), row.get('stress_level'),
        row.get('age'), row.get('bmi'), row.get('resting_hr'),
        row.get('predicted_efficiency'), row.get('timestamp')
    ))
    conn.commit()
    conn.close()

def fetch_history(limit=None):
    conn = sqlite3.connect(DB_PATH)
    q = "SELECT * FROM history ORDER BY timestamp DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(q, conn)
    conn.close()
    return df

def insert_todo(task_text):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO todos (task, done, created) VALUES (?, 0, datetime('now'))", (task_text,))
    conn.commit()
    conn.close()

def fetch_todos():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM todos ORDER BY id", conn)
    conn.close()
    return df

def set_todo_done(todo_id, done_flag):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE todos SET done=? WHERE id=?", (1 if done_flag else 0, int(todo_id)))
    conn.commit()
    conn.close()

def delete_todo(todo_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM todos WHERE id=?", (int(todo_id),))
    conn.commit()
    conn.close()

# initialize DB on app start
init_db()

# -------------------------
# Model & Scaler loader
# -------------------------
model = None
scaler = None
model_messages = []

model_candidates = [
    Path("mlp_efficiency_model.keras"),
    Path("mlp_efficiency_model.h5"),
    Path("models/mlp_efficiency_model.keras"),
    Path("models/mlp_efficiency_model.h5"),
]
scaler_candidates = [Path("scaler.save"), Path("models/scaler.save"), Path("scaler/scaler.save")]

for p in model_candidates:
    if p.exists():
        try:
            model = load_model(str(p), compile=False)
            model_messages.append(f"Loaded model from {p.name}")
            break
        except Exception as e:
            model_messages.append(f"Found {p.name} but failed to load: {e}")

for p in scaler_candidates:
    if p.exists():
        try:
            scaler = joblib.load(str(p))
            model_messages.append(f"Loaded scaler from {p.name}")
            break
        except Exception as e:
            model_messages.append(f"Found {p.name} but failed to load scaler: {e}")

if model is None:
    model_messages.append("No model found — running in demo mode.")
if scaler is None:
    model_messages.append("No scaler found — using demo scaling.")
    
# -------------------------
# Page config and CSS
# -------------------------
st.set_page_config(page_title="AURA: AI Wellness & Efficiency", layout="wide")
PRIMARY_GREEN = "#66d0bd"
PRIMARY_GREEN_2 = "#2aa79b"
CONTRAST_TEXT = "#eafaf7"

st.markdown(f"""
<style>
:root {{
  --bg: #05201d;
  --card: #063331;
  --accent: {PRIMARY_GREEN};
  --accent2: {PRIMARY_GREEN_2};
  --muted: #9aa6a3;
  --txt: {CONTRAST_TEXT};
}}
html, body, .stApp {{ background: var(--bg); color: var(--txt); }}
.card {{ background: var(--card); padding:18px; border-radius:12px; border:1px solid rgba(255,255,255,0.03); }}
.header-title {{ font-size:34px; font-weight:800; color:var(--accent); letter-spacing:1px; }}
.header-sub {{ color: var(--txt); opacity:0.9; margin-top:6px; }}
button.stButton>button {{ background: linear-gradient(90deg, var(--accent), var(--accent2)); color: #041311; font-weight:700; }}
footer {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header + horizontal tabs
# -------------------------
st.markdown(f"<div style='display:flex;align-items:center;gap:20px'><div class='header-title'>AURA</div><div class='header-sub'>AI for User Resilience & Activity — Human Efficiency Predictor</div></div>", unsafe_allow_html=True)
st.markdown("")

# create horizontal tabs (these appear across top)
tabs = st.tabs(["Home", "Daily Check", "Focus Mode", "Weekly Report", "Settings"])

# small helper to attempt a rerun in a compatible way
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.rerun()
        except Exception:
            # unable to rerun on this Streamlit version — ignore (UI will refresh next interaction)
            pass

# ---------------------------------------
# HOME tab
# ---------------------------------------
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:var(--accent)'>Welcome to AURA</h3>", unsafe_allow_html=True)
    st.write("A compact wellness & efficiency assistant. Use Daily Check to compute instant efficiency scores, view the ANN simulator, use Focus Mode to do focus on the work, and see your personalized Weekly Reports.")
    st.markdown("Model status:")
    for m in model_messages:
        st.markdown(f"- {m}")
    # include uploaded resource path (as requested)
    st.markdown("<div style='margin-top:8px'><b>Uploaded HTML resource:</b><br><code>/mnt/data/8a562eb2-84b1-4eb7-ad14-888463115b71.html</code></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------
# DAILY CHECK tab
# ---------------------------------------
with tabs[1]:
    st.markdown("<div class='card'><h3 style='color:var(--accent)'>Daily Check — Efficiency Score</h3></div>", unsafe_allow_html=True)
    left, right = st.columns([1, 2])
    with left:
        st.markdown("Enter your values")
        sleep = st.slider("Sleep Hours", 0, 12, 7)
        sleep_q = st.slider("Sleep Quality (0-1)", 0.0, 1.0, 0.8, step=0.01)
        steps = st.number_input("Steps Walked", 0, 30000, 8000)
        sitting = st.slider("Sitting Hours", 0, 16, 7)
        exercise = st.number_input("Exercise Minutes", 0, 180, 30)
        calories = st.number_input("Calories Intake", 0, 5000, 2200)
        water = st.slider("Water Intake (L)", 0.0, 5.0, 2.0, step=0.1)
        screen_time = st.slider("Screen Time (hrs)", 0.0, 16.0, 4.0, step=0.1)
        stress = st.slider("Stress Level (0-10)", 0.0, 10.0, 3.0, step=0.1)
        age = st.number_input("Age", 10, 100, 25)
        bmi = st.slider("BMI", 10.0, 50.0, 23.5, step=0.1)
        hr = st.number_input("Resting Heart Rate", 30, 140, 68)

        feature_names = ["sleep_hours","sleep_quality","steps","sitting_hours","exercise_mins","calories","water_l","screen_time","stress_level","age","bmi","resting_hr"]
        sample = np.array([[sleep, sleep_q, steps, sitting, exercise, calories, water, screen_time, stress, age, bmi, hr]], dtype=float)

        if st.button("Predict Efficiency"):
            try:
                # scaling
                if scaler is not None:
                    sample_scaled = scaler.transform(sample)
                else:
                    mn = np.min(sample); mx = np.max(sample)
                    sample_scaled = (sample - mn) / (mx - mn + 1e-6)

                # predict
                if model is not None:
                    raw = model.predict(sample_scaled)
                    pred = float(raw[0][0]) if raw.ndim == 2 else float(raw[0])
                    if pred <= 1.01:
                        pred *= 100.0
                    score = float(np.clip(pred, 0, 100))
                else:
                    score = float(np.clip(50 + (sleep*3 - stress*4), 0, 100))

                st.success(f"Predicted Efficiency Score: {score:.1f}/100")

                # store into DB
                entry = {
                    'sleep_hours': sleep, 'sleep_quality': sleep_q, 'steps': steps,
                    'sitting_hours': sitting, 'exercise_mins': exercise, 'calories': calories,
                    'water_l': water, 'screen_time': screen_time, 'stress_level': stress,
                    'age': age, 'bmi': bmi, 'resting_hr': hr,
                    'predicted_efficiency': score, 'timestamp': pd.Timestamp.now().isoformat()
                }
                insert_history_row(entry)

                # small guidance
                if screen_time > 6:
                    st.warning("Phone usage high. Consider a short break. Put the phone away for 20 minutes.")
                elif stress > 7:
                    st.warning("Stress is high. Try a 5-minute breathing break.")
                else:
                    st.info("Nice, you're doing well today! Keep it up")

            except Exception:
                st.error("Prediction error")
                st.text(traceback.format_exc())

        # Download CSV of latest 100 entries
        if st.button("Download last 100 history rows (CSV)"):
            df_hist = fetch_history(limit=100)
            csv = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "aura_history_100.csv", "text/csv")

    # Right: ANN simulator (fixed, safe)
    with right:
        st.markdown("<div class='card'><b>ANN Simulator</b></div>", unsafe_allow_html=True)
        # Build demo activations or real activations if model available
        activations = None
        layer_sizes = []
        layer_names = []
        try:
            if model is not None:
                import tensorflow as tf
                try:
                    _ = model.predict(np.zeros((1, sample.shape[1])))
                except Exception:
                    pass
                dense_outputs = [layer.output for layer in model.layers if getattr(layer, 'output_shape', None) is not None and len(layer.output_shape) >= 2]
                if dense_outputs:
                    activation_model = tf.keras.Model(inputs=model.input, outputs=dense_outputs)
                    sample_scaled = scaler.transform(sample) if scaler is not None else sample
                    raw_acts = activation_model.predict(sample_scaled)
                    if isinstance(raw_acts, list):
                        activations = [np.asarray(a).tolist()[0] for a in raw_acts]
                        layer_sizes = [int(np.asarray(a).shape[-1]) for a in raw_acts]
                        layer_names = [layer.name for layer in model.layers if getattr(layer, 'output_shape', None) is not None and len(layer.output_shape) >= 2][:len(activations)]
        except Exception:
            activations = None

        if activations is None:
            layer_sizes = [len(feature_names), 12, 8, 1]
            layer_names = ["input_layer"] + [f"demo_dense_{i}" for i in range(1, len(layer_sizes))]
            norm = (sample - np.min(sample)) / (np.max(sample) - np.min(sample) + 1e-6)
            input_act = norm[0].tolist()
            rng = np.random.default_rng(int(np.sum(sample)) + 1)
            hidden1 = (rng.random(12) * 1.5 * (0.5 + np.mean(norm))).tolist()
            hidden2 = (rng.random(8) * 1.8 * (0.5 + np.mean(norm))).tolist()
            out_act = [float(np.clip(0.5 + (np.mean(hidden2) - 0.5), 0, 1))]
            activations = [input_act, hidden1, hidden2, out_act]

        sim_payload = {"layer_names": layer_names, "layer_sizes": layer_sizes, "activations": activations, "feature_names": feature_names}
        # JSON dump and escape for embedding
        sim_json = json.dumps(sim_payload).replace("</", "<\\/")  # small safety trick
        
                # ANN simulator HTML + JS (dark friendly)
        ann_simulator_html = f"""
        <div style="font-family: Inter, Roboto, sans-serif;">
          <canvas id="annCanvas" width="1200" height="320" style="width:100%;border-radius:12px;background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));"></canvas>
          <div style="display:flex;gap:12px;align-items:center;margin-top:8px;">
            <button id="pulseBtn" style="padding:8px 12px;border-radius:8px;border:none;background:linear-gradient(90deg,#2aa79b,#66d0bd);color:white;cursor:pointer">Pulse</button>
            <div style="color:var(--muted);font-weight:600">Layers:</div>
            <div id="layersText" style="color:var(--muted)"></div>
          </div>
        </div>
        <script>
        const payload = {sim_json};
        const canvas = document.getElementById("annCanvas");
        const ctx = canvas.getContext("2d");
        let W = canvas.width;
        let H = canvas.height;
        const layers = payload.layer_sizes;
        const acts = payload.activations.map(l => l.map(v => Number(v) || 0));
        document.getElementById('layersText').innerText = payload.layer_names.join(' → ');
        function resizeCanvas() {{
          const styleW = canvas.clientWidth;
          canvas.width = styleW * devicePixelRatio;
          canvas.height = 320 * devicePixelRatio;
          W = canvas.width; H = canvas.height;
          ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
          draw();
        }}
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
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
              const r = 6 + Math.abs(act) * 20;
              ctx.beginPath();
              const grad = ctx.createRadialGradient(colX, y, r*0.2, colX, y, r*1.7);
              grad.addColorStop(0, 'rgba(155,231,213,' + (0.9 * (0.4+act*0.6)) + ')');
              grad.addColorStop(1, 'rgba(42,167,155,' + (0.3 * (0.3+act*0.7)) + ')');
              ctx.fillStyle = grad;
              ctx.shadowColor = 'rgba(42,167,155,0.6)';
              ctx.shadowBlur = 10 * Math.abs(act);
              ctx.arc(colX, y, r, 0, Math.PI*2);
              ctx.fill();
              ctx.shadowBlur = 0;
              ctx.fillStyle = '#071615';
              ctx.font = '10px Inter, Roboto, sans-serif';
              const txt = (Math.round(act*100)/100).toString();
              ctx.fillText(txt, colX - 10, y+4);
            }}
          }}
          ctx.globalAlpha = 0.09;
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
                ctx.strokeStyle = 'rgba(10,40,36,0.08)';
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
          for (let i=0;i<acts.length;i++) {{
            for (let j=0;j<acts[i].length;j++) {{
              acts[i][j] = Math.max(0, Math.min(1, acts[i][j] + 0.02 * Math.sin((t + i*10 + j*3)/6)));
            }}
          }}
          draw(); t++; window.requestAnimationFrame(animate);
        }}
        document.getElementById('pulseBtn').addEventListener('click', ()=> {{
          for (let i=0;i<acts.length;i++) {{
            for (let j=0;j<acts[i].length;j++) {{
              acts[i][j] = Math.min(1, acts[i][j] + 0.25*Math.random());
            }}
          }}
        }});
        animate();
        </script>
        """
        st.components.v1.html(ann_simulator_html, height=420, scrolling=False)

        # Simple contribution bar chart
        st.markdown("<div style='margin-top:12px'/>", unsafe_allow_html=True)
        try:
            # compute simple perturbation contributions if model+scaler available
            if model is not None and scaler is not None:
                base_scaled = scaler.transform(sample)
                base_pred = model.predict(base_scaled)
                if np.asarray(base_pred).ndim == 2:
                    base_val = float(base_pred[0][0])
                else:
                    base_val = float(base_pred[0])
                contributions = []
                for i, fname in enumerate(feature_names):
                    pert = sample.copy()
                    delta = max(abs(sample[0,i]) * 0.1, 1.0)
                    pert[0,i] = pert[0,i] + delta
                    pert_scaled = scaler.transform(pert)
                    p = model.predict(pert_scaled)
                    pval = float(p[0][0]) if np.asarray(p).ndim==2 else float(p[0])
                    contributions.append((pval - base_val) * (100 if base_val <= 1.01 else 1.0))
                y_vis = np.abs(np.array(contributions))
            else:
                arr = np.array(list([sleep, sleep_q, steps, sitting, exercise, calories, water, screen_time, stress, age, bmi, hr]), dtype=float)
                arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
                y_vis = (arr_norm * 100 * (0.5 + np.random.random(len(arr))*0.8))
        except Exception:
            y_vis = np.random.randint(5, 40, size=len(feature_names))

        df_feat = pd.DataFrame({"feature": feature_names, "contribution": y_vis})
        fig = px.bar(df_feat, x="feature", y="contribution", title="Estimated Feature Contribution", labels={"feature":"Feature","contribution":"Est. influence"})
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------
# FOCUS MODE tab (timer + todos stored in DB)
# ---------------------------------------
with tabs[2]:
    st.markdown("<div class='card'><h3 style='color:var(--accent)'>Focus Mode — Timer & Tasks</h3></div>", unsafe_allow_html=True)
    left, right = st.columns([2,1])
    with left:
        st.markdown("To-Do List (saved to DB)")
        new_task = st.text_input("Add a task", key="new_task_input")
        if st.button("Add Task"):
            if new_task.strip():
                insert_todo(new_task.strip())
                safe_rerun()
        # show todos from DB
        todos_df = fetch_todos()
        for _, row in todos_df.iterrows():
            cols = st.columns([0.06, 0.8, 0.14])
            done = cols[0].checkbox("", value=bool(row['done']), key=f"td_{row['id']}")
            if done != bool(row['done']):
                set_todo_done(row['id'], done)
                safe_rerun()
            cols[1].write(row['task'])
            if cols[2].button("Delete", key=f"del_{row['id']}"):
                delete_todo(row['id'])
                safe_rerun()

    with right:
        st.markdown("### Focus Timer")
        if 'timer_on' not in st.session_state:
            st.session_state['timer_on'] = False
        duration = st.number_input("Minutes", min_value=5, max_value=180, value=25, key="focus_duration")
        if not st.session_state['timer_on']:
            if st.button("Start Focus"):
                st.session_state['timer_on'] = True
                st.session_state['timer_end'] = time.time() + duration*60
                safe_rerun()
        else:
            remaining = int(st.session_state['timer_end'] - time.time())
            if remaining <= 0:
                st.success("Session complete! Great job 🎉")
                st.session_state['timer_on'] = False
            else:
                mins = remaining // 60
                secs = remaining % 60
                st.markdown(f"Time left: **{mins:02d}:{secs:02d}**")
                if st.button("Stop session (give up)"):
                    st.session_state['timer_on'] = False
                    safe_rerun()

# ---------------------------------------
# WEEKLY REPORT tab
# ---------------------------------------
with tabs[3]:
    st.markdown("<div class='card'><h3 style='color:var(--accent)'>Weekly Performance</h3></div>", unsafe_allow_html=True)
    hist_df = fetch_history()
    if hist_df.empty:
        st.info("No history yet. Use Daily Check or import CSV in Settings.")
    else:
        st.dataframe(hist_df.head(200))
        last_week = hist_df.head(7)
        if not last_week.empty:
            st.metric("Avg Efficiency", f"{last_week['predicted_efficiency'].mean():.1f}")
            st.metric("Avg Sleep (hrs)", f"{last_week['sleep_hours'].mean():.1f}")
            st.metric("Avg Stress", f"{last_week['stress_level'].mean():.1f}")
        # export
        csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download history CSV", csv, "aura_history.csv", "text/csv")

# ---------------------------------------
# SETTINGS tab (CSV import into DB)
# ---------------------------------------
with tabs[4]:
    st.markdown("<div class='card'><h3 style='color:var(--accent)'>Settings & Data Import</h3></div>", unsafe_allow_html=True)
    st.write("You can import a CSV of real participants to populate the DB. Required columns:")
    st.write("`sleep_hours,sleep_quality,steps,sitting_hours,exercise_mins,calories,water_l,screen_time,stress_level,age,bmi,resting_hr,efficiency_score`")
    uploaded = st.file_uploader("Upload CSV (features + efficiency_score)", type=['csv'])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        required = ['sleep_hours','sleep_quality','steps','sitting_hours','exercise_mins','calories','water_l','screen_time','stress_level','age','bmi','resting_hr','efficiency_score']
        if not all(col in df_up.columns for col in required):
            st.error("CSV is missing required columns. See the list above.")
        else:
            count = 0
            for _, r in df_up.iterrows():
                entry = {
                    'sleep_hours': float(r['sleep_hours']),
                    'sleep_quality': float(r['sleep_quality']),
                    'steps': int(r['steps']),
                    'sitting_hours': float(r['sitting_hours']),
                    'exercise_mins': int(r['exercise_mins']),
                    'calories': int(r['calories']),
                    'water_l': float(r['water_l']),
                    'screen_time': float(r['screen_time']),
                    'stress_level': float(r['stress_level']),
                    'age': int(r['age']),
                    'bmi': float(r['bmi']),
                    'resting_hr': int(r['resting_hr']),
                    'predicted_efficiency': float(r['efficiency_score']),
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                insert_history_row(entry)
                count += 1
            st.success(f"Imported {count} rows into the DB.")
            st.info("Now go to Weekly Report to view the data.")

st.markdown("---")
st.markdown(f"<div style='color:var(--muted)'>AURA — Made by you. DB persistence via aura.db. Model loaded: {('yes' if model is not None else 'no')}</div>", unsafe_allow_html=True)

