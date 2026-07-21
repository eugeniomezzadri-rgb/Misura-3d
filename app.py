import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
import re
import tempfile
import datetime
from fpdf import FPDF

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Report CMM Best-Fit 3D", layout="wide")
st.title("Report CMM Best-Fit 3D")

# --- INIZIALIZZAZIONE DELLO STATO (Prima di qualsiasi widget) ---
for k in ['dx', 'dy', 'dz', 'rx', 'ry', 'rz']:
    if k not in st.session_state:
        st.session_state[k] = 0.0

if 'best_fit_active' not in st.session_state:
    st.session_state.best_fit_active = False

# --- FUNZIONI CALLBACK PER I PULSANTI ---
def azzera_slider():
    st.session_state.dx = 0.0
    st.session_state.dy = 0.0
    st.session_state.dz = 0.0
    st.session_state.rx = 0.0
    st.session_state.ry = 0.0
    st.session_state.rz = 0.0

def cb_best_fit():
    st.session_state.best_fit_active = True
    azzera_slider()

def cb_reset():
    st.session_state.best_fit_active = False
    azzera_slider()

# --- FUNZIONE PER CREARE SFERE 3D ---
def create_sphere_mesh(x, y, z, radius=0.5, color='blue', n_subdiv=10):
    phi = np.linspace(0, np.pi, n_subdiv)
    theta = np.linspace(0, 2 * np.pi, n_subdiv)
    phi, theta = np.meshgrid(phi, theta)
    xs = x + radius * np.sin(phi) * np.cos(theta)
    ys = y + radius * np.sin(phi) * np.sin(theta)
    zs = z + radius * np.cos(phi)
    return go.Surface(
        x=xs, y=ys, z=zs,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        lighting=dict(ambient=0.6, diffuse=0.4),
        hoverinfo='skip'
    )

# --- ALGORITMO BEST-FIT (KABSCH) ---
def best_fit_alignment(real_pts, target_pts):
    centroid_real = np.mean(real_pts, axis=0)
    centroid_target = np.mean(target_pts, axis=0)
    p_centered = real_pts - centroid_real
    q_centered = target_pts - centroid_target
    H = p_centered.T @ q_centered
    U, S, Vt = np.linalg.svd(H)
    if np.linalg.det(Vt.T @ U.T) < 0:
        Vt[-1, :] *= -1
    rot_matrix = Vt.T @ U.T
    return (p_centered @ rot_matrix.T) + centroid_target

# --- PARSER TXT CMM ---
def parse_cmm_txt(content_str):
    blocks = content_str.split("3D POINT PROBING - MEASURING LOG")
    data = []
    for block in blocks:
        real_match = re.search(r'REAL:\s+X\s+([-\d.]+)\s+Y\s+([-\d.]+)\s+Z\s+([-\d.]+)', block)
        target_match = re.search(r'TARGET:\s+X\s+([-\d.]+)\s+Y\s+([-\d.]+)\s+Z\s+([-\d.]+)', block)
        if real_match and target_match:
            data.append({
                "Punto": len(data) + 1,
                "Real_X": float(real_match.group(1)),
                "Real_Y": float(real_match.group(2)),
                "Real_Z": float(real_match.group(3)),
                "Target_X": float(target_match.group(1)),
                "Target_Y": float(target_match.group(2)),
                "Target_Z": float(target_match.group(3))
            })
    return pd.DataFrame(data)

# --- GENERAZIONE PDF ---
def genera_pdf(df_tabella, fig, errore_rms, nome_file="Report_CMM"):
    try:
        pdf = FPDF(orientation="L", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=False, margin=0)
        pdf.add_page()
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "Report CMM Best-Fit 3D", align="C", new_x="LMARGIN", new_y="NEXT")
        
        data_oggi = datetime.datetime.now().strftime("%d/%m/%Y")
        pdf.set_font("helvetica", "I", 10)
        pdf.cell(0, 6, f"File: {nome_file}  |  Data: {data_oggi}  |  RMS Globale: {errore_rms:.4f} mm", align="C", new_x="LMARGIN", new_y="NEXT")

        fig_top = go.Figure(fig)
        fig_top.update_layout(
            scene_camera=dict(eye=dict(x=0, y=0.01, z=2.5), up=dict(x=0, y=1, z=0)),
            title="Vista dall'alto (Asse Z+)", showlegend=False
        )
        
        img_bytes = fig_top.to_image(format="png", width=1000, height=1000, scale=1)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(img_bytes)
            tmp_path = tmp_file.name

        pdf.image(tmp_path, x=10, y=32, w=130)

        start_y = 32
        left_table_margin = 145
        pdf.set_xy(left_table_margin, start_y)
        
        num_righe = len(df_tabella)
        spazio_disponibile_y = 170.0  
        altezza_riga = min(6.0, spazio_disponibile_y / (num_righe + 1))
        dim_font = max(4.0, min(8.0, altezza_riga * 1.3)) 
        
        headers = ["Pt.", "Tg X", "Tg Y", "Tg Z", "Rl X", "Rl Y", "Rl Z", "Err 3D", "Stato"]
        col_widths = [10, 15, 15, 15, 15, 15, 15, 16, 14]

        pdf.set_font("helvetica", "B", dim_font)
        pdf.set_fill_color(230, 230, 230)
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], altezza_riga, h, border=1, align="C", fill=True)
        pdf.ln()

        pdf.set_font("helvetica", "", dim_font)
        for _, row in df_tabella.iterrows():
            pdf.set_x(left_table_margin)
            is_ok = "OK" in str(row["Stato"])
            pdf.set_fill_color(225, 245, 225) if is_ok else pdf.set_fill_color(255, 230, 230)

            pdf.cell(col_widths[0], altezza_riga, str(int(row["Punto"])), border=1, align="C", fill=True)
            pdf.cell(col_widths[1], altezza_riga, f"{row['Target_X']:.3f}", border=1, align="C", fill=True)
            pdf.cell(col_widths[2], altezza_riga, f"{row['Target_Y']:.3f}", border=1, align="C", fill=True)
            pdf.cell(col_widths[3], altezza_riga, f"{row['Target_Z']:.3f}", border=1, align="C", fill=True)
            pdf.cell(col_widths[4], altezza_riga, f"{row['Real_X']:.3f}", border=1, align="C", fill=True)
            pdf.cell(col_widths[5], altezza_riga, f"{row['Real_Y']:.3f}", border=1, align="C", fill=True)
            pdf.cell(col_widths[6], altezza_riga, f"{row['Real_Z']:.3f}", border=1, align="C", fill=True)
            pdf.cell(col_widths[7], altezza_riga, f"{row['Errore_3D (mm)']:.3f}", border=1, align="C", fill=True)
            
            pdf.set_text_color(0, 120, 0) if is_ok else pdf.set_text_color(180, 0, 0)
            pdf.cell(col_widths[8], altezza_riga, "OK" if is_ok else "KO", border=1, align="C", fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln()

        return bytes(pdf.output())
    except Exception as e:
        st.error(f"Errore durante la generazione del PDF. Assicurati che 'kaleido' sia installato (`pip install kaleido`). Dettaglio: {e}")
        return None

# --- CARICAMENTO FILE ---
uploaded_file = st.file_uploader("Carica il file dei dati CMM", type=["csv", "xlsx", "txt"])

if uploaded_file is not None:
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'csv':
        df = pd.read_csv(uploaded_file)
    elif ext == 'xlsx':
        df = pd.read_excel(uploaded_file)
    elif ext == 'txt':
        df = parse_cmm_txt(uploaded_file.read().decode('utf-8', errors='ignore'))

    if df.empty:
        st.error("Nessun dato trovato nel file caricato. Verifica la struttura del file.")
    else:
        st.success(f"File caricato! Trovati {len(df)} punti.")

        # Normalizzazione colonne
        rename_map = {}
        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in ['pt', 'punto', 'point']: rename_map[c] = 'Punto'
            elif cl in ['tg x', 'target_x', 'x_target']: rename_map[c] = 'Target_X'
            elif cl in ['tg y', 'ty', 'target_y', 'y_target']: rename_map[c] = 'Target_Y'
            elif cl in ['tg z', 'tz', 'target_z', 'z_target']: rename_map[c] = 'Target_Z'
            elif cl in ['rix', 'rl x', 'real_x', 'x_real']: rename_map[c] = 'Real_X'
            elif cl in ['riy', 'rl y', 'real_y', 'y_real']: rename_map[c] = 'Real_Y'
            elif cl in ['riz', 'rl z', 'real_z', 'z_real']: rename_map[c] = 'Real_Z'
        
        df = df.rename(columns=rename_map)
        if 'Punto' not in df.columns:
            df['Punto'] = range(1, len(df) + 1)

        # --- SIDEBAR CONTROLLI ---
        st.sidebar.header("⚙️ Parametri & Tolleranza")
        tolleranza = st.sidebar.number_input("Tolleranza Errore 3D (mm)", value=0.05, step=0.01)

        st.sidebar.markdown("---")
        st.sidebar.header("🎯 Best-Fit & Reset")
        col_b1, col_b2 = st.sidebar.columns(2)
        col_b1.button("Esegui Best-Fit", on_click=cb_best_fit)
        col_b2.button("Reset", on_click=cb_reset)

        st.sidebar.markdown("---")
        st.sidebar.header("🎛️ Aggiustamenti Manuali")
        
        # SLIDER LEGATI ESCLUSIVAMENTE ALLA KEY (Senza parametro 'value')
        dx = st.sidebar.slider("Delta X (mm)", -20.0, 20.0, key="dx", step=0.05)
        dy = st.sidebar.slider("Delta Y (mm)", -20.0, 20.0, key="dy", step=0.05)
        dz = st.sidebar.slider("Delta Z (mm)", -20.0, 20.0, key="dz", step=0.05)
        rx = st.sidebar.slider("Rotazione A (°)", -45.0, 45.0, key="rx", step=0.1)
        ry = st.sidebar.slider("Rotazione B (°)", -45.0, 45.0, key="ry", step=0.1)
        rz = st.sidebar.slider("Rotazione C (°)", -45.0, 45.0, key="rz", step=0.1)

        # CALCOLI
        target_pts = df[['Target_X', 'Target_Y', 'Target_Z']].values
        raw_real_pts = df[['Real_X', 'Real_Y', 'Real_Z']].values

        aligned_pts = best_fit_alignment(raw_real_pts, target_pts) if st.session_state.best_fit_active else raw_real_pts.copy()

        centroid = np.mean(aligned_pts, axis=0)
        r = R.from_euler('xyz', [rx, ry, rz], degrees=True)
        real_pts = (r.apply(aligned_pts - centroid) + centroid) + np.array([dx, dy, dz])

        errori_3d = np.linalg.norm(target_pts - real_pts, axis=1)
        errore_rms = np.sqrt(np.mean(errori_3d ** 2))

        st.metric(label="📉 Errore RMS Globale", value=f"{errore_rms:.4f} mm")

        # GRAFICO 3D
        point_colors = ['#2ecc71' if err <= tolleranza else '#e74c3c' for err in errori_3d]
        fig = go.Figure()

        for pt in target_pts:
            fig.add_trace(create_sphere_mesh(pt[0], pt[1], pt[2], radius=0.2, color='blue'))

        fig.add_trace(go.Scatter3d(
            x=real_pts[:, 0], y=real_pts[:, 1], z=real_pts[:, 2],
            mode='markers+text',
            text=[str(int(p)) for p in df['Punto']],
            textposition='top center',
            marker=dict(size=5, color=point_colors),
            name='Punti Reali'
        ))

        for t_pt, r_pt in zip(target_pts, real_pts):
            fig.add_trace(go.Scatter3d(
                x=[t_pt[0], r_pt[0]], y=[t_pt[1], r_pt[1]], z=[t_pt[2], r_pt[2]],
                mode='lines', line=dict(color='gray', width=2), showlegend=False
            ))

        fig.update_layout(
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            margin=dict(l=0, r=0, b=0, t=30), height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        # TABELLA
        st.markdown("---")
        df_tabella = df.copy()
        df_tabella["Real_X"] = real_pts[:, 0]
        df_tabella["Real_Y"] = real_pts[:, 1]
        df_tabella["Real_Z"] = real_pts[:, 2]
        df_tabella["Errore_3D (mm)"] = errori_3d
        df_tabella["Stato"] = ["✅ OK" if err <= tolleranza else "❌ KO" for err in errori_3d]
        
        st.dataframe(df_tabella.style.format("{:.4f}", subset=["Real_X", "Real_Y", "Real_Z", "Target_X", "Target_Y", "Target_Z", "Errore_3D (mm)"]), use_container_width=True, hide_index=True)

        # PDF
        st.markdown("---")
        if st.button("🔧 Prepara PDF"):
            with st.spinner("Generazione PDF..."):
                pdf_bytes = genera_pdf(df_tabella, fig, errore_rms, nome_file=uploaded_file.name)
                if pdf_bytes:
                    st.session_state.pdf_data = pdf_bytes
                
        if "pdf_data" in st.session_state:
            st.download_button("⬇️ Scarica PDF", data=st.session_state.pdf_data, file_name=f"Report_{uploaded_file.name}.pdf", mime="application/pdf", type="primary")
