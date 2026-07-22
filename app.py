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

# --- INIZIALIZZAZIONE STATO ---
for k in ['dx', 'dy', 'dz', 'rx', 'ry', 'rz']:
    if k not in st.session_state:
        st.session_state[k] = 0.0

for k in ['en_dx', 'en_dy', 'en_dz', 'en_rx', 'en_ry', 'en_rz']:
    if k not in st.session_state:
        st.session_state[k] = True  # Di default tutti gli assi sono abilitati

if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

# --- CALLBACKS PER I PULSANTI ---
def azzera_slider():
    st.session_state.dx = 0.0
    st.session_state.dy = 0.0
    st.session_state.dz = 0.0
    st.session_state.rx = 0.0
    st.session_state.ry = 0.0
    st.session_state.rz = 0.0

def cb_best_fit(df_local):
    target_pts = df_local[['Target_X', 'Target_Y', 'Target_Z']].values
    raw_real_pts = df_local[['Real_X', 'Real_Y', 'Real_Z']].values
    
    centroid_real = np.mean(raw_real_pts, axis=0)
    centroid_target = np.mean(target_pts, axis=0)
    
    p_centered = raw_real_pts - centroid_real
    q_centered = target_pts - centroid_target
    
    H = p_centered.T @ q_centered
    U, S, Vt = np.linalg.svd(H)
    
    if np.linalg.det(Vt.T @ U.T) < 0:
        Vt[-1, :] *= -1
        
    rot_matrix = Vt.T @ U.T
    
    r = R.from_matrix(rot_matrix)
    rx_full, ry_full, rz_full = r.as_euler('xyz', degrees=True)
    dx_full, dy_full, dz_full = centroid_target - centroid_real
    
    # Applica i risultati solo se l'asse corrispondente è spuntato/abilitato
    st.session_state.dx = float(dx_full) if st.session_state.en_dx else 0.0
    st.session_state.dy = float(dy_full) if st.session_state.en_dy else 0.0
    st.session_state.dz = float(dz_full) if st.session_state.en_dz else 0.0
    st.session_state.rx = float(rx_full) if st.session_state.en_rx else 0.0
    st.session_state.ry = float(ry_full) if st.session_state.en_ry else 0.0
    st.session_state.rz = float(rz_full) if st.session_state.en_rz else 0.0

def cb_reset():
    azzera_slider()

# --- FUNZIONI DI SUPPORTO GEOMETRICO ---
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

# --- CARICAMENTO E CACHING DATI ---
@st.cache_data
def load_data(file_obj):
    file_obj.seek(0)
    ext = file_obj.name.split('.')[-1].lower()
    if ext == 'csv':
        df = pd.read_csv(file_obj)
    elif ext == 'xlsx':
        df = pd.read_excel(file_obj)
    elif ext == 'txt':
        content = file_obj.read().decode('utf-8', errors='ignore')
        df = parse_cmm_txt(content)
    else:
        df = pd.DataFrame()
    return df

# --- GENERAZIONE PDF ---
def genera_pdf(df_tabella, fig, errore_rms, dx, dy, dz, rx, ry, rz, nome_file="Report_CMM"):
    try:
        dx = float(dx) if dx is not None else 0.0
        dy = float(dy) if dy is not None else 0.0
        dz = float(dz) if dz is not None else 0.0
        rx = float(rx) if rx is not None else 0.0
        ry = float(ry) if ry is not None else 0.0
        rz = float(rz) if rz is not None else 0.0
        errore_rms = float(errore_rms) if errore_rms is not None else 0.0
        nome_file = str(nome_file) if nome_file else "Report_CMM"

        pdf = FPDF(orientation="L", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=False, margin=0)
        pdf.add_page()
        
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "Report CMM Best-Fit 3D", align="C", new_x="LMARGIN", new_y="NEXT")
        
        data_oggi = datetime.datetime.now().strftime("%d/%m/%Y")
        pdf.set_font("helvetica", "I", 10)
        pdf.cell(0, 6, f"File: {nome_file}  |  Data: {data_oggi}  |  RMS Globale: {errore_rms:.4f} mm", align="C", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("helvetica", "", 9)
        testo_correzioni = f"Correzioni applicate - Offset (mm): dX={dx:.4f} | dY={dy:.4f} | dZ={dz:.4f}   --   Rotazioni (°): rx={rx:.3f} | ry={ry:.3f} | rz={rz:.3f}"
        pdf.cell(0, 6, testo_correzioni, align="C", new_x="LMARGIN", new_y="NEXT")

        fig_top = go.Figure(fig)
        fig_top.update_layout(
            scene_camera=dict(eye=dict(x=0, y=0.01, z=2.5), up=dict(x=0, y=1, z=0)),
            title="Vista dall'alto (Asse Z+)", showlegend=False
        )
        
        img_bytes = fig_top.to_image(format="png", width=1000, height=1000, scale=1)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(img_bytes)
            tmp_path = tmp_file.name

        pdf.image(tmp_path, x=10, y=38, w=125)

        start_y = 38
        left_table_margin = 140
        pdf.set_xy(left_table_margin, start_y)
        
        num_righe = len(df_tabella)
        spazio_disponibile_y = 160.0  
        altezza_riga = min(6.0, spazio_disponibile_y / (num_righe + 1))
        dim_font = max(4.0, min(8.0, altezza_riga * 1.3)) 
        
        headers = ["Pt.", "Tg X", "Tg Y", "Tg Z", "Rl X", "Rl Y", "Rl Z", "Err 3D", "Stato"]
        col_widths = [8, 16, 16, 16, 16, 16, 16, 15, 12]

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
            pdf.cell(col_widths[1], altezza_riga, f"{row['Target_X']:.4f}", border=1, align="C", fill=True)
            pdf.cell(col_widths[2], altezza_riga, f"{row['Target_Y']:.4f}", border=1, align="C", fill=True)
            pdf.cell(col_widths[3], altezza_riga, f"{row['Target_Z']:.4f}", border=1, align="C", fill=True)
            pdf.cell(col_widths[4], altezza_riga, f"{row['Real_X']:.4f}", border=1, align="C", fill=True)
            pdf.cell(col_widths[5], altezza_riga, f"{row['Real_Y']:.4f}", border=1, align="C", fill=True)
            pdf.cell(col_widths[6], altezza_riga, f"{row['Real_Z']:.4f}", border=1, align="C", fill=True)
            pdf.cell(col_widths[7], altezza_riga, f"{row['Errore_3D (mm)']:.4f}", border=1, align="C", fill=True)
            
            pdf.set_text_color(0, 120, 0) if is_ok else pdf.set_text_color(180, 0, 0)
            pdf.cell(col_widths[8], altezza_riga, "OK" if is_ok else "KO", border=1, align="C", fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln()

        return bytes(pdf.output())
    except Exception as e:
        st.error(f"Errore generazione PDF. Assicurati che 'kaleido' sia installato (`pip install kaleido`). Dettaglio: {e}")
        return None

# --- UI PRINCIPALE ---
uploaded_file = st.file_uploader("Carica il file dei dati CMM", type=["csv", "xlsx", "txt"])

if uploaded_file is not None:
    if st.session_state.last_uploaded_file != uploaded_file.name:
        azzera_slider()
        st.session_state.last_uploaded_file = uploaded_file.name

    df_raw = load_data(uploaded_file)

    if df_raw.empty:
        st.error("Nessun dato trovato nel file. Verifica la struttura del documento.")
    else:
        df = df_raw.copy()

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

        st.success(f"File caricato con successo! Trovati {len(df)} punti.")

        # --- CONTROLLI SIDEBAR ---
        st.sidebar.header("⚙️ Parametri & Tolleranza")
        tolleranza = st.sidebar.number_input("Tolleranza Errore 3D (mm)", value=0.05, step=0.01)

        st.sidebar.markdown("---")
        st.sidebar.header("🎯 Best-Fit & Reset")
        col_b1, col_b2 = st.sidebar.columns(2)
        
        col_b1.button("Esegui Best-Fit", on_click=cb_best_fit, args=(df,))
        col_b2.button("Reset", on_click=cb_reset)

        st.sidebar.markdown("---")
        st.sidebar.header("🎛️ Aggiustamenti & Assi Abilitati")
        
        # Layout con checkbox per abilitare/disabilitare i singoli assi nel Best-Fit e nei controlli
        col_ax1, col_ax2 = st.sidebar.columns(2)
        with col_ax1:
            st.checkbox("Abilita X", key="en_dx")
            st.checkbox("Abilita Y", key="en_dy")
            st.checkbox("Abilita Z", key="en_dz")
        with col_ax2:
            st.checkbox("Abilita Rot A", key="en_rx")
            st.checkbox("Abilita Rot B", key="en_ry")
            st.checkbox("Abilita Rot C", key="en_rz")

        # Se un asse viene disabilitato manualmente dall'utente, blocchiamo il suo slider a 0
        dx = st.sidebar.slider("Delta X (mm)", -20.0, 20.0, key="dx", step=0.0005, format="%.4f", disabled=not st.session_state.en_dx)
        dy = st.sidebar.slider("Delta Y (mm)", -20.0, 20.0, key="dy", step=0.0005, format="%.4f", disabled=not st.session_state.en_dy)
        dz = st.sidebar.slider("Delta Z (mm)", -20.0, 20.0, key="dz", step=0.0005, format="%.4f", disabled=not st.session_state.en_dz)
        rx = st.sidebar.slider("Rotazione A (°)", -45.0, 45.0, key="rx", step=0.001, format="%.3f", disabled=not st.session_state.en_rx)
        ry = st.sidebar.slider("Rotazione B (°)", -45.0, 45.0, key="ry", step=0.001, format="%.3f", disabled=not st.session_state.en_ry)
        rz = st.sidebar.slider("Rotazione C (°)", -45.0, 45.0, key="rz", step=0.001, format="%.3f", disabled=not st.session_state.en_rz)

        # Se l'asse è disabilitato forziamo il valore a zero nei calcoli geometrici
        val_dx = dx if st.session_state.en_dx else 0.0
        val_dy = dy if st.session_state.en_dy else 0.0
        val_dz = dz if st.session_state.en_dz else 0.0
        val_rx = rx if st.session_state.en_rx else 0.0
        val_ry = ry if st.session_state.en_ry else 0.0
        val_rz = rz if st.session_state.en_rz else 0.0

        # --- CALCOLI GEOMETRICI ---
        target_pts = df[['Target_X', 'Target_Y', 'Target_Z']].values
        raw_real_pts = df[['Real_X', 'Real_Y', 'Real_Z']].values

        centroid = np.mean(raw_real_pts, axis=0)
        r = R.from_euler('xyz', [val_rx, val_ry, val_rz], degrees=True)
        
        real_pts = (r.apply(raw_real_pts - centroid) + centroid) + np.array([val_dx, val_dy, val_dz])

        errori_3d = np.linalg.norm(target_pts - real_pts, axis=1)
        errore_rms = np.sqrt(np.mean(errori_3d ** 2))

        st.metric(label="📉 Errore RMS Globale", value=f"{errore_rms:.4f} mm")

        # --- GRAFICO 3D ---
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

        # --- TABELLA ---
        st.markdown("---")
        df_tabella = df.copy()
        df_tabella["Real_X"] = real_pts[:, 0]
        df_tabella["Real_Y"] = real_pts[:, 1]
        df_tabella["Real_Z"] = real_pts[:, 2]
        df_tabella["Errore_3D (mm)"] = errori_3d
        df_tabella["Stato"] = ["✅ OK" if err <= tolleranza else "❌ KO" for err in errori_3d]
        
        st.dataframe(df_tabella.style.format("{:.4f}", subset=["Real_X", "Real_Y", "Real_Z", "Target_X", "Target_Y", "Target_Z", "Errore_3D (mm)"]), use_container_width=True, hide_index=True)

        # --- REPORT PDF ---
        st.markdown("---")
        if st.button("🔧 Prepara PDF"):
            with st.spinner("Generazione PDF in corso..."):
                pdf_bytes = genera_pdf(
                    df_tabella=df_tabella, 
                    fig=fig, 
                    errore_rms=errore_rms, 
                    dx=val_dx, dy=val_dy, dz=val_dz, 
                    rx=val_rx, ry=val_ry, rz=val_rz, 
                    nome_file=uploaded_file.name
                )
                if pdf_bytes:
                    st.session_state.pdf_data = pdf_bytes
                
        if "pdf_data" in st.session_state:
            st.download_button(
                label="⬇️ Scarica PDF", 
                data=st.session_state.pdf_data, 
                file_name=f"Report_{uploaded_file.name}.pdf", 
                mime="application/pdf", 
                type="primary"
            )
