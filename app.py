import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
import re
import io
import tempfile
import datetime
from fpdf import FPDF

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Report CMM Best-Fit 3D", layout="wide")

st.title("Report CMM Best-Fit 3D")

# --- FUNZIONE PER CREARE SFERE 3D NEL GRAFICO ---
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

# --- FUNZIONE GENERAZIONE PDF ---
def genera_pdf(df_tabella, fig, nome_file="Report_CMM.csv"):
    pdf = FPDF(orientation="L", unit="mm", format="A4") # A4 Landscape
    
    # Disabilita il salto pagina automatico per forzare tutto su un unico foglio
    pdf.set_auto_page_break(auto=False, margin=0)
    pdf.add_page()
    
    # Titolo principale
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Report CMM Best-Fit 3D", align="C", new_x="LMARGIN", new_y="NEXT")
    
    # Sottotitolo con il nome file reale e la data odierna
    data_oggi = datetime.datetime.now().strftime("%d/%m/%Y")
    pdf.set_font("helvetica", "I", 10)
    pdf.cell(0, 6, f"File: {nome_file}  |  Data: {data_oggi}", align="C", new_x="LMARGIN", new_y="NEXT")

    # 1. Configurazione Telecamera per vista Z+ (dall'alto)
    fig_top = go.Figure(fig)
    fig_top.update_layout(
        scene_camera=dict(
            eye=dict(x=0, y=0.01, z=2.5), 
            up=dict(x=0, y=1, z=0)
        ),
        title="Vista dall'alto (Asse Z+)",
        showlegend=False
    )
    
    # Esportazione grafico in immagine PNG
    img_bytes = fig_top.to_image(format="png", width=1000, height=1000, scale=1)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(img_bytes)
        tmp_path = tmp_file.name

    # Inserisce l'immagine nella metà di sinistra del PDF
    pdf.image(tmp_path, x=10, y=32, w=130)

    # 2. Configurazione Tabella Autoadattiva
    start_y = 32
    left_table_margin = 145
    pdf.set_xy(left_table_margin, start_y)
    
    num_righe = len(df_tabella)
    spazio_disponibile_y = 170.0  
    
    altezza_riga = min(6.0, spazio_disponibile_y / (num_righe + 1))
    dim_font = max(4.0, min(8.0, altezza_riga * 1.3)) 
    
    headers = ["Pt.", "Tg X", "Tg Y", "Tg Z", "Rl X", "Rl Y", "Rl Z", "Err 3D", "Stato"]
    col_widths = [10, 15, 15, 15, 15, 15, 15, 16, 14]

    # Intestazioni Tabella
    pdf.set_font("helvetica", "B", dim_font)
    pdf.set_fill_color(230, 230, 230)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], altezza_riga, h, border=1, align="C", fill=True)
    pdf.ln()

    # Dati Tabella con sfondi colorati in base allo stato
    pdf.set_font("helvetica", "", dim_font)
    for _, row in df_tabella.iterrows():
        pdf.set_x(left_table_margin)
        
        is_ok = "OK" in str(row["Stato"])
        if is_ok:
            pdf.set_fill_color(225, 245, 225)  # Verde chiaro pastello
            stato_txt = "OK"
        else:
            pdf.set_fill_color(255, 230, 230)  # Rosso chiaro pastello
            stato_txt = "KO"

        pdf.cell(col_widths[0], altezza_riga, str(int(row["Punto"])), border=1, align="C", fill=True)
        pdf.cell(col_widths[1], altezza_riga, f"{row['Target_X']:.3f}", border=1, align="C", fill=True)
        pdf.cell(col_widths[2], altezza_riga, f"{row['Target_Y']:.3f}", border=1, align="C", fill=True)
        pdf.cell(col_widths[3], altezza_riga, f"{row['Target_Z']:.3f}", border=1, align="C", fill=True)
        pdf.cell(col_widths[4], altezza_riga, f"{row['Real_X']:.3f}", border=1, align="C", fill=True)
        pdf.cell(col_widths[5], altezza_riga, f"{row['Real_Y']:.3f}", border=1, align="C", fill=True)
        pdf.cell(col_widths[6], altezza_riga, f"{row['Real_Z']:.3f}", border=1, align="C", fill=True)
        pdf.cell(col_widths[7], altezza_riga, f"{row['Errore_3D (mm)']:.3f}", border=1, align="C", fill=True)
        
        if is_ok:
            pdf.set_text_color(0, 120, 0)
        else:
            pdf.set_text_color(180, 0, 0)
            
        pdf.cell(col_widths[8], altezza_riga, stato_txt, border=1, align="C", fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln()

    return bytes(pdf.output())


# --- INTERFACCIA UTENTE (UI) ---
uploaded_file = st.file_uploader("Carica il file dei dati CMM", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success(f"File '{uploaded_file.name}' caricato con successo!")

    # --- NORMALIZZAZIONE INTELLIGENTE DELLE COLONNE ---
    rename_mapping = {}
    for col in df.columns:
        c_clean = col.strip().lower()
        if c_clean in ['pt', 'punto', 'point']:
            rename_mapping[col] = 'Punto'
        elif c_clean in ['tg x', 'target_x', 'x_target']:
            rename_mapping[col] = 'Target_X'
        elif c_clean in ['tg y', 'ty', 'target_y', 'y_target']:
            rename_mapping[col] = 'Target_Y'
        elif c_clean in ['tg z', 'tz', 'target_z', 'z_target']:
            rename_mapping[col] = 'Target_Z'
        elif c_clean in ['rix', 'rl x', 'real_x', 'x_real']:
            rename_mapping[col] = 'Real_X'
        elif c_clean in ['riy', 'rl y', 'real_y', 'y_real']:
            rename_mapping[col] = 'Real_Y'
        elif c_clean in ['riz', 'rl z', 'real_z', 'z_real']:
            rename_mapping[col] = 'Real_Z'

    df = df.rename(columns=rename_mapping)

    if 'Punto' not in df.columns:
        df['Punto'] = range(1, len(df) + 1)

    # Sidebar per la tolleranza
    st.sidebar.header("Parametri di Controllo")
    tolleranza = st.sidebar.number_input("Tolleranza Errore 3D (mm)", value=0.5, step=0.05)

    # Calcoli geometrici errori
    target_pts = df[['Target_X', 'Target_Y', 'Target_Z']].values
    real_pts = df[['Real_X', 'Real_Y', 'Real_Z']].values
    errori_3d = np.linalg.norm(target_pts - real_pts, axis=1)

    # Costruzione Grafico 3D
    fig = go.Figure()

    for pt in target_pts:
        fig.add_trace(create_sphere_mesh(pt[0], pt[1], pt[2], radius=0.2, color='blue'))

    fig.add_trace(go.Scatter3d(
        x=real_pts[:, 0], y=real_pts[:, 1], z=real_pts[:, 2],
        mode='markers',
        marker=dict(size=4, color='red'),
        name='Punti Reali'
    ))

    for t_pt, r_pt in zip(target_pts, real_pts):
        fig.add_trace(go.Scatter3d(
            x=[t_pt[0], r_pt[0]], y=[t_pt[1], r_pt[1]], z=[t_pt[2], r_pt[2]],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(xaxis_title='Asse X (mm)', yaxis_title='Asse Y (mm)', zaxis_title='Asse Z (mm)'),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- TABELLA DATI ---
    st.markdown("---")
    st.subheader("📋 Dettaglio Punti e Scostamenti")
    
    df_tabella = df.copy()
    df_tabella["Errore_3D (mm)"] = errori_3d
    df_tabella["Stato"] = ["✅ OK" if err <= tolleranza else "❌ KO" for err in errori_3d]
    
    st.dataframe(
        df_tabella.style.format({
            "Real_X": "{:.4f}", "Real_Y": "{:.4f}", "Real_Z": "{:.4f}",
            "Target_X": "{:.4f}", "Target_Y": "{:.4f}", "Target_Z": "{:.4f}",
            "Errore_3D (mm)": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # --- ZONA ESPORTAZIONE PDF ---
    st.markdown("---")
    st.subheader("📥 Esporta Report")
    
    if st.button("🔧 Prepara PDF (Immagine + Dati)"):
        with st.spinner("Generazione PDF in corso..."):
            st.session_state.pdf_data = genera_pdf(df_tabella, fig, nome_file=uploaded_file.name)
            
    if "pdf_data" in st.session_state:
        st.download_button(
            label="⬇️ Scarica il Report in formato PDF",
            data=st.session_state.pdf_data,
            file_name=f"Report_{uploaded_file.name}.pdf",
            mime="application/pdf",
            type="primary"
        )
