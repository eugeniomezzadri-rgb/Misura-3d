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
import plotly.graph_objects as go

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="CMM Best-Fit Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- INIZIALIZZAZIONE VARIABILI DI STATO ---
for var in ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']:
    if var not in st.session_state:
        st.session_state[var] = 0.0

# --- FUNZIONE HELPER: CREA UNA SFERA GEOMETRICA PERFETTAMENTE ROTONDA ---
def create_sphere_mesh(x, y, z, radius=0.4, color='green', resolution=15):
    """Genera una mesh 3D sferica per evitare punti spigolosi o quadrati."""
    phi = np.linspace(0, 2 * np.pi, resolution)
    theta = np.linspace(0, np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)

    x_sphere = radius * np.sin(theta) * np.cos(phi) + x
    y_sphere = radius * np.sin(theta) * np.sin(phi) + y
    z_sphere = radius * np.cos(theta) + z

    return go.Mesh3d(
        x=x_sphere.flatten(), y=y_sphere.flatten(), z=z_sphere.flatten(),
        alphahull=0,
        color=color,
        flatshading=False,  # Attiva lo smoothing delle facce geometriche
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.5, roughness=0.3),
        showlegend=False,
        hoverinfo='skip'
    )

# --- MOTORE MATEMATICO ---
def calcola_bestfit_completo(target, real):
    """Calcola Traslazione e Rotazione (Angoli di Eulero ABC) ottimali."""
    c_target = np.mean(target, axis=0)
    c_real = np.mean(real, axis=0)
    
    t_centrati = target - c_target
    r_centrati = real - c_real
    
    H = np.dot(r_centrati.T, t_centrati)
    U, S, Vt = np.linalg.svd(H)
    R_mat = np.dot(Vt.T, U.T)
    
    if np.linalg.det(R_mat) < 0:
        Vt[2, :] *= -1
        R_mat = np.dot(Vt.T, U.T)
        
    T_vec = c_target - np.dot(R_mat, c_real)
    
    rot = R.from_matrix(R_mat)
    angoli = rot.as_euler('xyz', degrees=True)
    
    return T_vec, angoli

def applica_rototraslazione(punti, tx, ty, tz, rx, ry, rz):
    """Applica gli offset impostati ai punti grezzi."""
    rot = R.from_euler('xyz', [rx, ry, rz], degrees=True)
    matrice_R = rot.as_matrix()
    vettore_T = np.array([tx, ty, tz])
    return np.dot(punti, matrice_R.T) + vettore_T

# --- PARSER AUTOMATICO E BLINDATO DEL LOG CMM ---
def carica_log(file_obj):
    """Legge i dati gestendo in automatico le codifiche dei file di log industriali (UTF-8/UTF-16)."""
    byte_data = file_obj.getvalue()
    
    # Tenta la decodifica intelligente per intercettare i formati ANSI/UTF-16 tipici dei CNC
    try:
        contenuto = byte_data.decode('utf-8')
        if "REAL:" not in contenuto or "TARGET:" not in contenuto:
            contenuto = byte_data.decode('utf-16')
    except Exception:
        contenuto = byte_data.decode('utf-16', errors='ignore')
    
    if "REAL:" in contenuto and "TARGET:" in contenuto:
        pattern_real = r'REAL:\s+X\s+([-]?\d+\.\d+)\s+Y\s+([-]?\d+\.\d+)\s+Z\s+([-]?\d+\.\d+)'
        pattern_target = r'TARGET:\s+X\s+([-]?\d+\.\d+)\s+Y\s+([-]?\d+\.\d+)\s+Z\s+([-]?\d+\.\d+)'
        
        reals = re.findall(pattern_real, contenuto)
        targets = re.findall(pattern_target, contenuto)
        
        punti = []
        for i in range(min(len(reals), len(targets))):
            rx, ry, rz = map(float, reals[i])
            tx, ty, tz = map(float, targets[i])
            punti.append({
                "Punto": i + 1, 
                "Real_X": rx, "Real_Y": ry, "Real_Z": rz,
                "Target_X": tx, "Target_Y": ty, "Target_Z": tz
            })
        
        if not punti:
            st.error("⚠️ Struttura log non riconosciuta. Controlla il formato del file.")
            return pd.DataFrame()
            
        return pd.DataFrame(punti)
    else:
        # Se si tratta di un file CSV classico, riposiziona il puntatore e leggilo normalmente
        file_obj.seek(0)
        return pd.read_csv(file_obj)

# --- CALLBACK PER AZIONI INTERFACCIA ---
def esegui_bestfit_callback():
    target = st.session_state.df[["Target_X", "Target_Y", "Target_Z"]].values
    real = st.session_state.df[["Real_X", "Real_Y", "Real_Z"]].values
    T_vec, angoli = calcola_bestfit_completo(target, real)
    
    st.session_state.tx = float(T_vec[0])
    st.session_state.ty = float(T_vec[1])
    st.session_state.tz = float(T_vec[2])
    st.session_state.rx = float(angoli[0])
    st.session_state.ry = float(angoli[1])
    st.session_state.rz = float(angoli[2])

def resetta_offset():
    for var in ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']:
        st.session_state[var] = 0.0



# --- GENERAZIONE PDF (SINGOLA PAGINA FORZATA) ---
def genera_pdf(df_tabella, fig, nome_file="Dati_Misurazione.csv"):
    pdf = FPDF(orientation="L", unit="mm", format="A4") # A4 Landscape
    
    # 🔴 IL FIX: Disabilita il salto pagina automatico per forzare tutto su un foglio!
    pdf.set_auto_page_break(auto=False, margin=0)
    
    pdf.add_page()
    
    # Titolo
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Report CMM Best-Fit 3D", align="C", new_x="LMARGIN", new_y="NEXT")
    
    # Sottotitolo (Nome File e Data)
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

    # Inserisce l'immagine nella metà di sinistra del PDF (leggermente più in basso per il sottotitolo)
    pdf.image(tmp_path, x=10, y=32, w=130)

    # 2. Configurazione Tabella Autoadattiva
    start_y = 32
    left_table_margin = 145
    pdf.set_xy(left_table_margin, start_y)
    
    # --- CALCOLO DINAMICO DELLE DIMENSIONI ---
    num_righe = len(df_tabella)
    spazio_disponibile_y = 170.0  # Millimetri verticali massimi
    
    # Calcola l'altezza della riga (massimo 6mm, si restringe se ci sono tante righe)
    altezza_riga = min(6.0, spazio_disponibile_y / (num_righe + 1))
    # Calcola la dimensione del font in base all'altezza della riga (massimo font size 8)
    dim_font = max(4.0, min(8.0, altezza_riga * 1.3)) 
    
    headers = ["Pt.", "Tg X", "Tg Y", "Tg Z", "Rl X", "Rl Y", "Rl Z", "Err 3D", "Stato"]
    col_widths = [10, 15, 15, 15, 15, 15, 15, 16, 14]

    # Intestazioni Tabella
    pdf.set_font("helvetica", "B", dim_font)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], altezza_riga, h, border=1, align="C")
    pdf.ln()

    # Dati Tabella
    pdf.set_font("helvetica", "", dim_font)
    for _, row in df_tabella.iterrows():
        pdf.set_x(left_table_margin)
        pdf.cell(col_widths[0], altezza_riga, str(int(row["Punto"])), border=1, align="C")
        pdf.cell(col_widths[1], altezza_riga, f"{row['Target_X']:.3f}", border=1, align="C")
        pdf.cell(col_widths[2], altezza_riga, f"{row['Target_Y']:.3f}", border=1, align="C")
        pdf.cell(col_widths[3], altezza_riga, f"{row['Target_Z']:.3f}", border=1, align="C")
        pdf.cell(col_widths[4], altezza_riga, f"{row['Real_X']:.3f}", border=1, align="C")
        pdf.cell(col_widths[5], altezza_riga, f"{row['Real_Y']:.3f}", border=1, align="C")
        pdf.cell(col_widths[6], altezza_riga, f"{row['Real_Z']:.3f}", border=1, align="C")
        pdf.cell(col_widths[7], altezza_riga, f"{row['Errore_3D (mm)']:.3f}", border=1, align="C")

        # Logica di colore Verde/Rosso
        if "OK" in row["Stato"]:
            pdf.set_text_color(0, 150, 0) # Verde
            stato_txt = "OK"
        else:
            pdf.set_text_color(200, 0, 0) # Rosso
            stato_txt = "KO"
            
        pdf.cell(col_widths[8], altezza_riga, stato_txt, border=1, align="C")
        pdf.set_text_color(0, 0, 0) # Resetta a nero
        pdf.ln()

    return bytes(pdf.output())

# --- INTERFACCIA UTENTE (UI) ---
st.title("📊 CMM Best-Fit 3D Dashboard")

with st.sidebar:
    st.header("1. Carica Dati")
    file_caricato = st.file_uploader("Carica file CMM (Log/CSV)", type=["txt", "log", "csv"])
    
    st.markdown("---")
    st.header("2. Allineamento")
    
    if file_caricato is not None:
        st.button("✨ ESEGUI BEST-FIT AUTOMATICO", on_click=esegui_bestfit_callback, type="primary", use_container_width=True)
        st.button("🔄 Resetta Offset", on_click=resetta_offset, use_container_width=True)
        
        st.markdown("### Offset Manuale (Rototraslazione)")
        st.markdown("**Traslazioni (XYZ in mm)**")
        st.number_input("Tx (Offset X)", step=0.01, format="%.5f", key="tx")
        st.number_input("Ty (Offset Y)", step=0.01, format="%.5f", key="ty")
        st.number_input("Tz (Offset Z)", step=0.01, format="%.5f", key="tz")
        
        st.markdown("**Rotazioni (ABC in Gradi)**")
        st.number_input("Rx (Angolo A)", step=0.01, format="%.5f", key="rx")
        st.number_input("Ry (Angolo B)", step=0.01, format="%.5f", key="ry")
        st.number_input("Rz (Angolo C)", step=0.01, format="%.5f", key="rz")
        
        tolleranza = st.number_input("Tolleranza (mm)", value=0.05, step=0.01)
    else:
        st.warning("Carica un file per abilitare i controlli.")

if file_caricato is not None:
    if 'df' not in st.session_state or st.session_state.get('last_file') != file_caricato.name:
        st.session_state.df = carica_log(file_caricato)
        st.session_state.last_file = file_caricato.name

    df = st.session_state.df
    
    if not df.empty:
        target_pts = df[["Target_X", "Target_Y", "Target_Z"]].values
        real_raw = df[["Real_X", "Real_Y", "Real_Z"]].values
        
        real_aligned = applica_rototraslazione(real_raw, st.session_state.tx, st.session_state.ty, st.session_state.tz, 
                                               st.session_state.rx, st.session_state.ry, st.session_state.rz)
        
        errori_3d = np.linalg.norm(target_pts - real_aligned, axis=1)
        rms_globale = np.sqrt(np.mean(errori_3d**2))
        punti_ko = np.sum(errori_3d > tolleranza)
        punti_ok = len(df) - punti_ko
        
        # Pannello Metriche
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Punti Totali", len(df))
        col2.metric("RMS Globale", f"{rms_globale:.5f} mm")
        col3.metric("Punti in Tolleranza", punti_ok, delta=f"{punti_ok} OK", delta_color="normal")
        col4.metric("Punti KO", punti_ko, delta=f"{punti_ko} KO", delta_color="inverse")
        
        if punti_ko == 0:
            st.success(f"✅ PEZZO CONFORME - Tutti i punti rientrano nella tolleranza di {tolleranza} mm")
        else:
            st.error(f"❌ PEZZO NON CONFORME - {punti_ko} punti fuori tolleranza")
            
        st.markdown("---")
        
        # --- CREAZIONE SCENA 3D AVANZATA ---
        fig = go.Figure()
        
        # Traccia fittizia solo per mostrare la legenda corretta a destra
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                   marker=dict(size=10, color='blue'), name='Target (Teorico)'))
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                   marker=dict(size=10, color='green'), name='Misurato (OK)'))
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                   marker=dict(size=10, color='red'), name='Misurato (KO)'))

        # Generazione Etichette numeriche dei punti
        etichette_punti = [str(i+1) for i in range(len(target_pts))]
        
        # 1. Aggiunta Sfere Target (Blu)
        for pt in target_pts:
            fig.add_trace(create_sphere_mesh(pt[0], pt[1], pt[2], radius=0.3, color='blue'))
            
        # Testo sopra i punti target
        fig.add_trace(go.Scatter3d(
            x=target_pts[:, 0], y=target_pts[:, 1], z=target_pts[:, 2],
            mode='text', text=etichette_punti, textposition='top center',
            textfont=dict(size=13, color='black'), showlegend=False, hoverinfo='skip'
        ))
        
        # 2. Aggiunta Sfere Misurate (Verdi / Rosse) + Dati di Hover avanzati
        testi_hover = [f"<b>Punto {i+1}</b><br>Errore: {err:.4f} mm<br>ΔX: {(r[0]-t[0]):.4f}<br>ΔY: {(r[1]-t[1]):.4f}<br>ΔZ: {(r[2]-t[2]):.4f}" 
                       for i, (err, t, r) in enumerate(zip(errori_3d, target_pts, real_aligned))]
        
        for i, pt in enumerate(real_aligned):
            colore_sfera = 'green' if errori_3d[i] <= tolleranza else 'red'
            sfera_mesh = create_sphere_mesh(pt[0], pt[1], pt[2], radius=0.5, color=colore_sfera)
            sfera_mesh.update(hoverinfo='text', text=testi_hover[i], hovertemplate="%{text}<extra></extra>")
            fig.add_trace(sfera_mesh)
        
        # 3. Linee di Scostamento (Tratteggiate)
        for i in range(len(target_pts)):
            fig.add_trace(go.Scatter3d(
                x=[target_pts[i, 0], real_aligned[i, 0]],
                y=[target_pts[i, 1], real_aligned[i, 1]],
                z=[target_pts[i, 2], real_aligned[i, 2]],
                mode='lines', showlegend=False,
                line=dict(color='darkgray', width=2.5, dash='dot'),
                hoverinfo='skip'
            ))
            
        fig.update_layout(
            title="Visualizzazione Scostamenti 3D Metrologica (Sfere Reali)",
            scene=dict(
                xaxis_title='Asse X (mm)', yaxis_title='Asse Y (mm)', zaxis_title='Asse Z (mm)',
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        # --- TABELLA DATI ---
        st.markdown("---")
        st.subheader("📋 Dettaglio Punti e Scostamenti")
        
        # Creiamo una tabella arricchita con gli errori calcolati in tempo reale
        df_tabella = df.copy()
        df_tabella["Errore_3D (mm)"] = errori_3d
        df_tabella["Stato"] = ["✅ OK" if err <= tolleranza else "❌ KO" for err in errori_3d]
        
        # Mostriamo la tabella interattiva
        st.dataframe(
            df_tabella.style.format({
                "Real_X": "{:.4f}", "Real_Y": "{:.4f}", "Real_Z": "{:.4f}",
                "Target_X": "{:.4f}", "Target_Y": "{:.4f}", "Target_Z": "{:.4f}",
                "Errore_3D (mm)": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )
        # --- ZONA ESPORTAZIONE ---
        st.markdown("---")
        st.subheader("📥 Esporta Report")
        
        # Pulsante preparatorio per non appesantire l'app ricaricando l'immagine a ogni click a vuoto
        if st.button("🔧 Prepara PDF (Immagine + Dati)"):
            with st.spinner("Scatto immagine Z+ e generazione PDF in corso... (potrebbe richiedere qualche secondo)"):
                st.session_state.pdf_data = genera_pdf(df_tabella, fig)
                
        # Mostra il pulsante di download effettivo solo quando il file è pronto in memoria
        if "pdf_data" in st.session_state:
            st.download_button(
                label="⬇️ Scarica il Report in formato PDF",
                data=st.session_state.pdf_data,
                file_name="Report_CMM_ZPlus.pdf",
                mime="application/pdf",
                type="primary"
            )
