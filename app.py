import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
import re

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
