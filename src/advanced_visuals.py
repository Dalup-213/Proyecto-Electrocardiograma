# src/advanced_visuals.py
"""
Funciones avanzadas de visualización para ECG Analyzer
EXTENSIONES para mejorar interfaz sin romper el flujo principal
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
from typing import Dict, Tuple
from src.ui_styles import COLORS


# ═══════════════════════════════════════════════════════════════════════════
# MAPEO DE DIAGNÓSTICOS
# ═══════════════════════════════════════════════════════════════════════════

DIAGNOSIS_CONFIG = {
    'norm': {
        'emoji': '✓',
        'color': COLORS['success'],
        'estado': 'NORMAL',
        'explicacion': 'Registro electrocardiográfico dentro de los parámetros normales. No se detectaron hallazgos patológicos significativos.',
        'riesgo': 'BAJO',
        'alertas': []
    },
    'MI': {
        'emoji': '⚠️',
        'color': COLORS['danger'],
        'estado': 'INFARTO MIOCÁRDICO',
        'explicacion': 'Cambios indicativos de infarto miocárdico. Puede ser agudo o antiguo según patrón de onda Q y cambios ST-T.',
        'riesgo': 'CRÍTICO',
        'alertas': [
            '🔴 Onda Q patológica detectada',
            '🔴 Cambios ST-T consistentes con infarto',
            '⚠️ Requiere evaluación cardíaca urgente'
        ]
    },
    'STTC': {
        'emoji': '⚠️',
        'color': COLORS['warning'],
        'estado': 'CAMBIOS ST-T',
        'explicacion': 'Cambios no específicos en el segmento ST y onda T. Pueden asociarse a isquemia, hipertrofia o variantes normales.',
        'riesgo': 'MODERADO',
        'alertas': [
            '⚠️ Cambios ST-T detectados',
            '⚠️ Requiere correlación clínica',
            '• Considerar troponina y ecocardiografía'
        ]
    },
    'CD': {
        'emoji': '⚠️',
        'color': COLORS['warning'],
        'estado': 'ENFERMEDAD CARDÍACA',
        'explicacion': 'Evidencia de enfermedad cardíaca estructural o funcional. Pueden observarse cambios de voltaje, duración o morfología.',
        'riesgo': 'ALTO',
        'alertas': [
            '⚠️ Hallazgos de enfermedad cardíaca',
            '⚠️ Evaluación cardiológica recomendada',
            '• Monitoreo electrocardiográfico continuo'
        ]
    },
    'HYP': {
        'emoji': '⚠️',
        'color': COLORS['warning'],
        'estado': 'HIPERTROFIA',
        'explicacion': 'Criterios de hipertrofia ventricular izquierda. Indicador de aumento de masa miocárdica, frecuentemente por hipertensión crónica.',
        'riesgo': 'MODERADO',
        'alertas': [
            '⚠️ Voltajes aumentados compatibles con HVI',
            '• Control de presión arterial recomendado',
            '• Ecocardiografía para confirmación'
        ]
    },
    'AFIB': {
        'emoji': '🔴',
        'color': COLORS['danger'],
        'estado': 'FIBRILACIÓN AURICULAR',
        'explicacion': 'Ritmo auricular irregular. Existe pérdida de la contracción auricular coordinada con ondas P irregulares o ausentes.',
        'riesgo': 'ALTO',
        'alertas': [
            '🔴 Ausencia de onda P regular',
            '🔴 Intervalos RR irregulares',
            '⚠️ Riesgo aumentado de accidente cerebrovascular',
            '• Considerar anticoagulación'
        ]
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIÓN: Gráfico de latido anotado
# ═══════════════════════════════════════════════════════════════════════════

def plot_annotated_heartbeat(heartbeat: np.ndarray, 
                             r_peak_idx: int,
                             features: Dict,
                             fs: int = 250) -> go.Figure:
    """
    Crea gráfico de latido promedio con anotaciones de ondas y segmentos
    
    Args:
        heartbeat: Señal del latido
        r_peak_idx: Índice del pico R
        features: Características extraídas
        fs: Frecuencia de muestreo
    
    Returns:
        Figura plotly anotada
    """
    hb = heartbeat.squeeze()
    time_ms = np.arange(len(hb)) / fs * 1000
    
    fig = go.Figure()
    
    # Latido completo
    fig.add_trace(go.Scatter(
        x=time_ms, y=hb,
        mode='lines',
        name='Latido',
        line=dict(color=COLORS['primary'], width=3),
        hovertemplate='<b>Tiempo:</b> %{x:.1f}ms<br><b>Amplitud:</b> %{y:.4f}mV<extra></extra>'
    ))
    
    # Pico R (rojo grande)
    r_time = time_ms[r_peak_idx]
    r_value = hb[r_peak_idx]
    fig.add_trace(go.Scatter(
        x=[r_time], y=[r_value],
        mode='markers+text',
        name='Pico R',
        marker=dict(size=15, color=COLORS['danger'], symbol='diamond'),
        text=['<b>R</b>'],
        textposition='top center',
        hovertemplate='<b>PICO R</b><br>Tiempo: %{x:.1f}ms<br>Amplitud: %{y:.4f}mV<extra></extra>'
    ))
    
    # Onda Q (si existe)
    q_idx = int(features.get('qt_q_idx', r_peak_idx - 20))
    if q_idx > 0:
        q_time = time_ms[q_idx]
        q_value = hb[q_idx]
        fig.add_trace(go.Scatter(
            x=[q_time], y=[q_value],
            mode='markers+text',
            name='Onda Q',
            marker=dict(size=10, color=COLORS['secondary'], symbol='diamond'),
            text=['<b>Q</b>'],
            textposition='bottom center',
            hovertemplate='<b>ONDA Q</b><br>Duración: %{x:.1f}ms<extra></extra>'
        ))
    
    # Onda T (si existe)
    t_idx = int(features.get('t_peak_idx', min(r_peak_idx + 60, len(hb) - 1)))
    if t_idx < len(hb):
        t_time = time_ms[t_idx]
        t_value = hb[t_idx]
        fig.add_trace(go.Scatter(
            x=[t_time], y=[t_value],
            mode='markers+text',
            name='Onda T',
            marker=dict(size=10, color=COLORS['warning'], symbol='diamond'),
            text=['<b>T</b>'],
            textposition='top center',
            hovertemplate='<b>ONDA T</b><br>Amplitud: %{y:.4f}mV<extra></extra>'
        ))
    
    # Segmento ST resaltado
    j_point_offset = int(0.04 * fs)
    j_idx = min(r_peak_idx + j_point_offset, len(hb) - 1)
    st_end_idx = min(r_peak_idx + int(0.12 * fs), len(hb) - 1)
    
    if st_end_idx > j_idx:
        st_time = time_ms[j_idx:st_end_idx]
        st_signal = hb[j_idx:st_end_idx]
        
        fig.add_trace(go.Scatter(
            x=st_time, y=st_signal,
            fill='tozeroy',
            fillcolor='rgba(255, 127, 0, 0.15)',
            line=dict(color='rgba(255, 127, 0, 0)', width=0),
            name='Segmento ST',
            hoverinfo='skip'
        ))
        
        # Etiqueta ST
        st_mid = int((j_idx + st_end_idx) / 2)
        fig.add_annotation(
            x=time_ms[st_mid],
            y=hb[st_mid],
            text='<b>ST</b>',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=COLORS['warning'],
            ax=30,
            ay=-40,
            bgcolor=COLORS['card_bg'],
            bordercolor=COLORS['warning'],
            borderwidth=1,
            font=dict(color=COLORS['warning'], size=11)
        )
    
    # Intervalo QT resaltado (líneas verticales)
    q_idx_int = int(features.get('qt_q_idx', r_peak_idx - 20))
    t_idx_int = int(features.get('qt_t_idx', min(r_peak_idx + 100, len(hb) - 1)))
    
    if q_idx_int > 0 and t_idx_int < len(hb):
        # Línea Q
        fig.add_vline(x=time_ms[q_idx_int], line_dash="dash", 
                      line_color=COLORS['secondary'], opacity=0.5)
        # Línea T
        fig.add_vline(x=time_ms[t_idx_int], line_dash="dash",
                      line_color=COLORS['secondary'], opacity=0.5)
        
        qt_duration = features.get('qt_qt_duration', 0)
        fig.add_annotation(
            x=(time_ms[q_idx_int] + time_ms[t_idx_int]) / 2,
            y=np.max(hb) * 0.9,
            text=f'<b>QT: {qt_duration:.0f}ms</b>',
            showarrow=False,
            bgcolor=COLORS['card_bg'],
            bordercolor=COLORS['secondary'],
            borderwidth=2,
            font=dict(color=COLORS['secondary'], size=10)
        )
    
    # Duración QRS
    qrs_duration = features.get('qrs_qrs_duration', 0)
    
    fig.update_layout(
        title=f"<b>❤️ Latido Promedio Anotado</b> | QRS: {qrs_duration:.0f}ms",
        xaxis_title="Tiempo (ms)",
        yaxis_title="Amplitud (mV)",
        template="plotly_dark",
        height=450,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=11, color=COLORS['text_primary']),
        paper_bgcolor=COLORS['dark_bg'],
        plot_bgcolor=COLORS['card_bg'],
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            bgcolor=COLORS['card_bg'],
            bordercolor=COLORS['primary'],
            borderwidth=1,
            x=0.02,
            y=0.98
        )
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIÓN: Panel de información de diagnóstico
# ═══════════════════════════════════════════════════════════════════════════

def render_diagnosis_panel(diagnosis: str):

    config = DIAGNOSIS_CONFIG.get(diagnosis, DIAGNOSIS_CONFIG['norm'])

    color = config['color']

    html_content = f"""
    <div style="
        background: linear-gradient(135deg, {color}20 0%, {color}08 100%);
        border: 2px solid {color};
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 25px;
    ">

    <h2 style="color:{color}; margin-top:0;">
    {config['emoji']} {config['estado']}
    </h2>

    <p style="color:#E0E0E0; font-size:15px;">
    {config['explicacion']}
    </p>

    <p style="
        color:{color};
        font-weight:bold;
        font-size:16px;
    ">
    🎯 NIVEL DE RIESGO: {config['riesgo']}
    </p>

    </div>
    """

    st.markdown(html_content, unsafe_allow_html=True)

    if config['alertas']:
        st.markdown("### ⚠️ Hallazgos Clínicos")

        for alerta in config['alertas']:
            st.warning(alerta)

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIÓN: Comparación de dos pacientes
# ═══════════════════════════════════════════════════════════════════════════

def plot_patient_comparison(ecg1: np.ndarray, 
                            ecg2: np.ndarray,
                            label1: str,
                            label2: str,
                            r_peaks1: np.ndarray,
                            r_peaks2: np.ndarray,
                            fs: int = 250) -> go.Figure:
    """
    Crea gráfico comparativo de dos ECG
    
    Args:
        ecg1: Primera señal ECG
        ecg2: Segunda señal ECG
        label1: Etiqueta primera señal
        label2: Etiqueta segunda señal
        r_peaks1: Picos R primera señal
        r_peaks2: Picos R segunda señal
        fs: Frecuencia de muestreo
    
    Returns:
        Figura plotly comparativa
    """
    time1 = np.arange(len(ecg1)) / fs
    time2 = np.arange(len(ecg2)) / fs
    
    fig = go.Figure()
    
    # ECG 1
    fig.add_trace(go.Scatter(
        x=time1, y=ecg1.squeeze(),
        mode='lines',
        name=f'Paciente 1: {label1}',
        line=dict(color=COLORS['primary'], width=2),
        opacity=0.8,
        hovertemplate='<b>ECG 1:</b> %{y:.4f}mV<extra></extra>'
    ))
    
    # ECG 2
    fig.add_trace(go.Scatter(
        x=time2, y=ecg2.squeeze(),
        mode='lines',
        name=f'Paciente 2: {label2}',
        line=dict(color=COLORS['secondary'], width=2),
        opacity=0.8,
        hovertemplate='<b>ECG 2:</b> %{y:.4f}mV<extra></extra>'
    ))
    
    # Picos R paciente 1
    peaks_time1 = r_peaks1 / fs
    peaks_value1 = ecg1.squeeze()[r_peaks1]
    
    fig.add_trace(go.Scatter(
        x=peaks_time1, y=peaks_value1,
        mode='markers',
        name='Picos R (Pac. 1)',
        marker=dict(color=COLORS['primary'], size=8, symbol='diamond'),
        hovertemplate='<b>Pico R</b><extra></extra>'
    ))
    
    # Picos R paciente 2
    peaks_time2 = r_peaks2 / fs
    peaks_value2 = ecg2.squeeze()[r_peaks2]
    
    fig.add_trace(go.Scatter(
        x=peaks_time2, y=peaks_value2,
        mode='markers',
        name='Picos R (Pac. 2)',
        marker=dict(color=COLORS['secondary'], size=8, symbol='diamond'),
        hovertemplate='<b>Pico R</b><extra></extra>'
    ))
    
    fig.update_layout(
        title="<b>📊 Comparación de ECG de Dos Pacientes</b>",
        xaxis_title="Tiempo (segundos)",
        yaxis_title="Amplitud (mV)",
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(size=11, color=COLORS['text_primary']),
        paper_bgcolor=COLORS['dark_bg'],
        plot_bgcolor=COLORS['card_bg'],
        hovermode='x unified',
        legend=dict(
            bgcolor=COLORS['card_bg'],
            bordercolor=COLORS['primary'],
            borderwidth=1
        )
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIÓN: Tabla comparativa de métricas
# ═══════════════════════════════════════════════════════════════════════════

def show_metrics_comparison(metrics1: Dict, metrics2: Dict, labels: Tuple[str, str]):
    """
    Muestra tabla comparativa de métricas entre dos pacientes
    
    Args:
        metrics1: Métricas paciente 1
        metrics2: Métricas paciente 2
        labels: (label1, label2)
    """
    import pandas as pd
    
    metrics_to_compare = {
        'Frecuencia Cardíaca (BPM)': ('hr', 'hr'),
        'QRS (ms)': ('qrs_qrs_duration', 'qrs_qrs_duration'),
        'QT (ms)': ('qt_qt_duration', 'qt_qt_duration'),
        'QTc (ms)': ('qt_qtc_duration', 'qt_qtc_duration'),
        'SDNN (ms)': ('sdnn', 'sdnn'),
        'RMSSD (ms)': ('rmssd', 'rmssd'),
        'Elevación ST (µV)': ('st_st_elevation_j', 'st_st_elevation_j'),
    }
    
    data = []
    for metric_name, (key1, key2) in metrics_to_compare.items():
        # Extraer valores anidados si es necesario
        val1 = metrics1.get('rr_metrics', {}).get(key1) if 'rr_metrics' in str(metrics1) else metrics1.get(key1, 0)
        val2 = metrics2.get('rr_metrics', {}).get(key2) if 'rr_metrics' in str(metrics2) else metrics2.get(key2, 0)
        
        if isinstance(metrics1.get('rr_metrics'), dict):
            val1 = metrics1['rr_metrics'].get(key1, metrics1.get(key1, 0))
        if isinstance(metrics2.get('rr_metrics'), dict):
            val2 = metrics2['rr_metrics'].get(key2, metrics2.get(key2, 0))
        
        data.append([metric_name, f"{val1:.2f}", f"{val2:.2f}"])
    
        col1 = labels[0]
        col2 = labels[1]

            # Evitar nombres duplicados
        if col1 == col2:
            col2 = f"{col2} (Comparación)"

        df = pd.DataFrame(data, columns=['Métrica', col1, col2])
    
    st.markdown("### 📊 Tabla Comparativa de Métricas")
    st.dataframe(df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIÓN: Tooltips de métricas
# ═══════════════════════════════════════════════════════════════════════════

METRIC_TOOLTIPS = {
    'FC (BPM)': {
        'name': 'Frecuencia Cardíaca',
        'description': 'Número de latidos por minuto',
        'normal': '60-100 BPM',
        'interpretation': 'Valores menores pueden indicar bradicardia; mayores indican taquicardia'
    },
    'QRS (ms)': {
        'name': 'Duración del Complejo QRS',
        'description': 'Tiempo desde el inicio de Q hasta el final de S',
        'normal': '80-120 ms',
        'interpretation': 'Duración prolongada puede indicar bloqueo de rama'
    },
    'QT (ms)': {
        'name': 'Intervalo QT',
        'description': 'Tiempo desde inicio de Q hasta final de T, refleja repolarización ventricular',
        'normal': 'Hombres <440ms, Mujeres <460ms',
        'interpretation': 'QT prolongado aumenta riesgo de arritmias peligrosas'
    },
    'ST (µV)': {
        'name': 'Elevación/Depresión del Segmento ST',
        'description': 'Desplazamiento del segmento ST respecto a línea isoeléctrica',
        'normal': '±50-100 µV',
        'interpretation': 'Elevación significativa sugiere infarto agudo; depresión sugiere isquemia'
    },
    'SDNN (ms)': {
        'name': 'Desv. Estándar de Intervalos RR',
        'description': 'Variabilidad general de la frecuencia cardíaca en período largo',
        'normal': '50-100 ms',
        'interpretation': 'SDNN bajo indica disfunción autonómica y mayor riesgo'
    },
    'RMSSD (ms)': {
        'name': 'Raíz de Media de Diferencias RR Sucesivas',
        'description': 'Variabilidad a corto plazo, refleja actividad parasimpática',
        'normal': '20-100 ms',
        'interpretation': 'RMSSD bajo puede indicar estrés o fatiga'
    }
}

def show_metric_tooltip(metric_key: str):
    """
    Muestra tooltip de una métrica
    
    Args:
        metric_key: Clave de métrica (ej: 'FC (BPM)')
    """
    if metric_key in METRIC_TOOLTIPS:
        info = METRIC_TOOLTIPS[metric_key]
        st.info(f"""
        **{info['name']}**
        
        {info['description']}
        
        📌 **Rango Normal:** {info['normal']}
        
        💡 **Interpretación:** {info['interpretation']}
        """)