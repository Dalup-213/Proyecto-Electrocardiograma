# src/ui_styles.py
"""
Estilos y temas modernos para la interfaz
"""

import streamlit as st
from typing import Dict, Tuple

# Paleta de colores médica moderna
COLORS = {
    'primary': '#00A8E8',      # Azul médico
    'secondary': '#00C9A7',    # Verde médico
    'danger': '#E63946',       # Rojo alerta
    'warning': '#F77F00',      # Naranja advertencia
    'success': '#06D6A0',      # Verde éxito
    'dark_bg': "#000F66",      # Fondo oscuro
    'card_bg': '#1A1E2E',      # Fondo tarjeta
    'text_primary': '#E0E0E0', # Texto principal
    'text_secondary': '#A0A0A0', # Texto secundario
    'border': '#2D3748',       # Bordes
}

def setup_page_style():
    """Configura el estilo global de la página"""
    st.set_page_config(
        page_title="🏥 ECG Analyzer Pro",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None
    )
    
    # CSS personalizado
    custom_css = f"""
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        [data-testid="stMainBlockContainer"] {{
            background-color: {COLORS['dark_bg']};
            color: {COLORS['text_primary']};
        }}
        
        [data-testid="stSidebar"] {{
            background-color: {COLORS['card_bg']};
            border-right: 2px solid {COLORS['border']};
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, {COLORS['card_bg']} 0%, {COLORS['primary']}15 100%);
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            border-color: {COLORS['primary']};
            box-shadow: 0 8px 32px {COLORS['primary']}20;
            transform: translateY(-4px);
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 12px;
            letter-spacing: 0.5px;
        }}
        
        .status-normal {{
            background-color: {COLORS['success']}20;
            color: {COLORS['success']};
            border: 1px solid {COLORS['success']};
        }}
        
        .status-warning {{
            background-color: {COLORS['warning']}20;
            color: {COLORS['warning']};
            border: 1px solid {COLORS['warning']};
        }}
        
        .status-danger {{
            background-color: {COLORS['danger']}20;
            color: {COLORS['danger']};
            border: 1px solid {COLORS['danger']};
        }}
        
        h1 {{
            color: {COLORS['primary']};
            text-shadow: 0 0 20px {COLORS['primary']}30;
            margin-bottom: 10px;
        }}
        
        h2, h3 {{
            color: {COLORS['secondary']};
            margin-top: 20px;
        }}
        
        [data-testid="stMetricLabel"] {{
            color: {COLORS['text_secondary']};
        }}
        
        .st-expander {{
            border: 1px solid {COLORS['border']} !important;
            border-radius: 8px;
        }}
        
        hr {{
            border-color: {COLORS['border']};
            margin: 20px 0;
        }}
    </style>
    """
    
    st.markdown(custom_css, unsafe_allow_html=True)

def get_status_color(value: float, thresholds: Dict) -> Tuple[str, str]:
    """
    Retorna color y estado basado en umbrales
    
    Args:
        value: Valor a evaluar
        thresholds: Dict con keys 'normal', 'warning', 'danger'
    
    Returns:
        (color, estado) tuple
    """
    if value <= thresholds.get('normal', 0):
        return COLORS['success'], '✓ NORMAL'
    elif value <= thresholds.get('warning', 0):
        return COLORS['warning'], '⚠️ ADVERTENCIA'
    else:
        return COLORS['danger'], '🔴 CRÍTICO'

def render_metric_card(label: str, value: str, unit: str = "", 
                       status: str = "normal", icon: str = ""):
    """Renderiza una tarjeta de métrica moderna"""
    status_html = f'<span class="status-badge status-{status}">{status.upper()}</span>'
    
    html = f"""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p style="color: {COLORS['text_secondary']}; margin: 0; font-size: 12px; text-transform: uppercase;">
                    {icon} {label}
                </p>
                <p style="color: {COLORS['text_primary']}; margin: 8px 0 0 0; font-size: 24px; font-weight: bold;">
                    {value} <span style="font-size: 14px; color: {COLORS['text_secondary']}">{unit}</span>
                </p>
            </div>
            <div>{status_html}</div>
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

def render_header():
    """Renderiza el encabezado profesional"""
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.markdown("""
        # ❤️ ECG Analyzer Pro
        ### Sistema Inteligente de Análisis Electrocardiográfico
        """)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: right; color: {COLORS['text_secondary']}; padding-top: 20px;">
            <p style="margin: 0; font-size: 12px;">🔬 v1.0 | 🏥 Análisis Médico</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f'<hr style="border-color: {COLORS["border"]}">', unsafe_allow_html=True)