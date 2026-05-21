import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Importar módulos del proyecto
sys.path.insert(0, str(Path(__file__).parent))

from src.ui_styles import setup_page_style, COLORS, render_header, render_metric_card
from src.data_manager import ECGDataManager
from src.signal_processor import ECGProcessor
from src.feature_extractor import FeatureExtractor
from src.report_generator import ReportGenerator

# NUEVA IMPORTACIÓN: Funciones avanzadas de visualización
# NUEVA IMPORTACIÓN: Funciones avanzadas de visualización
try:
    from src.advanced_visuals import (
        plot_annotated_heartbeat,
        render_diagnosis_panel,
        plot_patient_comparison,
        show_metrics_comparison,
        DIAGNOSIS_CONFIG,
        show_metric_tooltip
    )

    ADVANCED_VISUALS_AVAILABLE = True

except ImportError as e:
    ADVANCED_VISUALS_AVAILABLE = False
    print(f"⚠️ advanced_visuals no disponible: {e}")
# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE SESIÓN
# ═══════════════════════════════════════════════════════════════════════════

def initialize_session():
    """Inicializa variables de sesión"""
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = ECGDataManager()
    if 'current_ecg' not in st.session_state:
        st.session_state.current_ecg = None
    if 'current_features' not in st.session_state:
        st.session_state.current_features = None
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    # NUEVO: Almacenamiento para comparación
    if 'comparison_patient' not in st.session_state:
        st.session_state.comparison_patient = None

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONES DE GRÁFICOS (existentes, sin cambios)
# ═══════════════════════════════════════════════════════════════════════════

def plot_ecg_signal(ecg_original, ecg_filtered, r_peaks, fs=250):
    """Gráfico interactivo del ECG - SIN CAMBIOS"""
    time = np.arange(len(ecg_filtered)) / fs
    
    fig = go.Figure()
    
    # ECG filtrado
    fig.add_trace(go.Scatter(
        x=time, y=ecg_filtered.squeeze(),
        mode='lines',
        name='ECG Filtrado',
        line=dict(color=COLORS['primary'], width=2),
        hovertemplate='<b>Tiempo:</b> %{x:.3f}s<br><b>Amplitud:</b> %{y:.4f}mV<extra></extra>'
    ))
    
    # Picos R
    peaks_time = r_peaks / fs
    peaks_value = ecg_filtered.squeeze()[r_peaks]
    
    fig.add_trace(go.Scatter(
        x=peaks_time, y=peaks_value,
        mode='markers',
        name='Picos R',
        marker=dict(color=COLORS['danger'], size=12, symbol='diamond'),
        hovertemplate='<b>Pico R</b><br>Tiempo: %{x:.3f}s<br>Amplitud: %{y:.4f}mV<extra></extra>'
    ))
    
    fig.update_layout(
        title="<b>📊 Señal ECG Procesada</b>",
        xaxis_title="Tiempo (segundos)",
        yaxis_title="Amplitud (mV)",
        template="plotly_dark",
        hovermode="x unified",
        height=400,
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(size=11, color=COLORS['text_primary']),
        paper_bgcolor=COLORS['dark_bg'],
        plot_bgcolor=COLORS['card_bg']
    )
    
    return fig

def plot_heartbeat_average(heartbeats):
    """Gráfico del latido promedio - SIN CAMBIOS"""
    if heartbeats.shape[0] == 0:
        return None
    
    avg_beat = np.mean(heartbeats, axis=0)
    x = np.arange(len(avg_beat))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=avg_beat,
        fill='tozeroy',
        name='Latido Promedio',
        line=dict(color=COLORS['secondary'], width=3),
        fillcolor=f"rgba({int(COLORS['secondary'][1:3], 16)}, {int(COLORS['secondary'][3:5], 16)}, {int(COLORS['secondary'][5:7], 16)}, 0.3)",
        hovertemplate='Muestra: %{x}<br>Amplitud: %{y:.4f}mV<extra></extra>'
    ))
    
    fig.update_layout(
        title="<b>❤️ Latido Promedio</b>",
        xaxis_title="Muestras",
        yaxis_title="Amplitud (mV)",
        template="plotly_dark",
        height=350,
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(size=11, color=COLORS['text_primary']),
        paper_bgcolor=COLORS['dark_bg'],
        plot_bgcolor=COLORS['card_bg']
    )
    
    return fig

def plot_rr_histogram(r_peaks, fs=250):
    """Histograma de intervalos RR - SIN CAMBIOS"""
    if len(r_peaks) < 2:
        return None
    
    rr_intervals = np.diff(r_peaks) / fs * 1000  # ms
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=rr_intervals,
        nbinsx=30,
        name='Intervalos RR',
        marker_color=COLORS['primary'],
        hovertemplate='<b>RR (ms):</b> %{x:.1f}<br><b>Frecuencia:</b> %{y}<extra></extra>'
    ))
    
    # Media
    rr_mean = np.mean(rr_intervals)
    fig.add_vline(
        x=rr_mean,
        line_dash="dash",
        line_color=COLORS['secondary'],
        annotation_text=f"Media: {rr_mean:.0f}ms",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="<b>📈 Distribución de Intervalos RR</b>",
        xaxis_title="Intervalo RR (ms)",
        yaxis_title="Frecuencia",
        template="plotly_dark",
        height=350,
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(size=11, color=COLORS['text_primary']),
        paper_bgcolor=COLORS['dark_bg'],
        plot_bgcolor=COLORS['card_bg'],
        showlegend=False
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Función principal de la app"""
    
    # Setup
    setup_page_style()
    initialize_session()
    
    # Header
    render_header()
    
    # ───────────────────────────────────────────────────────────────────────
    # SIDEBAR - SELECCIÓN DE PACIENTES (MEJORADO)
    # ───────────────────────────────────────────────────────────────────────
    
    with st.sidebar:
        st.markdown("## 🔍 SELECTOR DE PACIENTES")
        st.markdown("---")
        
        # Estadísticas de BD
        stats = st.session_state.data_manager.get_database_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Total Registros", stats['total_records'])
        with col2:
            st.metric("👥 Edad Promedio", f"{stats['age_mean']:.0f} años")
        
        st.markdown("---")
        
        # MEJORA: Filtros mejorados con grupos de edad
        st.subheader("⚙️ Filtros de Búsqueda")
        
        # Nuevo: Grupo de edad predefinido
        age_groups = st.session_state.data_manager.get_age_groups()
        selected_age_group = st.selectbox(
            "Grupo de edad:",
            ["Personalizado"] + list(age_groups.keys())
        )
        
        # Si selecciona personalizado
        if selected_age_group == "Personalizado":
            col1, col2 = st.columns(2)
            with col1:
                age_min = st.slider("Edad mín:", 0, 100, 20, key="age_min")
            with col2:
                age_max = st.slider("Edad máx:", 0, 100, 80, key="age_max")
        else:
            age_min, age_max = age_groups[selected_age_group]
            st.info(f"📅 Rango seleccionado: {age_min}-{age_max} años")
        
        col1, col2 = st.columns(2)
        with col1:
            sex = st.selectbox("Sexo:", ["Todos", "M", "F"])
        with col2:
            diagnoses = st.session_state.data_manager.get_all_diagnoses()
            diagnosis = st.selectbox("Diagnóstico:", ["Todos"] + diagnoses)
        
        # Botón de búsqueda
        if st.button("🔎 Buscar Pacientes", use_container_width=True):
            sex_filter = None if sex == "Todos" else sex
            diag_filter = None if diagnosis == "Todos" else diagnosis
            
            records = st.session_state.data_manager.get_filtered_records(
                age_range=(age_min, age_max),
                sex=sex_filter,
                diagnosis=diag_filter,
                num_records=10  # Aumentado a 10 para más opciones
            )
            
            if records:
                st.session_state.filtered_records = records
                st.success(f"✓ {len(records)} pacientes encontrados")
            else:
                st.warning("⚠️  No se encontraron pacientes")
        
        st.markdown("---")
        
        # Seleccionar paciente
        if hasattr(st.session_state, 'filtered_records') and st.session_state.filtered_records:
            st.subheader("👤 Seleccionar Paciente")
            
            patient_options = [
                f"Paciente {r['patient_id']} - {r['age']}a {r['sex']} - {r['diagnostic']}"
                for r in st.session_state.filtered_records
            ]
            
            selected = st.selectbox("Paciente:", patient_options)
            
            if selected:
                idx = patient_options.index(selected)
                selected_record = st.session_state.filtered_records[idx]
                
                if st.button("📥 Cargar Señal ECG", use_container_width=True):
                    with st.spinner("Descargando y procesando..."):
                        try:
                            # Descargar señal
                            ecg, fs = st.session_state.data_manager.download_signal(
                                selected_record['filename_hr']
                            )
                            
                            # Procesar
                            processor = ECGProcessor(sampling_rate=fs)
                            results = processor.process_complete(ecg)
                            
                            # Características
                            if results['heartbeats'].shape[0] > 0:
                                extractor = FeatureExtractor(
                                    sampling_rate=fs,
                                    sex=selected_record['sex']
                                )
                                features = extractor.extract_all_features(
                                    results['heartbeats'][0],
                                    int(0.2 * fs)
                                )
                            else:
                                features = {}
                            
                            # Guardar en sesión
                            st.session_state.current_ecg = ecg
                            st.session_state.processing_results = results
                            st.session_state.current_features = features
                            st.session_state.selected_patient = selected_record
                            st.session_state.fs = fs
                            
                            st.success("✓ Señal cargada exitosamente")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
    
    # ───────────────────────────────────────────────────────────────────────
    # ÁREA PRINCIPAL
    # ───────────────────────────────────────────────────────────────────────
    
    if st.session_state.current_ecg is not None:
        
        patient = st.session_state.selected_patient
        results = st.session_state.processing_results
        features = st.session_state.current_features
        fs = st.session_state.fs
        
        # ─── PANEL DE DIAGNÓSTICO (NUEVO)
        if ADVANCED_VISUALS_AVAILABLE:
            diagnosis = patient.get('diagnostic', 'norm')
            render_diagnosis_panel(diagnosis)
        
        # ─── RESUMEN CLÍNICO
        st.markdown("## 📋 RESUMEN CLÍNICO")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "❤️ Frecuencia Cardíaca",
                f"{results['heart_rate']:.0f}",
                "BPM"
            )
        
        with col2:
            st.metric(
                "📊 Picos R",
                len(results['r_peaks']),
                "detectados"
            )
        
        with col3:
            sdnn = results['rr_metrics'].get('sdnn', 0)
            st.metric(
                "📈 SDNN",
                f"{sdnn:.1f}",
                "ms"
            )
        
        with col4:
            rmssd = results['rr_metrics'].get('rmssd', 0)
            st.metric(
                "💓 RMSSD",
                f"{rmssd:.1f}",
                "ms"
            )
        
        st.markdown("---")
        
        # ─── VISUALIZACIONES (MEJORADAS)
        st.markdown("## 📊 GRÁFICOS INTERACTIVOS")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔷 ECG Completo",
            "❤️ Latido Anotado",
            "📈 Intervalos RR",
            "🔬 Análisis ST/QT",
            "📋 Características"
        ])
        
        with tab1:
            fig_ecg = plot_ecg_signal(
                results['ecg_original'],
                results['ecg_filtered'],
                results['r_peaks'],
                fs=fs
            )
            st.plotly_chart(fig_ecg, use_container_width=True)
        
        with tab2:
            # NUEVO: Latido anotado con ondas identificadas
            if ADVANCED_VISUALS_AVAILABLE and results['heartbeats'].shape[0] > 0:
                fig_annotated = plot_annotated_heartbeat(
                    results['heartbeats'][0],
                    int(0.2 * fs),
                    features,
                    fs=fs
                )
                st.plotly_chart(fig_annotated, use_container_width=True)
            else:
                fig_beat = plot_heartbeat_average(results['heartbeats'])
                if fig_beat:
                    st.plotly_chart(fig_beat, use_container_width=True)
        
        with tab3:
            fig_rr = plot_rr_histogram(results['r_peaks'], fs=fs)
            if fig_rr:
                st.plotly_chart(fig_rr, use_container_width=True)
        
        with tab4:
            if results['heartbeats'].shape[0] > 0:
                # Aquí pueden ir gráficos de ST y QT
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 📊 Segmento ST")
                    st.write(f"**Elevación:** {features.get('st_st_elevation_j', 0):.1f} µV")
                    st.write(f"**Estado:** {'⚠️ Elevado' if features.get('st_st_elevation_flag', 0) else '✓ Normal'}")
                
                with col2:
                    st.markdown("### ⏱️ Intervalo QT")
                    st.write(f"**QT:** {features.get('qt_qt_duration', 0):.1f} ms")
                    st.write(f"**QTc:** {features.get('qt_qtc_duration', 0):.1f} ms")
        
        with tab5:
            st.markdown("### 🔍 CARACTERÍSTICAS ELECTROCARDIOGRÁFICAS")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### QRS")
                st.write(f"**Duración:** {features.get('qrs_qrs_duration', 0):.1f} ms")
                st.write(f"**Amplitud R:** {features.get('qrs_r_amplitude', 0):.3f} mV")
                st.write(f"**Amplitud Q:** {features.get('qrs_q_amplitude', 0):.3f} mV")
                st.write(f"**Q Patológica:** {'🔴 SÍ' if features.get('qrs_pathological_q', 0) else '✓ No'}")
            
            with col2:
                st.markdown("#### ST & T")
                st.write(f"**Elevación ST:** {features.get('st_st_elevation_j', 0):.1f} µV")
                st.write(f"**Depresión:** {'🔴 SÍ' if features.get('st_st_depression_flag', 0) else '✓ No'}")
                st.write(f"**Amplitud T:** {features.get('t_t_amplitude', 0):.3f} mV")
                st.write(f"**Inversión T:** {'⚠️ SÍ' if features.get('t_t_inverted', 0) else '✓ No'}")
            
            with col3:
                st.markdown("#### QT & P")
                st.write(f"**QT:** {features.get('qt_qt_duration', 0):.1f} ms")
                st.write(f"**QTc:** {features.get('qt_qtc_duration', 0):.1f} ms")
                st.write(f"**PR Interval:** {features.get('p_pr_interval', 0):.1f} ms")
                st.write(f"**Amplitud P:** {features.get('p_p_amplitude', 0):.3f} mV")
        
        st.markdown("---")
        
        
        # ─── NUEVA SECCIÓN: COMPARACIÓN DE PACIENTES
        if ADVANCED_VISUALS_AVAILABLE:
            st.markdown("## 🔄 COMPARACIÓN DE PACIENTES")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.checkbox("Comparar con otro paciente"):
                    # Selector para segundo paciente
                    if hasattr(st.session_state, 'filtered_records'):
                        comparison_options = [
                            f"Paciente {r['patient_id']} - {r['age']}a {r['sex']} - {r['diagnostic']}"
                            for r in st.session_state.filtered_records
                        ]
                        
                        if len(comparison_options) > 1:
                            selected_comp = st.selectbox(
                                "Seleccionar segundo paciente:",
                                comparison_options,
                                key="comparison_select"
                            )
                            
                            if st.button("📥 Cargar para Comparación"):
                                try:
                                    idx_comp = comparison_options.index(selected_comp)
                                    comp_record = st.session_state.filtered_records[idx_comp]

                                    # Descargar y procesar
                                    ecg_comp, fs_comp = st.session_state.data_manager.download_signal(
                                        comp_record['filename_hr']
                                    )

                                    processor_comp = ECGProcessor(sampling_rate=fs_comp)
                                    results_comp = processor_comp.process_complete(ecg_comp)

                                    # Extraer características del paciente de comparación
                                    if results_comp['heartbeats'].shape[0] > 0:

                                        extractor_comp = FeatureExtractor(
                                            sampling_rate=fs_comp,
                                            sex=comp_record['sex']
                                        )

                                        features_comp = extractor_comp.extract_all_features(
                                            results_comp['heartbeats'][0],
                                            int(0.2 * fs_comp)
                                        )

                                    else:
                                        features_comp = {}

                                    # Guardar para comparación
                                    st.session_state.comparison_patient = {
                                        'ecg': ecg_comp,
                                        'results': results_comp,
                                        'record': comp_record,
                                        'features': features_comp,
                                        'fs': fs_comp
                                    }

                                    st.success("✓ Paciente cargado para comparación")
                                    st.rerun()

                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
            
            # Mostrar comparación si existe
            if st.session_state.comparison_patient is not None:
                comp = st.session_state.comparison_patient
                
                st.markdown(f"""
                **Paciente 1:** {patient['patient_id']} ({patient['age']}a {patient['sex']})
                **Paciente 2:** {comp['record']['patient_id']} ({comp['record']['age']}a {comp['record']['sex']})
                """)
                
                # Gráfico comparativo
                fig_comp = plot_patient_comparison(
                    results['ecg_filtered'],
                    comp['results']['ecg_filtered'],
                    f"Pac. {patient['patient_id']} ({patient['diagnostic']})",
                    f"Pac. {comp['record']['patient_id']} ({comp['record']['diagnostic']})",
                    results['r_peaks'],
                    comp['results']['r_peaks'],
                    fs=fs
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Tabla comparativa
                st.markdown("### 📊 Comparación de Métricas")
                comp_metrics1 = {
                    'hr': results['heart_rate'],
                    'rr_metrics': results['rr_metrics'],
                    **features
                }
                comp_metrics2 = {
                    'hr': comp['results']['heart_rate'],
                    'rr_metrics': comp['results']['rr_metrics'],
                    **comp['features']
                }
                    
                show_metrics_comparison(
                    comp_metrics1,
                    comp_metrics2,
                    (f"Pac. {patient['patient_id']}", f"Pac. {comp['record']['patient_id']}")
                )
        
        st.markdown("---")
        
        # ─── REPORTE CLÍNICO
        st.markdown("## 📄 REPORTE CLÍNICO")
        
        reporter = ReportGenerator(
            patient_id=str(patient['patient_id']),
            sex=patient['sex'],
            age=int(patient['age'])
        )
        
        report_text = reporter.generate_clinical_report(
            results,
            features,
            diagnosis=patient['diagnostic']
        )
        
        st.text_area("Reporte ECG:", value=report_text, height=400, disabled=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Descargar Reporte (.txt)",
                data=report_text,
                file_name=f"ecg_reporte_{patient['patient_id']}.txt",
                mime="text/plain"
            )
        
        
    
    else:
        # Si no hay datos cargados
        st.info(
            "👈 **Usa el panel lateral para:**\n\n"
            "1. 🔍 Buscar pacientes por edad, sexo y diagnóstico\n"
            "2. 📥 Cargar la señal ECG\n\n"
            "El sistema descargará automáticamente los datos de PhysioNet PTB-XL"
        )
        
        st.markdown("---")
        
        st.markdown("""
        ### 🏥 Acerca de ECG Analyzer Pro
        
        **Sistema profesional de análisis electrocardiográfico** basado en IA.
        
        #### ✨ Características:
        - ✓ Análisis automático de señales ECG
        - ✓ Detección inteligente de arritmias
        - ✓ Identificación de patrones de infarto
        - ✓ Cálculo de métricas HRV
        - ✓ Generación de reportes clínicos
        - ✓ Interfaz profesional y moderna
        - ✓ Comparación entre pacientes
        - ✓ Visualización anotada de latidos
        
        #### 📊 Tecnología:
        - Python, Streamlit, Plotly
        - Procesamiento digital de señales (SciPy, NumPy)
        - Base de datos: PhysioNet PTB-XL
        
        #### ⚠️ Descargo de Responsabilidad:
        *Este sistema es **informativo** y **NO constituye diagnóstico médico**.
        Los resultados deben ser validados por un cardiólogo certificado.*
        """)

if __name__ == "__main__":
    main()