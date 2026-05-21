# src/report_generator.py
"""
Generación de reportes médicos profesionales
"""

import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

class ReportGenerator:
    """Genera reportes clínicos profesionales"""
    
    def __init__(self, patient_id: str = "N/A", sex: str = "N/A", age: int = 0):
        """
        Inicializa generador de reportes
        
        Args:
            patient_id: ID del paciente
            sex: Sexo (M/F)
            age: Edad
        """
        self.patient_id = patient_id
        self.sex = sex if sex else "N/A"
        self.age = age if age else 0
        self.timestamp = datetime.now()
    
    def generate_clinical_report(self, 
                                processing_results: Dict,
                                features: Dict,
                                diagnosis: str = "No especificado") -> str:
        """
        Genera reporte clínico completo
        
        Args:
            processing_results: Resultados del procesamiento
            features: Características extraídas
            diagnosis: Diagnóstico
        
        Returns:
            String con reporte formateado
        """
        
        report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    REPORTE ELECTROCARDIOGRÁFICO (ECG)                      ║
║                         Sistema Análisis ECG Pro                           ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─ INFORMACIÓN DEL PACIENTE ──────────────────────────────────────────────────┐
│
│  ID Paciente:           {str(self.patient_id):<45}
│  Sexo:                  {str(self.sex):<45}
│  Edad:                  {str(self.age)} años{'':<39}
│  Fecha/Hora:            {self.timestamp.strftime('%d/%m/%Y %H:%M:%S'):<45}
│
└─────────────────────────────────────────────────────────────────────────────┘

┌─ PARÁMETROS FUNDAMENTALES ──────────────────────────────────────────────────┐
│
│  Frecuencia Cardíaca (FC):        {processing_results.get('heart_rate', 0):.1f} BPM
│  SDNN (Variabilidad RR):          {processing_results.get('rr_metrics', {}).get('sdnn', 0):.2f} ms
│  RMSSD (HRV):                     {processing_results.get('rr_metrics', {}).get('rmssd', 0):.2f} ms
│  Picos R detectados:              {len(processing_results.get('r_peaks', []))} latidos
│  Duración registro:               {len(processing_results.get('ecg_filtered', [])) / 250:.1f} segundos
│
└─────────────────────────────────────────────────────────────────────────────┘

┌─ CARACTERÍSTICAS ELECTROCARDIOGRÁFICAS ──────────────────────────────────────┐
│
│  COMPLEJO QRS:
│  ├─ Duración QRS:                 {features.get('qrs_qrs_duration', 0):.1f} ms
│  ├─ Amplitud R:                   {features.get('qrs_r_amplitude', 0):.3f} mV
│  ├─ Amplitud Q:                   {features.get('qrs_q_amplitude', 0):.3f} mV
│  └─ Onda Q patológica:            {'SÍ ⚠️' if features.get('qrs_pathological_q', 0) else 'No ✓'}
│
│  INTERVALO QT:
│  ├─ QT:                           {features.get('qt_qt_duration', 0):.1f} ms
│  ├─ QTc (corregido):              {features.get('qt_qtc_duration', 0):.1f} ms
│  └─ QT prolongado:                {'SÍ ⚠️' if features.get('qt_qt_prolonged_flag', 0) else 'No ✓'}
│
│  SEGMENTO ST:
│  ├─ Elevación ST (punto J):       {features.get('st_st_elevation_j', 0):.1f} µV
│  ├─ Depresión ST:                 {'SÍ ⚠️' if features.get('st_st_depression_flag', 0) else 'No ✓'}
│  └─ Elevación/Infarto:            {'SÍ ⚠️' if features.get('st_st_elevation_flag', 0) else 'No ✓'}
│
│  ONDAS P Y T:
│  ├─ Amplitud P:                   {features.get('p_p_amplitude', 0):.3f} mV
│  ├─ Intervalo PR:                 {features.get('p_pr_interval', 0):.1f} ms
│  ├─ Amplitud T:                   {features.get('t_t_amplitude', 0):.3f} mV
│  └─ Inversión T:                  {'SÍ ⚠️' if features.get('t_t_inverted', 0) else 'No ✓'}
│
└─────────────────────────────────────────────────────────────────────────────┘

┌─ ANÁLISIS E INTERPRETACIÓN ─────────────────────────────────────────────────┐
│
"""
        
        # Análisis automático
        warnings = []
        normal_flags = []
        
        # Revisar onda Q patológica
        if features.get('qrs_pathological_q', 0):
            warnings.append("⚠️  ONDA Q PATOLÓGICA: Sugiere infarto anterior/previo")
        else:
            normal_flags.append("✓ Onda Q normal")
        
        # Revisar QRS
        if features.get('qrs_qrs_duration', 0) > 120:
            warnings.append("⚠️  QRS PROLONGADO: Posible bloqueo de rama")
        
        # Revisar ST
        if features.get('st_st_elevation_flag', 0):
            threshold = 100 if self.sex == 'M' else 150
            if features.get('st_st_elevation_j', 0) > threshold:
                warnings.append("🔴 ELEVACIÓN ST SIGNIFICATIVA: Sugerente de infarto AGUDO")
            else:
                warnings.append("⚠️  Elevación ST leve")
        
        if features.get('st_st_depression_flag', 0):
            warnings.append("⚠️  DEPRESIÓN ST: Sugerente de isquemia")
        
        # Revisar QT
        if features.get('qt_qt_prolonged_flag', 0):
            warnings.append("⚠️  QT PROLONGADO: Riesgo de arritmias ventriculares")
        
        # Revisar T
        if features.get('t_t_inverted', 0):
            warnings.append("⚠️  INVERSIÓN DE ONDA T: Posible isquemia/infarto")
        
        # Revisar P
        if features.get('p_p_absent', 0):
            warnings.append("⚠️  ONDA P AUSENTE: Posible fibrilación auricular")
        
        # Mostrar hallazgos
        if warnings:
            for warning in warnings:
                report += f"│  {warning}\n"
        else:
            report += f"│  ✓ REGISTRO NORMAL - Sin hallazgos patológicos significativos\n"
        
        report += f"""│
│  Diagnóstico reportado:  {diagnosis}
│
└─────────────────────────────────────────────────────────────────────────────┘

┌─ CONCLUSIONES Y RECOMENDACIONES ───────────────────────────────────────────┐
│
│  El análisis electrocardiográfico ha sido realizado automáticamente por el
│  sistema ECG Analyzer. Los resultados deben ser validados por un
│  cardiólogo experimentado antes de tomar decisiones clínicas.
│
│  Recomendaciones:
│  • En caso de elevación ST o depresión: Evaluar urgentemente
│  • En caso de QT prolongado: Revisar medicamentos, electrolitos
│  • En caso de onda P ausente: ECG de 12 derivaciones completo
│  • Seguimiento según hallazgos y presentación clínica
│
└─────────────────────────────────────────────────────────────────────────────┘

┌─ INFORMACIÓN TÉCNICA ──────────────────────────────────────────────────────┐
│
│  Software:              ECG Analyzer v1.0
│  Frecuencia muestreo:   250 Hz
│  Filtro:                Pasa banda 0.5-40 Hz (Butterworth, orden 4)
│  Método detección R:    Análisis de picos con prominencia
│  
└─────────────────────────────────────────────────────────────────────────────┘

Generado: {datetime.now().strftime('%d/%m/%Y a las %H:%M:%S')}
⚠️  DESCARGO DE RESPONSABILIDAD: Este reporte es INFORMATIVO y NO constituye 
diagnóstico médico. Consulte a un cardiólogo certificado.

"""
        
        return report
    
    def get_risk_classification(self, features: Dict) -> Tuple[str, str]:
        """
        Clasifica nivel de riesgo del paciente
        
        Args:
            features: Características extraídas
        
        Returns:
            (nivel_riesgo, descripción)
        """
        
        risk_score = 0
        
        # Puntuación de riesgo
        if features.get('qrs_pathological_q', 0):
            risk_score += 30
        
        if features.get('st_st_elevation_flag', 0):
            risk_score += 40
        
        if features.get('st_st_depression_flag', 0):
            risk_score += 25
        
        if features.get('qt_qt_prolonged_flag', 0):
            risk_score += 20
        
        if features.get('t_t_inverted', 0):
            risk_score += 15
        
        if features.get('p_p_absent', 0):
            risk_score += 20
        
        if features.get('qrs_qrs_duration', 0) > 120:
            risk_score += 15
        
        # Clasificación
        if risk_score == 0:
            return "BAJO", "Registro normal sin factores de riesgo detectados"
        elif risk_score <= 20:
            return "LEVE", "Hallazgos menores, requiere seguimiento"
        elif risk_score <= 50:
            return "MODERADO", "Cambios significativos, evaluación recomendada"
        elif risk_score <= 80:
            return "ALTO", "Hallazgos concernientes, evaluación urgente"
        else:
            return "CRÍTICO", "Cambios severos, intervención inmediata recomendada"