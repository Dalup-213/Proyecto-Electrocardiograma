# src/feature_extractor.py
"""
Extracción de características clínicas del ECG
QT, QTc, QRS, amplitudes, etc.
"""

import numpy as np
from scipy import signal
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

class FeatureExtractor:
    """Extrae características clínicas de ECG"""
    
    def __init__(self, sampling_rate: int = 250, sex: str = 'M'):
        """
        Inicializa extractor de características
        
        Args:
            sampling_rate: Frecuencia de muestreo
            sex: Sexo del paciente ('M' o 'F')
        """
        self.fs = sampling_rate
        self.sex = sex.upper() if sex else 'M'
    
    def extract_qrs_complex(self, heartbeat: np.ndarray, 
                            r_peak_idx: int) -> Dict[str, float]:
        """
        Extrae características del complejo QRS
        
        Args:
            heartbeat: Latido individual
            r_peak_idx: Índice del pico R
        
        Returns:
            Diccionario de características QRS
        """
        hb = heartbeat.squeeze()
        
        # Buscar Q y S
        window_qrs = int(0.08 * self.fs)  # 80ms
        
        # Q: valle antes de R
        q_start = max(0, r_peak_idx - window_qrs)
        q_region = hb[q_start:r_peak_idx]
        q_idx = np.argmin(q_region) + q_start if len(q_region) > 0 else r_peak_idx
        q_value = hb[q_idx] if q_idx < len(hb) else 0
        
        # S: valle después de R
        s_end = min(len(hb), r_peak_idx + window_qrs)
        s_region = hb[r_peak_idx:s_end]
        s_idx = np.argmin(s_region) + r_peak_idx if len(s_region) > 0 else r_peak_idx
        s_value = hb[s_idx] if s_idx < len(hb) else 0
        
        # Duraciones
        qrs_duration = (s_idx - q_idx) / self.fs * 1000  # ms
        q_duration = (r_peak_idx - q_idx) / self.fs * 1000
        
        # Amplitudes
        r_amplitude = hb[r_peak_idx]
        q_amplitude = abs(q_value)
        s_amplitude = abs(s_value)
        
        return {
            'qrs_duration': qrs_duration,
            'qrs_duration_sec': qrs_duration / 1000,
            'q_duration': q_duration,
            'r_amplitude': r_amplitude,
            'q_amplitude': q_amplitude,
            's_amplitude': s_amplitude,
            'qs_ratio': q_amplitude / (r_amplitude + 1e-6),
            'pathological_q': 1 if q_duration > 40 and q_amplitude > 0.25 * r_amplitude else 0
        }
    
    def extract_st_segment(self, heartbeat: np.ndarray, 
                           r_peak_idx: int, 
                           s_idx: int = None) -> Dict[str, float]:
        """
        Analiza el segmento ST
        
        Args:
            heartbeat: Latido individual
            r_peak_idx: Índice del pico R
            s_idx: Índice del punto S
        
        Returns:
            Características del segmento ST
        """
        hb = heartbeat.squeeze()
        
        # Punto J: justo después de S (inicio de ST)
        j_point_offset = int(0.04 * self.fs)  # 40ms después de R
        j_idx = min(r_peak_idx + j_point_offset, len(hb) - 1)
        j_value = hb[j_idx]
        
        # Isoelectric line: promedio antes de onda P
        iso_start = max(0, r_peak_idx - int(0.15 * self.fs))
        iso_end = r_peak_idx - int(0.05 * self.fs)
        isoelectric = np.mean(hb[iso_start:iso_end]) if iso_end > iso_start else 0
        
        # Elevación/Depresión ST en punto J
        st_elevation = (j_value - isoelectric) * 1000  # mV a µV
        
        # Elevación en punto medio del ST (80ms después de J)
        st_mid_idx = min(j_idx + int(0.04 * self.fs), len(hb) - 1)
        st_mid_value = hb[st_mid_idx]
        st_elevation_mid = (st_mid_value - isoelectric) * 1000
        
        # Slope ST
        if st_mid_idx > j_idx:
            st_slope = (st_mid_value - j_value) / ((st_mid_idx - j_idx) / self.fs)
        else:
            st_slope = 0
        
        # Criterios de infarto
        threshold_elevation_male = 0.1  # mV para hombres
        threshold_elevation_female = 0.15  # mV para mujeres
        threshold = threshold_elevation_male if self.sex == 'M' else threshold_elevation_female
        
        elevation_flag = abs(st_elevation) > threshold * 1000
        
        return {
            'st_elevation_j': st_elevation,
            'st_elevation_mid': st_elevation_mid,
            'st_slope': st_slope,
            'st_elevation_flag': int(elevation_flag),
            'st_depression_flag': int(st_elevation < -100),
            'j_point': j_value
        }
    
    def extract_qt_interval(self, heartbeat: np.ndarray, 
                           r_peak_idx: int) -> Dict[str, float]:
        """
        Calcula intervalo QT y QTc (corregido)
        
        Args:
            heartbeat: Latido individual
            r_peak_idx: Índice del pico R
        
        Returns:
            Características QT
        """
        hb = heartbeat.squeeze()
        
        # Buscar inicio Q
        window_back = int(0.1 * self.fs)
        q_start = max(0, r_peak_idx - window_back)
        q_region = hb[q_start:r_peak_idx]
        q_idx = np.argmin(q_region) + q_start if len(q_region) > 0 else r_peak_idx
        
        # Buscar fin T (primero detectar T)
        t_start = r_peak_idx + int(0.1 * self.fs)
        t_end = min(len(hb), r_peak_idx + int(0.4 * self.fs))
        
        if t_end > t_start:
            t_region = hb[t_start:t_end]
            t_peaks, properties = signal.find_peaks(t_region, prominence=0.05 * (np.max(hb) - np.min(hb) + 1e-6))
            
            if len(t_peaks) > 0:
                t_idx = t_peaks[0] + t_start
            else:
                # Si no encuentra pico, busca donde la señal vuelve a cruzar isoelectric
                t_idx = t_end
        else:
            t_idx = t_end
        
        # Duraciones
        qt_duration = (t_idx - q_idx) / self.fs * 1000  # ms
        
        # QTc (Bazett): QTc = QT / sqrt(RR)
        # Asumimos RR medio de 1s (60 BPM) por defecto
        rr_mean = 1.0  # segundos
        qtc_duration = qt_duration / np.sqrt(rr_mean * self.fs / 250)
        
        # Límites normales
        qt_normal_max = 440 if self.sex == 'M' else 460  # ms
        qtc_flag = 1 if qtc_duration > 450 else 0  # ms
        
        return {
            'qt_duration': qt_duration,
            'qtc_duration': qtc_duration,
            'qt_prolonged_flag': int(qt_duration > qt_normal_max),
            'qtc_prolonged_flag': qtc_flag,
            'q_idx': q_idx,
            't_idx': t_idx
        }
    
    def extract_t_wave(self, heartbeat: np.ndarray, 
                       r_peak_idx: int) -> Dict[str, float]:
        """
        Características de la onda T
        
        Args:
            heartbeat: Latido individual
            r_peak_idx: Índice del pico R
        
        Returns:
            Características T
        """
        hb = heartbeat.squeeze()
        
        # Buscar T wave
        t_start = r_peak_idx + int(0.1 * self.fs)
        t_end = min(len(hb), r_peak_idx + int(0.45 * self.fs))
        
        if t_end > t_start:
            t_region = hb[t_start:t_end]
            
            # Amplitud T
            t_max_idx = np.argmax(t_region) + t_start
            t_amplitude = hb[t_max_idx]
            
            # Duración T
            t_duration = (t_end - t_start) / self.fs * 1000
        else:
            t_amplitude = 0
            t_duration = 0
            t_max_idx = r_peak_idx
        
        # Inversión de T (negativa)
        t_inverted = 1 if t_amplitude < -0.05 else 0
        
        return {
            't_amplitude': t_amplitude,
            't_duration': t_duration,
            't_inverted': t_inverted,
            't_peak_idx': t_max_idx
        }
    
    def extract_p_wave(self, heartbeat: np.ndarray, 
                       r_peak_idx: int) -> Dict[str, float]:
        """
        Características de la onda P
        
        Args:
            heartbeat: Latido individual
            r_peak_idx: Índice del pico R
        
        Returns:
            Características P
        """
        hb = heartbeat.squeeze()
        
        # Buscar P antes de Q
        p_start = max(0, r_peak_idx - int(0.15 * self.fs))
        p_end = r_peak_idx - int(0.04 * self.fs)
        
        if p_end > p_start:
            p_region = hb[p_start:p_end]
            
            # Amplitud P
            p_max_idx = np.argmax(p_region) + p_start
            p_amplitude = hb[p_max_idx]
            
            # Duración P
            p_duration = (p_end - p_start) / self.fs * 1000
        else:
            p_amplitude = 0
            p_duration = 0
            p_max_idx = max(0, r_peak_idx - int(0.1 * self.fs))
        
        # PR interval (inicio P a inicio QRS)
        pr_interval = (r_peak_idx - p_start) / self.fs * 1000
        
        return {
            'p_amplitude': p_amplitude,
            'p_duration': p_duration,
            'pr_interval': pr_interval,
            'p_peak_idx': p_max_idx,
            'p_absent': int(p_amplitude < 0.03)
        }
    
    def extract_all_features(self, heartbeat: np.ndarray, 
                            r_peak_idx: int,
                            rr_mean: float = 1.0) -> Dict[str, float]:
        """
        Extrae TODAS las características de un latido
        
        Args:
            heartbeat: Latido individual
            r_peak_idx: Índice del pico R dentro del latido
            rr_mean: Intervalo RR medio para QTc
        
        Returns:
            Diccionario completo de características
        """
        features = {}
        
        # QRS
        qrs_features = self.extract_qrs_complex(heartbeat, r_peak_idx)
        features.update({'qrs_' + k: v for k, v in qrs_features.items()})
        
        # ST
        st_features = self.extract_st_segment(heartbeat, r_peak_idx)
        features.update({'st_' + k: v for k, v in st_features.items()})
        
        # QT
        qt_features = self.extract_qt_interval(heartbeat, r_peak_idx)
        features.update({'qt_' + k: v for k, v in qt_features.items()})
        
        # T
        t_features = self.extract_t_wave(heartbeat, r_peak_idx)
        features.update({'t_' + k: v for k, v in t_features.items()})
        
        # P
        p_features = self.extract_p_wave(heartbeat, r_peak_idx)
        features.update({'p_' + k: v for k, v in p_features.items()})
        
        return features