# src/signal_processor.py
"""
Procesamiento avanzado de señales ECG
Filtrado, detección de picos, segmentación
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt, find_peaks, medfilt
from typing import Tuple, List, Dict
import warnings

warnings.filterwarnings('ignore')

class ECGProcessor:
    """Procesador profesional de señales ECG"""
    
    def __init__(self, sampling_rate: int = 250, verbose: bool = True):
        """
        Inicializa el procesador
        
        Args:
            sampling_rate: Frecuencia de muestreo en Hz
            verbose: Mostrar mensajes de procesamiento
        """
        self.fs = sampling_rate
        self.verbose = verbose
        self.ecg_original = None
        self.ecg_filtered = None
        self.r_peaks = None
        self.heart_rate = None
    
    def _log(self, msg: str):
        """Log condicional"""
        if self.verbose:
            print(f"[ECGProcessor] {msg}")
    
    def bandpass_filter(self, ecg: np.ndarray, 
                       low_freq: float = 0.5, 
                       high_freq: float = 40.0,
                       order: int = 4) -> np.ndarray:
        """
        Filtro pasa banda IIR (Butterworth)
        
        Args:
            ecg: Señal ECG
            low_freq: Frecuencia baja (Hz)
            high_freq: Frecuencia alta (Hz)
            order: Orden del filtro
        
        Returns:
            Señal filtrada
        """
        self._log(f"Aplicando filtro pasa banda {low_freq}-{high_freq} Hz")
        
        # Normalizar frecuencias
        nyquist = self.fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Asegurar rangos válidos
        low = np.clip(low, 0.001, 0.999)
        high = np.clip(high, 0.001, 0.999)
        
        if low >= high:
            low, high = low / 2, high * 0.8
        
        # Crear filtro
        b, a = butter(order, [low, high], btype='band')
        
        # Aplicar filtro bidireccional (sin desfase)
        filtered = filtfilt(b, a, ecg.squeeze())
        
        return filtered
    
    def remove_baseline_wander(self, ecg: np.ndarray, window_ms: int = 200) -> np.ndarray:
        """
        Elimina el error de línea base (drift)
        
        Args:
            ecg: Señal ECG
            window_ms: Tamaño de ventana en ms
        
        Returns:
            Señal sin línea base
        """
        self._log(f"Eliminando wandering de línea base")
        
        window_samples = int(self.fs * window_ms / 1000)
        if window_samples % 2 == 0:
            window_samples += 1
        
        # Mediana móvil para detectar tendencia
        baseline = medfilt(ecg.squeeze(), kernel_size=window_samples)
        
        return ecg.squeeze() - baseline
    
    def detect_r_peaks(self, ecg: np.ndarray, 
                       prominence: float = 0.3) -> np.ndarray:
        """
        Detecta los picos R usando análisis de picos
        
        Args:
            ecg: Señal ECG filtrada
            prominence: Prominencia mínima relativa
        
        Returns:
            Índices de los picos R
        """
        self._log("Detectando picos R")
        
        # Normalizar para robustez
        ecg_norm = ecg.squeeze()
        ecg_norm = (ecg_norm - np.mean(ecg_norm)) / (np.std(ecg_norm) + 1e-6)
        
        # Detectar picos
        distance = int(0.4 * self.fs)  # Mínimo 0.4s entre picos
        height = np.percentile(ecg_norm, 70)
        prom = prominence * (np.max(ecg_norm) - np.min(ecg_norm))
        
        peaks, properties = find_peaks(
            ecg_norm,
            height=height,
            distance=distance,
            prominence=prom
        )
        
        if len(peaks) < 2:
            self._log(f"⚠️  Apenas {len(peaks)} picos detectados, ajustando parámetros")
            peaks, _ = find_peaks(
                ecg_norm,
                height=np.percentile(ecg_norm, 50),
                distance=int(0.3 * self.fs)
            )
        
        self._log(f"✓ {len(peaks)} picos R detectados")
        
        return peaks
    
    def calculate_heart_rate(self, r_peaks: np.ndarray) -> float:
        """
        Calcula la frecuencia cardíaca
        
        Args:
            r_peaks: Índices de picos R
        
        Returns:
            Frecuencia cardíaca en BPM
        """
        if len(r_peaks) < 2:
            return 0.0
        
        # RR intervals en segundos
        rr_intervals = np.diff(r_peaks) / self.fs
        
        # BPM = 60 / RR_mean
        heart_rate = 60 / np.mean(rr_intervals)
        
        return heart_rate
    
    def segment_heartbeats(self, ecg: np.ndarray, r_peaks: np.ndarray,
                          pre_samples: int = None, 
                          post_samples: int = None) -> Tuple[np.ndarray, List[int]]:
        """
        Segmenta la señal en latidos individuales
        
        Args:
            ecg: Señal ECG
            r_peaks: Índices de picos R
            pre_samples: Muestras antes del pico R
            post_samples: Muestras después del pico R
        
        Returns:
            (matriz de latidos, índices válidos)
        """
        self._log("Segmentando latidos")
        
        if pre_samples is None:
            pre_samples = int(0.2 * self.fs)  # 200ms antes
        if post_samples is None:
            post_samples = int(0.4 * self.fs)  # 400ms después
        
        heartbeats = []
        valid_indices = []
        
        for i, peak in enumerate(r_peaks):
            start = peak - pre_samples
            end = peak + post_samples
            
            # Validar límites
            if start >= 0 and end <= len(ecg):
                hb = ecg.squeeze()[start:end]
                heartbeats.append(hb)
                valid_indices.append(i)
        
        if heartbeats:
            heartbeats = np.array(heartbeats)
        else:
            heartbeats = np.array([]).reshape(0, pre_samples + post_samples)
        
        return heartbeats, valid_indices
    
    def compute_rr_metrics(self, r_peaks: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas basadas en intervalos RR
        
        Args:
            r_peaks: Índices de picos R
        
        Returns:
            Diccionario de métricas
        """
        self._log("Calculando métricas RR")
        
        if len(r_peaks) < 2:
            return {'sdnn': 0, 'rmssd': 0, 'hr_mean': 0, 'hr_std': 0}
        
        # Intervalos RR en ms
        rr_intervals = np.diff(r_peaks) / self.fs * 1000
        
        # SDNN: Desviación estándar de RR
        sdnn = np.std(rr_intervals)
        
        # RMSSD: Raíz de la media cuadrada de diferencias sucesivas
        rr_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(rr_diffs ** 2)) if len(rr_diffs) > 0 else 0
        
        # HR en BPM
        hr_mean = 60000 / np.mean(rr_intervals)
        hr_std = 60 / np.std(rr_intervals) if np.std(rr_intervals) > 0 else 0
        
        return {
            'sdnn': sdnn,
            'rmssd': rmssd,
            'hr_mean': hr_mean,
            'hr_std': hr_std,
            'rr_mean': np.mean(rr_intervals),
            'rr_std': np.std(rr_intervals)
        }
    
    def find_waves(self, heartbeat: np.ndarray, r_peak_idx: int) -> Dict[str, int]:
        """
        Identifica ondas P, Q, R, S, T en un latido
        
        Args:
            heartbeat: Señal de un latido
            r_peak_idx: Índice del pico R dentro del latido
        
        Returns:
            Diccionario con índices de puntos característicos
        """
        # Normalizar
        hb = heartbeat.squeeze()
        hb_norm = (hb - np.mean(hb)) / (np.std(hb) + 1e-6)
        
        results = {}
        
        # R ya está identificado
        results['R'] = r_peak_idx
        
        # Q: valle antes de R
        if r_peak_idx > 5:
            q_region = hb_norm[:r_peak_idx]
            q_idx = np.argmin(q_region)
            results['Q'] = q_idx
        
        # S: valle después de R
        if r_peak_idx < len(hb) - 5:
            s_region = hb_norm[r_peak_idx:r_peak_idx + int(0.1 * self.fs)]
            if len(s_region) > 0:
                s_idx = np.argmin(s_region) + r_peak_idx
                results['S'] = s_idx
        
        # P: pico antes de QRS
        if r_peak_idx > int(0.1 * self.fs):
            p_region = hb_norm[:r_peak_idx - int(0.05 * self.fs)]
            if len(p_region) > 0:
                p_peaks, _ = find_peaks(p_region, prominence=0.1)
                if len(p_peaks) > 0:
                    results['P'] = int(p_peaks[-1])
        
        # T: pico después de S
        if r_peak_idx < len(hb) - int(0.1 * self.fs):
            t_region = hb_norm[r_peak_idx + int(0.1 * self.fs):]
            if len(t_region) > 0:
                t_peaks, _ = find_peaks(t_region, prominence=0.05)
                if len(t_peaks) > 0:
                    results['T'] = int(t_peaks[0] + r_peak_idx + int(0.1 * self.fs))
        
        return results
    
    def process_complete(self, ecg: np.ndarray) -> Dict:
        """
        Procesamiento completo: filtrado, detección, segmentación
        
        Args:
            ecg: Señal ECG cruda
        
        Returns:
            Diccionario con todos los resultados
        """
        self.ecg_original = ecg.copy()
        
        # 1. Filtrado
        self.ecg_filtered = self.bandpass_filter(ecg)
        self.ecg_filtered = self.remove_baseline_wander(self.ecg_filtered)
        
        # 2. Detección de picos R
        self.r_peaks = self.detect_r_peaks(self.ecg_filtered)
        
        # 3. Frecuencia cardíaca
        self.heart_rate = self.calculate_heart_rate(self.r_peaks)
        
        # 4. Segmentación
        heartbeats, valid_idx = self.segment_heartbeats(self.ecg_filtered, self.r_peaks)
        
        # 5. Métricas RR
        rr_metrics = self.compute_rr_metrics(self.r_peaks)
        
        return {
            'ecg_original': self.ecg_original,
            'ecg_filtered': self.ecg_filtered,
            'r_peaks': self.r_peaks,
            'heart_rate': self.heart_rate,
            'heartbeats': heartbeats,
            'valid_r_peaks': self.r_peaks[valid_idx],
            'rr_metrics': rr_metrics
        }