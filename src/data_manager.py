# src/data_manager.py - VERSIÓN MEJORADA
"""
Gestor de datos ECG desde PhysioNet PTB-XL
Descarga automática y gestión de base de datos
MEJORA: Mejor soporte para rangos de edad
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import wfdb
import json

class ECGDataManager:
    """Gestor centralizado de datos ECG"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Inicializa el gestor de datos
        
        Args:
            data_dir: Directorio para almacenar datos
        """
        self.data_dir = Path(data_dir)
        self.ptb_xl_dir = self.data_dir / "ptb_xl"
        self.ptb_xl_url = "https://physionet.org/files/ptb-xl/1.0.3/"
        
        # Crear directorios
        self.data_dir.mkdir(exist_ok=True)
        self.ptb_xl_dir.mkdir(exist_ok=True)
        
        self.records = None
        self.signals_dict = {}
    
    # RANGOS DE EDAD PREDEFINIDOS
    AGE_GROUPS = {
        '0-10': (0, 10),
        '11-20': (11, 20),
        '21-30': (21, 30),
        '31-40': (31, 40),
        '41-50': (41, 50),
        '51-60': (51, 60),
        '61+': (61, 120),
    }
    
    def download_ptb_xl_metadata(self) -> pd.DataFrame:
        """
        Descarga metadatos de PTB-XL
        
        Returns:
            DataFrame con información de pacientes
        """
        csv_path = self.ptb_xl_dir / "ptb_xl_database.csv"
        
        # Si ya existe, cargar
        if csv_path.exists():
            print("✓ Metadatos encontrados localmente")
            return pd.read_csv(csv_path)
        
        print("📥 Descargando metadatos de PTB-XL...")
        try:
            url = self.ptb_xl_url + "ptb_xl_database.csv"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(csv_path, 'w') as f:
                f.write(response.text)
            
            print("✓ Metadatos descargados correctamente")
            return pd.read_csv(csv_path)
        
        except Exception as e:
            print(f"❌ Error descargando metadatos: {e}")
            print("⚠️  Usando base de datos de demostración...")
            return self._create_dummy_database()
    
    def _create_dummy_database(self) -> pd.DataFrame:
        """Crea una BD ficticia para demostración"""
        print("⚠️  Creando base de datos de demostración")
        
        data = []
        age_ranges = [5, 15, 25, 35, 45, 55, 65, 75, 85]
        sexes = ['M', 'F']
        diagnoses = ['norm', 'MI', 'STTC', 'CD', 'HYP', 'AFIB']
        
        idx = 1
        for age in age_ranges:
            for sex in sexes:
                for diag in diagnoses:
                    data.append({
                        'filename_hr': f'records100/00{idx:03d}_hr',
                        'patient_id': idx,
                        'age': age,
                        'sex': sex,
                        'diagnostic': diag,
                        'scp_codes': f'{{"{diag}": {{"scp_code": "{diag}"}}}}'
                    })
                    idx += 1
        
        df = pd.DataFrame(data)
        csv_path = self.ptb_xl_dir / "ptb_xl_database.csv"
        df.to_csv(csv_path, index=False)
        
        return df
    
    def get_filtered_records(self, 
                            age_range: Tuple[int, int] = (0, 100),
                            sex: str = None,
                            diagnosis: str = None,
                            num_records: int = 5) -> List[Dict]:
        """
        Obtiene registros filtrados por criterios
        
        Args:
            age_range: Rango de edad (min, max)
            sex: 'M' o 'F'
            diagnosis: Tipo de diagnóstico
            num_records: Número de registros a retornar
        
        Returns:
            Lista de registros filtrados
        """
        if self.records is None:
            self.records = self.download_ptb_xl_metadata()
        
        df = self.records.copy()
        
        # Filtrar por edad
        df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
        
        # Filtrar por sexo
        if sex:
            df = df[df['sex'] == sex]
        
        # Filtrar por diagnóstico
        if diagnosis and diagnosis != 'Todos':
            df = df[df['diagnostic'].str.contains(diagnosis, case=False, na=False)]
        
        # Retornar N registros
        return df.head(num_records).to_dict('records')
    
    def get_records_by_age_group(self, 
                                 age_group: str,
                                 sex: str = None,
                                 diagnosis: str = None,
                                 num_records: int = 5) -> List[Dict]:
        """
        Obtiene registros por grupo de edad predefinido
        
        Args:
            age_group: Grupo de edad ('0-10', '11-20', etc.)
            sex: 'M' o 'F'
            diagnosis: Diagnóstico
            num_records: Cantidad de registros
        
        Returns:
            Lista de registros
        """
        if age_group not in self.AGE_GROUPS:
            return []
        
        age_range = self.AGE_GROUPS[age_group]
        return self.get_filtered_records(age_range, sex, diagnosis, num_records)
    
    def download_signal(self, record_name: str, force_redownload: bool = False) -> Tuple[np.ndarray, int]:
        """
        Descarga una señal ECG específica
        
        Args:
            record_name: Nombre del registro (ej: 'records100/00001_hr')
            force_redownload: Forzar descarga
        
        Returns:
            (señal, sampling_rate) tuple
        """
        # Crear ruta local
        local_path = self.ptb_xl_dir / record_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Si existe y no forzamos redescargar
        if not force_redownload:
            try:
                record = wfdb.rdrecord(str(local_path.with_suffix('')))
                print(f"✓ Señal cargada desde caché: {record_name}")
                return record.p_signal, record.fs
            except:
                pass
        
        # Descargar archivos necesarios
        try:
            print(f"📥 Descargando señal: {record_name}...")
            
            for ext in ['.dat', '.hea']:
                url = self.ptb_xl_url + record_name + ext
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                file_path = local_path.parent / (local_path.name + ext)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            
            # Leer el registro
            record = wfdb.rdrecord(str(local_path.with_suffix('')))
            print(f"✓ Señal descargada: {record_name}")
            
            return record.p_signal, record.fs
        
        except Exception as e:
            print(f"⚠️  Error descargando {record_name}: {e}")
            print("Generando señal sintética...")
            return self._generate_synthetic_ecg()
    
    def _generate_synthetic_ecg(self, duration: float = 10.0, fs: int = 250) -> Tuple[np.ndarray, int]:
        """
        Genera una señal ECG sintética realista
        
        Returns:
            (señal, sampling_rate) tuple
        """
        t = np.arange(0, duration, 1/fs)
        
        # Componentes del ECG
        heart_rate = np.random.randint(60, 100)
        heart_freq = heart_rate / 60
        
        # Onda P, QRS, T sinusoidal
        ecg = (
            0.15 * np.sin(2 * np.pi * heart_freq * t) +
            0.3 * np.sin(2 * np.pi * heart_freq * 3 * t) +
            0.1 * np.sin(2 * np.pi * heart_freq * 2 * t) +
            np.random.normal(0, 0.05, len(t))
        )
        
        # Retornar como array 2D (canal único)
        return ecg.reshape(-1, 1), fs
    
    def get_all_diagnoses(self) -> List[str]:
        """Retorna lista de diagnósticos disponibles"""
        if self.records is None:
            self.records = self.download_ptb_xl_metadata()
        
        return sorted(self.records['diagnostic'].unique().tolist())
    
    def get_database_stats(self) -> Dict:
        """Retorna estadísticas de la BD"""
        if self.records is None:
            self.records = self.download_ptb_xl_metadata()
        
        return {
            'total_records': len(self.records),
            'age_mean': self.records['age'].mean(),
            'age_std': self.records['age'].std(),
            'male_count': (self.records['sex'] == 'M').sum(),
            'female_count': (self.records['sex'] == 'F').sum(),
            'diagnoses': self.records['diagnostic'].value_counts().to_dict()
        }
    
    def get_age_groups(self) -> Dict[str, Tuple[int, int]]:
        """Retorna los grupos de edad disponibles"""
        return self.AGE_GROUPS.copy()