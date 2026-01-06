#!/usr/bin/env python3
"""
Análisis de escalabilidad temporal de experimentos K-LBERTO
Determina si el tiempo de ejecución crece lineal, logarítmica o exponencialmente

Autor: Omar Velázquez
Fecha: 2026-01-06
"""

import os
import numpy as np
from datetime import datetime

class TimingAnalyzer:
    def __init__(self, results_dir='resultados'):
        self.results_dir = results_dir
        self.data = []
        
    def collect_data(self):
        """Recolectar tiempos de experimentos completados"""
        datasets = [500, 750, 1000, 1500, 2000, 2500, 3000]
        configs = ['base_seed42', 'sqrt_seed42', 'base_seed43', 
                   'sqrt_seed43', 'base_seed44', 'sqrt_seed44']
        
        for dataset_size in datasets:
            for config in configs:
                exp_name = f"RQ1_D{dataset_size}_{config}"
                config_file = f"{self.results_dir}/{exp_name}_config.json"
                metrics_file = f"{self.results_dir}/{exp_name}_metrics.csv"
                
                if os.path.exists(config_file) and os.path.exists(metrics_file):
                    start_time = os.path.getmtime(config_file)
                    end_time = os.path.getmtime(metrics_file)
                    duration_minutes = (end_time - start_time) / 60
                    self.data.append((dataset_size, duration_minutes))
        
        if not self.data:
            raise ValueError("No se encontraron datos de experimentos completados")
    
    def print_data_summary(self):
        """Imprimir resumen de datos recolectados"""
        print("="*80)
        print("DATOS REALES DE EXPERIMENTOS")
        print("="*80)
        print(f"Total experimentos analizados: {len(self.data)}")
        
        # Agrupar por dataset
        datasets = sorted(set([d[0] for d in self.data]))
        print(f"\nPor dataset size:")
        for ds in datasets:
            times = [y for x, y in self.data if x == ds]
            if times:
                mean = np.mean(times)
                std = np.std(times)
                min_t = np.min(times)
                max_t = np.max(times)
                print(f"  D{ds:4d}: {len(times)} exp, "
                      f"μ={mean:5.1f} min, σ={std:4.1f}, "
                      f"range=[{min_t:.1f}, {max_t:.1f}]")
    
    def fit_models(self):
        """Ajustar diferentes modelos y comparar"""
        from numpy.polynomial import Polynomial
        
        X = np.array([d[0] for d in self.data])
        Y = np.array([d[1] for d in self.data])
        
        print("\n" + "="*80)
        print("AJUSTE DE MODELOS")
        print("="*80)
        
        # Modelo 1: Lineal (y = a*x + b)
        p_linear = Polynomial.fit(X, Y, 1)
        y_pred_linear = p_linear(X)
        r2_linear = 1 - np.sum((Y - y_pred_linear)**2) / np.sum((Y - np.mean(Y))**2)
        
        a_lin, b_lin = p_linear.convert().coef
        print(f"\n1. MODELO LINEAL")
        print(f"   Ecuación: y = {a_lin:.6f}*x + {b_lin:.4f}")
        print(f"   R² = {r2_linear:.6f}")
        print(f"   Interpretación: Cada 1000 samples agregan {a_lin*1000:.1f} minutos")
        
        # Modelo 2: Logarítmico (y = a*log(x) + b)
        X_log = np.log(X)
        p_log = Polynomial.fit(X_log, Y, 1)
        y_pred_log = p_log(X_log)
        r2_log = 1 - np.sum((Y - y_pred_log)**2) / np.sum((Y - np.mean(Y))**2)
        
        a_log, b_log = p_log.convert().coef
        print(f"\n2. MODELO LOGARÍTMICO")
        print(f"   Ecuación: y = {a_log:.4f}*log(x) + {b_log:.4f}")
        print(f"   R² = {r2_log:.6f}")
        print(f"   Interpretación: Crecimiento desacelera con dataset size")
        
        # Modelo 3: Potencial (y = a*x^b)
        X_log = np.log(X)
        Y_log = np.log(Y)
        p_power = Polynomial.fit(X_log, Y_log, 1)
        b_pow, log_a = p_power.convert().coef
        a_pow = np.exp(log_a)
        
        y_pred_power = a_pow * (X ** b_pow)
        r2_power = 1 - np.sum((Y - y_pred_power)**2) / np.sum((Y - np.mean(Y))**2)
        
        print(f"\n3. MODELO POTENCIAL")
        print(f"   Ecuación: y = {a_pow:.6f}*x^{b_pow:.6f}")
        print(f"   R² = {r2_power:.6f}")
        
        if b_pow < 1:
            print(f"   Interpretación: Crecimiento SUBLINEAL (exp={b_pow:.3f}<1)")
            print(f"                   Datasets grandes más eficientes")
        elif b_pow > 1:
            print(f"   Interpretación: Crecimiento SUPERLINEAL (exp={b_pow:.3f}>1)")
            print(f"                   ⚠️ Datasets grandes desproporcionadamente lentos")
        else:
            print(f"   Interpretación: Crecimiento LINEAL (exp≈1)")
        
        # Determinar mejor modelo
        models = {
            'Lineal': (r2_linear, lambda x: a_lin*x + b_lin),
            'Logarítmico': (r2_log, lambda x: a_log*np.log(x) + b_log),
            'Potencial': (r2_power, lambda x: a_pow * (x ** b_pow))
        }
        
        best_name = max(models.items(), key=lambda m: m[1][0])[0]
        best_r2, best_func = models[best_name]
        
        print("\n" + "="*80)
        print(f"MEJOR MODELO: {best_name} (R² = {best_r2:.6f})")
        print("="*80)
        
        return best_name, best_func, best_r2
    
    def project_times(self, model_func, timeout_minutes=240):
        """Proyectar tiempos para datasets grandes"""
        print("\n" + "="*80)
        print("PROYECCIÓN PARA DATASETS GRANDES")
        print("="*80)
        print(f"Timeout configurado: {timeout_minutes} minutos\n")
        
        results = []
        for ds in [2000, 2500, 3000]:
            pred = model_func(ds)
            margin = timeout_minutes - pred
            pct_used = (pred / timeout_minutes) * 100
            
            if pred < timeout_minutes * 0.8:
                status = "✅ HOLGADO"
            elif pred < timeout_minutes * 0.95:
                status = "⚠️ AJUSTADO"
            else:
                status = "❌ RIESGO"
            
            results.append({
                'dataset': ds,
                'predicted': pred,
                'margin': margin,
                'pct_used': pct_used,
                'status': status
            })
            
            print(f"D{ds}:")
            print(f"  Tiempo predicho: {pred:6.1f} min ({pred/60:.2f} h)")
            print(f"  Uso del timeout: {pct_used:5.1f}%")
            print(f"  Margen:          {margin:6.1f} min")
            print(f"  Estado:          {status}")
            print()
        
        return results
    
    def generate_recommendations(self, results, timeout_minutes=240):
        """Generar recomendaciones basadas en proyección"""
        print("="*80)
        print("RECOMENDACIONES")
        print("="*80)
        
        max_pred = max([r['predicted'] for r in results])
        min_margin = min([r['margin'] for r in results])
        
        if min_margin < 0:
            recommended = int(max_pred * 1.15)  # 15% margen
            print(f"\n❌ TIMEOUT INSUFICIENTE")
            print(f"   Máximo predicho: {max_pred:.1f} min")
            print(f"   Timeout actual:  {timeout_minutes} min")
            print(f"   Recomendado:     {recommended} min (15% margen)")
        elif min_margin < timeout_minutes * 0.1:
            recommended = int(max_pred * 1.15)
            print(f"\n⚠️ MARGEN MUY AJUSTADO")
            print(f"   Margen mínimo: {min_margin:.1f} min ({(min_margin/timeout_minutes)*100:.1f}%)")
            print(f"   Recomendado aumentar a: {recommended} min")
        else:
            print(f"\n✅ TIMEOUT ADECUADO")
            print(f"   Margen mínimo: {min_margin:.1f} min ({(min_margin/timeout_minutes)*100:.1f}%)")
            print(f"   No requiere ajuste")
        
        print("\nNotas:")
        print("  • Proyección basada en datos de D500-D1500")
        print("  • Variabilidad real puede diferir ±10%")
        print("  • Monitorear primeros experimentos D2000+ para ajustar")
    
    def run_full_analysis(self, timeout_minutes=240):
        """Ejecutar análisis completo"""
        print("\n" + "="*80)
        print("ANÁLISIS DE ESCALABILIDAD TEMPORAL - K-LBERTO RQ1")
        print("="*80)
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Directorio: {self.results_dir}\n")
        
        # Recolectar datos
        print("Recolectando datos de experimentos completados...")
        self.collect_data()
        
        # Resumen de datos
        self.print_data_summary()
        
        # Ajustar modelos
        best_name, best_func, best_r2 = self.fit_models()
        
        # Proyectar tiempos
        results = self.project_times(best_func, timeout_minutes)
        
        # Recomendaciones
        self.generate_recommendations(results, timeout_minutes)
        
        print("\n" + "="*80)
        print("ANÁLISIS COMPLETADO")
        print("="*80)


def main():
    """Función principal"""
    import sys
    
    # Parsear timeout de argumentos
    timeout = 240
    if len(sys.argv) > 1:
        try:
            timeout = int(sys.argv[1])
        except ValueError:
            print(f"Advertencia: Timeout inválido '{sys.argv[1]}', usando 240 min")
    
    try:
        analyzer = TimingAnalyzer()
        analyzer.run_full_analysis(timeout_minutes=timeout)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
