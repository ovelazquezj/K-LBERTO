#!/usr/bin/env python3
"""
Análisis de resultados RQ1: Validación de √-Scaling
Autor: Omar Velázquez
Fecha: 2026-01-04
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns
import sys
import os

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

class RQ1Analyzer:
    def __init__(self, csv_file='consolidated_results.csv'):
        """Inicializar analizador con archivo de resultados"""
        if not os.path.exists(csv_file):
            print(f"ERROR: {csv_file} no existe")
            print("Ejecuta primero: python consolidate_results.py")
            sys.exit(1)
        
        self.df = pd.read_csv(csv_file)
        self.dataset_sizes = sorted(self.df['dataset_size'].unique())
        
        # Filtrar solo RQ1
        self.df = self.df[self.df['rq'] == 'RQ1'].copy()
        
        if len(self.df) == 0:
            print("ERROR: No hay experimentos RQ1 en el archivo")
            sys.exit(1)
        
        print(f"✓ Cargados {len(self.df)} experimentos RQ1")
        print(f"  Dataset sizes: {self.dataset_sizes}")
        print(f"  Configs: {sorted(self.df['config_type'].unique())}")
        print(f"  Seeds: {sorted(self.df['seed'].unique())}")
    
    def summary_statistics(self):
        """Calcular estadísticas resumen por configuración"""
        print("\n" + "="*80)
        print("ESTADÍSTICAS RESUMEN - F1 Overall")
        print("="*80)
        
        summary = self.df.groupby(['dataset_size', 'config_type'])['f1_overall'].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('n', 'count')
        ]).round(4)
        
        print(summary)
        
        # Guardar tabla
        summary.to_csv('rq1_summary_statistics.csv')
        print("\n✓ Guardado: rq1_summary_statistics.csv")
        
        return summary
    
    def plot_f1_comparison(self):
        """Gráfica principal: F1 vs Dataset Size"""
        print("\n" + "="*80)
        print("GRÁFICA: F1 vs Dataset Size")
        print("="*80)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Calcular medias y std
        summary = self.df.groupby(['dataset_size', 'config_type'])['f1_overall'].agg(['mean', 'std'])
        
        # Plot base
        base_data = summary.loc[(slice(None), 'base'), :]
        ax.errorbar(
            self.dataset_sizes, 
            base_data['mean'].values,
            yerr=base_data['std'].values,
            marker='o', 
            markersize=8,
            linewidth=2,
            capsize=5,
            label='Base LR (fixed 2e-5)',
            color='#1f77b4'
        )
        
        # Plot sqrt
        sqrt_data = summary.loc[(slice(None), 'sqrt'), :]
        ax.errorbar(
            self.dataset_sizes,
            sqrt_data['mean'].values,
            yerr=sqrt_data['std'].values,
            marker='s',
            markersize=8,
            linewidth=2,
            capsize=5,
            label='√-scaled LR',
            color='#ff7f0e'
        )
        
        ax.set_xlabel('Dataset Size (samples)', fontsize=14)
        ax.set_ylabel('F1 Score (Overall)', fontsize=14)
        ax.set_title('RQ1: Learning Rate Scaling Strategy Comparison', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rq1_f1_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Guardado: rq1_f1_comparison.png")
        plt.close()
    
    def power_law_fit(self):
        """Ajustar power law y estimar α"""
        print("\n" + "="*80)
        print("POWER LAW FIT: F1 = c × D^α")
        print("="*80)
        
        # Datos sqrt (solo estos siguen power law teóricamente)
        sqrt_data = self.df[self.df['config_type'] == 'sqrt'].groupby('dataset_size')['f1_overall'].mean()
        
        X = np.array(self.dataset_sizes)
        Y = sqrt_data.values
        
        # Función power law
        def power_law(x, c, alpha):
            return c * (x ** alpha)
        
        # Fit
        try:
            params, covariance = curve_fit(power_law, X, Y)
            c, alpha = params
            
            # Calcular R²
            y_pred = power_law(X, c, alpha)
            residuals = Y - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((Y - np.mean(Y))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            print(f"\nResultados del fit:")
            print(f"  c (constante) = {c:.6f}")
            print(f"  α (exponente)  = {alpha:.4f}")
            print(f"  R² = {r_squared:.4f}")
            print(f"\nFórmula ajustada: F1 = {c:.6f} × D^{alpha:.4f}")
            
            # Verificar hipótesis
            print(f"\nValidación de hipótesis:")
            in_range = -0.55 <= alpha <= -0.45
            print(f"  ✓ α en [-0.55, -0.45]: {in_range} (α = {alpha:.4f})")
            print(f"  ✓ R² > 0.90: {r_squared > 0.90} (R² = {r_squared:.4f})")
            
            # Gráfica del fit
            fig, ax = plt.subplots(figsize=(12, 7))
            
            ax.scatter(X, Y, s=100, alpha=0.7, label='Datos observados (√-scaled)', color='#ff7f0e')
            
            # Línea de fit
            X_smooth = np.linspace(X.min(), X.max(), 100)
            Y_smooth = power_law(X_smooth, c, alpha)
            ax.plot(X_smooth, Y_smooth, 'r--', linewidth=2, 
                   label=f'Fit: F1 = {c:.4f} × D^{alpha:.3f} (R²={r_squared:.3f})')
            
            ax.set_xlabel('Dataset Size (samples)', fontsize=14)
            ax.set_ylabel('F1 Score (Overall)', fontsize=14)
            ax.set_title('Power Law Fit: √-scaled LR Strategy', fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('rq1_power_law_fit.png', dpi=300, bbox_inches='tight')
            print("\n✓ Guardado: rq1_power_law_fit.png")
            plt.close()
            
            return {'c': c, 'alpha': alpha, 'r_squared': r_squared}
            
        except Exception as e:
            print(f"ERROR en power law fit: {e}")
            return None
    
    def statistical_tests(self):
        """T-tests entre base vs sqrt por dataset size"""
        print("\n" + "="*80)
        print("TESTS ESTADÍSTICOS: base vs sqrt")
        print("="*80)
        
        results = []
        
        for size in self.dataset_sizes:
            base_f1 = self.df[(self.df['dataset_size'] == size) & 
                             (self.df['config_type'] == 'base')]['f1_overall'].values
            sqrt_f1 = self.df[(self.df['dataset_size'] == size) & 
                             (self.df['config_type'] == 'sqrt')]['f1_overall'].values
            
            if len(base_f1) > 0 and len(sqrt_f1) > 0:
                t_stat, p_value = stats.ttest_ind(base_f1, sqrt_f1)
                
                base_mean = base_f1.mean()
                sqrt_mean = sqrt_f1.mean()
                improvement = ((sqrt_mean - base_mean) / base_mean) * 100
                
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                
                results.append({
                    'dataset_size': size,
                    'base_mean': base_mean,
                    'sqrt_mean': sqrt_mean,
                    'improvement_%': improvement,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': sig
                })
                
                print(f"\nD={size:4d}: base={base_mean:.4f}, sqrt={sqrt_mean:.4f}, "
                      f"Δ={improvement:+.2f}%, p={p_value:.4f} {sig}")
        
        # Guardar tabla
        df_tests = pd.DataFrame(results)
        df_tests.to_csv('rq1_statistical_tests.csv', index=False)
        print("\n✓ Guardado: rq1_statistical_tests.csv")
        
        # Resumen
        sig_count = sum(1 for r in results if r['p_value'] < 0.05)
        print(f"\nResumen: {sig_count}/{len(results)} comparaciones significativas (p < 0.05)")
        
        return df_tests
    
    def entity_breakdown(self):
        """Análisis por tipo de entidad (PER/LOC/ORG)"""
        print("\n" + "="*80)
        print("BREAKDOWN POR TIPO DE ENTIDAD")
        print("="*80)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        entities = ['PER', 'LOC', 'ORG']
        
        for idx, entity in enumerate(entities):
            ax = axes[idx]
            col = f'f1_{entity}'
            
            # Calcular medias
            summary = self.df.groupby(['dataset_size', 'config_type'])[col].mean()
            
            base_data = summary.loc[(slice(None), 'base')]
            sqrt_data = summary.loc[(slice(None), 'sqrt')]
            
            ax.plot(self.dataset_sizes, base_data.values, 'o-', label='Base', linewidth=2)
            ax.plot(self.dataset_sizes, sqrt_data.values, 's-', label='√-scaled', linewidth=2)
            
            ax.set_xlabel('Dataset Size')
            ax.set_ylabel(f'F1 Score')
            ax.set_title(f'{entity} Entity Type', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Performance by Entity Type', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('rq1_entity_breakdown.png', dpi=300, bbox_inches='tight')
        print("\n✓ Guardado: rq1_entity_breakdown.png")
        plt.close()
    
    def generate_report(self):
        """Generar reporte completo"""
        print("\n" + "="*80)
        print("REPORTE RQ1 - VALIDACIÓN √-SCALING")
        print("="*80)
        
        with open('rq1_report.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("RQ1: VALIDACIÓN DE √-SCALING PARA LEARNING RATE\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total experimentos: {len(self.df)}\n")
            f.write(f"Dataset sizes: {self.dataset_sizes}\n")
            f.write(f"Seeds: {sorted(self.df['seed'].unique())}\n")
            f.write(f"Configs: base (LR fijo), sqrt (LR escalado)\n\n")
            
            # Summary statistics
            summary = self.df.groupby(['dataset_size', 'config_type'])['f1_overall'].agg(['mean', 'std'])
            f.write("RESUMEN DE RESULTADOS:\n")
            f.write(summary.to_string())
            f.write("\n\n")
            
            f.write("ARCHIVOS GENERADOS:\n")
            f.write("  - rq1_summary_statistics.csv\n")
            f.write("  - rq1_f1_comparison.png\n")
            f.write("  - rq1_power_law_fit.png\n")
            f.write("  - rq1_statistical_tests.csv\n")
            f.write("  - rq1_entity_breakdown.png\n")
            f.write("  - rq1_report.txt (este archivo)\n")
        
        print("✓ Guardado: rq1_report.txt")
    
    def run_full_analysis(self):
        """Ejecutar análisis completo"""
        print("\n" + "="*80)
        print("INICIANDO ANÁLISIS COMPLETO RQ1")
        print("="*80)
        
        self.summary_statistics()
        self.plot_f1_comparison()
        fit_results = self.power_law_fit()
        self.statistical_tests()
        self.entity_breakdown()
        self.generate_report()
        
        print("\n" + "="*80)
        print("ANÁLISIS COMPLETADO")
        print("="*80)
        print("\nArchivos generados:")
        print("  📊 rq1_f1_comparison.png")
        print("  📈 rq1_power_law_fit.png")
        print("  📊 rq1_entity_breakdown.png")
        print("  📄 rq1_summary_statistics.csv")
        print("  📄 rq1_statistical_tests.csv")
        print("  📄 rq1_report.txt")
        
        if fit_results:
            print(f"\nRESULTADO CLAVE:")
            print(f"  α estimado = {fit_results['alpha']:.4f}")
            print(f"  ¿Valida √-scaling (α ≈ -0.5)? {abs(fit_results['alpha'] + 0.5) < 0.05}")


def main():
    analyzer = RQ1Analyzer('consolidated_results.csv')
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
