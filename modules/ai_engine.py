# modules/ai_engine.py
"""
Moteur d'IA avec capacité d'apprentissage et d'analyse contextuelle
Version améliorée avec génération de phrases naturelles
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
from openai import OpenAI 
from pathlib import Path
from dotenv import load_dotenv


env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
USE_OPENAI = bool(OPENAI_API_KEY)
class AIAnalyzer:
    """
    Moteur d'IA avancé avec capacité d'apprentissage et de mémoire
    """
    def __init__(self, memory_path="ai_memory.pkl"):
        self.memory_path = memory_path
        self.memory = self._load_memory()
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        self.client_climate = None
        if USE_OPENAI:
            self.client_climate = OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.client_climate = None
        self.cache = {}
        # Plafonds réglementaires
        self.plafonds = {
            "BL": 90,
            "RG": 90,
            "RS": 100
        }
        
    def _load_memory(self) -> Dict:
        """Charge la mémoire de l'IA depuis le disque"""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return self._init_memory() 
        return self._init_memory()
    
    def _init_memory(self) -> Dict:
        """Initialise une nouvelle mémoire"""
        return {
            'analyses': [],
            'patterns': {},
            'thresholds': {},
            'feedback': [],
            'learning_history': []
        }
    
    def _save_memory(self):
        """Sauvegarde la mémoire de l'IA"""
        with open(self.memory_path, 'wb') as f:
            pickle.dump(self.memory, f)
    
    def _detect_patterns(self, data: pd.DataFrame, column: str) -> Dict:
        """Détecte des patterns dans les données"""
        patterns = {}
        
        if column in data.columns and len(data) > 3:
            # Préparation des données
            if 'annee' in data.columns:
                data_sorted = data.sort_values('annee')
                y = data_sorted[column].values
                x = data_sorted['annee'].values
            else:
                y = data[column].values
                x = np.arange(len(y))
            
            if len(y) > 2 and len(np.unique(y)) > 1:
                # Tendance (pente par an)
                coeffs = np.polyfit(x, y, 1)
                patterns['trend'] = coeffs[0]
                patterns['trend_pct'] = (coeffs[0] / np.mean(y)) * 100 if np.mean(y) != 0 else 0
                
                # Volatilité
                patterns['volatility'] = np.std(y)
                patterns['cv'] = (np.std(y) / np.mean(y)) * 100 if np.mean(y) != 0 else 0
                
                # Classification de la stabilité
                if patterns['cv'] < 15:
                    patterns['stability'] = 'très stable'
                elif patterns['cv'] < 25:
                    patterns['stability'] = 'stable'
                elif patterns['cv'] < 35:
                    patterns['stability'] = 'modérément variable'
                else:
                    patterns['stability'] = 'très variable'
                
                patterns['min_year'] = x[np.argmin(y)]
                patterns['max_year'] = x[np.argmax(y)]
                patterns['min_value'] = np.min(y)
                patterns['max_value'] = np.max(y)
                patterns['range'] = patterns['max_value'] - patterns['min_value']
        
        return patterns
    
    def _generate_rendement_analysis(self, data: Dict) -> str:
        """Génère une analyse en langage naturel pour les rendements"""
        
        lines = []
        stats = data.get('stats', {})
        patterns = data.get('patterns', {})
        risks = data.get('risks', [])
        recommendations = data.get('recommendations', [])
        comparisons = data.get('comparisons', {})
        
        if not stats:
            return "Données insuffisantes pour l'analyse."
        
        # Calcul des moyennes globales
        means = [s['mean'] for s in stats.values()]
        stds = [s['std'] for s in stats.values()]
        
        global_mean = np.mean(means)
        global_std = np.mean(stds)
        
        # Introduction
        lines.append(f"**Analyse globale des rendements**")
        
        if len(stats) == 1:
            groupe = list(stats.keys())[0]
            lines.append(f"Pour la catégorie **{groupe}**, le rendement moyen observé est de **{global_mean:.1f} hl/ha**.")
        else:
            lines.append(f"L'analyse porte sur {len(stats)} catégories. Le rendement moyen global est de **{global_mean:.1f} hl/ha**, avec une variabilité moyenne de **{global_std:.1f} hl/ha**.")
        
        lines.append("")
        
        # Analyse par groupe
        lines.append(f"**Analyse détaillée par catégorie :**")
        lines.append("")
        
        for groupe, stat in stats.items():
            lines.append(f"**{groupe}** :")
            
            # Position par rapport à la moyenne
            if len(stats) > 1:
                if stat['mean'] > global_mean + 3:
                    lines.append(f"  • Rendement **supérieur** à la moyenne de {stat['mean'] - global_mean:.1f} hl/ha.")
                elif stat['mean'] < global_mean - 3:
                    lines.append(f"  • Rendement **inférieur** à la moyenne de {global_mean - stat['mean']:.1f} hl/ha.")
                else:
                    lines.append(f"  • Rendement **dans la moyenne** globale.")
            
            # Variabilité
            cv = (stat['std'] / stat['mean'] * 100) if stat['mean'] > 0 else 0
            
            if cv > 35:
                lines.append(f"  • **Forte variabilité** interannuelle (écart-type de {stat['std']:.1f} hl/ha, Indice de stabilité du rendement={cv:.0f}%). Cette irrégularité complique la planification et peut indiquer une sensibilité aux aléas climatiques.")
            elif cv > 25:
                lines.append(f"  • **Variabilité marquée** (écart-type de {stat['std']:.1f} hl/ha, Indice de stabilité du rendement={cv:.0f}%). La production fluctue significativement selon les millésimes.")
            elif cv > 15:
                lines.append(f"  • **Variabilité modérée** (écart-type de {stat['std']:.1f} hl/ha, Indice de stabilité du rendement={cv:.0f}%).")
            else:
                lines.append(f"  • **Très bonne stabilité** (écart-type de {stat['std']:.1f} hl/ha, Indice de stabilité du rendement={cv:.0f}%).")
            
            # Étendue
            lines.append(f"  • Les rendements s'échelonnent de **{stat['min']:.0f}** à **{stat['max']:.0f} hl/ha**, soit une amplitude de {stat['max'] - stat['min']:.0f} hl/ha.")
            
            # Analyse des minimums
            if stat['min'] < 20:
                lines.append(f"  • Les années de très faible production (<20 hl/ha) évoquent des **épisodes de gel printanier** ou de **sécheresse sévère**.")
            
            # Proximité des plafonds
            plafond = self.plafonds.get(groupe, 90)
            if stat['max'] > plafond:
                lines.append(f"  • Le rendement maximum ({stat['max']:.0f} hl/ha) **dépasse le plafond réglementaire** de {plafond} hl/ha. Ces années présentent un risque de déclassement.")
            elif stat['max'] > plafond * 0.9:
                lines.append(f"  • Le rendement maximum ({stat['max']:.0f} hl/ha) est **proche du plafond** de {plafond} hl/ha.")
            
            # Analyse de la tendance
            if groupe in patterns and 'trend' in patterns[groupe]:
                trend = patterns[groupe]['trend']
                if abs(trend) > 0.3:
                    if trend > 0:
                        lines.append(f"  • **Tendance à la hausse** : +{trend:.2f} hl/ha par an.")
                    else:
                        lines.append(f"  • **Tendance à la baisse** : {trend:.2f} hl/ha par an.")
            
            lines.append("")
        
        # Comparaisons
        if comparisons:
            lines.append(f"**Comparaison entre catégories :**")
            lines.append("")
            lines.append(f"  • La catégorie la plus productive est **{comparisons['best']}** avec {comparisons['best_value']:.1f} hl/ha.")
            lines.append(f"  • La catégorie la moins productive est **{comparisons['worst']}** avec {comparisons['worst_value']:.1f} hl/ha.")
            lines.append(f"  • L'écart est de **{comparisons['gap']:.1f} hl/ha** ({comparisons['gap_percent']:.0f}%).")
            lines.append("")
        
        # Risques
        if risks:
            lines.append(f"**Synthèse des risques identifiés :**")
            lines.append("")
            for risk in risks[:5]:  # Limiter à 5 risques
                lines.append(f"  • {risk}")
            lines.append("")
        
        # Recommandations
        if recommendations:
            lines.append(f"**Recommandations personnalisées :**")
            lines.append("")
            for rec in recommendations[:3]:  # Limiter à 3 recommandations
                lines.append(f"  • {rec}")
            lines.append("")
        
        return "\n".join(lines)
    
    def analyze_rendement(self, data: pd.DataFrame, mode: str, selections: List[str]) -> Dict[str, Any]:
        """
        Analyse approfondie des rendements
        """
        df_analysis = data.copy()
        
        # Mapping des modes vers les noms de colonnes réels
        mode_to_col = {
            "Couleur": "code_couleur",
            "Departement": "code_departement",  # Sans accent
            "Zone": "Zone"
        }
        
        col_name = mode_to_col.get(mode, mode)
        
        stats = {}
        patterns = {}
        recommendations = []
        risks = []
        comparisons = {}
        
        for selection in selections:
            subset = df_analysis[df_analysis[col_name] == selection]
            
            if not subset.empty:
                rendements = subset['rendement'].dropna()
                
                if len(rendements) > 0:
                    stats[selection] = {
                        'mean': rendements.mean(),
                        'median': rendements.median(),
                        'std': rendements.std(),
                        'min': rendements.min(),
                        'max': rendements.max(),
                        'q1': rendements.quantile(0.25),
                        'q3': rendements.quantile(0.75),
                        'n': len(rendements),
                        'n_years': len(subset['annee'].unique())
                    }
                    
                    # Patterns
                    time_data = subset.groupby('annee')['rendement'].mean().reset_index()
                    patterns[selection] = self._detect_patterns(time_data, 'rendement')
                    
                    # Risques
                    cv = (stats[selection]['std'] / stats[selection]['mean'] * 100) if stats[selection]['mean'] > 0 else 0
                    
                    if cv > 35:
                        risks.append(f"**{selection}** : Très forte variabilité (CV={cv:.0f}%)")
                    
                    if stats[selection]['min'] < 20:
                        risks.append(f"**{selection}** : Années avec rendements <20 hl/ha")
                    
                    plafond = self.plafonds.get(selection, 90)
                    if stats[selection]['max'] > plafond:
                        risks.append(f"**{selection}** : Dépassement du plafond de {plafond} hl/ha")
                    
                    # Tendances
                    if selection in patterns and 'trend' in patterns[selection]:
                        trend = patterns[selection]['trend']
                        absolute_trend = abs(trend)
                        if absolute_trend > 0.5:
                            if trend > 0:
                                recommendations.append(f"**{selection}** : Tendance haussière de {trend:.2f} hl/ha/an")
                                
                            else:
                                recommendations.append(f"**{selection}** : Tendance baissière de {absolute_trend:.2f} hl/ha/an")

        # Comparaisons
        if len(selections) > 1:
            means = {k: v['mean'] for k, v in stats.items()}
            best = max(means, key=means.get)
            worst = min(means, key=means.get)
            
            comparisons = {
                'best': best,
                'worst': worst,
                'best_value': means[best],
                'worst_value': means[worst],
                'gap': means[best] - means[worst],
                'gap_percent': ((means[best] - means[worst]) / means[worst] * 100) if means[worst] > 0 else 0
            }
        
        # Génération analyse
        analysis_data = {
            'stats': stats,
            'patterns': patterns,
            'risks': risks,
            'recommendations': recommendations,
            'comparisons': comparisons
        }
        
        natural_analysis = self._generate_rendement_analysis(analysis_data)
        
        analysis = {
            'title': f"Analyse des rendements par {mode}",
            'timestamp': datetime.now().isoformat(),
            'stats': stats,
            'patterns': patterns,
            'comparisons': comparisons,
            'risks': risks,
            'recommendations': recommendations,
            'natural_analysis': natural_analysis
        }
        
        return analysis
    
    def analyze_volume(self, data: pd.DataFrame, mode: str, selections: List[str]) -> Dict[str, Any]:
        """Analyse des volumes"""
        
        df_analysis = data.copy()
        
        # Mapping des modes vers les noms de colonnes réels
        mode_to_col = {
            "Couleur": "code_couleur",
            "Departement": "code_departement",  # Sans accent
            "Zone": "Zone",
            "Cepage": "code_cepage"  # Sans accent
        }
        
        col_name = mode_to_col.get(mode, mode)
        
        stats = {}
        patterns = {}
        recommendations = []
        
        for selection in selections:
            subset = df_analysis[df_analysis[col_name] == selection]
            
            if not subset.empty:
                volumes = subset['volume'].dropna()
                
                if len(volumes) > 0:
                    total_volume = volumes.sum()
                    mean_volume = volumes.mean()
                    
                    stats[selection] = {
                        'total': total_volume,
                        'mean': mean_volume,
                        'median': volumes.median(),
                        'std': volumes.std(),
                        'cv': (volumes.std() / mean_volume * 100) if mean_volume > 0 else 0,
                        'n_years': len(subset['annee'].unique())
                    }
                    
                    time_data = subset.groupby('annee')['volume'].sum().reset_index()
                    patterns[selection] = self._detect_patterns(time_data, 'volume')
        
        # Analyse croisée
        cross_analysis = {}
        if len(selections) > 1:
            totals = {k: v['total'] for k, v in stats.items()}
            total_global = sum(totals.values())
            for k, v in totals.items():
                cross_analysis[k] = {
                    'share': (v / total_global * 100) if total_global > 0 else 0,
                    'total': v
                }
        
        # Génération analyse simplifiée
        lines = []
        lines.append(f"**Analyse des volumes par {mode}**")
        lines.append("")
        
        for groupe, stat in stats.items():
            share = cross_analysis.get(groupe, {}).get('share', 0)
            lines.append(f"**{groupe}** :")
            lines.append(f"  • Volume total : **{stat['total']:,.0f} hl** ({share:.1f}% du total)")
            lines.append("")
        
        natural_analysis = "\n".join(lines)
        
        analysis = {
            'title': f"Analyse des volumes par {mode}",
            'timestamp': datetime.now().isoformat(),
            'stats': stats,
            'patterns': patterns,
            'cross_analysis': cross_analysis,
            'recommendations': recommendations,
            'natural_analysis': natural_analysis
        }
        
        return analysis
    
    def get_learning_summary(self) -> Dict:
        """Résumé de l'apprentissage"""
        n_analyses = len(self.memory['analyses'])
        n_feedback = len(self.memory['feedback'])
        
        return {
            'n_analyses': n_analyses,
            'n_feedback': n_feedback,
            'insights': [
                f"{n_analyses} analyses réalisées",
                f"{n_feedback} retours utilisateur"
            ] if n_analyses > 0 else ["Prêt à analyser"]
        }
    
    def _learn_from_feedback(self, analysis_id: str, feedback: Dict):
        """Apprend des retours"""
        self.memory['feedback'].append({
            'analysis_id': analysis_id,
            'feedback': feedback,
            'timestamp': datetime.now()
        })
        self._save_memory()
    #code IA Julien
    def _load_fallback_climate_data(self):
        if hasattr(self, "_climate_fallback_cache"):
            return self._climate_fallback_cache
        path_climate_AI = Path("climate_zones_AI_generate.json")
        if not path_climate_AI.exists():
            return None
        with open(path_climate_AI, "r", encoding="utf-8") as f:
            self._climate_fallback_cache = json.load(f)
        return self._climate_fallback_cache

    def agent_climate_similarity(self, zone_id, temperature_moyenne, precipitations_totales):
        prompt_climate = f"""
            Vous êtes une IA d'analyse climatique spécialisé dans la viticulture. Votre tâche est d'analyser toutes les zones en une seule réponse. 
            Vous devez évaluer la similarité entre une zone donnée et les zones de référence en fonction de deux indicateurs clés : 
            la température moyenne annuelle et les précipitations totales annuelles.
            CONTRAINTES STRICTES :
            - Comparer avec le Languedoc (ma région sur laquelle je travaille)
            - Climat méditerranéen en Languedoc-Roussillon avec des influences variées citées ci-dessous:
                Détails de chaque zone pour rappel pour avoir analyse fiable:
                Zone 1: zone humide de l'arrière-pays,
                Zone 2: zone de montagne avec des sols acides et peu profonds,
                Zone 3: zone de piémont avec une réserve utile limitante,
                Zone 4: zone froide et sèche autour du Pic Saint-Loup,
                Zone 5: zone de sols de qualité moyenne dans l’arrière-pays,
                Zone 6: *zone de sols profonds sur côtes tempérées,
                Zone 7: *zone avec le plus grand nombre de jours très chauds mais sols profonds,
            - Utilise uniquement des régions viticoles réelles internationales hors France 
            - Mettre des sous-régions idéalement (Californie ou Afrique du Sud sont trop vastes mais cite une zone)
            - Liste les cépages dominants (en vin rouge, vin blanc et en rosé) utilisés dans ces sous-régions 
            - Cépages en vin rouge, vin blanc et en rosé adaptés à ce type de climat pour lutter contre le dérèglement climatique
            - Utiliser les cépages autorisées en Pays d'Oc IGP pour ta réponse sur les cépages résistants et qui sont autorisées dans ces sous-régions pour la partie
            - Cépages en vin rouge, vin blanc et en rosé adaptés à ce type de climat et résistants pour lutter contre le dérèglement climatique
            - https://info.agriculture.gouv.fr/boagri/document_administratif-9231994c-2220-475c-b784-b51d291c7c7c
            - Ne pas mettre de régions viticoles françaises dans tes réponses 
            - Mettre des sous-régions idéalement (Californie ou Afrique du Sud sont trop vastes mais cite une zone)
            - Etre fiable sur la topographie de la région
            - Ne pas inventer de noms de régions et de sous-régions dans la partie "Cépages utilisés dans ces sous-régions"
            - Facteurs limitantes dans ces sous-régions (manque d'eau ou autre chose)
            - Détaille sur besoin en irrigation si c'est critique dis le et donne les raisons
            - S'il y a déjà eu des épisodes de sécheresse accrue dans ces sous-régions, mentionne les
            - dans la partie Détails sur ces sous-régions sur le rendement moyen (vin rouge, blanc, rose), volume total (vin rouge, blanc, rose), et précipitations annuelles moyennes
            - Va voir les bonnes sources de données fiables locales/internationales en les confrontant (viticoles ou autre) pour récupérer le rendement moyen/couleur et le volume total/couleur, les températures annuelles et les précipitations annuelles
            - (donner si possible volume total (rouge, blanc, rose) en hl et rendement moyen par couleur (rouge, blanc, rose) en hl/ha)
            - Mettre "non connu" si les rendements moyen annuels sont trop faibles (en-dessous de 30 hl/ha)
            - Donne le volume total si tu arrives à trouver (en hl)
            - Détails sur ces sous-régions sur le rendement moyen en hl/ha et précipitations annuelles moyennes en mm (donner températures moyennes annuelles)
            - Justifie chaque analogie par le climat
            - Zone : {zone_id}
            - Température moyenne : {temperature_moyenne} °C
            - Précipitations totales : {precipitations_totales} mm
        Réponds à mon besoin en précisant les points suivants :
        1. Type de climat :
        2. Quels sont les 2 régions viticoles mondiales avec de fortes similitudes (sous-régions idéalement) avec la zone {zone_id} :
        3. Quelles sont les raisons de cette similarité ?
        4. Détails sur ces sous-régions sur le rendement moyen et précipitations annuelles moyennes :
        5. Quels sont les cépages en vin rouge, vin blanc et en rosé utilisés dans ces sous-régions ?
        6. Quels sont les cépages en vin rouge, vin blanc et en rosé adaptés à ce type de climat et résistants pour lutter contre le dérèglement climatique ?
        7. Quels sont les facteurs limitants dans ces sous-régions ?
        8. Quel est le besoin en irrigation (faible/modéré/fort) dans ces régions ? 
        9. Quelles mesures ont été prises pour garantir une bonne gestion de l'eau ?
        """
        key = f"{zone_id}_{temperature_moyenne}_{precipitations_totales}"
        if key in self.cache:
            return self.cache[key]
        if not self.client_climate:
            fallback = self._load_fallback_climate_data()
            if fallback and str(zone_id) in fallback:
                return fallback[str(zone_id)]
            return self._fallback_climate_analysis(
                zone_id,
                temperature_moyenne,
                precipitations_totales
            )
        try:
            response_AI = self.client_climate.chat.completions.create(model="gpt-4o-mini",
                                                                        messages=[{"role": "user", "content": prompt_climate}],
                                                                        max_tokens=1000,
                                                                        temperature=0.85
                                                                      ) 
            result = response_AI.choices[0].message.content
            self.cache[key] = result
            return result
        except Exception as e:
            fallback = self._load_fallback_climate_data()
            if fallback and str(zone_id) in fallback:
                return fallback[str(zone_id)]
            return self._fallback_climate_analysis(
                zone_id,
                temperature_moyenne,
                precipitations_totales
            )