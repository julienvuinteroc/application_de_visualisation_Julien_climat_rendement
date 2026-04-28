Détails du projet de modélisation climatique en Pays d'Oc IGP
-----------
Ce projet est une application Streamlit pour obtenir la cartographie de Pays d'Oc IGP.
A partir des données climatiques, nous souhaitons simuler des cartes géographiques afin d'anticiper les aléas climatiques (chaleurs, gel, sécheresses) sur plusieurs années. 
Pour activer l'environnement virtuel, taper la commande suivante:
.\.venv\Scripts\Activate.bat
Nous avons formaté le code informatique avec black et ruff:
Pour black --check dossier/
ex: black --check src/ ou black --check tests/
black dossier/
Pour ruff, voici les commandes ci-dessous:
ruff check . --fix
ruff check .

Pour tester les fichiers du dossier tests/, il faut taper la commande suivante dans le terminal:
pytest

Version python 3.12
Streamlit entre 1.18 et la version 2



