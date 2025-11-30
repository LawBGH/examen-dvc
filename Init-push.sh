# Dans le repo
dvc init

# Étape de téléchargement : on versionne data via DVC
dvc add data/raw/raw.csv
git add data/.gitignore raw.csv.dvc
git commit -m "Track raw data with DVC"

# Configurer le remote DagsHub (remplace <user>/<repo> par ton dépôt)
# Sur DagsHub, crée un repo vide, copie l’URL 'Data Remote' (par défaut S3 ou DagsHub storage)
dvc remote add -d dagshub https://dagshub.com/LawBGH/examen-dvc.dvc
git add .dvc/config
git commit -m "Configure DVC remote (DagsHub)"
