# Air Quality API

API FastAPI pour exposer les modèles de qualité de l’air et les scénarios contrefactuels.

## Routes

- `GET /` : état du service
- `GET /config` : variables, statistiques et classes attendues par le frontend
- `POST /predict` : prédiction de classe, probabilités, valeur estimée et CATE
- `POST /whatif?pct=-20` : simulation d’une variation du traitement NO2

## Lancer localement

```bash
python -m pip install -r requirements.txt
uvicorn api.index:app --reload
```

Les modèles sont chargés depuis `models/`. Le frontend doit utiliser la variable `NEXT_PUBLIC_API_BASE` avec l’URL publique de cette API.
