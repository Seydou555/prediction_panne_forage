# validation Pydantic des entrées

import json
from pydantic import BaseModel, create_model
from src.config import ROOT, CAT_FEATURES

def get_prediction_schema():
    """Crée dynamiquement un schéma Pydantic en associant le bon type (str ou float) 
    à chaque feature en se basant sur metadata.json et src/config.py.
    """
    metadata_path = ROOT / "models" / "metadata.json"
    
    features = []
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        features = metadata.get("features", [])
    
    if features:
        fields = {}
        for feat in features:
            # Si la variable est catégorielle, on exige une chaîne de caractères (str)
            if feat in CAT_FEATURES:
                fields[feat] = (str, ...)
            else:
                # Sinon, c'est une variable numérique (float)
                fields[feat] = (float, ...)
                
        return create_model("ForageInput", **fields)
    
    # Schéma par défaut de secours si le modèle n'est pas encore entraîné
    class DefaultInput(BaseModel):
        pass
    
    return DefaultInput

# Schéma prêt à être importé dans l'API
ForageInput = get_prediction_schema()