"""
Utilitaires pour la préparation des données pour le classificateur d'opinions.
Ce module fournit des fonctions et classes pour :
- Convertir les labels textuels en indices numériques
- Créer des datasets PyTorch pour l'entraînement
- Gérer le padding dynamique des batches
"""

from typing import Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# Mapping des labels vers indices pour les 4 classes d'opinion
LABEL_TO_IDX = {
    "Positive": 0,
    "Négative": 1,
    "Neutre": 2,
    "NE": 3
}

# Mapping inverse : indices vers labels
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

# Liste des aspects à classifier
ASPECTS = ["Prix", "Cuisine", "Service"]


class OpinionDataset(Dataset):
    """
    Dataset PyTorch pour les avis de restaurants avec annotations d'opinions.
    
    Attributes:
        encodings: Les textes tokenisés et encodés
        labels: Dictionnaire avec les labels pour chaque aspect (Prix, Cuisine, Service)
    """
    
    def __init__(
        self, 
        texts: list[str], 
        tokenizer: PreTrainedTokenizer,
        labels: Optional[dict[str, list[int]]] = None,
        max_length: int = 256
    ):
        """
        Initialise le dataset.
        
        Args:
            texts: Liste des textes d'avis
            tokenizer: Tokenizer du modèle pré-entraîné
            labels: Dictionnaire optionnel avec les labels par aspect
            max_length: Longueur maximale des séquences
        """
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx: int) -> dict:
        """Retourne un exemple avec ses encodings et labels."""
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        
        if self.labels is not None:
            for aspect in ASPECTS:
                item[f"label_{aspect.lower()}"] = torch.tensor(
                    self.labels[aspect][idx], dtype=torch.long
                )
        
        return item


def prepare_labels(data: list[dict]) -> dict[str, list[int]]:
    """
    Prépare les labels numériques à partir des données d'entraînement.
    
    Args:
        data: Liste de dictionnaires avec les colonnes Avis, Prix, Cuisine, Service
        
    Returns:
        Dictionnaire avec les labels numériques pour chaque aspect
    """
    labels = {aspect: [] for aspect in ASPECTS}
    
    for item in data:
        for aspect in ASPECTS:
            label_text = item[aspect]
            # Gestion des variations possibles dans les labels
            if label_text in LABEL_TO_IDX:
                labels[aspect].append(LABEL_TO_IDX[label_text])
            else:
                # Par défaut, considérer comme NE si label inconnu
                labels[aspect].append(LABEL_TO_IDX["NE"])
    
    return labels


def get_texts(data: list[dict]) -> list[str]:
    """
    Extrait les textes d'avis des données.
    
    Args:
        data: Liste de dictionnaires avec la colonne Avis
        
    Returns:
        Liste des textes d'avis
    """
    return [item["Avis"] for item in data]


class DataCollatorWithPadding:
    """
    Collator pour le padding dynamique des batches.
    Adapte la longueur des séquences au maximum du batch.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, padding: bool = True):
        self.tokenizer = tokenizer
        self.padding = padding
    
    def __call__(self, features: list[dict]) -> dict:
        """
        Combine une liste de features en un batch avec padding.
        
        Args:
            features: Liste de dictionnaires avec les features
            
        Returns:
            Dictionnaire avec les tensors batchés
        """
        # Séparer les input_ids, attention_mask et labels
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        }
        
        # Ajouter les labels si présents
        for aspect in ASPECTS:
            key = f"label_{aspect.lower()}"
            if key in features[0]:
                batch[key] = torch.stack([f[key] for f in features])
        
        return batch
