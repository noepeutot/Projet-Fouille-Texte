"""
Classificateur d'opinions multi-aspects basé sur CamemBERT-large.

Architecture V9: CamemBERT-large + Mean Pooling + 3 classificateurs MLP
(1024→384→GELU→LayerNorm→4) + Mixed Precision (FP16).
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer, 
    AutoModel,
    get_cosine_schedule_with_warmup
)

from config import Config
from data_utils import (
    OpinionDataset, 
    DataCollatorWithPadding,
    prepare_labels, 
    get_texts,
    ASPECTS, 
    IDX_TO_LABEL
)


class MultiHeadClassifier(nn.Module):
    """
    Modèle de classification multi-aspects basé sur CamemBERT-large.
    
    Architecture optimisée V9:
    - Mean Pooling sur tous les tokens (meilleur que [CLS] seul)
    - Couche cachée avec LayerNorm
    - 3 têtes MLP indépendantes
    """
    
    def __init__(self, plm_name: str = "camembert/camembert-large", 
                 hidden_dim: int = 384, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(plm_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Têtes de classification avec couche cachée + LayerNorm
        self.classifiers = nn.ModuleDict({
            aspect: nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 4)
            ) for aspect in ASPECTS
        })
    
    def mean_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling sur les tokens (meilleur que [CLS] seul)."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        return {aspect: self.classifiers[aspect](pooled) for aspect in ASPECTS}


class PLMClassifier:
    """
    Wrapper pour le classificateur d'opinions multi-aspects.
    Configuration : CamemBERT-large, Mean Pooling, FP16, 4 epochs.
    """
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # Configuration
        self.plm_name = "camembert/camembert-large"
        self.max_length = 256
        self.batch_size = 12
        self.learning_rate = 3e-5
        self.num_epochs = 4
        self.warmup_ratio = 0.1
        self.hidden_dim = 384
        self.dropout = 0.1
        self.label_smoothing = 0.1
        self.weight_decay = 0.01
        self.patience = 2
        
        print(f"Chargement de {self.plm_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.plm_name)
        self.model = MultiHeadClassifier(
            plm_name=self.plm_name,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.device = None
        self.scaler = None  # Pour FP16
    
    def _get_device(self, device: int) -> torch.device:
        if device >= 0 and torch.cuda.is_available():
            return torch.device(f"cuda:{device}")
        return torch.device("cpu")
    
    def train(self, train_data: list[dict], val_data: list[dict], device: int = -1) -> None:
        self.device = self._get_device(device)
        self.model.to(self.device)
        
        # Activer FP16 si GPU disponible
        use_fp16 = self.device.type == "cuda"
        if use_fp16:
            self.scaler = GradScaler('cuda')
        
        print(f"Entraînement sur {self.device} (FP16: {use_fp16})")
        print(f"Train: {len(train_data)} | Val: {len(val_data)}")
        
        # Préparer les datasets
        train_dataset = OpinionDataset(
            get_texts(train_data), self.tokenizer,
            labels=prepare_labels(train_data), max_length=self.max_length
        )
        val_dataset = OpinionDataset(
            get_texts(val_data), self.tokenizer,
            labels=prepare_labels(val_data), max_length=self.max_length
        )
        
        collator = DataCollatorWithPadding(self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collator)
        
        # Optimiseur avec weight decay
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Cosine scheduler avec warmup
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        
        # Entraînement avec early stopping
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.num_epochs} ---")
            
            self.model.train()
            total_train_loss = 0.0
            
            n_batches = len(train_loader)
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}/{n_batches}")
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                optimizer.zero_grad()
                
                if use_fp16:
                    with autocast('cuda'):
                        logits = self.model(input_ids, attention_mask)
                        loss = sum(
                            self.criterion(logits[aspect], batch[f"label_{aspect.lower()}"].to(self.device))
                            for aspect in ASPECTS
                        )
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(input_ids, attention_mask)
                    loss = sum(
                        self.criterion(logits[aspect], batch[f"label_{aspect.lower()}"].to(self.device))
                        for aspect in ASPECTS
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                total_train_loss += loss.item()
            
            avg_loss = total_train_loss / len(train_loader)
            print(f"Train Loss: {avg_loss:.4f}")
            
            # Validation
            val_acc, val_details = self._evaluate(val_loader)
            print(f"Validation: {val_acc:.2f}% | {val_details}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping")
                    break
        
        # Restaurer le meilleur modèle
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)
        
        print(f"\nMeilleure exactitude de validation: {best_val_acc:.2f}%")
    
    def _evaluate(self, dataloader: DataLoader) -> tuple[float, dict]:
        self.model.eval()
        correct = {aspect: 0 for aspect in ASPECTS}
        total = 0
        
        use_fp16 = self.device.type == "cuda"
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                if use_fp16:
                    with autocast('cuda'):
                        logits = self.model(input_ids, attention_mask)
                else:
                    logits = self.model(input_ids, attention_mask)
                
                for aspect in ASPECTS:
                    labels = batch[f"label_{aspect.lower()}"].to(self.device)
                    preds = torch.argmax(logits[aspect], dim=-1)
                    correct[aspect] += (preds == labels).sum().item()
                
                total += input_ids.size(0)
        
        details = {aspect: round(100 * correct[aspect] / total, 2) for aspect in ASPECTS}
        avg_acc = sum(details.values()) / len(ASPECTS)
        return avg_acc, details
    
    def predict(self, texts: list[str], device: int = -1) -> list[dict[str, str]]:
        if self.device is None:
            self.device = self._get_device(device)
            self.model.to(self.device)
        
        self.model.eval()
        predictions = []
        use_fp16 = self.device.type == "cuda"
        
        for i in range(0, len(texts), 32):
            batch_texts = texts[i:i + 32]
            
            encodings = self.tokenizer(
                batch_texts, truncation=True, padding=True,
                max_length=self.max_length, return_tensors="pt"
            )
            
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            
            with torch.no_grad():
                if use_fp16:
                    with autocast('cuda'):
                        logits = self.model(input_ids, attention_mask)
                else:
                    logits = self.model(input_ids, attention_mask)
            
            for j in range(len(batch_texts)):
                prediction = {
                    aspect: IDX_TO_LABEL[torch.argmax(logits[aspect][j]).item()]
                    for aspect in ASPECTS
                }
                predictions.append(prediction)
        
        return predictions
