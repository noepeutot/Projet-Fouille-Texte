from config import Config

from plm_classifier import PLMClassifier

class ClassifierWrapper:

    # METTRE LA BONNE VALEUR ci-dessous en fonction de la méthode utilisée
    METHOD: str = 'PLMFT'  # 'LLM' ou 'PLMFT' (for Pretrained Language Model Fine-Tuning)

    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # Initialiser le classificateur PLM (CamemBERT)
        self.classifier = PLMClassifier(cfg)


    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def train(self, train_data: list[dict], val_data: list[dict], device: int) -> None:
        # Entraîner le modèle PLM sur les données
        self.classifier.train(train_data, val_data, device)



    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def predict(self, texts: list[str], device: int) -> list[dict]:
        """
        :param texts:
        :param device: device: un nombre qui identifie le numéro de la gpu sur laquelle le traitement doit se faire
        -1 veut deire que le device est la cpu, et un nombre entier >= 0 indiquera le numéro de la gpu à utiliser
        :return:
        """
        # Utiliser le classificateur PLM qui traite par batch
        all_opinions = self.classifier.predict(texts, device)
        return all_opinions
