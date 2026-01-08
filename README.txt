=============================================================================
            PROJET : FOUILLE D'OPINIONS DANS LES AVIS DE RESTAURANTS
=============================================================================

AUTEURS
-------
Noé PEUTOT et Yanis GIRARDIN


DESCRIPTION DU CLASSIFIEUR
--------------------------
Pour ce projet, on a choisi d'utiliser la méthode PLMFT (fine-tuning d'un 
modèle pré-entraîné). On utilise CamemBERT-large comme modèle de base car 
c'est un modèle français performant. 
Le principe est simple : on prend le texte d'un avis, on le fait passer dans 
CamemBERT pour obtenir une représentation vectorielle, puis on utilise trois 
classificateurs (un par aspect : Prix, Cuisine, Service) pour prédire 
l'opinion.

Pour la représentation du texte, nous utilisons le tokenizer de CamemBERT 
avec une longueur maximale de 256 tokens et du padding dynamique. Au lieu 
de ne garder que le token [CLS], on effectue un Mean Pooling sur l'ensemble 
des tokens de la séquence, ce qui permet de mieux capturer l'information 
globale de l'avis. Chaque classificateur est un petit réseau de neurones 
avec deux couches (1024 -> 384 -> 4) et des activations GELU.

Recherches et évolution du modèle :
On a testé plusieurs versions avant d'arriver à notre modèle final. On a 
commencé avec camembert-base (88.72% en 50 min), mais c'était trop long. 
On a essayé de geler les couches de CamemBERT pour accélérer, mais 
l'accuracy a chuté à 74.37%. Mauvaise idée. On est passé à camembert-large 
(88.43% en 37 min), puis on a ajouté l'early stopping (87.94% en 29 min). 
Finalement, on a combiné le Mean Pooling et la Mixed Precision (FP16) pour 
accélérer les calculs sur GPU : 88.72% en seulement 12 minutes. C'est notre
meilleure version.

Hyperparamètres finaux utilisés :
  - Learning rate    : 3e-5
  - Batch size       : 12
  - Epochs           : 4 (avec early stopping, patience=2)
  - Dropout          : 0.1
  - Optimiseur       : AdamW
  - Scheduler        : Cosine avec warmup

Ressources utilisées :
  - PyTorch
  - Transformers (HuggingFace) pour CamemBERT

EXACTITUDE MOYENNE SUR LES DONNEES DE DEV
-----------------------------------------
  - Prix              : 88.00%
  - Cuisine           : 88.17%
  - Service           : 90.00%
  - Exactitude moyenne: 88.72%

Entraînement réalisé sur Google Colab avec GPU T4 en environ 12 minutes.

===============================================================================
