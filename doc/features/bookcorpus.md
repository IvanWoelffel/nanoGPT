# Plan d'implémentation - BookCorpus Dataset

## Objectif
Créer un script Python pour importer le dataset BookCorpus depuis HuggingFace (https://huggingface.co/datasets/SamuelYang/bookcorpus) et le stocker dans `data/bookcorpus/` dans le format utilisé par nanoGPT.

## Analyse du Dataset BookCorpus

### Caractéristiques du dataset
- **Source** : SamuelYang/bookcorpus sur HuggingFace
- **Taille** : 
  - 74M+ de lignes (documents/livres)
  - 4.84 GB en format Arrow (original)
  - ~4.8 GB de texte brut
- **Splits** : Uniquement 'train' disponible (pas de val/test)
- **Feature** : 'text' (string) - texte des livres
- **Description** : Corpus de livres utilisé pour entraîner BERT et GPT-N
- **Téléchargement total** : ~1.18 GB (compressé), ~6 GB (décompressé)

### Utilisation historique
- Utilisé pour entraîner les modèles BERT (Google), GPT-N (OpenAI)
- ~11,038 livres de livres gratuits écrits par des auteurs inédits
- Filtre: livres avec >20K mots

## Analyse du Codebase nanoGPT

### Pattern existant pour les datasets
nanoGPT utilise une structure cohérente pour la préparation des datasets :

#### openwebtext (dataset de référence)
- `data/openwebtext/prepare.py` : Script de préparation
- Utilise huggingface `datasets` pour le téléchargement
- Tokenisation avec `tiktoken.get_encoding("gpt2")`
- Multiprocessing pour la tokenisation (`num_proc=8`)
- Sauvegarde en `.bin` avec `numpy.memmap`
- Split train/val créé manuellement (90%/10%)
- `data/openwebtext/readme.md` : Documentation des résultats

#### shakespeare (dataset simple)
- `data/shakespeare/prepare.py` : Script plus simple
- Téléchargement direct via requests
- Même pattern de tokenisation et sauvegarde

#### Format de sortie standard
- `train.bin` : données d'entraînement tokenisées
- `val.bin` : données de validation tokenisées
- Format : `numpy.array(dtype=np.uint16)` (tokènes GPT-2 BPE)
- Compatible avec le script `train.py`

## Plan d'Implémentation

### Étape 1 : Création de la structure de répertoire
```
data/bookcorpus/
├── prepare.py      # Script de préparation principal
├── readme.md       # Documentation des résultats
├── train.bin       # Données d'entraînement tokenisées
└── val.bin         # Données de validation tokenisées
```

### Étape 2 : Création de `data/bookcorpus/prepare.py`

#### Structure du script
```python
"""
Prepare the BookCorpus dataset for GPT training.
Similar to openwebtext/prepare.py pattern.
"""

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# Configuration
num_proc = 8  # nombre de workers pour .map()
num_proc_load_dataset = num_proc
enc = tiktoken.get_encoding("gpt2")

# Téléchargement du dataset
dataset = load_dataset("SamuelYang/bookcorpus", num_proc=num_proc_load_dataset)

# Création du split train/val (dataset n'a que 'train')
split_dataset = dataset["train"].train_test_split(
    test_size=0.0005,  # ~37K documents pour validation
    seed=2357,
    shuffle=True
)
split_dataset['val'] = split_dataset.pop('test')

# Fonction de tokenisation
def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    return {'ids': ids, 'len': len(ids)}

# Tokenisation avec multiprocessing
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

# Sauvegarde en .bin avec memmap
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    
    # Écriture par batches pour éviter OOM
    total_batches = 1024
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True)
        arr_batch = np.concatenate(batch['ids'])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()
```

#### Points clés
- Suivre exactement le pattern `openwebtext/prepare.py`
- Adapter pour le dataset BookCorpus (nom du dataset, structure)
- Utiliser `train_test_split` pour créer val (90/10 ou 99.9/0.1 comme openwebtext)
- Multiprocessing crucial pour 74M+ documents
- Écriture par batches pour éviter problèmes mémoire

### Étape 3 : Création de `data/bookcorpus/readme.md`

#### Contenu attendu
```markdown
## BookCorpus dataset

After running `prepare.py`:

- train.bin is ~X.XX GB, val.bin ~X.XX GB
- train has ~X.XX billion tokens
- val has ~X.XX million tokens

This came from 74,004,228 documents in total.

References:
- BookCorpus paper: Zhu et al. (2015) "Aligning Books and Movies"
- Original dataset: https://yknzhu.wixsite.com/mbweb
- HuggingFace: https://huggingface.co/datasets/SamuelYang/bookcorpus
```

### Étape 4 : Mise à jour de `AGENTS.md` (si existe)

Ajouter les commandes pour BookCorpus :
```bash
# Préparation du dataset
python data/bookcorpus/prepare.py

# Entraînement avec BookCorpus
python train.py --dataset=bookcorpus --block_size=1024 --batch_size=12
```

## Considérations Techniques

### Performance et Mémoire
1. **Multiprocessing** : Utiliser `num_proc=8` (ou ajuster selon CPU)
2. **Memory efficiency** : 
   - Écriture par batches (1024 shards)
   - Utilisation de `numpy.memmap` pour éviter charger tout en mémoire
   - `remove_columns=['text']` après tokenisation pour libérer mémoire
3. **Estimation temps** :
   - Téléchargement: ~1-2 GB (10-30 min selon connexion)
   - Tokenisation: Plusieurs heures (74M documents)
   - Écriture: ~1-2 heures selon disque

### Contraintes
- Espace disque requis: ~6-8 GB temporaires + ~X GB binaires finaux
- RAM requise: ~4-8 GB (multiprocessing)
- Compatible Python 3.x, torch, numpy, tiktoken, datasets, tqdm

### Risques potentiels
- Dataset très grand (74M documents) - temps de traitement long
- Split val/actuel peut être trop petit avec 0.0005 (37K docs) → ajuster si nécessaire
- Attention aux droits d'auteur (documenté dans les papiers BookCorpus)
- Possibles doublons dans le dataset (documenté)

## Fichiers à Créer

1. **`data/bookcorpus/prepare.py`** (~80 lignes)
   - Script principal de préparation
   - Basé sur `data/openwebtext/prepare.py`

2. **`data/bookcorpus/readme.md`** (~20 lignes)
   - Documentation des résultats
   - Statistiques après exécution

## Tests et Validation

### Tests manuels après implémentation
1. Vérifier téléchargement complet
2. Vérifier création des splits train/val
3. Vérifier fichiers .bin créés
4. Test rapide de lecture:
   ```python
   import numpy as np
   train = np.memmap('data/bookcorpus/train.bin', dtype=np.uint16, mode='r')
   print(f"Tokens: {len(train):,}")
   ```
5. Test d'entraînement minimal:
   ```bash
   python train.py --dataset=bookcorpus --max_iters=100 --eval_iters=10
   ```

### Validation attendue
- Fichiers train.bin et val.bin créés
- Taille des fichiers raisonnable (estimation: ~10-15 GB train, ~50-100 MB val)
- Pas d'erreurs lors de la lecture avec numpy.memmap
- Compatible avec train.py existant

## Questions pour l'utilisateur

1. **Proportion split train/val** : Préférer-vous 90/10 ou 99.9/0.1 (comme openwebtext) ?
   - 90/10: Plus de données de validation (~7.4M docs), mais moins d'entraînement
   - 99.9/0.1: Comme openwebtext (~37K docs val), maximise entraînement

2. **Numéro de workers** : `num_proc=8` est-il approprié pour votre machine ?

3. **Espace disque** : Avez-vous ~10-15 GB d'espace disque disponible ?

## Références
- OpenWebText prepare.py: `data/openwebtext/prepare.py`
- nanoGPT train.py: `train.py`
- HuggingFace Datasets: https://huggingface.co/docs/datasets
- BookCorpus dataset: https://huggingface.co/datasets/SamuelYang/bookcorpus
- Tiktoken: https://github.com/openai/tiktoken
