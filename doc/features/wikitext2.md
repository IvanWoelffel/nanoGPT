# Plan d'implémentation - WikiText-2 Dataset

## Objectif
Créer un script Python pour importer le dataset WikiText-2 depuis HuggingFace (https://huggingface.co/datasets/mindchain/wikitext2) et le stocker dans `data/wikitext2/` dans le format utilisé par nanoGPT.

## Analyse du Dataset WikiText-2

### Caractéristiques du dataset
- **Source** : mindchain/wikitext2 sur HuggingFace
- **Taille** :
  - 44,836 documents au total (36,718 train + 3,760 validation + 4,358 test)
  - ~4.72 MB téléchargé (format raw)
  - ~13.54 MB généré (format Arrow)
  - ~18.26 MB total sur disque
- **Splits** : 'train', 'validation', 'test' disponibles (pas besoin de créer split)
- **Feature** : 'text' (string) - articles Wikipedia
- **Description** : Collection de 2+ millions de tokens extraits d'articles Wikipedia vérifiés
- **Téléchargement total** : ~5 MB (compressé), ~13.5 MB (décompressé)

### Utilisation historique
- Dataset de référence pour le language modeling (plus populaire que Penn Treebank)
- Utilisé dans de nombreux papiers de recherche (Pointer Sentinel Mixture Models, etc.)
- Idéal pour modèles avec dépendances à long terme (articles complets)
- Variants disponibles : raw (préserve casse, ponctuation, nombres) vs non-raw (tokens inconnus remplacés par <unk>)

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

#### bookcorpus (pattern récent)
- `data/bookcorpus/prepare.py` : Script similaire à openwebtext
- Même pattern de tokenisation et sauvegarde
- Split train/val créé avec `train_test_split`
- Adapté pour dataset très grand (74M documents)

#### Format de sortie standard
- `train.bin` : données d'entraînement tokenisées
- `val.bin` : données de validation tokenisées
- Format : `numpy.array(dtype=np.uint16)` (tokènes GPT-2 BPE)
- Compatible avec le script `train.py`

#### Situation actuelle wikitext2
- **Existe déjà** : `data/wikitext2/train.txt`, `val.txt`, `test.txt`
- Format brut : text files (47,534 lignes train, 4,922 val, 5,782 test)
- **Besoin** : Convertir en format .bin tokenisé nanoGPT

## Plan d'Implémentation

### Étape 1 : Création de la structure de répertoire
```
data/wikitext2/
├── prepare.py      # Script de préparation principal (NOUVEAU)
├── readme.md       # Documentation des résultats (NOUVEAU)
├── train.bin       # Données d'entraînement tokenisées (NOUVEAU)
├── val.bin         # Données de validation tokenisées (NOUVEAU)
├── train.txt       # Données brutes existantes
├── val.txt         # Données brutes existantes
└── test.txt        # Données brutes existantes
```

### Étape 2 : Création de `data/wikitext2/prepare.py`

#### Structure du script
```python
# saves the wikitext2 dataset to a binary file for training.

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # WikiText-2 dataset - ~45K documents, ~2M tokens
    # Using 'wikitext-2-raw-v1' to preserve case, punctuation, numbers
    dataset = load_dataset("mindchain/wikitext2", "wikitext-2-raw-v1", num_proc=num_proc_load_dataset)

    # wikitext2 already has 'train', 'validation', 'test' splits
    # We'll use 'validation' as 'val' to match nanoGPT convention
    split_dataset = {
        'train': dataset['train'],
        'val': dataset['validation']
    }

    # Note: 'test' split exists but we won't use it for training
    # Could be saved separately if needed for evaluation

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = {}
    for split_name, dset in split_dataset.items():
        tokenized[split_name] = dset.map(
            process,
            remove_columns=['text'],
            desc=f"tokenizing {split_name}",
            num_proc=num_proc,
        )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin is ~X.XX MB, val.bin ~X.XX MB
    # train has ~X.XX M tokens
    # val has ~X.XX K tokens

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
```

#### Points clés
- Suivre exactement le pattern `openwebtext/prepare.py` et `bookcorpus/prepare.py`
- Utiliser config `wikitext-2-raw-v1` pour préserver casse, ponctuation, nombres (comme PTB)
- Utiliser splits existants ('train' et 'validation' comme 'val')
- Multiprocessing optionnel (dataset petit: 45K documents vs 74M pour bookcorpus)
- Adaptation: moins de workers nécessaires (num_proc=4 suffisant)
- Conserver test.txt existant pour référence

#### Variantes possibles
1. **wikitext-2-raw-v1** (recommandé): Préserve casse, ponctuation, nombres
2. **wikitext-2-v1**: Remplace OOV tokens par `<unk>`, plus adapté au word-level modeling

### Étape 3 : Création de `data/wikitext2/readme.md`

#### Contenu attendu
```markdown
## WikiText-2 dataset

After running `prepare.py`:

- train.bin is ~X.XX MB, val.bin ~X.XX MB
- train has ~X.XX M tokens
- val has ~X.XX K tokens

This came from 36,718 training documents and 3,760 validation documents in total.

The WikiText dataset is a collection of over 2 million tokens extracted from the set of verified Good and Featured articles on Wikipedia.

References:
- WikiText paper: Merity et al. (2016) "Pointer Sentinel Mixture Models"
- HuggingFace: https://huggingface.co/datasets/mindchain/wikitext2
- Original: https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/
```

### Étape 4 : Mise à jour de `AGENTS.md` (optionnel)

Ajouter les commandes pour WikiText-2 :
```bash
# Préparation du dataset
python data/wikitext2/prepare.py

# Entraînement avec WikiText-2
python train.py --dataset=wikitext2 --block_size=1024 --batch_size=12
```

## Considérations Techniques

### Performance et Mémoire
1. **Multiprocessing** : `num_proc=4` suffisant (dataset petit: 45K documents)
2. **Memory efficiency** :
   - Écriture par batches (1024 shards)
   - Utilisation de `numpy.memmap` pour éviter charger tout en mémoire
   - `remove_columns=['text']` après tokenisation pour libérer mémoire
3. **Estimation temps** :
   - Téléchargement: ~1-5 min (~5 MB)
   - Tokenisation: ~5-15 min (45K documents)
   - Écriture: ~1-5 min selon disque

### Contraintes
- Espace disque requis: ~50-100 MB temporaires + ~50-100 MB binaires finaux
- RAM requise: ~2-4 GB (multiprocessing)
- Compatible Python 3.x, torch, numpy, tiktoken, datasets, tqdm

### Avantages vs datasets locaux
- Standardisation avec autres datasets nanoGPT
- Utilisation de tiktoken GPT-2 BPE (cohérence avec GPT-2)
- Format .bin optimisé pour `train.py`
- Compatible avec multiprocessing et DDP
- Facile à reproduire via HuggingFace

## Risques potentiels
- Dataset petit (45K docs) → peut être limité pour entraînement large modèle
- Test split existant non utilisé → considérer option pour sauvegarder test.bin
- Licences multiples (cc-by-sa-3.0, gfdl) → attention à l'utilisation commerciale
- Articles Wikipedia peuvent être biaisés → considérer pour research

## Fichiers à Créer

1. **`data/wikitext2/prepare.py`** (~70 lignes)
   - Script principal de préparation
   - Basé sur `data/openwebtext/prepare.py` et `data/bookcorpus/prepare.py`
   - Adapté pour wikitext2 (splits existants, config raw)

2. **`data/wikitext2/readme.md`** (~20 lignes)
   - Documentation des résultats
   - Statistiques après exécution

## Tests et Validation

### Tests manuels après implémentation
1. Vérifier téléchargement complet depuis HuggingFace
2. Vérifier création des splits train/val
3. Vérifier fichiers .bin créés
4. Test rapide de lecture:
   ```python
   import numpy as np
   train = np.memmap('data/wikitext2/train.bin', dtype=np.uint16, mode='r')
   print(f"Tokens: {len(train):,}")
   ```
5. Test d'entraînement minimal:
   ```bash
   python train.py --dataset=wikitext2 --max_iters=100 --eval_iters=10
   ```

### Validation attendue
- Fichiers train.bin et val.bin créés
- Taille des fichiers raisonnable (estimation: ~20-50 MB train, ~2-5 MB val)
- Pas d'erreurs lors de la lecture avec numpy.memmap
- Compatible avec train.py existant
- Nombre de tokens correspond à documentation (~2M train)

## Comparaison avec fichiers existants

| Aspect | Fichiers txt existants | Fichiers .bin cibles |
|--------|----------------------|---------------------|
| Format | Text brut (lignes) | Binaire tokenisé |
| Lecture | Directe | numpy.memmap |
| Compatible train.py | Non | Oui |
| Taille (est.) | ~15 MB | ~50 MB |
| Utilisation | Référence | Entraînement |

## Questions pour l'utilisateur

1. **Configuration à utiliser** : Préférez-vous `wikitext-2-raw-v1` (préserve casse/ponctuation) ou `wikitext-2-v1` (tokens <unk>) ?
   - raw-v1: Meilleur pour character-level, préserve information (recommandé)
   - v1: Meilleur pour word-level, plus simple

2. **Test split** : Souhaitez-vous également sauvegarder test.bin (en plus de train/val) ?

3. **Numéro de workers** : `num_proc=4` est-il approprié pour votre machine ? (dataset petit)

4. **Fichiers txt** : Souhaitez-vous supprimer train.txt/val.txt/test.txt après création des .bin ?

## Références
- OpenWebText prepare.py: `data/openwebtext/prepare.py`
- BookCorpus prepare.py: `data/bookcorpus/prepare.py`
- nanoGPT train.py: `train.py`
- HuggingFace Datasets: https://huggingface.co/docs/datasets
- WikiText dataset: https://huggingface.co/datasets/mindchain/wikitext2
- Tiktoken: https://github.com/openai/tiktoken
- WikiText paper: Merity et al. (2016) "Pointer Sentinel Mixture Models"
