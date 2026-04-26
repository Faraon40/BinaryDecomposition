# CLI Návod na použitie

Nástroj umožňuje spúšťanie algoritmov pre dekompozíciu binárnych obrázkov priamo z príkazového riadku bez potreby upravovať Python skripty.

## Spustenie

```bash
# Aktivácia prostredia
source .venv/bin/activate

# Základný formát
python -m src.cli <príkaz> [možnosti]
```

---

## Príkazy

### `list` — zoznam algoritmov

```bash
python -m src.cli list
```

Vypíše tabuľku všetkých dostupných algoritmov s ich rýchlosťou.

### `help` — detailná nápoveda

```bash
python -m src.cli help <algoritmus>

# Príklady
python -m src.cli help dm
python -m src.cli help ga_gdm
python -m src.cli help quadtree
```

Vypíše všetky dostupné parametre a príklad volania pre daný algoritmus.

### `preprocess` — binarizácia obrázkov

```bash
python -m src.cli preprocess --input <cesta> --output <priečinok> [možnosti]
```

Konvertuje PNG/JPG obrázky do binárnej podoby a uloží ich ako `.npy` polia a/alebo binárne PNG.

| Možnosť | Popis | Predvolená |
|---------|-------|-----------|
| `--input` | Vstupný obrázok alebo priečinok | — |
| `--output` | Výstupný priečinok | — |
| `--format` | `npy`, `png`, alebo `both` | `both` |
| `--invert` | Tmavý objekt na svetlom pozadí (list na bielom) | zapnuté |
| `--no-invert` | Svetlý objekt na tmavom pozadí | — |

```bash
# Príklady
python -m src.cli preprocess --input raw_images/ --output data/datasets/moje/
python -m src.cli preprocess --input foto.png --output data/datasets/moje/ --format npy
python -m src.cli preprocess --input scany/ --output data/ --no-invert
```

### `run` — spustenie algoritmu

```bash
python -m src.cli run --algo <algoritmus> --input <cesta> --output <priečinok> [možnosti]
```

Spustí vybraný algoritmus na jednom obrázku alebo celom priečinku.

---

## Dostupné algoritmy

### Deterministické

| Kľúč | Názov | Rýchlosť |
|------|-------|----------|
| `dm` | Delta Method | rýchly |
| `gdm` | Generalized Delta Method | rýchly |
| `quadtree` | Quadtree | rýchly |
| `largest_rect` | Largest Rectangle | rýchly |
| `graph_based` | Graph-Based (Optimálny) | **veľmi pomalý** (~2.5 min/obr.) |

### Genetický algoritmus (GA)

| Kľúč | Inicializácia |
|------|--------------|
| `ga_dm` | Delta Method |
| `ga_gdm` | Generalized DM — **najlepšie výsledky** |
| `ga_random` | Náhodná |
| `ga_qtd` | Quadtree |
| `ga_lrf` | Largest Rectangle |

---

## Spoločné možnosti pre `run`

| Možnosť | Popis |
|---------|-------|
| `--algo` | Kľúč algoritmu (povinné) |
| `--input` | Vstupný `.npy` súbor, PNG/JPG, alebo priečinok |
| `--output` | Výstupný priečinok |
| `--no-viz` | Nevytvárať vizualizáciu PNG |
| `--no-csv` | Nelogovať výsledky do CSV |
| `--verbose` | Zobrazovať výstup algoritmu |
| `--limit N` | Spracovať maximálne N obrázkov (pre testovanie) |
| `--dataset-name` | Vlastný názov datasetu v výstupných cestách |
| `--invert` | Tmavý objekt na svetlom pozadí (default pre PNG) |
| `--no-invert` | Svetlý objekt na tmavom pozadí |

---

## Parametre jednotlivých algoritmov

### `quadtree`

```bash
python -m src.cli run --algo quadtree --input leafs/ --output out/
                      [--no-full-decomposition]
                      [--no-trim]
```

| Možnosť | Popis |
|---------|-------|
| `--no-full-decomposition` | Adaptívny min_size namiesto delenia až na veľkosť 2 |
| `--no-trim` | Preskočiť GDM trim zmiešaných listov |

### `largest_rect`

```bash
python -m src.cli run --algo largest_rect --input leafs/ --output out/
                      [--coverage-threshold 0.95]
```

| Možnosť | Popis | Predvolená |
|---------|-------|-----------|
| `--coverage-threshold` | Zastaviť keď je pokrytých F% pixelov | `0.95` |

---

## Parametre GA algoritmov

GA algoritmy majú 3-vrstvový systém parametrov:

```
1. ExperimentConfig defaults  (najnižšia priorita)
         ↓
2. --ga-preset (TOML preset súbor)
         ↓
3. Individuálne --ga-* flagy  (najvyššia priorita)
```

### Predvolené presety

| Preset | pop_size | generations | patience | Použitie |
|--------|----------|-------------|----------|----------|
| `default` | 15 | 150 | 12 | Konfigurácia s najlepším pomerom čas/výkon — nájdená experimentmi |
| `fast` | 10 | 30 | 3 | Rýchle testovanie — znížená populácia, skoré zastavenie |
| `research` | 50 | 200 | 20 | Vynútená evolúcia — veľká populácia a vysoký stagnačný limit pre maximálnu kvalitu riešenia |

```bash
# Použitie presetu
python -m src.cli run --algo ga_gdm --ga-preset fast --input img.npy --output out/
python -m src.cli run --algo ga_gdm --ga-preset research --input leafs/ --output out/

# Preset + override konkrétnych parametrov
python -m src.cli run --algo ga_gdm --ga-preset fast --ga-pop-size 20 --input img.npy --output out/

# Vlastný TOML preset
python -m src.cli run --algo ga_gdm --ga-preset moj_preset.toml --input img.npy --output out/
```

### Všetky GA parametre

| Možnosť | Popis | Predvolená (default preset) |
|---------|-------|---------------------------|
| `--ga-preset` | Preset: `default`, `fast`, `research`, alebo cesta k `.toml` | `default` |
| `--ga-pop-size` | Veľkosť populácie | `30` |
| `--ga-generations` | Maximálny počet generácií | `100` |
| `--ga-patience` | Zastavenie po N generáciách bez zlepšenia | `5` |
| `--ga-seed` | Náhodný seed (ak nie je zadaný, generuje sa automaticky) | náhodný |
| `--ga-elite-size` | Počet elitných jedincov | `3` |
| `--ga-penalty` | Násobiteľ penalizácie za neplatné riešenia | `2.0` |
| `--ga-crossover` | Metóda kríženia | `subset_greedy` |

**Pravdepodobnosti mutácií:**

| Možnosť | Mutácia | Predvolená |
|---------|---------|-----------|
| `--ga-p-geometry` | Geometrická (G) | `0.30` |
| `--ga-p-merge` | Zlúčenie (M) | `0.10` |
| `--ga-p-local` | Lokálna repartícia (L) | `0.50` |
| `--ga-p-largest` | Largest-rect (R) | `0.20` |
| `--ga-p-delete` | Vymazanie (D) | `0.20` |
| `--ga-p-split` | Rozdelenie (S) | `0.20` |
| `--ga-p-shift` | Posun (H) | `0.05` |
| `--ga-p-repair` | Oprava pokrytia | `0.50` |

**Metódy kríženia** (`--ga-crossover`):
- `subset_greedy` — Subset Crossover s greedy rozšírením *(predvolené, najlepšie)*
- `single_point` — Jednobodové kríženie
- `two_point` — Dvojbodové kríženie
- `uniform` — Uniformné kríženie

---

## Vlastný TOML preset pre GA

Skopíruj niektorý zo vstavaných presetov a uprav ho:

```bash
# Zobraziť vstavaný preset
cat src/cli/presets/research.toml
```

```toml
# moj_preset.toml
pop_size = 15
patience = 12
generations = 150
elite_size = 3
penalty = 2.0
crossover_method = "subset_greedy_relaxed"
p_delete = 0.30
p_split = 0.30
p_geometry = 0.30
p_shift = 0.1
p_local = 0.50
p_largest = 0.20
p_merge = 0.20
p_repair = 0.50
```

```bash
python -m src.cli run --algo ga_gdm --ga-preset moj_preset.toml \
  --input data/datasets/leafs_selected/npy/ --output out/
```

---

## Vstupné formáty

Nástroj automaticky detekuje typ vstupu:

| Formát | Spracovanie |
|--------|-------------|
| `.npy` | Načíta priamo, normalizuje na {0, 1} |
| `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif` | Automaticky binarizuje (Otsu + morfológia) |
| Priečinok | Nájde všetky `.npy` a rasterové obrázky, zoradí ich |

Ak má obrázok tmavý objekt na **bielom pozadí** (napr. listy, siluety):
→ použi predvolené nastavenie (`--invert` je zapnuté)

Ak má obrázok svetlý objekt na **tmavom pozadí**:
→ použi `--no-invert`

---

## Výstupná štruktúra

```
<output>/
  csv/<algo>/<dataset>/
      results.csv          ← metriky pre každý obrázok
      generations.csv      ← história generácií (len GA)
      errors.log           ← chyby (ak nastali)
  rectangles/<algo>/<dataset>/
      <obr>_rects_N.json   ← súradnice obdĺžnikov
  visualizations/<algo>/<dataset>/
      <obr>_rects_N.png    ← vizualizácia riešenia
```

`results.csv` obsahuje:
- Pre deterministické algoritmy: `image_name, rectangle_count, execution_time_sec`
- Pre GA: navyše `seed, pop_size, final_fitness, generations_used`

---

## Príklady použitia

```bash
# Rýchly test DM na 3 obrázkoch
python -m src.cli run --algo dm \
  --input data/datasets/analysis/leafs_subset/npy/ \
  --output /tmp/test/ \
  --max-images 3

# GDM na celom datasete bez vizualizácií
python -m src.cli run --algo gdm \
  --input data/datasets/leafs_selected/npy/ \
  --output experiments/results/my_run/ \
  --no-viz

# GA s fast presetom na jednom obrázku
python -m src.cli run --algo ga_gdm \
  --ga-preset fast \
  --ga-seed 42 \
  --input data/datasets/leafs_selected/npy/leaf_001.npy \
  --output /tmp/ga_test/

# GA s research presetom, vlastný názov datasetu
python -m src.cli run --algo ga_gdm \
  --ga-preset research \
  --input data/datasets/leafs_selected/npy/ \
  --output experiments/results/ \
  --dataset-name leafs_selected

# Binarizácia vlastných obrázkov
python -m src.cli preprocess \
  --input ~/Downloads/moje_listy/ \
  --output data/datasets/moje_listy/ \
  --format both

# Následné spustenie algoritmu na binarizovaných obrázkoch
python -m src.cli run --algo gdm \
  --input data/datasets/moje_listy/npy/ \
  --output results/
```
