# CLI Nástroj: Binary Image Rectangle Decomposition

## Kontext

Projekt obsahuje viacero implementovaných algoritmov pre dekompozíciu binárnych obrázkov. Doteraz sa experimenty spúšťali cez dedikované Python skripty (`run_dm.py`, `run_ga.py`, atď.) s hardcodovanými parametrami. Cieľom je vytvoriť jednotný CLI nástroj, ktorý umožní interaktívne spúšťanie ľubovoľného algoritmu s voľbou vstupného/výstupného priečinka a parametrov — vrátane automatickej binarizácie vstupných obrázkov.

---

## Stav implementácie

| Krok | Súbor | Stav |
|------|-------|------|
| 1 | `experiments/src/runner.py` — `run_single_experiment_from_array()` | [x] |
| 2 | `src/cli/registry.py` — ALGORITHM_REGISTRY | [x] |
| 3 | `src/cli/presets/*.toml` — GA presety | [x] |
| 4 | `src/cli/arg_groups/ga_args.py` — GA argument group | [x] |
| 5 | `src/cli/input_handler.py` — InputSpec + resolve_input() | [x] |
| 6 | `src/cli/config_builder.py` — build_config() | [x] |
| 7 | `src/cli/output_manager.py` — OutputManager | [x] |
| 8 | `src/cli/commands/list_cmd.py` — list + help | [x] |
| 9 | `src/cli/commands/preprocess_cmd.py` — binarizácia | [x] |
| 10 | `src/cli/commands/run_cmd.py` — execution loop | [x] |
| 11 | `src/cli/dispatcher.py` + `__main__.py` — routing | [x] |

---

## Architektúra: Vrstvy systému

```
┌─────────────────────────────────┐
│         POUŽÍVATEĽ (shell)       │
│  python -m src.cli run --algo dm │
└────────────────┬────────────────┘
                 │
         ┌───────▼────────┐
         │  dispatcher.py  │  ← routing subpríkazov
         └───────┬─────────┘
    ┌────────────┼────────────┐
    │            │            │
┌───▼───┐  ┌────▼────┐  ┌───▼────┐
│run_cmd│  │list_cmd │  │pre_cmd │
└───┬───┘  └─────────┘  └────────┘
    │
┌───┴──────────────────────────────┐
│        CLI Infrastructure         │
│  input_handler  config_builder    │
│  output_manager  progress         │
│  registry       ga_args           │
└───┬──────────────────────────────┘
    │
┌───▼──────────────────────────────┐
│     Existujúca infraštruktúra     │
│  runner.py   config.py  logger.py │
│  preprocessing.py  visualization  │
└───┬──────────────────────────────┘
    │
┌───▼──────────────────────────────┐
│         Algoritmy                 │
│  dm  gdm  quadtree  largest_rect  │
│  graph_based  ga_*                │
└──────────────────────────────────┘
```

---

## Príkazy (Command Structure)

### `list` — zoznam algoritmov
```
python -m src.cli list
```
Vypíše tabuľku: kľúč algoritmu | meno | popis | dostupné presety (GA)

### `help <algo>` — detail algoritmu
```
python -m src.cli help ga_gdm
python -m src.cli help dm
```
Vypíše všetky parametre, presety, príklad volania.

### `preprocess` — binarizácia vstupných obrázkov
```
python -m src.cli preprocess --input <dir_alebo_subor> --output <dir>
                              [--invert | --no-invert]
                              [--format npy|png|both]
```

### `run` — spustenie algoritmu

**Spoločné flagy pre všetky algoritmy:**
```
python -m src.cli run --algo <kľúč>
                      --input <subor_alebo_dir>
                      --output <dir>
                      [--no-viz]
                      [--no-csv]
                      [--verbose]
                      [--max-images N]
                      [--dataset-name <meno>]
                      [--invert | --no-invert]
```

**Deterministické algoritmy (dm, gdm, graph_based):**
```
python -m src.cli run --algo dm --input img.npy --output out/
python -m src.cli run --algo gdm --input img.npy --output out/
python -m src.cli run --algo graph_based --input img.npy --output out/
```

**Quadtree:**
```
python -m src.cli run --algo quadtree --input img.npy --output out/
                      [--no-full-decomposition] [--no-trim]
```

**Largest-Rect:**
```
python -m src.cli run --algo largest_rect --input img.npy --output out/
                      [--coverage-threshold 0.95]
```

**Genetický algoritmus — skrátené kľúče (zodpovedajú ExperimentConfig.algorithm):**
```
# Kľúče: ga_dm | ga_gdm | ga_random | ga_qtd | ga_lrf
python -m src.cli run --algo ga_gdm
                      [--ga-preset default|fast|research|<cesta.toml>]
                      [--ga-pop-size 20]
                      [--ga-generations 100]
                      [--ga-patience 5]
                      [--ga-seed 42]
                      [--ga-crossover subset_greedy|single_point|two_point|uniform]
                      [--ga-penalty 2.0]
                      [--ga-p-geometry 0.3] [--ga-p-merge 0.1]
                      [--ga-p-local 0.5]    [--ga-p-largest 0.2]
                      [--ga-p-delete 0.2]   [--ga-p-split 0.2]
                      [--ga-p-shift 0.05]   [--ga-p-repair 0.5]
                      --input img.npy --output out/
```

ILP algoritmy ostávajú len v `experiments/scripts/run_ilp_*.py`.

---

## GA Parameter Handling: 3-vrstvový systém

```
1. ExperimentConfig defaults (najnižšia priorita)
        ↓
2. TOML preset súbor (--ga-preset fast.toml)
        ↓
3. Individuálne --ga-* CLI flagy (najvyššia priorita)
```

**Vstavané presety** (`src/cli/presets/`):
- `default.toml` — zrkadlí ExperimentConfig defaults
- `fast.toml` — nízky pop_size, nízka patience (rýchle testovanie)
- `research.toml` — najlepšie známe hyperparametre z experimentov

Vlastný preset: `--ga-preset myconfig.toml`

---

## Spracovanie vstupov (Input Handling)

`input_handler.resolve_input()` pre každý súbor vykoná:

```
súbor.npy → np.load() → kontrola hodnôt
    unique values ⊆ {0,1}    → OK, priamo použiteľné
    unique values ⊆ {0,255}  → normalize (img > 0).astype(int)
    inak (grayscale)          → WARNING + image_to_binary()

súbor.png/.jpg → vždy image_to_binary() → (img > 0).astype(int)

priečinok → nájdi *.npy, *.png, *.jpg → klasifikuj každý súbor
```

`--invert` flag: tmavý objekt na svetlom pozadí (default=True).

---

## Výstupná štruktúra (Output)

Zachováva existujúcu hierarchiu (kompatibilné s dashboardom):

```
<output_dir>/
  csv/<algo>/<dataset>/
      results.csv
      generations.csv        (len GA)
  rectangles/<algo>/<dataset>/
      <stem>_rects_N.json
  visualizations/<algo>/<dataset>/
      <stem>_rects_N.png
```

Pre GA: vloží `seed_<N>/` vrstvu (rovnaký vzor ako `experiments/results/`).

---

## Štruktúra súborov (Module Layout)

```
src/
  cli/
    __init__.py
    __main__.py              ← python -m src.cli vstupný bod
    dispatcher.py            ← routing subpríkazov (argparse)
    registry.py              ← ALGORITHM_REGISTRY (single source of truth)
    input_handler.py         ← resolve_input(), InputSpec dataclass
    config_builder.py        ← build_config(algo, namespace) → ExperimentConfig
    output_manager.py        ← OutputManager (JSON + PNG + CSV)
    progress.py              ← ASCII progress pre batch mode
    commands/
      __init__.py
      run_cmd.py             ← hlavný execution loop
      preprocess_cmd.py      ← binarizácia príkaz
      list_cmd.py            ← list + help
    arg_groups/
      __init__.py
      ga_args.py             ← GA argument group + preset resolver
    presets/
      default.toml
      fast.toml
      research.toml
```

---

## Kritická zmena v existujúcej infraštruktúre

`runner.py` aktuálne prijíma len cestu k súboru. Pre non-.npy vstupy (PNG po binarizácii) treba pridať:

```python
# experiments/src/runner.py
def run_single_experiment_from_array(
    img: np.ndarray,
    config: ExperimentConfig,
) -> Tuple[object, Dict, List[int]]:
    """Rovnaký výstup ako run_single_experiment, ale prijíma pole namiesto cesty."""
    ...
```

---

## Rozhodnutia (finalizované)

| Otázka | Rozhodnutie |
|--------|-------------|
| CLI framework | `argparse` (stdlib, žiadne nové závislosti) |
| GA kľúč | Skrátene: `--algo ga_gdm`, `--algo ga_dm`, atď. |
| ILP v CLI | Nie — ostáva len v `experiments/scripts/run_ilp_*.py` |
| Umiestnenie | `src/cli/` balíček, `python -m src.cli` |
| --invert v batch | Jeden globálny flag pre celý priečinok |

---

## Overenie (Verification)

```bash
# Zoznam algoritmov
python -m src.cli list
python -m src.cli help ga_gdm

# Single image (.npy)
python -m src.cli run --algo dm \
  --input data/datasets/leafs_selected/npy/leaf_001.npy \
  --output /tmp/out/

# PNG vstup (auto-binarizácia)
python -m src.cli run --algo gdm \
  --input some_image.png \
  --output /tmp/out/

# GA s presetom + override
python -m src.cli run --algo ga_gdm \
  --ga-preset fast --ga-pop-size 50 \
  --input data/datasets/analysis/leafs_subset/npy/ \
  --output /tmp/out/

# Batch mode
python -m src.cli run --algo quadtree \
  --input data/datasets/leafs_selected/npy/ \
  --output /tmp/out/

# Preprocess PNG → npy
python -m src.cli preprocess \
  --input raw_images/ \
  --output data/datasets/my_dataset/ \
  --format both
```
