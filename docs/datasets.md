# LapisLLM datasets

LAPIS keeps training corpora outside the Git repository. The canonical registry is:

```text
configs/data/datasets.yaml
```

The registry currently covers the following sources:

| Source | Stage | Hugging Face dataset | Purpose |
|---|---|---|---|
| FineWeb | pretrain | `HuggingFaceFW/fineweb` | general web text |
| FineWeb-Edu | pretrain | `HuggingFaceFW/fineweb-edu` | educational/high-quality web text |
| FineWeb2 | pretrain | `HuggingFaceFW/fineweb-2` | multilingual web text |
| Dolma | pretrain | `allenai/dolma` | broad aggregate corpus |
| RedPajama V2 | pretrain | `togethercomputer/RedPajama-Data-V2` | broad multilingual web corpus |
| The Stack | pretrain | `bigcode/the-stack` | source code |
| The Stack v2 | pretrain | `bigcode/the-stack-v2` | large-scale source-code metadata/files pipeline |
| Wikipedia | pretrain | `wikimedia/wikipedia` | encyclopedic knowledge |
| Wikibooks | pretrain | `wikimedia/wikibooks` | educational/reference books |
| peS2o | pretrain | `allenai/peS2o` | open academic/scientific text |
| OpenWebMath | pretrain | `open-web-math/open-web-math` | mathematics text |
| SmolTalk | sft | `HuggingFaceTB/smoltalk` | instruction/conversation SFT |
| OpenHermes 2.5 | sft | `teknium/OpenHermes-2.5` | instruction/conversation SFT |
| MetaMathQA | sft | `meta-math/MetaMathQA` | mathematical instruction tuning |
| NuminaMath-CoT | sft | `AI-MO/NuminaMath-CoT` | mathematical reasoning SFT |
| Self-OSS / StarCoder2 instruct | sft | `bigcode/self-oss-instruct-sc2-exec-filter-50k` | execution-validated code instruction |

## Download

Install the data extras:

```bash
pip install -e '.[data]'
```

Inspect the registry:

```bash
python scripts/download_datasets.py --list
```

Development profile:

```bash
python scripts/download_datasets.py --profile development
```

Recommended profile:

```bash
python scripts/download_datasets.py --profile recommended
```

Everything registered:

```bash
python scripts/download_datasets.py --profile all
```

For a bounded development run, use `--max-records` and optionally `--streaming`:

```bash
python scripts/download_datasets.py --profile recommended --max-records 10000 --streaming
```

A single source can be selected directly:

```bash
python scripts/download_datasets.py --source numinamath_cot
```

Source-specific licensing is deliberately blocked by default. After reviewing the
upstream dataset card and all relevant per-document metadata, an operator can
explicitly opt in:

```bash
python scripts/download_datasets.py --profile all --allow-source-specific-license
```

## Storage contract

Raw material is written to `data/raw/<source-id>/records.jsonl` and accompanied by
`manifest.json`. The manifests record the dataset identifier, stage, split, config,
license metadata, source URL, field mapping, record count, byte count, and a hash of
the registry entry used for the run.

Do not commit `data/raw/`, `data/cleaned/`, `data/deduplicated/`, or `data/tokenized/`.
These are generated artifacts and should stay local or in external object storage.

## Important scale rule

The registry intentionally includes very large corpora such as FineWeb2, Dolma,
RedPajama V2 and The Stack v2. `--profile all` means every registered source is part
of the pipeline; it does not mean a small workstation should download every byte in
one operation. Use dataset configs/language subsets, streaming and `--max-records`
for development.

## Reproducibility

Every training preparation run should retain:

1. the exact registry revision;
2. source configuration and split;
3. upstream dataset revision when available;
4. filtering/deduplication settings;
5. tokenizer version;
6. generated manifests.

Dataset quality and licensing are part of the model provenance and therefore part of
Lapis model cards.
