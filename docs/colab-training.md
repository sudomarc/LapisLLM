# LapisLLM — first Colab GPU training run

This is the supported path for the first bounded GPU pretraining run. It uses
`configs/colab.yaml`, streams a small set of pretraining sources, records provenance,
and writes the checkpoint locally in the Colab runtime.

## 1. Start Colab with GPU

In **Runtime → Change runtime type**, select a GPU runtime.

## 2. Clone and install

```python
!git clone https://github.com/sudomarc/LapisLLM.git
%cd LapisLLM
!pip install -e '.[data]'
!nvidia-smi
```

## 3. Build the bounded corpus

The first run uses FineWeb-Edu, French Wikipedia, and OpenWebMath. The builder streams
the sources and caps the combined corpus instead of downloading the full datasets.

```python
!python scripts/build_colab_corpus.py \
  --output training_data/colab_pretrain.txt \
  --manifest training_data/colab_pretrain_manifest.json \
  --max-chars 200000000
```

For a smoke test before the full run:

```python
!python scripts/build_colab_corpus.py --max-chars 5000000
```

## 4. Train the model

```python
!python -m lapis.dev.cli train \
  --config configs/colab.yaml \
  --device cuda \
  --data training_data/colab_pretrain.txt \
  --checkpoint checkpoints/colab-pretrain.pt \
  --epochs 1 \
  --monitor-interval 250
```

The trainer owns tokenizer creation for a fresh run and stores the tokenizer next to
the checkpoint. The Colab profile uses a larger model than `local-dev.yaml` and FP16 on
CUDA.

## 5. Resume

```python
!python scripts/train.py \
  --config configs/colab.yaml \
  --device cuda \
  --data training_data/colab_pretrain.txt \
  --resume checkpoints/colab-pretrain.pt \
  --checkpoint checkpoints/colab-pretrain.pt \
  --epochs 1
```

## 6. Inspect and generate

```python
!python -m lapis.dev.cli inspect --checkpoint checkpoints/colab-pretrain.pt
!python -m lapis.dev.cli generate \
  --checkpoint checkpoints/colab-pretrain.pt \
  --device cuda \
  --max-new-tokens 128 \
  'The future of artificial intelligence is'
```

## Notes

The corpus is deliberately bounded for a first Colab run. Increase the corpus size,
training steps, sequence length, or model size only after the complete pipeline is
stable and the checkpoint can be resumed successfully.

Do not commit `training_data/`, `data/raw/`, checkpoints, or tokenizer artifacts.
