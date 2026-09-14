# Tor onion training data

LapisLLM can collect text from explicitly allowlisted public `.onion` services through a local Tor client.

The collector is deliberately **not** a dark-web crawler. It does not discover onion services, consume arbitrary indexes, follow cross-site links, or authenticate to private onion services.

## Requirements

Install the data extra so Requests has SOCKS support:

```bash
pip install -e ".[data]"
```

Run a local Tor client. Tor's documented default SOCKS5 endpoint is `127.0.0.1:9050`; Lapis uses `socks5h://127.0.0.1:9050` by default so hostname resolution is delegated through Tor. See the [Tor Project SOCKS/DNS-leak guidance](https://support.torproject.org/little-t-tor/troubleshooting/check-for-leaks/).

## Source allowlist

Create a local source file from `configs/data/onion_sources.example.txt`:

```bash
cp configs/data/onion_sources.example.txt configs/data/onion_sources.txt
```

Add one public v3 `.onion` HTTP(S) URL per line. The collector requires a valid 56-character v3 onion hostname and rejects credentials, fragments, non-onion hosts, and non-HTTP(S) schemes.

Do not commit a local source list containing private, sensitive, or inappropriate targets.

## Collect

```bash
python scripts/fetch_onion_training_data.py \
  --source-file configs/data/onion_sources.txt
```

The command writes:

- `training_data/onion.txt` — cleaned text records;
- `training_data/onion_manifest.json` — source provenance, hashes, settings, and failures.

To explicitly add the collected records to the existing combined corpus:

```bash
python scripts/fetch_onion_training_data.py \
  --source-file configs/data/onion_sources.txt \
  --append-to-combined
```

## Safety and provenance

Treat onion content as untrusted external data. The collector strips active HTML elements and does not execute page scripts. It limits response size, accepts only text-like content types, and permits redirects only when they remain on the same onion host.

These controls do **not** determine whether source material is lawful, licensed, safe, or suitable for training. Dataset operators remain responsible for source selection, provenance, licensing, privacy, and legal compliance. Prefer public technical, research, journalism, privacy, and open-source material with clear provenance.

Generated corpora are experiment inputs and should not be committed by default.
