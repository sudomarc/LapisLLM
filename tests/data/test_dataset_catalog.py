from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "configs" / "data" / "datasets.yaml"
EXPECTED_IDS = {
    "fineweb",
    "fineweb_edu",
    "fineweb2",
    "dolma",
    "redpajama_v2",
    "the_stack",
    "the_stack_v2",
    "wikipedia",
    "wikibooks",
    "pes2o",
    "openwebmath",
    "smoltalk",
    "openhermes_25",
    "metamathqa",
    "numinamath_cot",
    "self_oss_instruct_sc2",
}


def load_catalog():
    return yaml.safe_load(CATALOG.read_text(encoding="utf-8"))


def test_dataset_catalog_contains_every_registered_source():
    catalog = load_catalog()
    sources = catalog["sources"]
    ids = {source["id"] for source in sources}
    assert ids == EXPECTED_IDS
    assert len(sources) == len(EXPECTED_IDS)


def test_every_source_has_provenance_and_license_metadata():
    for source in load_catalog()["sources"]:
        assert source["dataset"]
        assert source["stage"] in {"pretrain", "sft"}
        assert source["source_url"]
        assert source["license"]
        assert source["license_url"]


def test_all_profile_only_references_known_sources():
    catalog = load_catalog()
    ids = {source["id"] for source in catalog["sources"]}
    for profile in catalog["profiles"].values():
        assert set(profile["include"]).issubset(ids)
