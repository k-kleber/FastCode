from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _default_model_from_config() -> str:
    if not CONFIG_PATH.exists():
        return "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    embedding = config.get("embedding", {}) if isinstance(config, dict) else {}
    return embedding.get(
        "model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preload FastCode embedding model")
    parser.add_argument(
        "--model",
        default=_default_model_from_config(),
        help="Sentence-transformers model ID",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for preload (cpu/cuda/mps). Defaults to cpu.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned preload settings without downloading",
    )
    args = parser.parse_args()

    hf_home = os.getenv("HF_HOME", "<default>")
    hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE", "<default>")
    transformers_cache = os.getenv("TRANSFORMERS_CACHE", "<default>")
    st_home = os.getenv("SENTENCE_TRANSFORMERS_HOME", "<default>")

    print("FastCode embedding preload")
    print(f"- Model: {args.model}")
    print(f"- Device: {args.device}")
    print(f"- HF_HOME: {hf_home}")
    print(f"- HUGGINGFACE_HUB_CACHE: {hub_cache}")
    print(f"- TRANSFORMERS_CACHE: {transformers_cache}")
    print(f"- SENTENCE_TRANSFORMERS_HOME: {st_home}")

    if args.dry_run:
        print("Dry run complete (no download).")
        return

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, device=args.device)
    dim = model.get_sentence_embedding_dimension()
    print(f"Preload complete. Embedding dimension: {dim}")


if __name__ == "__main__":
    main()
