import argparse
import json
import os
import socket
import time
from typing import Dict, Iterable, List, Optional

import pandas as pd
import urllib.error
import urllib.parse
import urllib.request


def _strip_ensembl_version(ensembl_id: str) -> str:
    # ENSGxxxxxx.x -> ENSGxxxxxx
    if not isinstance(ensembl_id, str):
        return str(ensembl_id)
    return ensembl_id.split(".")[0]


def _chunks(xs: List[str], chunk_size: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), chunk_size):
        yield xs[i : i + chunk_size]


def _ensembl_lookup_id_batch(
    ids: List[str],
    cache: Dict[str, str],
    sleep_s: float,
    species: Optional[str],
    request_timeout: float,
    max_retries: int,
) -> None:
    """
    Ensembl REST POST /lookup/id
    - 官方文档：Maximum POST size = 1000
    - 返回通常是 dict: { "<id>": { ...gene fields... }, ... }
    """
    if not ids:
        return

    server = "https://rest.ensembl.org"
    ext = "/lookup/id"
    url = server + ext
    if species:
        # species 是 query 参数（不要塞进 POST body，否则可能 400）
        url = f"{url}?species={urllib.parse.quote(species)}"

    body_obj: Dict = {"ids": ids}

    data = json.dumps(body_obj).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; project_GRN/1.0; +https://ensembl.org)",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    payload = None
    for attempt in range(max(1, max_retries)):
        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            # 4xx：整批通常不可通过“多等几秒”解决，直接二分或记空
            if len(ids) > 1:
                mid = len(ids) // 2
                _ensembl_lookup_id_batch(
                    ids[:mid],
                    cache=cache,
                    sleep_s=sleep_s,
                    species=species,
                    request_timeout=request_timeout,
                    max_retries=max_retries,
                )
                _ensembl_lookup_id_batch(
                    ids[mid:],
                    cache=cache,
                    sleep_s=sleep_s,
                    species=species,
                    request_timeout=request_timeout,
                    max_retries=max_retries,
                )
                return
            cache[ids[0]] = ""
            time.sleep(sleep_s)
            return
        except (
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            OSError,
            json.JSONDecodeError,
        ):
            # 网络抖动 / 读超时：同一批次重试；仍失败则缩小批次
            if attempt < max_retries - 1:
                time.sleep(sleep_s * (2**attempt))
                continue
            if len(ids) > 1:
                mid = len(ids) // 2
                _ensembl_lookup_id_batch(
                    ids[:mid],
                    cache=cache,
                    sleep_s=sleep_s,
                    species=species,
                    request_timeout=request_timeout,
                    max_retries=max_retries,
                )
                _ensembl_lookup_id_batch(
                    ids[mid:],
                    cache=cache,
                    sleep_s=sleep_s,
                    species=species,
                    request_timeout=request_timeout,
                    max_retries=max_retries,
                )
                return
            cache[ids[0]] = ""
            time.sleep(sleep_s)
            return

    if payload is None:
        return

    if not isinstance(payload, dict):
        # 非预期格式：保守处理
        for ens in ids:
            cache.setdefault(ens, "")
        time.sleep(sleep_s)
        return

    for ens, gene in payload.items():
        if gene is None:
            cache[ens] = ""
            continue
        if not isinstance(gene, dict):
            cache[ens] = ""
            continue
        symbol = gene.get("display_name") or gene.get("external_name") or ""
        cache[ens] = str(symbol) if symbol else ""

    # 有些 id 可能不在返回里（未找到）
    for ens in ids:
        cache.setdefault(ens, "")

    time.sleep(sleep_s)


def main():
    parser = argparse.ArgumentParser(description="Prepare TCGA-LUAD aligned expression matrix.")
    parser.add_argument("--tcga-star-fpkm-gz", required=True, help="TCGA-LUAD.star_fpkm.tsv.gz path.")
    parser.add_argument("--exp-csv", required=True, help="Expression CSV used to provide gene_list order.")
    parser.add_argument("--output-csv", required=True, help="Write aligned expression CSV.")
    parser.add_argument(
        "--cache-json",
        default=None,
        help="Optional cache json for Ensembl->symbol mapping (default: <output_dir>/ensembl_to_symbol_cache.json).",
    )
    parser.add_argument("--api-sleep-sec", type=float, default=0.05)
    parser.add_argument(
        "--ensembl-species",
        default="homo_sapiens",
        help="Optional species hint for Ensembl POST /lookup/id (default: homo_sapiens).",
    )
    parser.add_argument(
        "--lookup-batch-size",
        type=int,
        default=200,
        help="Ensembl POST /lookup/id batch size (max 1000; 默认 200 降低超时概率).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="单次 HTTP 请求超时（秒），默认 600。",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="同一批次在网络超时/解析失败时的最大重试次数。",
    )
    parser.add_argument("--min-symbol-length", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    cache_path = args.cache_json or os.path.join(os.path.dirname(args.output_csv), "ensembl_to_symbol_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    # 1) gene_list order
    gene_df = pd.read_csv(args.exp_csv, usecols=[0])
    gene_list = gene_df.iloc[:, 0].astype(str).tolist()
    symbol_to_idx = {g: i for i, g in enumerate(gene_list)}
    num_genes = len(gene_list)

    # 2) load TCGA table
    # columns: Ensembl_ID, TCGA-xxxx, ...
    tcga_df = pd.read_csv(args.tcga_star_fpkm_gz, sep="\t", compression="gzip")
    tcga_cols = tcga_df.columns.tolist()
    if len(tcga_cols) < 2:
        raise ValueError("TCGA input looks invalid: need at least Ensembl_ID + one sample column.")

    sample_cols = tcga_cols[1:]

    # 3) map Ensembl_ID -> symbol
    tcga_ensembl = tcga_df["Ensembl_ID"].astype(str).map(_strip_ensembl_version).tolist()
    unique_ensembl = sorted(set(tcga_ensembl))

    print(f"[prepare_tcga_luad_aligned] unique Ensembl IDs: {len(unique_ensembl)}")
    batch_size = max(1, min(int(args.lookup_batch_size), 1000))
    pending = [ens for ens in unique_ensembl if ens not in cache]

    print(f"[prepare_tcga_luad_aligned] pending lookups (not cached): {len(pending)} | batch_size={batch_size}")

    for bi, chunk in enumerate(_chunks(pending, batch_size)):
        _ensembl_lookup_id_batch(
            chunk,
            cache=cache,
            sleep_s=args.api_sleep_sec,
            species=args.ensembl_species,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
        )

        if args.min_symbol_length > 1:
            for ens in chunk:
                sym = cache.get(ens, "")
                if not sym or len(sym) < args.min_symbol_length:
                    cache[ens] = ""

        if (bi + 1) % 5 == 0 or (bi + 1) * batch_size >= len(pending):
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            done = min((bi + 1) * batch_size, len(pending))
            print(f"  batch {bi+1} saved cache ... ({done}/{len(pending)})")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    # 4) fill aligned matrix in gene_list order
    aligned = [[0.0] * len(sample_cols) for _ in range(num_genes)]
    assigned = [False] * num_genes

    # iterate rows (TCGA is big but still manageable for one-time preprocessing)
    for row_idx, ens in enumerate(tcga_ensembl):
        symbol = cache.get(ens, "")
        if not symbol or symbol not in symbol_to_idx:
            continue
        gene_i = symbol_to_idx[symbol]
        if assigned[gene_i]:
            # if multiple Ensembl map to same symbol, keep the first row to avoid nondeterminism
            continue
        aligned[gene_i] = tcga_df.iloc[row_idx, 1:].astype(float).tolist()
        assigned[gene_i] = True

    # 5) write CSV: first column gene symbol, then samples
    out_df = pd.DataFrame(aligned, columns=sample_cols)
    out_df.insert(0, "Gene", gene_list)
    out_df.to_csv(args.output_csv, index=False)

    missing = assigned.count(False)
    print(f"[prepare_tcga_luad_aligned] done. missing genes filled as 0: {missing}/{num_genes}")


if __name__ == "__main__":
    main()

