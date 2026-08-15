# -*- coding: utf-8 -*-
"""sheets_queue.py — durable queue + paste plan for the Google Sheet.

Writing to the sheet goes through Claude in Chrome, which means it only works
while a browser is open. Training runs finish whenever they finish. So results
are never written directly: eval output lands in a local JSONL queue first, and
is flushed to the sheet whenever a browser is available. A run that completes at
04:00 with nobody watching still gets recorded.

    # after a run finishes (on Ibex, no browser needed)
    python sheets_queue.py add --results-csv sadeed_outputs/gemma-4-E4B-it-QLoRA__results.csv

    # refresh the snapshot of what's currently in the sheet (needs network)
    python sheets_queue.py snapshot

    # show what would be pasted where — always run this before writing
    python sheets_queue.py plan

    # after the paste succeeds, retire those entries
    python sheets_queue.py done --ids 3,4,5

The Fine-Tuning Models tab, confirmed against a live CSV export:

    A Notes   B DER_ce   C DER_noce   D WER_ce   E WER_noce   F Data Per-Processing
    G Results   H Assigned to   I Method   J Status   K Model

Rows 2-13 are pre-created, one per (Model x Method x Results) combination, so a
flush is an UPDATE of B:E on an existing row — never an append. The plan resolves
the row by matching G/I/K against a snapshot rather than trusting row numbers,
so inserting a row in the sheet doesn't silently send results to the wrong line.

Only the benchmark's Mean row is written to this tab; the MSA and CA breakdowns
go to the "Results of test models" tab, which has a Domain Track column for them.
"""

import argparse
import csv
import io
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
SHEET_ID = "1QXmHc2ut8DasxXoOWEaeZIgGzPePixEga1fW_Op6JGA"
FT_GID = "982162547"          # Fine-Tuning Models
QUEUE = os.path.join(HERE, "sheets_queue.jsonl")
SNAPSHOT = os.path.join(HERE, "finetuning_tab_snapshot.csv")

# Sheet column letters for the four metrics — contiguous, so one paste fills them.
METRIC_COLS = ("B", "C", "D", "E")
METRIC_KEYS = ("DER_ce", "DER_noce", "WER_ce", "WER_noce")

COL_RESULTS, COL_METHOD, COL_MODEL, COL_STATUS = 6, 8, 10, 9   # 0-indexed into the CSV


def col_letter(i):
    return chr(ord("A") + i)


# ============================================================
# Matching an eval label to a sheet row
# ============================================================
def infer_method(label, explicit):
    if explicit:
        return explicit
    low = label.lower()
    if "qlora" in low:
        return "QLoRA"
    if "lora" in low:
        return "LoRA"
    return ""


def normalise_model(name):
    """Collapse the cosmetic differences between our labels and the sheet's.

    The sheet writes 'Qwan 3.5 - 4B' (spaces, a typo'd 'Qwan'); we produce
    'Qwen/Qwen3.5-4B' and 'gemma-4-E4B-it-QLoRA'. Strip to lowercase alphanumerics
    and drop the method suffix so those meet in the middle.
    """
    n = name.lower()
    for suffix in ("-qlora-tashkeel", "-lora-tashkeel", "-qlora", "-lora"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    n = n.split("/")[-1]
    n = "".join(ch for ch in n if ch.isalnum())
    return n.replace("qwan", "qwen")


def method_matches(sheet_method, ours):
    """'QLoRA Fine-Tuning' vs 'QLoRA' — and LoRA must not match QLoRA."""
    s = sheet_method.lower().replace(" ", "")
    o = ours.lower().replace(" ", "")
    if not o:
        return False
    if o == "lora":
        return s.startswith("lora")          # 'qlorafine-tuning' does not
    return s.startswith(o)


# ============================================================
# Queue
# ============================================================
def load_queue():
    if not os.path.exists(QUEUE):
        return []
    out = []
    with io.open(QUEUE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_queue(items):
    with io.open(QUEUE, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def cmd_add(args):
    if not os.path.exists(args.results_csv):
        sys.exit(f"FATAL: no such file: {args.results_csv}")

    with io.open(args.results_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"FATAL: {args.results_csv} has no rows")

    items = load_queue()
    next_id = max((i["id"] for i in items), default=0) + 1
    added = 0

    for r in rows:
        split = (r.get("Results") or "").strip()
        track = (r.get("Domain Track") or "-").strip()
        # The Fine-Tuning tab has one row per split; the benchmark's per-domain
        # breakdown belongs on the other tab, so only Mean comes through here.
        if split == "SadeedDiac-25" and track != "Mean (MSA + CA)":
            continue
        if not split:
            continue

        label = (r.get("Model") or "").strip()
        entry = {
            "id": next_id,
            "queued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "pending",
            "source": os.path.basename(args.results_csv),
            "label": label,
            "sheet_model": args.sheet_model or label,
            "method": infer_method(label, args.method or (r.get("Method") or "").strip()),
            "split": split,
            "n": r.get("n", ""),
            "metrics": {k: r.get(k, "") for k in METRIC_KEYS},
        }
        items.append(entry)
        next_id += 1
        added += 1

    save_queue(items)
    print(f"queued {added} row(s) from {args.results_csv} -> {QUEUE}")
    for it in items[-added:] if added else []:
        m = it["metrics"]
        print(f"  #{it['id']:<3} {it['sheet_model']:<28} {it['method']:<6} "
              f"{it['split']:<14} DER_ce={m['DER_ce']}")


def cmd_snapshot(args):
    url = (f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"
           f"/export?format=csv&gid={FT_GID}")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read().decode("utf-8")
    except Exception as e:
        sys.exit(f"FATAL: could not fetch the tab ({type(e).__name__}: {e}).\n"
                 f"If this machine has no network, fetch it in a browser:\n  {url}\n"
                 f"and save it as {SNAPSHOT}")
    with io.open(SNAPSHOT, "w", encoding="utf-8", newline="") as f:
        f.write(data)
    n = len(data.splitlines()) - 1
    print(f"snapshot saved -> {SNAPSHOT} ({n} data rows)")


def read_snapshot():
    if not os.path.exists(SNAPSHOT):
        sys.exit(f"FATAL: no snapshot at {SNAPSHOT}. Run:  python sheets_queue.py snapshot")
    with io.open(SNAPSHOT, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def resolve_row(grid, item):
    """Sheet row number (1-indexed) for this queue entry, or None."""
    want_model = normalise_model(item["sheet_model"])
    hits = []
    for idx, row in enumerate(grid[1:], start=2):      # row 1 is the header
        if len(row) <= COL_MODEL:
            continue
        if (row[COL_RESULTS].strip() == item["split"]
                and normalise_model(row[COL_MODEL]) == want_model
                and method_matches(row[COL_METHOD], item["method"])):
            hits.append(idx)
    if len(hits) == 1:
        return hits[0], None
    if not hits:
        return None, "no matching row"
    return None, f"ambiguous — matches rows {hits}"


def cmd_plan(args):
    items = [i for i in load_queue() if i["status"] == "pending"]
    if not items:
        print("queue is empty — nothing to write")
        return

    grid = read_snapshot()
    print(f"{len(items)} pending row(s). Paste plan for the Fine-Tuning Models tab:\n")

    ok, bad = [], []
    for it in items:
        row, err = resolve_row(grid, it)
        if row is None:
            bad.append((it, err))
            continue
        values = [str(it["metrics"][k]) for k in METRIC_KEYS]
        ok.append((it, row, values))

    for it, row, values in ok:
        target = f"{METRIC_COLS[0]}{row}"
        current = grid[row - 1][1:5] if len(grid) >= row else []
        occupied = any(c.strip() for c in current)
        print(f"  #{it['id']:<3} {it['sheet_model']} / {it['method']} / {it['split']}")
        print(f"       select {target}  then paste (tab-separated):")
        print(f"       {chr(9).join(values)}")
        print(f"       fills {METRIC_COLS[0]}{row}:{METRIC_COLS[-1]}{row}"
              + ("   [!] OVERWRITES existing values " + str(current) if occupied else ""))
        print()

    if bad:
        print("UNRESOLVED — these will not be written:")
        for it, err in bad:
            print(f"  #{it['id']:<3} {it['sheet_model']} / {it['method']} / "
                  f"{it['split']}: {err}")
        print()

    print(f"resolved {len(ok)}/{len(items)}. After pasting, retire them with:")
    print(f"  python sheets_queue.py done --ids {','.join(str(i['id']) for i, _, _ in ok)}")


def cmd_done(args):
    ids = {int(x) for x in args.ids.split(",") if x.strip()}
    items = load_queue()
    hit = 0
    for it in items:
        if it["id"] in ids and it["status"] == "pending":
            it["status"] = "written"
            it["written_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            hit += 1
    save_queue(items)
    print(f"marked {hit} entr(y/ies) written; "
          f"{sum(1 for i in items if i['status'] == 'pending')} still pending")


def cmd_list(args):
    items = load_queue()
    if not items:
        print("queue is empty")
        return
    for it in items:
        m = it["metrics"]
        print(f"#{it['id']:<3} [{it['status']:<7}] {it['sheet_model']:<28} "
              f"{it['method']:<6} {it['split']:<14} "
              f"DER_ce={m['DER_ce']:<7} WER_ce={m['WER_ce']}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="queue an eval_sadeed.py results CSV")
    a.add_argument("--results-csv", required=True)
    a.add_argument("--sheet-model", default="",
                   help="override the Model cell to match (default: infer from label)")
    a.add_argument("--method", default="", help="LoRA / QLoRA (default: infer from label)")
    a.set_defaults(func=cmd_add)

    s = sub.add_parser("snapshot", help="refresh the local copy of the tab")
    s.set_defaults(func=cmd_snapshot)

    pl = sub.add_parser("plan", help="show target cells + paste blocks")
    pl.set_defaults(func=cmd_plan)

    d = sub.add_parser("done", help="retire queue entries after a successful paste")
    d.add_argument("--ids", required=True)
    d.set_defaults(func=cmd_done)

    ls = sub.add_parser("list", help="show the whole queue")
    ls.set_defaults(func=cmd_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
