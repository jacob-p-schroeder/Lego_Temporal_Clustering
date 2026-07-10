"""
build_lego_features.py

Builds a per-set feature table from a SQLite LEGO/Rebrickable-style database
(sets, inventories, inventory_parts, inventory_minifigs, minifigs, parts,
part_categories, colors).

Usage:
    python build_lego_features.py

Config (edit the constants near the top of this file):
    MIN_PARTS      - drop sets with fewer than this many parts (0 = no filter)
    HISTOGRAM      - True to also save a num_parts distribution histogram,
                     so you can check where MIN_PARTS lands before committing
    FILTER_SETS    - True to remove any sets that aren't "brick" building sets
    HISTOGRAM_OUT  - file path for that histogram

Design notes (see previous SQL version for the full rationale):
  - Uses the MAX(version) inventory per set as the canonical inventory.
  - total_part_count = non-spare part quantity (matches sets.num_parts).
  - rare part = appears in fewer than RARE_THRESHOLD distinct sets,
    computed across the ENTIRE inventory_parts table (non-spare rows).
  - "printed/decorative" and "technic" are name/category text-match
    heuristics — adjust PRINTED_PATTERN / TECHNIC_PATTERN if needed.
  - All composition/diversity stats are quantity-weighted.
  - Entropy uses natural log (nats); switch np.log -> np.log2 for bits.
"""

import sqlite3
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # write to file, no display needed
import matplotlib.pyplot as plt

RARE_THRESHOLD = 20
PRINTED_PATTERN = r"(?i)print|pattern|sticker"
TECHNIC_PATTERN = r"(?i)technic"

# --- user-editable config -------------------------------------------------
MIN_PARTS = 20          # drop sets with fewer than this many parts (0 = no filter)
HISTOGRAM = True        # True: also save a num_parts distribution histogram
FILTER_SETS = True
HISTOGRAM_OUT = os.path.join("num_parts_hist.png")  # where the histogram gets saved when HISTOGRAM=True
# ---------------------------------------------------------------------------


def to_bool(series: pd.Series) -> pd.Series:
    """Normalize SQLite boolean-ish columns (0/1, 't'/'f', 'true'/'false') to bool."""
    return series.astype(str).str.strip().str.lower().isin({"1", "t", "true", "yes"})


def shannon_entropy(quantities: pd.Series) -> float:
    """Shannon entropy (nats) of a distribution given raw counts/quantities."""
    total = quantities.sum()
    if total <= 0:
        return 0.0
    p = quantities / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def load_tables(conn: sqlite3.Connection):
    sets = pd.read_sql_query(
        "SELECT set_num, name AS set_name, year, theme_id, num_parts FROM sets", conn
    )
    inventories = pd.read_sql_query(
        "SELECT id AS inventory_id, set_num, version FROM inventories", conn
    )
    inv_parts = pd.read_sql_query(
        "SELECT inventory_id, part_num, color_id, quantity, is_spare FROM inventory_parts",
        conn,
    )
    inv_minifigs = pd.read_sql_query(
        "SELECT inventory_id, fig_num, quantity AS mf_qty FROM inventory_minifigs", conn
    )
    parts = pd.read_sql_query(
        "SELECT part_num, name AS part_name, part_cat_id FROM parts", conn
    )
    part_categories = pd.read_sql_query(
        "SELECT id AS part_cat_id, name AS cat_name FROM part_categories", conn
    )
    colors = pd.read_sql_query("SELECT id AS color_id, is_trans FROM colors", conn)

    inv_parts["is_spare"] = to_bool(inv_parts["is_spare"])
    colors["is_trans"] = to_bool(colors["is_trans"])

    return sets, inventories, inv_parts, inv_minifigs, parts, part_categories, colors


def plot_num_parts_histogram(sets: pd.DataFrame, out_path: str, min_parts: int) -> None:
    """
    Plot the distribution of sets.num_parts on a log-x histogram, with a
    vertical line at the MIN_PARTS cutoff, so you can see whether the
    cutoff lands in a sensible gap between "junk" (polybags, minifig
    packs) and genuine small sets.
    """
    vals = sets["num_parts"].clip(lower=1)  # avoid log(0) for 0-part entries

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: log-scale histogram, full range
    axes[0].hist(vals, bins=np.logspace(0, np.log10(vals.max() + 1), 60))
    axes[0].set_xscale("log")
    axes[0].axvline(min_parts, color="red", linestyle="--", label=f"MIN_PARTS = {min_parts}")
    axes[0].set_xlabel("num_parts (log scale)")
    axes[0].set_ylabel("count of sets")
    axes[0].set_title("Full distribution")
    axes[0].legend()

    # Right: zoomed-in linear histogram on the low-part-count region
    zoom = vals[vals <= 100]
    axes[1].hist(zoom, bins=range(0, 101, 2))
    axes[1].axvline(min_parts, color="red", linestyle="--", label=f"MIN_PARTS = {min_parts}")
    axes[1].set_xlabel("num_parts (0-100 range)")
    axes[1].set_ylabel("count of sets")
    axes[1].set_title("Zoomed: 0-100 parts")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    n_below = int((sets["num_parts"] < min_parts).sum())
    n_total = len(sets)
    print(
        f"Histogram saved to {out_path}. "
        f"MIN_PARTS={min_parts} would drop {n_below}/{n_total} sets "
        f"({n_below / n_total:.1%})."
    )


def get_canonical_inventories(inventories: pd.DataFrame) -> pd.DataFrame:
    """One inventory_id per set_num: the highest version."""
    idx = inventories.groupby("set_num")["version"].idxmax()
    return inventories.loc[idx, ["set_num", "inventory_id"]]


def build_features(conn: sqlite3.Connection) -> pd.DataFrame:
    sets, inventories, inv_parts, inv_minifigs, parts, part_categories, colors = load_tables(conn)

    canonical = get_canonical_inventories(inventories)
    canonical_ids = set(canonical["inventory_id"])

    # --- rare-part lookup: computed over ALL inventories, non-spare rows ---
    all_nonspare = inv_parts[~inv_parts["is_spare"]].merge(
        inventories[["inventory_id", "set_num"]], on="inventory_id", how="left"
    )
    part_set_counts = (
        all_nonspare.groupby("part_num")["set_num"].nunique().rename("num_sets").reset_index()
    )
    rare_parts = set(part_set_counts.loc[part_set_counts["num_sets"] < RARE_THRESHOLD, "part_num"])

    # --- restrict inventory_parts to the canonical (latest-version) inventory ---
    ip = inv_parts[inv_parts["inventory_id"].isin(canonical_ids)].merge(
        canonical, on="inventory_id", how="left"
    )
    ip = ip.merge(parts, on="part_num", how="left")
    ip = ip.merge(part_categories, on="part_cat_id", how="left")
    ip = ip.merge(colors, on="color_id", how="left")
    ip["is_rare"] = ip["part_num"].isin(rare_parts)
    ip["is_technic"] = ip["cat_name"].fillna("").str.contains(TECHNIC_PATTERN, regex=True)
    ip["is_printed"] = ip["part_name"].fillna("").str.contains(PRINTED_PATTERN, regex=True)

    nonspare = ip[~ip["is_spare"]]
    spare = ip[ip["is_spare"]]

    # --- scale + composition aggregates (per set) ---
    agg = nonspare.groupby("set_num").agg(
        total_part_count=("quantity", "sum"),
        unique_part_count=("part_num", "nunique"),
        n_distinct_colors=("color_id", "nunique"),
        n_distinct_categories=("part_cat_id", "nunique"),
    )
    agg["spare_part_count"] = spare.groupby("set_num")["quantity"].sum()
    agg["spare_part_count"] = agg["spare_part_count"].fillna(0)

    def weighted_qty(mask_col):
        return nonspare[nonspare[mask_col]].groupby("set_num")["quantity"].sum()

    agg["technic_qty"] = weighted_qty("is_technic")
    agg["printed_qty"] = weighted_qty("is_printed")
    agg["trans_qty"] = weighted_qty("is_trans")
    agg["rare_qty"] = weighted_qty("is_rare")
    agg[["technic_qty", "printed_qty", "trans_qty", "rare_qty"]] = agg[
        ["technic_qty", "printed_qty", "trans_qty", "rare_qty"]
    ].fillna(0)

    # --- entropy features ---
    h_cat = nonspare.groupby("set_num").apply(
        lambda g: shannon_entropy(g.groupby("part_cat_id")["quantity"].sum())
    )
    h_col = nonspare.groupby("set_num").apply(
        lambda g: shannon_entropy(g.groupby("color_id")["quantity"].sum())
    )
    agg["h_cat"] = h_cat
    agg["h_col"] = h_col

    # --- minifigure features ---
    mf = inv_minifigs[inv_minifigs["inventory_id"].isin(canonical_ids)].merge(
        canonical, on="inventory_id", how="left"
    )
    mf_agg = mf.groupby("set_num")["mf_qty"].sum().rename("minifig_count")
    agg = agg.join(mf_agg, how="left")
    agg["minifig_count"] = agg["minifig_count"].fillna(0)

    agg = agg.reset_index()

    # --- final ratios ---
    total = agg["total_part_count"].replace(0, np.nan)
    total_with_spare = (agg["total_part_count"] + agg["spare_part_count"]).replace(0, np.nan)

    agg["spare_part_ratio"] = agg["spare_part_count"] / total_with_spare
    agg["prop_technic"] = agg["technic_qty"] / total
    agg["prop_printed"] = agg["printed_qty"] / total
    agg["prop_transparent"] = agg["trans_qty"] / total
    agg["rare_part_ratio"] = agg["rare_qty"] / total
    agg["minifig_to_part_ratio"] = agg["minifig_count"] / total
    agg["has_minifigs"] = (agg["minifig_count"] > 0).astype(int)

    agg = agg.fillna(0)

    features = sets.merge(agg, on="set_num", how="left")

    final_cols = [
        "set_num", "set_name", "year", "theme_id", "num_parts",
        "total_part_count", "unique_part_count", "minifig_count", "spare_part_ratio",
        "h_cat", "h_col", "n_distinct_colors", "n_distinct_categories",
        "prop_technic", "prop_printed", "prop_transparent", "rare_part_ratio",
        "minifig_to_part_ratio", "has_minifigs",
    ]

    return features[final_cols]

def filter_sets(features: pd.DataFrame) -> pd.DataFrame:
    s = features["set_num"]
    remove = (
        # Contains any alphabetic character
        s.str.contains(r"[A-Za-z]", na=False)
        |
        # Suffix after hyphen is not exactly "1"
        ~s.str.match(r"^\d+-1$")
        |
        # More than 7 digits before the hyphen
        s.str.match(r"^\d{8,}-1$")
        |
        # Starts with 500xxxx (500 followed by exactly four digits)
        s.str.match(r"^500\d{4}-1$")
        |
        # Starts with 854xxx (854 followed by exactly three digits)
        s.str.match(r"^854\d{3}-1$")
        |
        # Exclude any set starting with 97, 98, or 99
        s.str.match(r"^(97|98|99)\d*-1$")
    )

    features = features[~remove]
    features = features.dropna()
    return features

def main():
    db_path = os.path.join("db", "lego.db")

    conn = sqlite3.connect(db_path)
    try:
        if HISTOGRAM:
            raw_sets = pd.read_sql_query("SELECT num_parts FROM sets", conn)
            plot_num_parts_histogram(raw_sets, HISTOGRAM_OUT, MIN_PARTS)

        features = build_features(conn)

        n_before = len(features)
        if FILTER_SETS:
            features = filter_sets(features)

        if MIN_PARTS > 0:
            features = features[features["num_parts"] >= MIN_PARTS].reset_index(drop=True)
        n_after = len(features)
        print(
            f"LEGO set filter: kept {n_after}/{n_before} sets "
            f"(dropped {n_before - n_after})"
        )

        feature_out = os.path.join("lego_cleaned_features.csv")
        features.to_csv(feature_out, index=False)
        print(f"Wrote {len(features)} rows to {feature_out}")

        features.to_sql("set_features", conn, if_exists="replace", index=False)
        print("Also wrote table 'set_features' back into the database")
    finally:
        conn.close()


if __name__ == "__main__":
    main()