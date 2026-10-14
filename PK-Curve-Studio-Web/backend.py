"""Model and PubChem utilities for PK Curve Studio.

This is a research prototype for rat PK exploration.  It uses the current
project's rat C-t data, predicted PK anchors, and a two-compartment simulator.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

DEFAULT_DB = Path("/Users/mac/Desktop/PK-AI/pk_database_plot.csv")
DEFAULT_LATENTS = Path("/Users/mac/Desktop/PK-AI/IV_PO_FINAL_DUALABS_SHAPE/cv_latent_parameters.csv")
DEMO_DATA_DIR = Path(__file__).parent / "demo_data"


@dataclass
class Compound:
    cid: int | None
    title: str
    smiles: str
    formula: str
    molecular_weight: float

    @property
    def image_url(self) -> str | None:
        if self.cid is None:
            return None
        return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{self.cid}/PNG"

    @property
    def pubchem_url(self) -> str:
        if self.cid is None:
            return f"https://pubchem.ncbi.nlm.nih.gov/#query={quote(self.title)}"
        return f"https://pubchem.ncbi.nlm.nih.gov/compound/{self.cid}"


class OOFProfileRepository:
    """Read-only access to the final scaffold-CV out-of-fold predictions.

    Each profile in these files was predicted by the model fold that did not
    train on that profile's Bemis-Murcko scaffold.  The class intentionally
    reads saved predictions rather than re-running a model at presentation
    time, so the result shown is exactly the one used for final CV reporting.
    """

    def __init__(self, data_dir: Path = DEMO_DATA_DIR):
        required = {
            "compounds": data_dir / "cv_compounds.csv",
            "profiles": data_dir / "cv_profile_metrics.csv",
            "points": data_dir / "cv_point_predictions.csv",
            "curves": data_dir / "cv_dense_curve_predictions.csv",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("OOF demonstration data is missing: " + ", ".join(missing))

        self.compounds = pd.read_csv(required["compounds"])
        self.profiles = pd.read_csv(required["profiles"])
        self.points = pd.read_csv(required["points"])
        self.curves = pd.read_csv(required["curves"])
        self.compounds["canonical_smiles"] = self.compounds["smiles"].map(canonical_smiles)
        self._drug_by_smiles = (
            self.compounds.groupby("canonical_smiles", sort=False)["drug"].agg(list).to_dict()
        )
        self._compound_by_normalized_name = {
            self._normalize_name(row.drug): row for row in self.compounds.itertuples(index=False)
        }

    @staticmethod
    def _normalize_name(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())

    def compound_from_dataset_name(self, name: str) -> Compound | None:
        """Return a local structure fallback when PubChem is temporarily unavailable."""
        row = self._compound_by_normalized_name.get(self._normalize_name(name))
        if row is None:
            return None
        mol = Chem.MolFromSmiles(row.smiles)
        return Compound(
            cid=None,
            title=row.drug,
            smiles=row.smiles,
            formula=rdMolDescriptors.CalcMolFormula(mol),
            molecular_weight=float(Descriptors.MolWt(mol)),
        )

    def matching_drugs(self, smiles: str) -> list[str]:
        return list(self._drug_by_smiles.get(canonical_smiles(smiles), []))

    def profiles_for_smiles(self, smiles: str) -> pd.DataFrame:
        drugs = self.matching_drugs(smiles)
        if not drugs:
            return self.profiles.iloc[0:0].copy()
        result = self.profiles[self.profiles["drug"].isin(drugs)].copy()
        return result.sort_values(["drug", "route", "dose_ugkg", "profile_id"]).reset_index(drop=True)

    def profile_payload(self, profile_id: str) -> dict[str, Any]:
        metric = self.profiles.loc[self.profiles["profile_id"] == profile_id]
        if metric.empty:
            raise KeyError(f"Unknown OOF profile: {profile_id}")
        metric_row = metric.iloc[0].to_dict()
        points = self.points.loc[self.points["profile_id"] == profile_id].sort_values("time_h").copy()
        curve = self.curves.loc[self.curves["profile_id"] == profile_id].sort_values("time_h").copy()

        if points.empty or curve.empty:
            raise ValueError(f"OOF points or dense curve missing for {profile_id}")

        obs = points["obs"].to_numpy(float)
        pred = points["pred"].to_numpy(float)
        ratio = np.maximum(obs, 1e-12) / np.maximum(pred, 1e-12)
        metric_row["within_3fold_pct"] = float(np.mean((ratio >= 1 / 3) & (ratio <= 3)) * 100)
        return {"metrics": metric_row, "points": points, "curve": curve}


@lru_cache(maxsize=1)
def get_oof_repository() -> OOFProfileRepository:
    return OOFProfileRepository()


def pubchem_lookup(name: str) -> Compound:
    endpoint = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quote(name.strip())}/property/ConnectivitySMILES,MolecularFormula,MolecularWeight/JSON"
    )
    response: requests.Response | None = None
    for attempt in range(3):
        response = requests.get(endpoint, timeout=12, headers={"User-Agent": "PK-Curve-Studio/1.0"})
        if response.status_code not in {429, 503}:
            break
        time.sleep(0.6 * (attempt + 1))
    if response is None:
        raise RuntimeError("PubChem 응답을 받지 못했습니다.")
    response.raise_for_status()
    row = response.json()["PropertyTable"]["Properties"][0]
    smiles = row.get("ConnectivitySMILES") or row.get("SMILES") or row.get("CanonicalSMILES")
    if not smiles:
        raise ValueError("PubChem 결과에서 SMILES를 찾지 못했습니다.")
    return Compound(
        cid=int(row["CID"]), title=name.strip(), smiles=canonical_smiles(smiles),
        formula=str(row.get("MolecularFormula", "")), molecular_weight=float(row.get("MolecularWeight", np.nan)),
    )


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("유효하지 않은 SMILES입니다.")
    return Chem.MolToSmiles(mol, canonical=True)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({".": np.nan, "": np.nan, "-": np.nan}), errors="coerce")


def feature_vector(smiles: str, fp_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(canonical_smiles(smiles))
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_bits)
    fp = generator.GetFingerprint(mol)
    fp_array = np.zeros((fp_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, fp_array)
    descriptors = []
    for _, fn in Descriptors._descList:
        try:
            value = float(fn(mol))
        except Exception:
            value = np.nan
        descriptors.append(value if np.isfinite(value) else np.nan)
    return np.concatenate([fp_array, np.asarray(descriptors, dtype=np.float32)])


def compound_target_table(plot_db: Path) -> pd.DataFrame:
    raw = pd.read_csv(plot_db)
    work = pd.DataFrame({
        "drug": raw["약물명"].astype(str).str.strip(),
        "smiles": raw["SMILES"],
        "route_raw": raw["Route"].astype(str).str.strip().str.upper(),
        "cl": numeric(raw["CL, CL/F (L/h/kg)"]),
        "vss": numeric(raw["Vss, Vz/F (L/kg)"]),
        "f": numeric(raw["F"]),
    })
    work["route"] = np.where(work["route_raw"].str.startswith("IV"), "IV", np.where(work["route_raw"].str.startswith("PO"), "PO", "OTHER"))
    work = work[work["route"].isin(["IV", "PO"])].copy()
    def safe_canonical(value: object) -> str | None:
        try:
            return canonical_smiles(str(value))
        except ValueError:
            return None

    work["smiles"] = work["smiles"].map(safe_canonical)
    work = work.dropna(subset=["smiles"])

    records = []
    for drug, group in work.groupby("drug", sort=True):
        iv, po = group[group.route == "IV"], group[group.route == "PO"]
        cl = iv.loc[iv.cl > 0, "cl"].to_numpy(float)
        vss = iv.loc[iv.vss > 0, "vss"].to_numpy(float)
        f = po.loc[po.f > 0, "f"].to_numpy(float)
        if len(cl) and len(vss) and len(f):
            records.append({"drug": drug, "smiles": group.smiles.iloc[0], "CL": float(np.median(cl)), "Vss": float(np.median(vss)), "F": float(np.median(f))})
    return pd.DataFrame(records)


def latent_target_table(latent_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(latent_csv)
    raw["dose"] = raw.profile_id.astype(str).str.split("::").str[2].astype(float)
    for col in ["fV", "CLD_L_h_kg", "ka_fast_1_h", "ka_slow_1_h", "fast_fraction"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw[(raw.fV > 0) & (raw.CLD_L_h_kg > 0)].copy()
    return raw


def _extra_trees(seed: int):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(n_estimators=700, min_samples_leaf=2, max_features=0.35, random_state=seed, n_jobs=-1)),
    ])


def train_models(output_dir: Path, plot_db: Path = DEFAULT_DB, latent_csv: Path = DEFAULT_LATENTS) -> None:
    import joblib

    output_dir.mkdir(parents=True, exist_ok=True)
    compounds = compound_target_table(plot_db)
    feature_map = {drug: feature_vector(smiles) for drug, smiles in zip(compounds.drug, compounds.smiles)}
    x = np.vstack([feature_map[d] for d in compounds.drug])
    anchors: dict[str, Any] = {}
    ranges: dict[str, list[float]] = {}
    for i, target in enumerate(["CL", "Vss", "F"]):
        model = _extra_trees(20260915 + i)
        y = compounds[target].to_numpy(float)
        model.fit(x, np.log(y))
        anchors[target] = model
        ranges[target] = [float(np.quantile(y, 0.02)), float(np.quantile(y, 0.98))]
    joblib.dump(anchors, output_dir / "anchor_models.joblib")

    latents = latent_target_table(latent_csv)
    latents = latents.merge(compounds[["drug", "smiles"]], on="drug", how="inner")
    base_x = np.vstack([feature_vector(s) for s in latents.smiles])
    latent_models: dict[str, Any] = {}
    for i, target in enumerate(["fV", "CLD_L_h_kg"]):
        model = _extra_trees(20261000 + i)
        values = latents[target].to_numpy(float)
        model.fit(base_x, np.log(values) if target != "fV" else values)
        latent_models[target] = model
    po = latents[(latents.route == "PO") & (latents.ka_fast_1_h > 0) & (latents.ka_slow_1_h > 0) & (latents.fast_fraction > 0) & (latents.fast_fraction < 1)].copy()
    po_x = np.column_stack([np.vstack([feature_vector(s) for s in po.smiles]), np.log(po.dose.to_numpy(float))])
    for i, target in enumerate(["ka_fast_1_h", "ka_slow_1_h"]):
        model = _extra_trees(20261100 + i)
        model.fit(po_x, np.log(po[target].to_numpy(float)))
        latent_models[target] = model
    model = _extra_trees(20261102)
    clipped_fraction = np.clip(po.fast_fraction.to_numpy(float), 1e-4, 1 - 1e-4)
    model.fit(po_x, np.log(clipped_fraction / (1 - clipped_fraction)))
    latent_models["fast_fraction"] = model
    joblib.dump(latent_models, output_dir / "latent_models.joblib")
    (output_dir / "metadata.json").write_text(json.dumps({"n_anchor_compounds": len(compounds), "n_latent_profiles": len(latents), "n_po_latent_profiles": len(po), "anchor_ranges": ranges}, indent=2), encoding="utf-8")


class PredictionEngine:
    def __init__(self, model_dir: Path):
        import joblib

        self.anchor_models = joblib.load(model_dir / "anchor_models.joblib")
        self.latent_models = joblib.load(model_dir / "latent_models.joblib")
        self.meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))

    def predict_anchors(self, smiles: str) -> dict[str, float]:
        x = feature_vector(smiles).reshape(1, -1)
        output = {}
        for name, model in self.anchor_models.items():
            value = float(np.exp(model.predict(x)[0]))
            lo, hi = self.meta["anchor_ranges"][name]
            output[name] = float(np.clip(value, lo, hi))
        return output

    def predict_latents(self, smiles: str, dose_ugkg: float, route: str) -> dict[str, float]:
        x = feature_vector(smiles).reshape(1, -1)
        fv = float(np.clip(self.latent_models["fV"].predict(x)[0], 0.08, 0.92))
        cld = float(np.exp(self.latent_models["CLD_L_h_kg"].predict(x)[0]))
        output = {"fV": fv, "CLD": max(cld, 1e-5)}
        if route == "PO":
            px = np.column_stack([x, [math.log(dose_ugkg)]])
            kfast = float(np.exp(self.latent_models["ka_fast_1_h"].predict(px)[0]))
            kslow = float(np.exp(self.latent_models["ka_slow_1_h"].predict(px)[0]))
            wlogit = float(self.latent_models["fast_fraction"].predict(px)[0])
            output.update({"ka_fast": max(kfast, 1e-4), "ka_slow": max(kslow, 1e-4), "w": float(1 / (1 + np.exp(-wlogit)))})
        return output

    def predict_curve(self, smiles: str, route: str, dose_ugkg: float, hours: float, overrides: dict[str, float] | None = None) -> dict[str, Any]:
        anchors = self.predict_anchors(smiles)
        if overrides:
            anchors.update({k: float(v) for k, v in overrides.items() if v is not None and float(v) > 0})
        latents = self.predict_latents(smiles, dose_ugkg, route)
        time = np.linspace(0, hours, int(hours * 8) + 1)
        conc = simulate_curve(time, route, dose_ugkg, anchors, latents)
        auc = float(np.trapezoid(conc, time))
        peak = int(np.argmax(conc))
        return {"time": time, "concentration": conc, "anchors": anchors, "latents": latents, "AUC": auc, "Cmax": float(conc[peak]), "Tmax": float(time[peak])}


def simulate_curve(time: np.ndarray, route: str, dose: float, anchors: dict[str, float], latents: dict[str, float]) -> np.ndarray:
    from scipy.linalg import expm

    cl, vss = max(anchors["CL"], 1e-6), max(anchors["Vss"], 1e-6)
    v1 = max(latents["fV"] * vss, 1e-6)
    v2 = max((1 - latents["fV"]) * vss, 1e-6)
    cld = max(latents["CLD"], 1e-6)
    if route == "IV":
        matrix = np.array([[-(cl + cld) / v1, cld / v2], [cld / v1, -cld / v2]])
        initial = np.array([dose, 0.0])
        return np.asarray([(expm(matrix * t) @ initial)[0] / v1 for t in time])
    f = max(anchors["F"], 1e-6)
    kfast, kslow, w = latents["ka_fast"], latents["ka_slow"], np.clip(latents["w"], 1e-4, 1 - 1e-4)
    matrix = np.array([
        [-kfast, 0, 0, 0], [0, -kslow, 0, 0],
        [kfast, kslow, -(cl + cld) / v1, cld / v2], [0, 0, cld / v1, -cld / v2],
    ])
    initial = np.array([f * dose * w, f * dose * (1 - w), 0.0, 0.0])
    return np.asarray([(expm(matrix * t) @ initial)[2] / v1 for t in time])
