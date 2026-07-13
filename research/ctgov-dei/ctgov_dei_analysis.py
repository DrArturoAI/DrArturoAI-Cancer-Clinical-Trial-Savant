#!/usr/bin/env python3
"""ClinicalTrials.gov diversity/equity language analysis.

Builds a current-snapshot, first-post-quarter reconstruction of human-participant
D/E commitments. Candidate studies are retrieved server-side; exact matching and
semantic classification are then performed locally with transparent rules.

Important limitation: current record text is assigned to the original first-post
quarter. Later amendments can therefore back-date language. Historical versions
are required before making causal claims about policy events.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import statsmodels.api as sm
except Exception:
    sm = None

API_URL = "https://clinicaltrials.gov/api/v2/studies"
USER_AGENT = "CTGov-DEI-Language-Research/1.0 (public-data analysis)"

SEARCH_TERMS: dict[str, str] = {
    "diversity": "diversity",
    "equity": "equity",
    "diverse": "diverse",
    "equitable": "equitable",
    "underrepresented": "underrepresented",
    "underserved": "underserved",
    "health_disparities": '"health disparities"',
    "racial_disparities": '"racial disparities"',
    "representative_enrollment": '"representative enrollment"',
    "inclusive_recruitment": '"inclusive recruitment"',
}

TERM_PATTERNS: dict[str, re.Pattern[str]] = {
    "diversity": re.compile(r"\bdiversity\b", re.I),
    "equity": re.compile(r"\bequity\b", re.I),
    "diverse": re.compile(r"\bdiverse\b", re.I),
    "equitable": re.compile(r"\bequitable\b", re.I),
    "underrepresented": re.compile(r"\bunder[- ]?represented\b", re.I),
    "underserved": re.compile(r"\bunder[- ]?served\b", re.I),
    "health_disparities": re.compile(r"\bhealth disparit(?:y|ies)\b", re.I),
    "racial_disparities": re.compile(r"\b(?:racial|ethnic|race|ethnicity)[ -]disparit(?:y|ies)\b", re.I),
    "representative_enrollment": re.compile(
        r"\brepresentative (?:enrollment|enrolment|recruitment|sample|cohort|population)\b", re.I
    ),
    "inclusive_recruitment": re.compile(
        r"\binclusive (?:recruitment|enrollment|enrolment|participation|trial)\b", re.I
    ),
}

HUMAN_CONTEXT_PATTERNS = [
    r"\brecruit(?:ment|ing|ed)?\b", r"\benrol+l(?:ment|ing|ed)?\b",
    r"\bparticipant(?:s)?\b", r"\bpatient(?:s)?\b", r"\bpopulation(?:s)?\b",
    r"\bcohort(?:s)?\b", r"\bcommunit(?:y|ies)\b", r"\baccess\b",
    r"\bbarrier(?:s)?\b", r"\bunder[- ]?represent(?:ed|ation)\b",
    r"\bunder[- ]?served\b", r"\bminority|minorities\b", r"\brace|racial\b",
    r"\bethnic|ethnicity\b", r"\bsocioeconomic\b", r"\bhealth disparit(?:y|ies)\b",
    r"\binequit(?:y|ies|able)\b", r"\bequit(?:y|able)\b", r"\brural\b",
    r"\blanguage access\b", r"\bgeographic(?:al)?\b", r"\bsex\b", r"\bgender\b",
    r"\bolder adult(?:s)?\b", r"\bpediatric|paediatric\b", r"\bdisabilit(?:y|ies)\b",
    r"\bLGBTQ(?:IA\+?)?\b", r"\brepresentative\b", r"\bgeneralizab(?:ility|le)\b",
]
ACTION_PATTERNS = [
    r"\baim(?:s|ed|ing)?\b", r"\bobjective(?:s)?\b", r"\bgoal(?:s)?\b",
    r"\brecruit(?:ment|ing|ed)?\b", r"\benrol+l(?:ment|ing|ed)?\b",
    r"\bensure(?:s|d|ing)?\b", r"\bincrease(?:s|d|ing)?\b",
    r"\bimprove(?:s|d|ment|ing)?\b", r"\benhance(?:s|d|ment|ing)?\b",
    r"\baddress(?:es|ed|ing)?\b", r"\breduce(?:s|d|ing)?\b",
    r"\bevaluat(?:e|es|ed|ing|ion)\b", r"\bassess(?:es|ed|ing|ment)?\b",
    r"\bfacilitat(?:e|es|ed|ing|ion)\b", r"\bpromot(?:e|es|ed|ing|ion)\b",
    r"\binclud(?:e|es|ed|ing|ion)\b", r"\brepresent(?:s|ed|ing|ation)?\b",
    r"\bengag(?:e|es|ed|ing|ement)\b", r"\bovercome\b", r"\bidentify\b",
    r"\btarget(?:s|ed|ing)?\b", r"\breflect(?:s|ed|ing)?\b",
]
BIOMEDICAL_PATTERNS = [
    r"\bmicrobiom(?:e|es)\b", r"\bmicrobial\b", r"\bbacteri(?:a|al)\b",
    r"\bspecies\b", r"\balpha[- ]diversity\b", r"\bbeta[- ]diversity\b",
    r"\bShannon(?:'s)?(?: diversity)? index\b", r"\bphylogenetic\b", r"\bflora\b",
    r"\bgut\b", r"\bgenetic diversity\b", r"\bgenomic diversity\b",
    r"\bmolecular diversity\b", r"\bclonal diversity\b", r"\brepertoire diversity\b",
    r"\bT[- ]?cell receptor diversity\b", r"\bimmune repertoire\b",
    r"\btumou?r diversity\b", r"\bdiversity index\b", r"\becological diversity\b",
    r"\bdietary diversity\b", r"\bfood diversity\b",
    r"\bdiversity of (?:drugs|interventions|conditions|symptoms|lesions|mutations|assays|measures|responses)\b",
]
STRONG_HUMAN_PHRASES = [
    r"\bethnically diverse\b", r"\bracially diverse\b", r"\bracially[/ -]ethnically diverse\b",
    r"\bdiverse (?:patient|participant|population|community|cohort|sample)s?\b",
    r"\bdiversity (?:in|of|among) (?:clinical trial|trial|participant|patient|enrollment|enrolment|recruitment|population|cohort)s?\b",
    r"\bhealth equity\b", r"\bequitable access\b", r"\bequitable enrollment\b",
    r"\bequitable enrolment\b", r"\bequitable recruitment\b", r"\bdigital (?:health )?equity\b",
    r"\bpharmaco[- ]equity\b", r"\btrial equity\b", r"\bresearch equity\b",
    r"\brepresentative enrollment\b", r"\binclusive recruitment\b",
]

ONCOLOGY_REGEX = re.compile(
    r"\b(cancer|neoplasm|tumou?r|carcinoma|sarcoma|leukemi?a|lymphoma|myeloma|melanoma|"
    r"glioma|blastoma|malignan|oncolog|myelodysplastic|myeloproliferative|mesothelioma|"
    r"metasta(?:sis|tic)|solid tumor|solid tumour)\b", re.I
)

POLICY_EVENTS = {
    "2020Q4": "FDA broadening-eligibility/diversity guidance",
    "2022Q2": "FDA draft diversity-plan guidance",
    "2022Q4": "FDORA enacted",
    "2024Q2": "FDA draft Diversity Action Plan guidance",
    "2025Q1": "Federal DEI-policy reversal / FDA page removals",
}


def setup_logging(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(outdir / "run.log", encoding="utf-8")],
    )


class CTGovClient:
    def __init__(self, min_interval: float = 1.25, timeout: int = 120):
        self.min_interval = min_interval
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self._last_request = 0.0
        self.request_count = 0

    def get(self, params: dict[str, Any], retries: int = 8) -> dict[str, Any]:
        for attempt in range(retries):
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            try:
                self._last_request = time.monotonic()
                response = self.session.get(API_URL, params=params, timeout=self.timeout)
                self.request_count += 1
                if response.status_code == 429:
                    wait = float(response.headers.get("Retry-After", 60))
                    logging.warning("Rate limited; waiting %.1f seconds", wait)
                    time.sleep(wait)
                    continue
                if response.status_code >= 500:
                    raise requests.HTTPError(f"Server error {response.status_code}", response=response)
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == retries - 1:
                    raise
                wait = min(90.0, 2.0 ** attempt + 0.5)
                logging.warning("Request failed (%s), retrying in %.1fs", exc, wait)
                time.sleep(wait)
        raise RuntimeError("Unreachable")

    def count(self, *, query_term: str | None = None, query_cond: str | None = None) -> int:
        params: dict[str, Any] = {"pageSize": 1, "countTotal": "true", "format": "json"}
        if query_term:
            params["query.term"] = query_term
        if query_cond:
            params["query.cond"] = query_cond
        data = self.get(params)
        if "totalCount" in data:
            return int(data["totalCount"])
        studies = data.get("studies", [])
        if not data.get("nextPageToken") and not data.get("pageToken"):
            return len(studies)
        raise RuntimeError("API response omitted totalCount")

    def iter_search(self, term: str) -> Iterable[dict[str, Any]]:
        page_token: str | None = None
        first = True
        while True:
            params: dict[str, Any] = {"query.term": term, "pageSize": 1000, "format": "json"}
            if first:
                params["countTotal"] = "true"
            if page_token:
                params["pageToken"] = page_token
            data = self.get(params)
            if first:
                logging.info("Search %r API totalCount=%s", term, data.get("totalCount", "not returned"))
            yield from data.get("studies", [])
            page_token = data.get("nextPageToken") or data.get("pageToken")
            first = False
            if not page_token:
                break


def nget(obj: dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(clean_text(v) for v in value if v is not None)
    if isinstance(value, dict):
        return "\n".join(clean_text(v) for v in value.values() if v is not None)
    text = re.sub(r"[`*_#>|]", " ", str(value))
    return re.sub(r"\s+", " ", text).strip()


def outcome_text(outcomes: list[dict[str, Any]] | None) -> str:
    if not outcomes:
        return ""
    chunks: list[str] = []
    for out in outcomes:
        chunks.extend([str(out.get("measure", "")), str(out.get("description", ""))])
    return clean_text(chunks)


def extract_text_fields(study: dict[str, Any]) -> dict[str, str]:
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    desc = p.get("descriptionModule", {})
    cond = p.get("conditionsModule", {})
    elig = p.get("eligibilityModule", {})
    arms = p.get("armsInterventionsModule", {})
    outcomes = p.get("outcomesModule", {})
    arm_desc = [a.get("description", "") for a in arms.get("armGroups", [])]
    int_desc: list[str] = []
    for intervention in arms.get("interventions", []):
        int_desc.extend([intervention.get("name", ""), intervention.get("description", "")])
    return {
        "title": clean_text([ident.get("briefTitle", ""), ident.get("officialTitle", "")]),
        "brief_summary": clean_text(desc.get("briefSummary", "")),
        "detailed_description": clean_text(desc.get("detailedDescription", "")),
        "keywords": clean_text(cond.get("keywords", [])),
        "eligibility": clean_text(elig.get("eligibilityCriteria", "")),
        "study_population": clean_text(elig.get("studyPopulation", "")),
        "arm_descriptions": clean_text(arm_desc),
        "intervention_descriptions": clean_text(int_desc),
        "primary_outcomes": outcome_text(outcomes.get("primaryOutcomes")),
        "secondary_outcomes": outcome_text(outcomes.get("secondaryOutcomes")),
        "other_outcomes": outcome_text(outcomes.get("otherOutcomes")),
    }


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    text = re.sub(r"\s*[•▪◦]\s*", ". ", text)
    parts = re.split(r"(?<=[.!?])\s+|\s*\n+\s*|(?<=;)\s+(?=[A-Z])", text)
    return [p.strip(" -;\t") for p in parts if p and p.strip(" -;\t")]


def count_patterns(patterns: list[str], text: str) -> int:
    return sum(1 for pat in patterns if re.search(pat, text, re.I))


def classify_context(context: str, matched_term: str) -> tuple[str, int, int, int, str]:
    human = count_patterns(HUMAN_CONTEXT_PATTERNS, context)
    action = count_patterns(ACTION_PATTERNS, context)
    biomedical = count_patterns(BIOMEDICAL_PATTERNS, context)
    strong_human = any(re.search(p, context, re.I) for p in STRONG_HUMAN_PHRASES)
    equity_like = matched_term in {"equity", "equitable"}
    if biomedical >= 2 and human == 0 and not strong_human:
        return "biomedical_false_positive", human, action, biomedical, "biomedical diversity context"
    if strong_human and (action >= 1 or equity_like):
        return "substantive_human_commitment", human, action, biomedical, "strong human phrase plus action/equity context"
    if equity_like and human >= 1 and biomedical == 0:
        label = "substantive_human_commitment" if action >= 1 else "human_context_noncommitment"
        return label, human, action, biomedical, "equity term in human/access context"
    if human >= 2 and action >= 1 and biomedical <= 1:
        return "substantive_human_commitment", human, action, biomedical, "human context with operational/action language"
    if biomedical >= 1 and human <= 1:
        return "biomedical_false_positive", human, action, biomedical, "biomedical/ecological use"
    if human >= 1:
        return "human_context_noncommitment", human, action, biomedical, "human context without clear operational commitment"
    return "generic_or_uncertain", human, action, biomedical, "insufficient human recruitment/access context"


def study_matches(study: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    sponsor = p.get("sponsorCollaboratorsModule", {})
    oversight = p.get("oversightModule", {})
    design = p.get("designModule", {})
    cond = p.get("conditionsModule", {})
    elig = p.get("eligibilityModule", {})
    locations = nget(p, "contactsLocationsModule", "locations", default=[]) or []
    derived = study.get("derivedSection", {})
    meshes = nget(derived, "conditionBrowseModule", "meshes", default=[]) or []
    ancestors = nget(derived, "conditionBrowseModule", "ancestors", default=[]) or []

    nct_id = ident.get("nctId", "")
    first_post = nget(status, "studyFirstPostDateStruct", "date", default="") or ""
    last_update = nget(status, "lastUpdatePostDateStruct", "date", default="") or ""
    conditions = [str(x) for x in cond.get("conditions", [])]
    mesh_terms = [str(x.get("term", "")) for x in meshes + ancestors if isinstance(x, dict)]
    condition_blob = " | ".join(conditions + mesh_terms)
    oncology_local = bool(ONCOLOGY_REGEX.search(condition_blob)) or any(x.strip().lower() == "neoplasms" for x in mesh_terms)
    countries = sorted({str(loc.get("country", "")) for loc in locations if isinstance(loc, dict) and loc.get("country")})
    us_site = "United States" in countries

    fields = extract_text_fields(study)
    match_rows: list[dict[str, Any]] = []
    labels_by_term: dict[str, list[str]] = defaultdict(list)
    term_counts: dict[str, int] = defaultdict(int)

    for field_name, text in fields.items():
        sentences = split_sentences(text)
        for idx, sentence in enumerate(sentences):
            for term_name, pattern in TERM_PATTERNS.items():
                if not pattern.search(sentence):
                    continue
                prev_s = sentences[idx - 1] if idx > 0 else ""
                next_s = sentences[idx + 1] if idx + 1 < len(sentences) else ""
                context = " ".join(x for x in [prev_s, sentence, next_s] if x)
                label, human, action, biomedical, reason = classify_context(context, term_name)
                labels_by_term[term_name].append(label)
                term_counts[term_name] += len(pattern.findall(sentence)) or 1
                match_rows.append({
                    "nct_id": nct_id,
                    "first_post_date": first_post,
                    "last_update_date": last_update,
                    "field": field_name,
                    "term": term_name,
                    "sentence": sentence[:2000],
                    "context": context[:5000],
                    "classification": label,
                    "human_context_score": human,
                    "action_score": action,
                    "biomedical_score": biomedical,
                    "classification_reason": reason,
                    "oncology_local": int(oncology_local),
                    "us_site": int(us_site),
                })

    exact_diversity = bool(labels_by_term.get("diversity"))
    exact_equity = bool(labels_by_term.get("equity"))
    exact_div_commit = "substantive_human_commitment" in labels_by_term.get("diversity", [])
    exact_eq_commit = "substantive_human_commitment" in labels_by_term.get("equity", [])
    broad_commit = any("substantive_human_commitment" in labels for labels in labels_by_term.values())
    any_human_context = any(
        any(label in {"substantive_human_commitment", "human_context_noncommitment"} for label in labels)
        for labels in labels_by_term.values()
    )
    any_biomedical = any("biomedical_false_positive" in labels for labels in labels_by_term.values())

    phases = design.get("phases", []) or []
    lead = sponsor.get("leadSponsor", {}) or {}
    try:
        quarter = str(pd.Period(first_post[:10], freq="Q")) if first_post else ""
    except Exception:
        quarter = ""
    row = {
        "nct_id": nct_id,
        "brief_title": ident.get("briefTitle", ""),
        "official_title": ident.get("officialTitle", ""),
        "first_post_date": first_post,
        "last_update_date": last_update,
        "quarter": quarter,
        "overall_status": status.get("overallStatus", ""),
        "study_type": design.get("studyType", ""),
        "phases": "|".join(phases),
        "late_phase": int(any(ph in {"PHASE2", "PHASE3", "PHASE4"} for ph in phases)),
        "enrollment": nget(design, "enrollmentInfo", "count", default=np.nan),
        "lead_sponsor_name": lead.get("name", ""),
        "lead_sponsor_class": lead.get("class", ""),
        "industry_sponsor": int(lead.get("class") == "INDUSTRY"),
        "fda_regulated_drug": int(bool(oversight.get("isFdaRegulatedDrug", False))),
        "fda_regulated_device": int(bool(oversight.get("isFdaRegulatedDevice", False))),
        "conditions": "|".join(conditions),
        "condition_mesh_terms": "|".join(mesh_terms),
        "oncology_local": int(oncology_local),
        "countries": "|".join(countries),
        "us_site": int(us_site),
        "sex": elig.get("sex", ""),
        "minimum_age": elig.get("minimumAge", ""),
        "maximum_age": elig.get("maximumAge", ""),
        "has_results": int(bool(study.get("hasResults", False))),
        "exact_diversity": int(exact_diversity),
        "exact_equity": int(exact_equity),
        "exact_diversity_or_equity": int(exact_diversity or exact_equity),
        "exact_diversity_commitment": int(exact_div_commit),
        "exact_equity_commitment": int(exact_eq_commit),
        "exact_human_commitment": int(exact_div_commit or exact_eq_commit),
        "broad_human_commitment": int(broad_commit),
        "any_human_context": int(any_human_context),
        "any_biomedical_false_positive": int(any_biomedical),
        "matched_terms": "|".join(sorted(labels_by_term)),
        "match_count": len(match_rows),
    }
    for term_name in TERM_PATTERNS:
        row[f"count_{term_name}"] = int(term_counts.get(term_name, 0))
    return row, match_rows


def quarter_bounds(start_year: int, end_date: date) -> list[tuple[pd.Period, str, str]]:
    first = pd.Period(f"{start_year}Q1", freq="Q")
    last = pd.Period(end_date, freq="Q")
    out = []
    for q in pd.period_range(first, last, freq="Q"):
        start = q.start_time.date()
        end = min(q.end_time.date(), end_date)
        out.append((q, start.isoformat(), end.isoformat()))
    return out


def build_denominators(client: CTGovClient, start_year: int, end_date: date, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    rows = []
    bounds = quarter_bounds(start_year, end_date)
    for i, (q, start, end) in enumerate(bounds, 1):
        date_expr = f"AREA[StudyFirstPostDate]RANGE[{start}, {end}]"
        total = client.count(query_term=date_expr)
        oncology = client.count(query_term=date_expr, query_cond="Neoplasms")
        rows.append({
            "quarter": str(q), "quarter_start": start, "quarter_end": end,
            "all_trials": total, "oncology_trials_proxy": oncology,
            "non_oncology_trials_proxy": max(total - oncology, 0),
            "partial_quarter": int(q == pd.Period(end_date, freq="Q") and end_date < q.end_time.date()),
        })
        logging.info("Denominator %d/%d %s all=%d oncology_proxy=%d", i, len(bounds), q, total, oncology)
    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    return df


def fetch_candidates(client: CTGovClient, cache_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    studies: dict[str, dict[str, Any]] = {}
    source_queries: dict[str, set[str]] = defaultdict(set)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                study = item["study"]
                nct = nget(study, "protocolSection", "identificationModule", "nctId", default="")
                if nct:
                    studies[nct] = study
                    source_queries[nct].update(item.get("source_queries", []))
        return studies, source_queries

    for label, query in SEARCH_TERMS.items():
        before = len(studies)
        try:
            iterator = client.iter_search(query)
            for study in iterator:
                nct = nget(study, "protocolSection", "identificationModule", "nctId", default="")
                if nct:
                    studies[nct] = study
                    source_queries[nct].add(label)
        except requests.HTTPError as exc:
            if '"' not in query:
                raise
            fallback = query.replace('"', "")
            logging.warning("Quoted query %r failed (%s); retrying %r", query, exc, fallback)
            for study in client.iter_search(fallback):
                nct = nget(study, "protocolSection", "identificationModule", "nctId", default="")
                if nct:
                    studies[nct] = study
                    source_queries[nct].add(label)
        logging.info("Candidate query %-26s added %d unique; cumulative %d", label, len(studies) - before, len(studies))

    with cache_path.open("w", encoding="utf-8") as f:
        for nct in sorted(studies):
            f.write(json.dumps({"source_queries": sorted(source_queries[nct]), "study": studies[nct]}, ensure_ascii=False) + "\n")
    return studies, source_queries


def aggregate_quarterly(studies_df: pd.DataFrame, denom_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "exact_diversity", "exact_equity", "exact_diversity_or_equity",
        "exact_diversity_commitment", "exact_equity_commitment", "exact_human_commitment",
        "broad_human_commitment", "any_human_context", "any_biomedical_false_positive",
    ]
    valid = studies_df[studies_df["quarter"].astype(str).str.match(r"^\d{4}Q[1-4]$")].copy()
    agg_all = valid.groupby("quarter")[metrics].sum().reset_index()
    agg_onc = valid[valid["oncology_local"] == 1].groupby("quarter")[metrics].sum().reset_index()
    agg_non = valid[valid["oncology_local"] == 0].groupby("quarter")[metrics].sum().reset_index()
    agg_onc = agg_onc.rename(columns={m: f"oncology_{m}" for m in metrics})
    agg_non = agg_non.rename(columns={m: f"non_oncology_{m}" for m in metrics})
    out = denom_df.merge(agg_all, on="quarter", how="left").merge(agg_onc, on="quarter", how="left").merge(agg_non, on="quarter", how="left")
    count_cols = [c for c in out.columns if c in metrics or c.startswith("oncology_") or c.startswith("non_oncology_")]
    out[count_cols] = out[count_cols].fillna(0).astype(int)
    for metric in metrics:
        out[f"rate_per_1000_{metric}"] = np.where(out["all_trials"] > 0, 1000.0 * out[metric] / out["all_trials"], np.nan)
        onc_col = f"oncology_{metric}"
        out[f"oncology_rate_per_1000_{metric}"] = np.where(out["oncology_trials_proxy"] > 0, 1000.0 * out[onc_col] / out["oncology_trials_proxy"], np.nan)
        non_col = f"non_oncology_{metric}"
        out[f"non_oncology_rate_per_1000_{metric}"] = np.where(out["non_oncology_trials_proxy"] > 0, 1000.0 * out[non_col] / out["non_oncology_trials_proxy"], np.nan)
        out[f"rolling4_rate_per_1000_{metric}"] = 1000.0 * out[metric].rolling(4, min_periods=1).sum() / out["all_trials"].rolling(4, min_periods=1).sum()
        out[f"oncology_rolling4_rate_per_1000_{metric}"] = 1000.0 * out[onc_col].rolling(4, min_periods=1).sum() / out["oncology_trials_proxy"].rolling(4, min_periods=1).sum()
    return out


def period_window_summary(qdf: pd.DataFrame) -> pd.DataFrame:
    windows = [
        ("2017-2019 pre-2020", "2017Q1", "2019Q4"),
        ("2020-2022", "2020Q1", "2022Q4"),
        ("2023-2024", "2023Q1", "2024Q4"),
        ("2025-current", "2025Q1", str(pd.Period(date.today(), freq="Q"))),
    ]
    metrics = ["exact_diversity", "exact_equity", "exact_human_commitment", "broad_human_commitment"]
    rows = []
    for label, q1, q2 in windows:
        sub = qdf[(qdf["quarter"] >= q1) & (qdf["quarter"] <= q2) & (qdf["partial_quarter"] == 0)]
        if sub.empty:
            continue
        for scope, denom_col, prefix in [("all", "all_trials", ""), ("oncology", "oncology_trials_proxy", "oncology_")]:
            denom = int(sub[denom_col].sum())
            for metric in metrics:
                count_col = f"{prefix}{metric}" if prefix else metric
                count = int(sub[count_col].sum())
                rows.append({
                    "window": label, "scope": scope, "metric": metric, "trials": denom,
                    "commitment_count": count,
                    "rate_per_1000": 1000.0 * count / denom if denom else np.nan,
                })
    return pd.DataFrame(rows)


def event_study_models(qdf: pd.DataFrame) -> pd.DataFrame:
    if sm is None:
        return pd.DataFrame()
    rows = []
    metric = "broad_human_commitment"
    for event_q, event_name in POLICY_EVENTS.items():
        event_p = pd.Period(event_q, freq="Q")
        for scope, count_col, denom_col in [
            ("all", metric, "all_trials"),
            ("oncology", f"oncology_{metric}", "oncology_trials_proxy"),
        ]:
            work = qdf[qdf["partial_quarter"] == 0].copy()
            work["period"] = work["quarter"].map(lambda x: pd.Period(x, freq="Q"))
            work["event_time"] = work["period"].map(lambda p: p.ordinal - event_p.ordinal)
            work = work[(work["event_time"] >= -8) & (work["event_time"] <= 8) & (work[denom_col] > 0)].copy()
            if len(work) < 10 or work[count_col].sum() < 5:
                continue
            work["post"] = (work["event_time"] >= 0).astype(int)
            work["time_after"] = work["event_time"].clip(lower=0)
            X = sm.add_constant(work[["event_time", "post", "time_after"]].astype(float), has_constant="add")
            y = work[count_col].astype(float)
            try:
                fit = sm.GLM(y, X, family=sm.families.Poisson(), offset=np.log(work[denom_col].astype(float))).fit(cov_type="HC0")
            except Exception as exc:
                logging.warning("Event model failed %s %s: %s", event_q, scope, exc)
                continue
            for coef, interpretation in [("post", "immediate level ratio"), ("time_after", "post-event quarterly slope ratio")]:
                est = float(fit.params[coef])
                se = float(fit.bse[coef])
                rows.append({
                    "event_quarter": event_q, "event": event_name, "scope": scope,
                    "outcome": metric, "parameter": coef, "interpretation": interpretation,
                    "log_rate_ratio": est, "rate_ratio": math.exp(est),
                    "ci95_low": math.exp(est - 1.96 * se), "ci95_high": math.exp(est + 1.96 * se),
                    "p_value": float(fit.pvalues[coef]), "quarters_in_model": int(len(work)),
                    "events_in_model": int(y.sum()),
                    "note": "Associational segmented Poisson model, ±8 quarters; current-snapshot text, robust SEs.",
                })
    return pd.DataFrame(rows)


def add_policy_lines(ax: Any) -> None:
    for q in POLICY_EVENTS:
        ax.axvline(pd.Period(q, freq="Q").start_time, linestyle="--", linewidth=0.8, alpha=0.55)


def plot_rates(qdf: pd.DataFrame, outdir: Path) -> None:
    plot = qdf.copy()
    plot["date"] = plot["quarter"].map(lambda x: pd.Period(x, freq="Q").start_time)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(plot["date"], plot["rolling4_rate_per_1000_exact_diversity"], label='Exact word “diversity”')
    ax.plot(plot["date"], plot["rolling4_rate_per_1000_exact_equity"], label='Exact word “equity”')
    ax.plot(plot["date"], plot["rolling4_rate_per_1000_broad_human_commitment"], label="Validated broader human-participant commitment", linewidth=2.2)
    add_policy_lines(ax)
    ax.set_title("Diversity and equity language in ClinicalTrials.gov registrations")
    ax.set_ylabel("Studies per 1,000 newly first-posted trials (4-quarter rolling)")
    ax.set_xlabel("First-post quarter")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "figure_1_all_trials_language_trend.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(plot["date"], plot["rolling4_rate_per_1000_broad_human_commitment"], label="All trials", linewidth=2.0)
    ax.plot(plot["date"], plot["oncology_rolling4_rate_per_1000_broad_human_commitment"], label="Oncology (Neoplasms denominator proxy)", linewidth=2.0)
    add_policy_lines(ax)
    ax.set_title("Human-participant diversity/equity commitments: oncology versus all trials")
    ax.set_ylabel("Studies per 1,000 newly first-posted trials (4-quarter rolling)")
    ax.set_xlabel("First-post quarter")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "figure_2_oncology_vs_all.png", dpi=220)
    plt.close(fig)


def plot_classification(matches_df: pd.DataFrame, outdir: Path) -> None:
    exact = matches_df[matches_df["term"].isin(["diversity", "equity"])].copy()
    if exact.empty:
        return
    counts = exact.groupby(["term", "classification"]).size().unstack(fill_value=0)
    order = ["substantive_human_commitment", "human_context_noncommitment", "biomedical_false_positive", "generic_or_uncertain"]
    counts = counts.reindex(columns=[c for c in order if c in counts.columns])
    ax = counts.plot(kind="bar", stacked=True, figsize=(9, 6))
    ax.set_title("Why exact-word counting needs semantic validation")
    ax.set_xlabel("Exact word")
    ax.set_ylabel("Matched text passages")
    ax.legend(title="Classification", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outdir / "figure_3_exact_word_classification.png", dpi=220)
    plt.close()


def build_manual_review_sample(matches_df: pd.DataFrame, outpath: Path, n_per_stratum: int = 30) -> None:
    rng = np.random.default_rng(20260713)
    exact = matches_df[matches_df["term"].isin(["diversity", "equity", "diverse", "equitable"])].copy()
    samples = []
    for term in sorted(exact["term"].unique()):
        for label in ["substantive_human_commitment", "human_context_noncommitment", "biomedical_false_positive", "generic_or_uncertain"]:
            sub = exact[(exact["term"] == term) & (exact["classification"] == label)]
            if sub.empty:
                continue
            take = min(n_per_stratum, len(sub))
            idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
            samples.append(sub.loc[idx])
    if not samples:
        pd.DataFrame().to_csv(outpath, index=False)
        return
    review = pd.concat(samples, ignore_index=True).drop_duplicates(["nct_id", "term", "sentence"])
    review["reviewer_label"] = ""
    review["reviewer_notes"] = ""
    review.to_csv(outpath, index=False)


def top_examples(studies_df: pd.DataFrame, matches_df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    human = matches_df[matches_df["classification"] == "substantive_human_commitment"].copy()
    if human.empty:
        return pd.DataFrame()
    human["first_post_date"] = pd.to_datetime(human["first_post_date"], errors="coerce")
    human = human.sort_values(["oncology_local", "first_post_date"], ascending=[False, False]).drop_duplicates(["nct_id", "term"])
    meta_cols = ["nct_id", "brief_title", "conditions", "lead_sponsor_name", "study_type", "phases"]
    return human.head(n).merge(studies_df[meta_cols], on="nct_id", how="left")


def write_methodology(outdir: Path, metadata: dict[str, Any]) -> None:
    text = f"""# ClinicalTrials.gov diversity/equity language analysis — methodology

## Snapshot and scope

- API snapshot/version holder: {metadata.get('version_holder', 'unknown')}
- Extraction time (UTC): {metadata.get('extraction_time_utc')}
- First-post window: {metadata.get('start_year')} through {metadata.get('end_date')}
- Candidate records retrieved: {metadata.get('candidate_records_retrieved')}
- Candidate records containing at least one local lexicon match: {metadata.get('candidate_records_with_local_match')}
- API requests: {metadata.get('api_requests')}

The primary exact-word outcomes are word-boundary matches for **diversity** and **equity** in selected public protocol fields. A broader index adds diverse, equitable, underrepresented, underserved, health/racial disparities, representative enrollment, and inclusive recruitment.

## Semantic classification

Every matched passage is assigned by transparent rules to one of four categories: substantive human-participant commitment; human demographic context without a clear operational commitment; biomedical false positive (for example microbiome, genetic, clonal, or T-cell repertoire diversity); or generic/uncertain. The rule outputs, scores, sentence, and surrounding context are exported for audit. `manual_review_sample.csv` is a stratified review template.

## Denominators and oncology

All-trial denominators are exact quarterly counts returned by ClinicalTrials.gov API date-range queries. Oncology denominators use the ClinicalTrials.gov condition-search query `Neoplasms` and are therefore a reproducible proxy rather than a pathology-adjudicated universe. Candidate records additionally receive a local oncology flag from condition and MeSH terms.

## Temporal assignment

Records are assigned to their first-post quarter, but text is taken from the current registry snapshot. Later amendments can therefore be back-dated. The time series is suitable for descriptive surveillance; causal claims require historical record versions or archived snapshots.

## Event models

Segmented Poisson models use ±8 quarters around each policy event with a denominator offset and robust standard errors. These estimates are associational, not causal.
"""
    (outdir / "methodology.md").write_text(text, encoding="utf-8")


def write_auto_report(outdir: Path, metadata: dict[str, Any], window_df: pd.DataFrame, studies_df: pd.DataFrame, matches_df: pd.DataFrame) -> None:
    primary = studies_df[studies_df["exact_diversity_or_equity"] == 1]
    commit = studies_df[studies_df["broad_human_commitment"] == 1]
    onc_commit = commit[commit["oncology_local"] == 1]
    biomedical = studies_df[studies_df["any_biomedical_false_positive"] == 1]
    exact_matches = matches_df[matches_df["term"].isin(["diversity", "equity"])]
    cls = exact_matches["classification"].value_counts().to_dict()
    lines = [
        "# Automated empirical results: diversity and equity in ClinicalTrials.gov", "",
        f"**Snapshot:** {metadata.get('version_holder', 'unknown')}  ",
        f"**Extracted:** {metadata.get('extraction_time_utc')}  ",
        f"**Candidate records retrieved:** {metadata.get('candidate_records_retrieved'):,}  ",
        f"**Records with exact diversity/equity:** {len(primary):,}  ",
        f"**Records classified as broader human-participant commitments:** {len(commit):,}  ",
        f"**Oncology commitments (local condition/MeSH flag):** {len(onc_commit):,}  ",
        f"**Records with at least one biomedical false-positive use:** {len(biomedical):,}", "",
        "## Exact-word passage classification", "",
    ]
    for key in ["substantive_human_commitment", "human_context_noncommitment", "biomedical_false_positive", "generic_or_uncertain"]:
        lines.append(f"- {key.replace('_', ' ').title()}: {cls.get(key, 0):,}")
    lines.extend(["", "## Period summaries", "", window_df.to_markdown(index=False) if not window_df.empty else "No period summary."])
    lines.extend(["", "## Interpretation guardrail", "", "This is a current-snapshot reconstruction by first-post quarter. Historical-version analysis is required before attributing changes to political or regulatory events."])
    (outdir / "automated_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="artifacts/ctgov_dei_analysis")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--request-interval", type=float, default=float(os.getenv("CTGOV_REQUEST_INTERVAL", "1.25")))
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    setup_logging(outdir)
    cache_dir = outdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = CTGovClient(min_interval=args.request_interval)
    end_date = date.today()

    candidate_cache = cache_dir / "candidate_studies.jsonl"
    denominator_cache = cache_dir / "quarterly_denominators.csv"
    if not args.reuse_cache:
        candidate_cache.unlink(missing_ok=True)
        denominator_cache.unlink(missing_ok=True)

    studies, source_queries = fetch_candidates(client, candidate_cache)
    logging.info("Classifying %d unique candidate records", len(studies))
    study_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    version_holders = set()
    for i, nct in enumerate(sorted(studies), 1):
        study = studies[nct]
        row, matches = study_matches(study)
        if matches:
            row["retrieval_queries"] = "|".join(sorted(source_queries[nct]))
            study_rows.append(row)
            match_rows.extend(matches)
        vh = nget(study, "derivedSection", "miscInfoModule", "versionHolder", default="")
        if vh:
            version_holders.add(str(vh))
        if i % 1000 == 0:
            logging.info("Classified %d/%d", i, len(studies))

    studies_df = pd.DataFrame(study_rows)
    matches_df = pd.DataFrame(match_rows)
    if studies_df.empty or matches_df.empty:
        raise RuntimeError("No local lexicon matches were found; inspect API search behavior")
    studies_df.to_csv(outdir / "candidate_studies_classified.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    matches_df.to_csv(outdir / "matched_passages.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    denom_df = build_denominators(client, args.start_year, end_date, denominator_cache)
    qdf = aggregate_quarterly(studies_df, denom_df)
    qdf.to_csv(outdir / "quarterly_rates.csv", index=False)
    window_df = period_window_summary(qdf)
    window_df.to_csv(outdir / "period_summary.csv", index=False)
    event_df = event_study_models(qdf)
    event_df.to_csv(outdir / "segmented_event_models.csv", index=False)
    top_examples(studies_df, matches_df).to_csv(outdir / "validated_examples_priority_review.csv", index=False)
    build_manual_review_sample(matches_df, outdir / "manual_review_sample.csv")
    plot_rates(qdf, outdir)
    plot_classification(matches_df, outdir)

    metadata = {
        "extraction_time_utc": datetime.now(timezone.utc).isoformat(),
        "end_date": end_date.isoformat(), "start_year": args.start_year,
        "api_url": API_URL, "api_requests": client.request_count,
        "version_holder": max(version_holders) if version_holders else "unknown",
        "all_version_holders_seen": sorted(version_holders), "search_terms": SEARCH_TERMS,
        "candidate_records_retrieved": len(studies),
        "candidate_records_with_local_match": len(studies_df),
        "matched_passages": len(matches_df),
        "exact_diversity_studies": int(studies_df["exact_diversity"].sum()),
        "exact_equity_studies": int(studies_df["exact_equity"].sum()),
        "exact_human_commitment_studies": int(studies_df["exact_human_commitment"].sum()),
        "broad_human_commitment_studies": int(studies_df["broad_human_commitment"].sum()),
        "oncology_broad_human_commitment_studies": int(studies_df.loc[studies_df["oncology_local"] == 1, "broad_human_commitment"].sum()),
        "methodology_hash_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_limitations": [
            "Current snapshot text is assigned to first-post quarter; later edits may be back-dated.",
            "Oncology denominator uses ClinicalTrials.gov condition-search query Neoplasms.",
            "Rule-based semantic classification requires manual validation; a review sample is exported.",
            "API full-text candidate retrieval may index fields beyond selected protocol text; local matching is authoritative.",
        ],
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_methodology(outdir, metadata)
    write_auto_report(outdir, metadata, window_df, studies_df, matches_df)
    if os.getenv("KEEP_RAW_CACHE", "0") != "1":
        candidate_cache.unlink(missing_ok=True)
    logging.info("Finished. Metadata: %s", json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
