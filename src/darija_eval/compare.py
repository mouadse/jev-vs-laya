from __future__ import annotations

import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import load_run
from .dataset import DEFAULT_SPLIT_PATH
from .statistics import paired_statistics


def compare_runs(
    first_run: Path,
    second_run: Path,
    results_root: Path = Path("results"),
    *,
    split_path: Path | None = DEFAULT_SPLIT_PATH,
) -> Path:
    runs = [_load_run(first_run), _load_run(second_run)]
    by_backend = {run["metrics"]["backend"]: run for run in runs}
    if set(by_backend) != {"jev", "laya"}:
        raise ValueError("comparison requires one Jev run and one Laya run")
    jev, laya = by_backend["jev"], by_backend["laya"]
    for field in ("dataset_fingerprint", "split_ids_sha256"):
        left = jev["metrics"].get("provenance", {}).get(field)
        right = laya["metrics"].get("provenance", {}).get(field)
        if left is not None and right is not None and left != right:
            raise ValueError(f"run provenance mismatch for {field}")
    pairs = _pair_predictions(jev["predictions"], laya["predictions"])
    comparison = _comparison_metrics(jev["metrics"], laya["metrics"], pairs)
    comparison["paired_statistics"] = paired_statistics(pairs)
    comparison["source_runs"] = {name: run["metrics"]["analysis_provenance"] for name, run in by_backend.items()}
    comparison["scope"] = {"status": "unverified", "note": "No frozen split file supplied; full-holdout membership unverified."}
    if split_path is not None:
        frozen = json.loads(split_path.read_text(encoding="utf-8"))
        ids = {row["id"] for row in jev["predictions"]}
        expected = set(frozen["eval"])
        if not ids <= expected:
            raise ValueError("comparison contains IDs outside the frozen eval split")
        split_hash = hashlib.sha256(json.dumps(frozen["eval"]).encode()).hexdigest()
        for run in runs:
            provenance = run["metrics"].get("provenance", {})
            for field, expected_value in (
                ("dataset_fingerprint", frozen["dataset_fingerprint"]),
                ("split_ids_sha256", split_hash),
                ("split_size", len(expected)),
            ):
                if field in provenance and provenance[field] != expected_value:
                    raise ValueError(f"run {field} disagrees with the frozen eval split")
        comparison["scope"] = {
            "status": "full" if ids == expected else "subset", "expected_n": len(expected),
            "dataset_fingerprint": frozen["dataset_fingerprint"],
            "note": "IDs checked against local frozen split; historical dataset and prompt provenance cannot be retroactively verified.",
        }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = results_root / f"compare_jev_laya_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paired_rows = [_pair_record(left, right) for left, right in pairs]
    _write_jsonl(output / "paired_predictions.jsonl", paired_rows)
    (output / "report.html").write_text(
        comparison_report(comparison, paired_rows), encoding="utf-8"
    )
    return output


def _load_run(path: Path) -> dict[str, Any]:
    run = load_run(path)
    metrics, predictions = run["metrics"], run["predictions"]
    if metrics.get("split") != "eval":
        raise ValueError(f"comparison requires eval runs: {path}")
    if metrics.get("api_failures") or metrics.get("successful") != len(predictions):
        raise ValueError(f"comparison requires runs without missing predictions: {path}")
    if len({row["id"] for row in predictions}) != len(predictions):
        raise ValueError(f"run contains duplicate prediction IDs: {path}")
    if not predictions:
        raise ValueError("comparison requires non-empty successful runs")
    if len(metrics.get("resolved_models", [])) != 1:
        raise ValueError("comparison requires one resolved model per run; mixed model versions are not comparable")
    return run


def _pair_predictions(left: list[dict], right: list[dict]) -> list[tuple[dict, dict]]:
    left_by_id = {row["id"]: row for row in left}
    right_by_id = {row["id"]: row for row in right}
    if set(left_by_id) != set(right_by_id):
        missing_left = sorted(set(right_by_id) - set(left_by_id))
        missing_right = sorted(set(left_by_id) - set(right_by_id))
        raise ValueError(
            f"runs do not contain the same successful IDs; missing from first={missing_left}, "
            f"missing from second={missing_right}"
        )
    pairs = []
    for identifier in sorted(left_by_id):
        a, b = left_by_id[identifier], right_by_id[identifier]
        for field in ("split", "text", "gold", "writing_style", "topic", "schema_version", "schema_fingerprint"):
            if a.get(field) != b.get(field):
                raise ValueError(f"run mismatch for id {identifier}, field {field}")
        pairs.append((a, b))
    return pairs


def _comparison_metrics(jev: dict, laya: dict, pairs: list[tuple[dict, dict]]) -> dict:
    counts = {"both_correct": 0, "jev_only": 0, "laya_only": 0, "both_wrong": 0}
    same_wrong = 0
    for left, right in pairs:
        if left["correct"] and right["correct"]:
            counts["both_correct"] += 1
        elif left["correct"]:
            counts["jev_only"] += 1
        elif right["correct"]:
            counts["laya_only"] += 1
        else:
            counts["both_wrong"] += 1
            same_wrong += left["predicted"] == right["predicted"]
    jev_errors = {left["id"] for left, _ in pairs if not left["correct"]}
    laya_errors = {right["id"] for _, right in pairs if not right["correct"]}
    laya_server = [
        float(right["raw"]["inference_ms"])
        for _, right in pairs
        if isinstance(right.get("raw"), dict) and right["raw"].get("inference_ms") is not None
    ]
    return {
        "format_version": 1,
        "split": jev["split"],
        "paired_examples": len(pairs),
        "models": {"jev": _model_summary(jev), "laya": _model_summary(laya)},
        "delta_laya_minus_jev": {
            "accuracy": laya["quality"]["accuracy"] - jev["quality"]["accuracy"],
            "macro_f1": laya["quality"]["macro_f1"] - jev["quality"]["macro_f1"],
            "arabic_accuracy": _difference(_style(laya, "Arabic"), _style(jev, "Arabic")),
            "arabizi_accuracy": _difference(_style(laya, "Arabizi"), _style(jev, "Arabizi")),
            "ece": _difference((laya.get("calibration") or {}).get("ece"), (jev.get("calibration") or {}).get("ece")),
            "brier_score": _difference((laya.get("calibration") or {}).get("brier_score"), (jev.get("calibration") or {}).get("brier_score")),
        },
        "outcomes": counts,
        "both_wrong_same_label": same_wrong,
        "both_wrong_different_label": counts["both_wrong"] - same_wrong,
        "errors": {
            "jev": len(jev_errors),
            "laya": len(laya_errors),
            "intersection": len(jev_errors & laya_errors),
            "union": len(jev_errors | laya_errors),
            "jaccard": len(jev_errors & laya_errors) / len(jev_errors | laya_errors)
            if jev_errors | laya_errors
            else 0.0,
        },
        "laya_server_inference_ms": _latency(laya_server),
        "note": (
            "Jev is a managed remote API; Laya is self-hosted on one Modal L4. "
            "Recorded round-trip latency includes different infrastructure, queueing and retries; cached rows retain original timings. It is not a controlled speed comparison."
        ),
    }


def _model_summary(metrics: dict) -> dict:
    calibration = metrics.get("calibration") or {}
    threshold = next(
        (row for row in calibration.get("thresholds", []) if row["threshold"] == 0.9),
        None,
    )
    return {
        "requested_model": metrics["requested_model"],
        "resolved_models": metrics["resolved_models"],
        "successful": metrics["successful"],
        "api_failures": metrics["api_failures"],
        "accuracy": metrics["quality"]["accuracy"],
        "macro_f1": metrics["quality"]["macro_f1"],
        "per_sentiment": metrics["quality"]["per_sentiment"],
        "arabic_accuracy": _style(metrics, "Arabic"),
        "arabizi_accuracy": _style(metrics, "Arabizi"),
        "style_delta": metrics["arabizi_minus_arabic_accuracy"],
        "accuracy_ci95": metrics["quality"].get("accuracy_ci95"),
        "observed_majority_baseline": metrics["quality"].get("observed_majority_baseline"),
        "writing_style": metrics["writing_style"],
        "ece": calibration.get("ece"),
        "brier_score": calibration.get("brier_score"),
        "log_loss": calibration.get("log_loss"),
        "confidence_90": threshold,
        "confidence_thresholds": calibration.get("thresholds", []),
        "latency_ms": metrics["latency_ms"],
        "latency_observation_note": metrics.get("latency_observation_note"),
        "cache": metrics.get("cache"),
    }


def _style(metrics: dict, name: str) -> float | None:
    return metrics["writing_style"].get(name, {}).get("accuracy")


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def _latency(values: list[float]) -> dict | None:
    if not values:
        return None
    array = np.asarray(values)
    return {
        "n": len(values),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _pair_record(jev: dict, laya: dict) -> dict:
    return {
        "id": jev["id"],
        "text": jev["text"],
        "writing_style": jev["writing_style"],
        "topic": jev["topic"],
        "gold": jev["gold"],
        "jev": {"predicted": jev["predicted"], "confidence": jev["confidence"], "correct": jev["correct"]},
        "laya": {"predicted": laya["predicted"], "confidence": laya["confidence"], "correct": laya["correct"]},
    }


def comparison_report(comparison: dict, pairs: list[dict]) -> str:
    jev, laya = comparison["models"]["jev"], comparison["models"]["laya"]
    delta = comparison["delta_laya_minus_jev"]
    disagreements = [row for row in pairs if not (row["jev"]["correct"] and row["laya"]["correct"])]
    cards = "".join(_comparison_card(row) for row in disagreements)
    server = comparison.get("laya_server_inference_ms")
    server_text = "Unavailable" if not server else f"{server['p50']:.1f} ms p50 / {server['p95']:.1f} ms p95"
    return _COMPARISON_DOCUMENT.replace("{{BODY}}", f"""
      <header class="hero wrap"><p class="kicker">Paired benchmark · {comparison['paired_examples']} reviews</p>
        <h1>Jev <i>versus</i> Laya</h1><p class="lede">The same Moroccan Darija reviews and reference labels. A paired comparison of two inference systems.</p>
        <div class="winner"><span>Accuracy delta · Laya − Jev</span><strong>{_signed(delta['accuracy'])}</strong></div>
      </header>
      <main>
        <section class="scoreboard wrap">{_model_column('Jev', jev)}{_model_column('Laya', laya)}</section>
        {_evidence_section(comparison)}
        <section class="split"><div class="wrap"><div><p class="kicker">Writing style</p><h2>Arabic and Arabizi tell different stories.</h2></div>
          <div class="style-compare">{_style_compare('Arabic script', jev['arabic_accuracy'], laya['arabic_accuracy'])}{_style_compare('Arabizi', jev['arabizi_accuracy'], laya['arabizi_accuracy'])}</div></div></section>
        <section class="wrap outcomes"><div><p class="kicker">Paired outcomes</p><h2>Who gets which reviews right?</h2><p>{html.escape(comparison['note'])}</p></div>
          <div class="quadrants">{''.join(_quadrant(key, comparison['outcomes'][key], comparison['paired_examples']) for key in ('both_correct','jev_only','laya_only','both_wrong'))}</div></section>
        <section class="calibration"><div class="wrap"><p class="kicker">Trust &amp; operations</p><h2>Confidence and speed</h2>
          <div class="measure-grid">{_measure('ECE', jev['ece'], laya['ece'], lower=True)}{_measure('Brier score', jev['brier_score'], laya['brier_score'], lower=True)}{_measure('Round-trip p50', jev['latency_ms']['p50'], laya['latency_ms']['p50'], suffix=' ms', decimals=0)}{_measure('Round-trip p95', jev['latency_ms']['p95'], laya['latency_ms']['p95'], suffix=' ms', decimals=0)}</div>
          <div class="thresholds"><h3>When confidence rises, what stays reliable?</h3><p>Coverage is the share of all {comparison['paired_examples']} reviews at or above the threshold; accuracy is measured only on those reviews.</p>
          <table><thead><tr><th>Confidence</th><th>Jev coverage</th><th>Jev accuracy</th><th>Laya coverage</th><th>Laya accuracy</th></tr></thead><tbody>{_threshold_rows(jev['confidence_thresholds'], laya['confidence_thresholds'])}</tbody></table></div>
          <p class="server">Laya GPU inference: <b>{server_text}</b>. Round-trip figures include network and queue time.</p></div></section>
        <section class="errors wrap"><div class="errors-head"><div><p class="kicker">Paired error analysis</p><h2>Where either model missed.</h2></div><p>{len(disagreements)} reviews with at least one error · {comparison['both_wrong_same_label']} identical wrong labels · Error overlap Jaccard {comparison['errors']['jaccard']:.2f}</p></div>
          <div class="filters"><button class="active" data-filter="all">All errors</button><button data-filter="jev">Jev right</button><button data-filter="laya">Laya right</button><button data-filter="neither">Both wrong</button><input id="search" type="search" aria-label="Search reviews" placeholder="Search review text…"></div>
          <div class="cards">{cards or '<p>No model errors.</p>'}</div></section>
      </main>
      <footer class="wrap"><p>Zero-shot · saved prediction analysis · reference labels unchanged</p><div><a href="comparison.json">Comparison JSON</a><a href="paired_predictions.jsonl">Paired predictions</a></div></footer>
    """)


def _model_column(name: str, model: dict) -> str:
    return f"""<article><div class="model-name"><span>{name}</span><small>{html.escape(', '.join(model['resolved_models']))}</small></div>
      <strong class="accuracy">{_pct(model['accuracy'])}</strong><span class="accuracy-label">Accuracy</span>
      <p>95% accuracy interval: {_interval(model.get('accuracy_ci95'))}</p>
      <dl><div><dt>Macro F1</dt><dd>{_pct(model['macro_f1'])}</dd></div><div><dt>Arabic</dt><dd>{_pct(model['arabic_accuracy'])}</dd></div><div><dt>Arabizi</dt><dd>{_pct(model['arabizi_accuracy'])}</dd></div><div><dt>Style gap</dt><dd>{_signed(model['style_delta'])}</dd></div></dl></article>"""


def _interval(value: dict | None) -> str:
    return "unavailable" if value is None else f"{_pct(value['low'])}–{_pct(value['high'])}"


def _evidence_section(comparison: dict) -> str:
    stats = comparison.get("paired_statistics")
    if not stats:
        return ""
    low, high = stats["accuracy_delta_ci95"]
    f_low, f_high = stats["macro_f1_delta_ci95"]
    models = comparison["models"]
    baseline = models["jev"].get("observed_majority_baseline") or {}
    scope = comparison.get("scope", {})
    class_rows = "".join(
        f"<tr><th>{label.title()}</th><td>{models['jev']['per_sentiment'][label]['support']}</td>"
        f"<td>{_pct(models['jev']['per_sentiment'][label]['f1'])}</td>"
        f"<td>{_pct(models['laya']['per_sentiment'][label]['f1'])}</td></tr>"
        for label in ("positive", "neutral", "negative")
    )
    supports = " · ".join(f"{html.escape(style)} N={group['n']}" for style, group in models["jev"]["writing_style"].items())
    return f"""<section class="wrap evidence thresholds"><p class="kicker">Evidence &amp; limits</p>
      <h2>How much evidence supports the gap?</h2>
      <p><b>Scope: {html.escape(scope.get('status', 'unverified'))}</b> · {comparison['paired_examples']} paired reviews · {supports}.</p>
      <p>Always predicting the most frequent reference label ({html.escape(baseline.get('label', 'unknown'))}) scores <b>{_pct(baseline.get('accuracy'))}</b> accuracy here. This is a descriptive class-frequency reference, not a model selected on development data.</p>
      <p>Laya − Jev accuracy gap: <b>{_signed(comparison['delta_laya_minus_jev']['accuracy'])}</b>, 95% paired bootstrap interval <b>{_signed(low)} to {_signed(high)}</b>.
      Macro F1 gap interval: {_signed(f_low)} to {_signed(f_high)}. Exact McNemar p = {stats['mcnemar_exact']['p_value']:.3g}.</p>
      <p>5,000 paired resamples, seed 42, preserving sentiment and style counts. {html.escape(stats['limitations'])}</p>
      <table><thead><tr><th>Sentiment</th><th>Reference N</th><th>Jev F1</th><th>Laya F1</th></tr></thead><tbody>{class_rows}</tbody></table>
      <p><b>Reference labels:</b> supplied dataset annotations, not independently adjudicated. Style groups contain different reviews and topic mixtures, so their gap does not isolate the effect of transliteration.</p>
      <p>{html.escape(scope.get('note', ''))} These saved predictions have already been inspected; further prompt development belongs on dev, with a new independent test set for confirmatory claims.</p>
      <p>Methods: <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html">paired bootstrap</a> · <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html">exact binomial test on discordant pairs</a>.</p></section>"""


def _style_compare(label: str, jev: float, laya: float) -> str:
    return f"""<div class="style-row"><h3>{label}</h3><div><span>Jev</span><i><b style="width:{(jev or 0)*100:.1f}%"></b></i><strong>{_pct(jev)}</strong></div><div><span>Laya</span><i><b style="width:{(laya or 0)*100:.1f}%"></b></i><strong>{_pct(laya)}</strong></div></div>"""


def _quadrant(key: str, value: int, total: int) -> str:
    labels = {"both_correct":"Both correct","jev_only":"Only Jev correct","laya_only":"Only Laya correct","both_wrong":"Both wrong"}
    return f'<div class="{key}"><span>{labels[key]}</span><strong>{value}</strong><small>{value/total:.1%} of paired reviews</small></div>'


def _measure(label: str, jev: float, laya: float, *, lower: bool = False, suffix: str = "", decimals: int = 3) -> str:
    fmt = lambda value: "Unavailable" if value is None else f"{value:.{decimals}f}{suffix}"
    winner = "Lower is better" if lower else ""
    return f"""<div><span>{label}<small>{winner}</small></span><p><b>Jev</b>{fmt(jev)}</p><p><b>Laya</b>{fmt(laya)}</p></div>"""


def _threshold_rows(jev: list[dict], laya: list[dict]) -> str:
    laya_by_threshold = {row["threshold"]: row for row in laya}
    rows = []
    for left in jev:
        right = laya_by_threshold.get(left["threshold"])
        if right is None:
            continue
        jev_accuracy = "n/a" if left["accuracy"] is None else _pct(left["accuracy"])
        laya_accuracy = "n/a" if right["accuracy"] is None else _pct(right["accuracy"])
        rows.append(
            f"<tr><th>≥ {left['threshold']:.2f}</th><td>{_pct(left['coverage'])}<small>N={left['n']}</small></td>"
            f"<td>{jev_accuracy}<small>95% CI {_interval(left.get('accuracy_ci95'))}</small></td><td>{_pct(right['coverage'])}<small>N={right['n']}</small></td>"
            f"<td>{laya_accuracy}<small>95% CI {_interval(right.get('accuracy_ci95'))}</small></td></tr>"
        )
    return "".join(rows) or '<tr><td colspan="5">Complete probabilities unavailable; calibration comparison skipped.</td></tr>'


def _comparison_card(row: dict) -> str:
    if row["jev"]["correct"]:
        outcome = "jev"
    elif row["laya"]["correct"]:
        outcome = "laya"
    else:
        outcome = "neither"
    confidence = lambda name: "n/a" if row[name].get("confidence") is None else f"{row[name]['confidence']:.2f}"
    return f"""<article data-outcome="{outcome}" data-search="{html.escape(row['text'].lower())}"><div class="meta"><span>{html.escape(row['writing_style'])}</span><span>{html.escape(row['topic'].title())}</span><span>ID {row['id']}</span></div><blockquote dir="auto">{html.escape(row['text'])}</blockquote><div class="preds"><div><small>Reference</small><b>{html.escape(row['gold'])}</b></div><div><small>Jev · {confidence('jev')}</small><b>{html.escape(row['jev']['predicted'])}</b></div><div><small>Laya · {confidence('laya')}</small><b>{html.escape(row['laya']['predicted'])}</b></div></div></article>"""


def _pct(value: float) -> str:
    return "n/a" if value is None else f"{value*100:.1f}%"


def _signed(value: float) -> str:
    return "n/a" if value is None else f"{value*100:+.1f} pp"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


_COMPARISON_DOCUMENT = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Jev vs Laya · Darija sentiment</title><style>
:root{--paper:#f4f0e5;--ink:#17221d;--jev:#c44b37;--laya:#35765c;--lime:#dce978;--line:rgba(23,34,29,.2);--serif:Georgia,'Times New Roman',serif;--sans:'Trebuchet MS','Gill Sans',sans-serif}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 var(--sans)}.wrap{width:min(1120px,calc(100% - 32px));margin:auto}.kicker{text-transform:uppercase;letter-spacing:.14em;font-size:.68rem;font-weight:800;color:var(--jev)}.hero{padding:48px 0 82px;border-bottom:1px solid var(--ink);position:relative}.hero h1{font:400 clamp(4rem,11vw,9rem)/.82 var(--serif);letter-spacing:-.075em;margin:55px 0 25px}.hero h1 i{font-weight:400;color:var(--laya)}.lede{font:italic clamp(1rem,2vw,1.35rem) var(--serif);max-width:590px}.winner{position:absolute;right:0;bottom:70px;display:grid;text-align:right}.winner span{text-transform:uppercase;font-size:.62rem;letter-spacing:.1em}.winner strong{font:400 3rem var(--serif);color:var(--laya)}.scoreboard{display:grid;grid-template-columns:1fr 1fr}.scoreboard article{padding:70px clamp(20px,6vw,70px) 85px;min-width:0}.scoreboard article:first-child{border-right:1px solid var(--ink)}.model-name{display:flex;justify-content:space-between;gap:20px;border-bottom:1px solid var(--line);padding-bottom:12px;min-width:0}.model-name span{font:400 2.2rem var(--serif)}.model-name small{text-align:right;max-width:260px;min-width:0;color:#66716b;overflow-wrap:anywhere}.accuracy{display:block;font:400 clamp(5rem,10vw,8rem)/1 var(--serif);letter-spacing:-.06em;margin-top:35px}.scoreboard article:first-child .accuracy{color:var(--jev)}.scoreboard article:last-child .accuracy{color:var(--laya)}.accuracy-label{text-transform:uppercase;letter-spacing:.1em;font-size:.66rem}.scoreboard dl{display:grid;grid-template-columns:1fr 1fr;margin:44px 0 0;border-top:1px solid var(--line)}.scoreboard dl div{padding:14px 0;border-bottom:1px solid var(--line);display:flex;justify-content:space-between}.scoreboard dl div:nth-child(odd){padding-right:18px}.scoreboard dl div:nth-child(even){padding-left:18px;border-left:1px solid var(--line)}dt{color:#68736d}dd{margin:0;font-family:var(--serif)}.split{background:var(--ink);color:var(--paper);padding:90px 0}.split>.wrap,.outcomes{display:grid;grid-template-columns:.7fr 1.3fr;gap:90px}.split h2,.outcomes h2,.calibration h2,.errors h2{font:400 clamp(2.2rem,5vw,4.2rem)/1 var(--serif);letter-spacing:-.045em;margin:20px 0}.style-compare{display:grid;gap:40px}.style-row h3{font:400 1.5rem var(--serif);margin:0 0 14px}.style-row>div{display:grid;grid-template-columns:45px 1fr 58px;gap:12px;align-items:center;margin:9px 0;font-size:.75rem}.style-row i{height:9px;background:#3e4943}.style-row i b{height:100%;display:block}.style-row div:nth-of-type(1) b{background:var(--jev)}.style-row div:nth-of-type(2) b{background:var(--lime)}.style-row strong{text-align:right}.outcomes{padding-top:110px;padding-bottom:110px;align-items:start}.outcomes>div:first-child>p:last-child{color:#69746e;max-width:370px}.quadrants{display:grid;grid-template-columns:1fr 1fr;border:1px solid var(--ink)}.quadrants>div{padding:30px;min-height:190px;display:grid;border:1px solid var(--paper);outline:1px solid var(--ink)}.quadrants strong{font:400 4rem var(--serif);align-self:end}.quadrants small{color:#69746e}.quadrants .jev_only strong{color:var(--jev)}.quadrants .laya_only strong{color:var(--laya)}.calibration{background:#e7dfcc;padding:100px 0}.measure-grid{display:grid;grid-template-columns:repeat(4,1fr);border-block:1px solid var(--ink);margin-top:50px}.measure-grid>div{padding:25px 22px;border-right:1px solid var(--line)}.measure-grid>div:last-child{border:0}.measure-grid>div>span{font:400 1.2rem var(--serif)}.measure-grid span small{display:block;font:normal .6rem var(--sans);text-transform:uppercase;color:#6c756f}.measure-grid p{display:flex;justify-content:space-between;margin:20px 0 0;border-bottom:1px solid var(--line);padding-bottom:7px}.measure-grid p b{text-transform:uppercase;font-size:.65rem}.server{color:#68736d}.errors{padding:110px 0}.errors-head{display:grid;grid-template-columns:1fr .7fr;align-items:end}.errors-head>p{color:#68736d}.filters{display:flex;gap:9px;align-items:center;border-bottom:1px solid var(--ink);padding-bottom:18px;margin:35px 0}.filters button{border:1px solid var(--ink);background:none;border-radius:99px;padding:8px 13px}.filters button.active{background:var(--ink);color:var(--paper)}.filters input{margin-left:auto;background:transparent;border:0;border-bottom:1px solid var(--ink);padding:9px;min-width:240px}.cards{display:grid;grid-template-columns:1fr 1fr;border:1px solid var(--line)}.cards article{padding:30px;border:1px solid var(--paper);outline:1px solid var(--line);min-width:0}.cards article[hidden]{display:none}.meta{display:flex;gap:12px;text-transform:uppercase;font-size:.58rem;color:#6a746e}.cards blockquote{font:400 1.25rem/1.4 var(--serif);margin:25px 0;min-height:80px}.preds{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;border-top:1px solid var(--line);padding-top:14px}.preds div{display:grid}.preds small{font-size:.6rem;color:#6a746e;text-transform:uppercase}.preds b{font-family:var(--serif)}footer{padding:35px 0;border-top:1px solid var(--ink);display:flex;justify-content:space-between;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em}footer div{display:flex;gap:20px}
.evidence{padding-bottom:70px}.evidence h2{font:400 clamp(2rem,4vw,3.4rem)/1.1 var(--serif);letter-spacing:-.035em}.evidence>p{max-width:85ch}.thresholds{margin-top:60px;overflow-x:auto}.thresholds small{display:block;font-size:.65rem;color:#68736d}.thresholds h3{font:400 clamp(1.6rem,3vw,2.4rem) var(--serif);margin:0}.thresholds>p{color:#68736d;margin:8px 0 24px}.thresholds table{border-collapse:collapse;width:100%;font-size:.85rem}.thresholds th,.thresholds td{text-align:right;padding:11px 8px;border-bottom:1px solid var(--line)}.thresholds th:first-child{text-align:left}.thresholds thead th{font-size:.66rem;text-transform:uppercase;letter-spacing:.05em}
@media(max-width:760px){.winner{position:static;text-align:left;margin-top:35px}.scoreboard,.split>.wrap,.outcomes,.errors-head{grid-template-columns:1fr}.scoreboard article:first-child{border-right:0;border-bottom:1px solid var(--ink)}.split>.wrap,.outcomes{gap:35px}.measure-grid{grid-template-columns:1fr 1fr}.measure-grid>div:nth-child(2){border-right:0}.cards{grid-template-columns:1fr}.filters{flex-wrap:wrap}.filters input{width:100%;margin:10px 0 0}.hero h1{font-size:4.4rem}footer{display:grid;gap:14px}.thresholds table{font-size:.72rem}.thresholds th,.thresholds td{padding:9px 2px}.thresholds thead th{font-size:.56rem}}
</style></head><body>{{BODY}}<script>(()=>{let f='all';const bs=[...document.querySelectorAll('[data-filter]')],cs=[...document.querySelectorAll('.cards article')],q=document.querySelector('#search');function a(){const s=(q.value||'').toLowerCase();cs.forEach(c=>c.hidden=!((f==='all'||c.dataset.outcome===f)&&(!s||c.dataset.search.includes(s))))}bs.forEach(b=>b.onclick=()=>{f=b.dataset.filter;bs.forEach(x=>x.classList.toggle('active',x===b));a()});q.oninput=a})()</script></body></html>'''
