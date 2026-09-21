from __future__ import annotations

import html
from collections import Counter
from typing import Any, Sequence

from .metrics import LABELS

LABEL_NAMES = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}


def html_report(
    metrics: dict[str, Any],
    predictions: Sequence[dict[str, Any]],
) -> str:
    """Build a standalone, human-readable HTML evaluation report."""
    metrics = dict(metrics)
    metrics["cached_predictions"] = sum(bool(row.get("cached")) for row in predictions)
    quality = metrics.get("quality")
    if not quality:
        body = _empty_report(metrics)
    else:
        errors = sorted(
            (row for row in predictions if not row["correct"]),
            key=lambda row: row.get("confidence") if row.get("confidence") is not None else -1,
            reverse=True,
        )
        body = "".join(
            [
                _hero(metrics),
                _quality_section(metrics),
                _style_section(metrics),
                _class_and_confusion(quality),
                _confidence_section(metrics),
                _topic_and_latency(metrics),
                _errors_section(errors),
                _methodology(metrics),
            ]
        )
    replacements = {
        "{{TITLE}}": html.escape(f"{metrics.get('backend', 'Model').title()} Darija evaluation"),
        "{{BODY}}": body,
    }
    result = _DOCUMENT
    for token, value in replacements.items():
        result = result.replace(token, value)
    return result


def _hero(metrics: dict[str, Any]) -> str:
    quality = metrics["quality"]
    display_name = str(metrics["backend"]).title()
    resolved = ", ".join(metrics.get("resolved_models") or [metrics.get("requested_model", "unknown")])
    delta = metrics.get("arabizi_minus_arabic_accuracy")
    weakest = min(
        ((label, values) for label, values in quality["per_sentiment"].items() if values["support"]),
        key=lambda item: item[1]["f1"],
    )
    if delta is None:
        finding = "Writing-style comparison is unavailable for this run."
    else:
        stronger = "Arabizi" if delta > 0 else "Arabic script"
        finding = (
            f"Observed {stronger} accuracy was higher by {abs(delta) * 100:.1f} percentage points. "
            f"{LABEL_NAMES[weakest[0]]} was the weakest sentiment class at "
            f"{weakest[1]['f1']:.1%} F1."
        )
    return f"""
    <header class="masthead wrap">
      <div class="kicker"><span>Field report</span><span>{html.escape(str(metrics.get('split', 'unknown')))} sample</span></div>
      <div class="hero-grid">
        <div>
          <p class="eyebrow">{html.escape(display_name)} × Moroccan Darija</p>
          <h1>Can a decision model<br>read the room?</h1>
          <p class="lede">A zero-shot sentiment benchmark across Arabic-script Darija and Arabizi.</p>
        </div>
        <div class="hero-score" aria-label="Overall accuracy {_pct(quality['accuracy'])}">
          <div class="score-ring" style="--score:{quality['accuracy'] * 360:.1f}deg">
            <div><strong>{_pct(quality['accuracy'])}</strong><span>accuracy</span></div>
          </div>
          <p>{html.escape(finding)}</p>
        </div>
      </div>
      <div class="runline">
        <span><b>{metrics['successful']} / {metrics['requested']}</b> successful / requested</span>
        <span><b>{metrics['api_failures']}</b> API failures</span>
        <span><b>{html.escape(resolved)}</b> resolved model</span>
        <span><b>{html.escape(metrics.get('split', 'unknown'))}</b> split</span>
      </div>
      <p class="plain-note">{_scope_note(metrics)}</p>
    </header>
    """


def _quality_section(metrics: dict[str, Any]) -> str:
    quality = metrics["quality"]
    interval = quality.get("accuracy_ci95")
    uncertainty = (
        f"Accuracy 95% Wilson interval: {_pct(interval['low'])}–{_pct(interval['high'])}. "
        "This interval assumes independent reviews; it does not measure label quality or dataset bias."
        if interval else "Accuracy uncertainty is unavailable in this saved run."
    )
    baseline = quality.get("observed_majority_baseline")
    baseline_note = (
        f"Observed majority-class reference: always predicting {html.escape(baseline['label'])} "
        f"would score {_pct(baseline['accuracy'])} on these same successful examples. "
        "This is a descriptive, post-hoc reference, not a baseline selected on dev."
        if baseline else ""
    )
    return f"""
    <section class="wrap section quality" id="quality">
      <div class="section-label"><span>01</span><h2>Quality at a glance</h2></div>
      <div class="stat-line">
        {_stat("Macro F1", quality['macro_f1'], "Balances all three sentiments")}
        {_stat("Macro precision", quality['macro_precision'], "How often class predictions land")}
        {_stat("Macro recall", quality['macro_recall'], "How much of each class is recovered")}
        {_stat("Weighted F1", quality['weighted_f1'], "Accounts for class frequency")}
      </div>
      <p class="plain-note"><b>{quality['correct']} correct</b> and {quality['incorrect']} incorrect out of {quality['n']} successful predictions. API failures are excluded from every quality metric.</p>
      <p class="plain-note">{uncertainty}</p>
      <p class="plain-note">{baseline_note}</p>
    </section>
    """


def _style_section(metrics: dict[str, Any]) -> str:
    styles = metrics.get("writing_style", {})
    rows = []
    for style in ("Arabic", "Arabizi"):
        values = styles.get(style)
        if not values:
            continue
        rows.append(
            f"""
            <div class="style-row">
              <div><strong>{html.escape(style)}</strong><span>{values['n']} reviews</span></div>
              <div class="track"><i style="width:{values['accuracy'] * 100:.1f}%"></i></div>
              <b>{_pct(values['accuracy'])}</b>
              <small>Macro F1 {_pct(values['macro_f1'])}{_group_note(values)}</small>
            </div>
            """
        )
    delta = metrics.get("arabizi_minus_arabic_accuracy")
    delta_text = "Unavailable" if delta is None else f"{delta * 100:+.1f} pp"
    return f"""
    <section class="ink-band section" id="writing-style">
      <div class="wrap style-grid">
        <div class="section-label light"><span>02</span><h2>Accuracy by writing style</h2></div>
        <div class="style-bars">{''.join(rows)}</div>
        <aside class="delta">
          <span>Arabizi − Arabic</span>
          <strong>{delta_text}</strong>
          <p>Observed group difference in this run. Different reviews, sentiments, and topics can contribute; this is not a causal effect of transliteration.</p>
        </aside>
      </div>
    </section>
    """


def _class_and_confusion(quality: dict[str, Any]) -> str:
    class_rows = []
    for label in LABELS:
        values = quality["per_sentiment"][label]
        class_rows.append(
            f"""
            <tr>
              <th><span class="sentiment-dot {label}"></span>{LABEL_NAMES[label]}</th>
              <td>{values['support']}</td>
              <td>{_bar_value(values['precision'])}</td>
              <td>{_bar_value(values['recall'])}</td>
              <td>{_bar_value(values['f1'])}</td>
            </tr>
            """
        )
    matrix = quality["confusion_matrix"]
    maximum = max(max(row) for row in matrix) or 1
    cells = []
    for gold_index, gold in enumerate(LABELS):
        cells.append(f'<div class="matrix-label row-label">{LABEL_NAMES[gold]}</div>')
        for predicted_index, _ in enumerate(LABELS):
            value = matrix[gold_index][predicted_index]
            correct = gold_index == predicted_index
            cells.append(
                f'<div class="matrix-cell {"hit" if correct else "miss"}" '
                f'style="--heat:{value / maximum:.3f}" '
                f'aria-label="Gold {gold}, predicted {LABELS[predicted_index]}: {value}">'
                f'<strong>{value}</strong></div>'
            )
    return f"""
    <section class="wrap section two-up" id="classes">
      <div>
        <div class="section-label"><span>03</span><h2>Where the model holds—and slips</h2></div>
        <div class="table-scroll">
          <table class="class-table">
            <thead><tr><th>Sentiment</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
            <tbody>{''.join(class_rows)}</tbody>
          </table>
        </div>
      </div>
      <div class="confusion-wrap">
        <div class="mini-heading"><span>Confusion matrix</span><small>Rows are gold labels</small></div>
        <div class="matrix">
          <div></div>
          {''.join(f'<div class="matrix-label col-label">{LABEL_NAMES[label]}</div>' for label in LABELS)}
          {''.join(cells)}
        </div>
        <div class="legend"><span>Fewer</span><i></i><span>More</span></div>
      </div>
    </section>
    """


def _confidence_section(metrics: dict[str, Any]) -> str:
    calibration = metrics.get("calibration")
    if not calibration:
        return f"""
        <section class="wrap section" id="confidence">
          <div class="section-label"><span>04</span><h2>Confidence &amp; calibration</h2></div>
          <p class="plain-note">{html.escape(metrics.get('warning') or 'Probability data was unavailable.')}</p>
        </section>
        """
    threshold_rows = []
    for row in calibration["thresholds"]:
        accuracy = "—" if row["accuracy"] is None else _pct(row["accuracy"])
        interval = row.get("accuracy_ci95")
        if interval:
            accuracy += f"<br><small>95%: {_pct(interval['low'])}–{_pct(interval['high'])}</small>"
        threshold_rows.append(
            f"""
            <tr>
              <th>≥ {row['threshold']:.2f}</th>
              <td><div class="coverage"><i style="width:{row['coverage'] * 100:.1f}%"></i></div><span>{_pct(row['coverage'])}</span></td>
              <td>{row['n']}</td><td>{accuracy}</td><td>{row['errors']}</td>
            </tr>
            """
        )
    return f"""
    <section class="paper-band section" id="confidence">
      <div class="wrap confidence-grid">
        <div>
          <div class="section-label"><span>04</span><h2>Confidence &amp; observed accuracy</h2></div>
          <p class="section-copy">Coverage is the share of successful predictions above each threshold. Accepted N shows the evidence behind its accuracy. These are descriptive results, not a guarantee of safe automation; choose thresholds on dev and validate on fresh data.</p>
          <div class="calibration-stats">
            <div><span>Average confidence</span><strong>{calibration['average_confidence']:.3f}</strong></div>
            <div><span>When correct</span><strong>{_number(calibration['confidence_correct'])}</strong></div>
            <div><span>When wrong</span><strong>{_number(calibration['confidence_incorrect'])}</strong></div>
            <div><span>ECE <em>lower is better</em></span><strong>{calibration['ece']:.3f}</strong></div>
            <div><span>Brier <em>lower is better</em></span><strong>{calibration['brier_score']:.3f}</strong></div>
            <div><span>Log loss <em>lower is better</em></span><strong>{calibration['log_loss']:.3f}</strong></div>
          </div>
        </div>
        <div class="threshold-wrap">
          <div class="mini-heading"><span>Confidence threshold</span><small>Successful predictions</small></div>
          <div class="table-scroll"><table class="threshold-table">
            <thead><tr><th>Confidence</th><th>Coverage</th><th>Accepted N</th><th>Accuracy</th><th>Errors</th></tr></thead>
            <tbody>{''.join(threshold_rows)}</tbody>
          </table></div>
        </div>
      </div>
    </section>
    """


def _topic_and_latency(metrics: dict[str, Any]) -> str:
    topics = metrics.get("topics", {})
    topic_rows = "".join(
        f"""
        <div class="topic-row"><span>{html.escape(topic.title())}<small>{values['n']} reviews</small></span>
        <div class="track"><i style="width:{values['accuracy'] * 100:.1f}%"></i></div><b>{_pct(values['accuracy'])}</b></div>
        """
        for topic, values in sorted(topics.items(), key=lambda item: item[1]["accuracy"], reverse=True)
    )
    latency = metrics.get("current_run_latency_ms") or metrics["latency_ms"]
    fresh_latency = metrics.get("current_run_latency_ms")
    latency_heading = "Fresh request latency" if fresh_latency else "Historical request latency"
    cache = metrics.get("cache") or {}
    latency_note = (
        f"{cache.get('reused', metrics.get('cached_predictions', 0))} reused predictions; "
        f"{cache.get('fresh', 'unknown')} fresh; {cache.get('unknown', 'unknown')} with unknown cache status. "
        + ("Timings above include only fresh requests in this run. " if fresh_latency else
           "No verified fresh-request timing is available; timings above are original stored observations. ")
        + "Cached timing is not cache lookup time. Request timing includes network and service overhead, not isolated GPU inference time."
    )
    return f"""
    <section class="wrap section topic-latency" id="operations">
      <div>
        <div class="section-label"><span>05</span><h2>Performance by topic</h2></div>
        <p class="section-copy">Only topics with at least {metrics['topic_min_n']} successful examples are shown. Small groups are omitted to avoid false precision.</p>
        <div class="topic-list">{topic_rows or '<p class="plain-note">No topic met the reporting threshold.</p>'}</div>
      </div>
      <div class="latency-panel">
        <div class="mini-heading"><span>{latency_heading}</span><small>Milliseconds</small></div>
        <div class="latency-main"><strong>{latency['p50']:.0f}</strong><span>p50</span></div>
        <div class="latency-rail">
          <div><span>mean</span><b>{latency['mean']:.0f}</b></div>
          <div><span>p90</span><b>{latency['p90']:.0f}</b></div>
          <div><span>p95</span><b>{latency['p95']:.0f}</b></div>
          <div><span>p99</span><b>{latency['p99']:.0f}</b></div>
        </div>
        <p>Total wall-clock time <b>{metrics['wall_clock_seconds']:.2f}s</b>.</p>
        <p>{latency_note}</p>
      </div>
    </section>
    """


def _errors_section(errors: Sequence[dict[str, Any]]) -> str:
    counts = Counter(row["writing_style"] for row in errors)
    cards = []
    for rank, row in enumerate(errors, 1):
        confidence = row.get("confidence")
        cards.append(
            f"""
            <article class="error-card" data-style="{html.escape(row['writing_style'])}" data-gold="{html.escape(row['gold'])}" data-search="{html.escape(row['text'].lower())}">
              <div class="error-meta">
                <span class="rank">#{rank:02d}</span>
                <span>{html.escape(row['writing_style'])}</span>
                <span>{html.escape(row['topic'].title())}</span>
                <span>ID {row['id']}</span>
              </div>
              <blockquote dir="auto">{html.escape(row['text'])}</blockquote>
              <div class="verdict">
                <div><small>Gold</small><b class="{row['gold']}">{LABEL_NAMES[row['gold']]}</b></div>
                <span aria-hidden="true">→</span>
                <div><small>Model</small><b class="{row['predicted']}">{LABEL_NAMES[row['predicted']]}</b></div>
                <div class="confidence"><small>Confidence</small><b>{_number(confidence)}</b></div>
              </div>
            </article>
            """
        )
    return f"""
    <section class="errors section" id="errors">
      <div class="wrap">
        <div class="errors-head">
          <div class="section-label light"><span>06</span><h2>Read the misses</h2></div>
          <p>{len(errors)} model errors · {counts.get('Arabic', 0)} Arabic · {counts.get('Arabizi', 0)} Arabizi. Sorted by confidence, highest first.</p>
        </div>
        <div class="filters" role="group" aria-label="Filter errors">
          <button class="active" data-filter="all">All <span>{len(errors)}</span></button>
          <button data-filter="Arabic">Arabic <span>{counts.get('Arabic', 0)}</span></button>
          <button data-filter="Arabizi">Arabizi <span>{counts.get('Arabizi', 0)}</span></button>
          <label><span>Search reviews</span><input id="error-search" type="search" placeholder="Type a word or phrase…"></label>
        </div>
        <div class="error-list">{''.join(cards) if cards else '<p class="no-errors">No model errors in this run.</p>'}</div>
        <p class="empty-filter" hidden>No errors match this filter.</p>
      </div>
    </section>
    """


def _methodology(metrics: dict[str, Any]) -> str:
    requested_model = html.escape(metrics.get("requested_model", "unknown"))
    resolved = html.escape(", ".join(metrics.get("resolved_models") or ["unknown"]))
    return f"""
    <footer class="wrap footer">
      <details>
        <summary>Methodology &amp; artifact details</summary>
        <div class="method-grid">
          <p><b>Task</b> One zero-shot Choice question classifies each review as positive, neutral, or negative.</p>
          <p><b>Scope</b> {_scope_note(metrics)} The persisted split manifest records the split strategy. Do not use eval errors to tune prompts or select thresholds.</p>
          <p><b>Inputs</b> The model receives only the review text. Gold label, writing style, and topic are used after prediction for analysis.</p>
          <p><b>Reference labels</b> Dataset labels are treated as ground truth and have not been independently adjudicated. Errors may reflect ambiguous sentiment or annotation issues as well as model limitations.</p>
          <p><b>Failures</b> API failures are counted separately and never treated as incorrect model predictions.</p>
          <p><b>Model</b> Requested <code>{requested_model}</code>; resolved <code>{resolved}</code>; schema <code>{html.escape(metrics.get('schema_version', 'unknown'))}</code>.</p>
          <p><b>Calibration</b> Confidence means the probability assigned to the predicted class. ECE uses ten equal-width bins.</p>
        </div>
      </details>
      <div class="artifact-links"><a href="predictions.jsonl">Predictions</a><a href="failures.jsonl">Model errors</a><a href="api_failures.jsonl">API failures</a><a href="metrics.json">Metrics JSON</a><a href="summary.md">Markdown summary</a></div>
      <p class="footer-mark">DARĪJA / DECISION REPORT <span>Generated locally</span></p>
    </footer>
    """


def _empty_report(metrics: dict[str, Any]) -> str:
    return f"""
    <main class="wrap empty-state">
      <p class="eyebrow">Evaluation report</p><h1>No successful predictions yet.</h1>
      <p>{metrics.get('api_failures', 0)} API failures were recorded from {metrics.get('requested', 0)} requested examples. Quality metrics need at least one successful prediction.</p>
      <p>{_scope_note(metrics)}</p>
      <a href="api_failures.jsonl">Inspect API failures</a>
    </main>
    """


def _scope_note(metrics: dict[str, Any]) -> str:
    split = metrics.get("split")
    if split == "dev" or split == "demo":
        return "Development sample. These results are for iteration and are not held-out evaluation evidence."
    if split == "eval":
        provenance = metrics.get("provenance") or {}
        if provenance.get("full_split") is True:
            return (
                "All frozen eval examples were requested. "
                "Quality metrics describe successful predictions only; review any failures before treating the evaluation as complete."
            )
        if provenance.get("full_split") is False:
            return (
                f"Limited eval run: {provenance.get('selected_size', metrics.get('requested', 0))} "
                f"of {provenance.get('split_size', 'unknown')} frozen eval examples requested. "
                "This is a smoke test, not a full holdout evaluation."
            )
        return "Results cover the requested eval examples only. A limited or incomplete run is not a full evaluation of the frozen holdout."
    return "Run scope is not recorded; do not treat these results as held-out evaluation evidence."


def _group_note(values: dict[str, Any]) -> str:
    interval = values.get("accuracy_ci95")
    note = (
        f" · Accuracy 95% interval {_pct(interval['low'])}–{_pct(interval['high'])}"
        if interval else ""
    )
    support = values.get("sentiment_support")
    if support:
        note += "<br>Gold support: " + ", ".join(
            f"{LABEL_NAMES[label]} {support.get(label, 0)}" for label in LABELS
        )
    return note


def _stat(label: str, value: float, note: str) -> str:
    return f'<div><span>{label}</span><strong>{_pct(value)}</strong><small>{note}</small></div>'


def _bar_value(value: float) -> str:
    return f'<div class="cell-bar"><i style="width:{value * 100:.1f}%"></i><span>{_pct(value)}</span></div>'


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


_DOCUMENT = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <link rel="icon" href="data:,">
  <title>{{TITLE}}</title>
  <style>
    :root{--paper:#f3efe3;--paper-2:#e9e1d0;--ink:#18241f;--muted:#657068;--red:#c24732;--green:#2e7658;--acid:#d9e56f;--line:rgba(24,36,31,.2);--serif:Georgia,'Times New Roman',serif;--sans:'Trebuchet MS','Gill Sans',sans-serif}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);font-size:16px;line-height:1.5}a{color:inherit}button,input{font:inherit}.wrap{width:min(1180px,calc(100% - 40px));margin-inline:auto}.section{padding-block:clamp(64px,9vw,128px)}
    .masthead{padding-top:24px}.kicker{display:flex;justify-content:space-between;border-bottom:1px solid var(--ink);padding-bottom:10px;text-transform:uppercase;letter-spacing:.14em;font-size:.72rem;font-weight:700}.hero-grid{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(300px,.7fr);gap:clamp(40px,8vw,120px);align-items:end;padding:clamp(54px,9vw,120px) 0 54px}.eyebrow{text-transform:uppercase;letter-spacing:.18em;font-size:.76rem;font-weight:800;color:var(--red);margin:0 0 24px}.hero-grid h1,.empty-state h1{font-family:var(--serif);font-weight:400;font-size:clamp(3.3rem,8vw,7.4rem);letter-spacing:-.065em;line-height:.84;margin:0;max-width:900px}.lede{font-family:var(--serif);font-style:italic;font-size:clamp(1.1rem,2vw,1.5rem);max-width:600px;margin:36px 0 0;color:#4a554f}.hero-score{display:grid;justify-items:start;gap:28px}.score-ring{width:210px;aspect-ratio:1;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--green) var(--score),var(--paper-2) 0);position:relative}.score-ring:after{content:"";position:absolute;inset:13px;border-radius:50%;background:var(--paper)}.score-ring div{position:relative;z-index:1;display:grid;text-align:center}.score-ring strong{font-family:var(--serif);font-size:3.6rem;letter-spacing:-.06em;font-weight:400;line-height:1}.score-ring span{text-transform:uppercase;letter-spacing:.13em;font-size:.68rem;font-weight:800;margin-top:8px}.hero-score p{margin:0;max-width:320px;font-size:.95rem}.runline{display:flex;gap:30px 44px;flex-wrap:wrap;border-block:1px solid var(--ink);padding:17px 0;font-size:.76rem;text-transform:uppercase;letter-spacing:.08em}.runline b{font-size:.9rem;letter-spacing:0;margin-right:5px}
    .section-label{display:flex;align-items:baseline;gap:18px;margin-bottom:42px}.section-label span{font-size:.72rem;border:1px solid currentColor;border-radius:50%;width:29px;height:29px;display:grid;place-items:center}.section-label h2{font-family:var(--serif);font-size:clamp(2rem,4.5vw,4.3rem);font-weight:400;letter-spacing:-.045em;line-height:1;margin:0}.section-label.light{color:var(--paper)}.stat-line{display:grid;grid-template-columns:repeat(4,1fr);border-block:1px solid var(--line)}.stat-line>div{padding:30px 24px 32px 0;display:grid;gap:7px;border-right:1px solid var(--line)}.stat-line>div+div{padding-left:24px}.stat-line>div:last-child{border:0}.stat-line span{text-transform:uppercase;letter-spacing:.11em;font-size:.67rem;font-weight:800}.stat-line strong{font-family:var(--serif);font-size:clamp(2.2rem,4vw,4rem);font-weight:400;line-height:1}.stat-line small{color:var(--muted);font-size:.77rem}.plain-note{max-width:680px;margin:28px 0 0;color:var(--muted)}
    .ink-band{background:var(--ink);color:var(--paper)}.style-grid{display:grid;grid-template-columns:.8fr 1.45fr .6fr;gap:clamp(36px,6vw,90px);align-items:center}.style-grid .section-label{align-self:start;margin:0}.style-bars{display:grid;gap:38px}.style-row{display:grid;grid-template-columns:110px 1fr 66px;gap:12px 20px;align-items:center}.style-row>div:first-child{display:grid}.style-row span,.style-row small{color:#aeb9b1}.style-row .track{height:12px;background:#3c4943}.track i{height:100%;display:block;background:var(--acid)}.style-row>b{font-family:var(--serif);font-size:1.4rem}.style-row small{grid-column:2/4}.delta{border-left:1px solid #59655e;padding-left:28px}.delta span{text-transform:uppercase;letter-spacing:.12em;font-size:.66rem}.delta strong{display:block;color:var(--acid);font:400 clamp(2.5rem,5vw,4.8rem)/1 var(--serif);letter-spacing:-.05em;margin:16px 0}.delta p{font-size:.8rem;color:#aeb9b1}
    .two-up{display:grid;grid-template-columns:1.25fr .75fr;gap:clamp(50px,8vw,120px);align-items:start}.two-up>*,.confidence-grid>*,.topic-latency>*{min-width:0}.table-scroll{overflow-x:auto;max-width:100%}.class-table,.threshold-table{border-collapse:collapse;width:100%;font-size:.84rem}.class-table th,.class-table td{padding:17px 15px;border-bottom:1px solid var(--line);text-align:left}.class-table thead th,.threshold-table thead th{text-transform:uppercase;letter-spacing:.08em;font-size:.62rem;color:var(--muted)}.class-table tbody th{font-size:.92rem}.sentiment-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:10px}.positive{color:var(--green)}.negative{color:var(--red)}.neutral{color:#9a7723}.sentiment-dot.positive{background:var(--green)}.sentiment-dot.negative{background:var(--red)}.sentiment-dot.neutral{background:#9a7723}.cell-bar{min-width:92px;position:relative}.cell-bar:before{content:"";position:absolute;left:0;right:0;bottom:0;height:3px;background:var(--paper-2)}.cell-bar i{position:absolute;left:0;bottom:0;height:3px;background:var(--ink)}.cell-bar span{position:relative}.mini-heading{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--ink);padding-bottom:10px;margin-bottom:22px}.mini-heading span{font-family:var(--serif);font-size:1.35rem}.mini-heading small{text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-size:.6rem}.matrix{display:grid;grid-template-columns:76px repeat(3,1fr);gap:5px}.matrix-label{font-size:.65rem;color:var(--muted);display:grid;place-items:center;text-transform:uppercase;letter-spacing:.04em}.row-label{justify-content:end;padding-right:8px}.matrix-cell{aspect-ratio:1;display:grid;place-items:center;position:relative}.matrix-cell:before{content:"";position:absolute;inset:0;background:var(--green);opacity:calc(.08 + var(--heat)*.82)}.matrix-cell.miss:before{background:var(--red)}.matrix-cell strong{z-index:1;font:400 1.6rem var(--serif)}.legend{display:flex;align-items:center;justify-content:flex-end;gap:8px;color:var(--muted);font-size:.62rem;margin-top:12px}.legend i{width:70px;height:5px;background:linear-gradient(90deg,var(--paper-2),var(--green))}
    .paper-band{background:var(--paper-2)}.confidence-grid{display:grid;grid-template-columns:.8fr 1.2fr;gap:clamp(60px,9vw,130px)}.section-copy{max-width:550px;color:var(--muted);margin:-18px 0 36px}.calibration-stats{display:grid;grid-template-columns:repeat(2,1fr);border-top:1px solid var(--line)}.calibration-stats>div{padding:17px 12px 17px 0;border-bottom:1px solid var(--line);display:grid}.calibration-stats span{font-size:.72rem;color:var(--muted)}.calibration-stats em{font-size:.6rem}.calibration-stats strong{font:400 1.7rem var(--serif)}.threshold-table th,.threshold-table td{padding:13px 10px;border-bottom:1px solid var(--line);text-align:right}.threshold-table th:first-child{text-align:left}.threshold-table td:nth-child(2){display:grid;grid-template-columns:1fr 50px;gap:10px;align-items:center}.coverage{height:7px;background:#d7ceba}.coverage i{display:block;height:100%;background:var(--red)}
    .topic-latency{display:grid;grid-template-columns:1.2fr .7fr;gap:clamp(60px,10vw,150px)}.topic-list{display:grid;gap:16px}.topic-row{display:grid;grid-template-columns:180px 1fr 55px;gap:16px;align-items:center}.topic-row>span{display:grid}.topic-row small{color:var(--muted);font-size:.68rem}.topic-row .track{height:8px;background:var(--paper-2)}.topic-row .track i{background:var(--green)}.topic-row b{text-align:right;font-family:var(--serif)}.latency-panel{background:var(--red);color:var(--paper);padding:clamp(28px,4vw,52px);align-self:start}.latency-panel .mini-heading{border-color:var(--paper)}.latency-panel .mini-heading small{color:var(--paper-2)}.latency-main{display:flex;align-items:baseline;gap:12px;margin:28px 0}.latency-main strong{font:400 clamp(4rem,8vw,7rem)/.8 var(--serif);letter-spacing:-.07em}.latency-main span{text-transform:uppercase;font-size:.7rem;letter-spacing:.1em}.latency-rail{display:grid;grid-template-columns:repeat(4,1fr);border-block:1px solid rgba(243,239,227,.4)}.latency-rail div{display:grid;padding:14px 5px}.latency-rail span{font-size:.65rem;text-transform:uppercase}.latency-rail b{font:400 1.2rem var(--serif)}.latency-panel>p{font-size:.74rem;margin:18px 0 0}
    .errors{background:#252e2a;color:var(--paper)}.errors-head{display:grid;grid-template-columns:1fr .7fr;gap:40px;align-items:end}.errors-head p{color:#acb5af;margin:0 0 42px}.filters{display:flex;align-items:end;gap:10px;border-bottom:1px solid #59615d;padding-bottom:18px;margin-bottom:30px}.filters button{appearance:none;border:1px solid #657069;background:transparent;color:var(--paper);padding:9px 14px;border-radius:100px;cursor:pointer}.filters button:hover,.filters button:focus-visible,.filters button.active{background:var(--acid);border-color:var(--acid);color:var(--ink)}.filters button span{opacity:.65;margin-left:4px}.filters label{margin-left:auto;display:grid;gap:5px}.filters label span{font-size:.62rem;text-transform:uppercase;letter-spacing:.1em}.filters input{width:min(280px,38vw);background:transparent;color:var(--paper);border:0;border-bottom:1px solid #77827b;padding:8px 2px;outline:0}.filters input:focus{border-color:var(--acid)}.error-list{display:grid;grid-template-columns:repeat(2,1fr);gap:1px;background:#59615d;border:1px solid #59615d}.error-card{background:#252e2a;padding:clamp(24px,4vw,42px);min-height:300px;display:flex;flex-direction:column}.error-card[hidden]{display:none}.error-meta{display:flex;gap:14px;flex-wrap:wrap;color:#aeb7b1;text-transform:uppercase;letter-spacing:.08em;font-size:.6rem}.error-meta .rank{color:var(--acid)}blockquote{font:400 clamp(1.2rem,2.2vw,1.8rem)/1.35 var(--serif);margin:34px 0;flex:1}.verdict{display:flex;gap:18px;align-items:end;border-top:1px solid #4c5650;padding-top:18px}.verdict div{display:grid}.verdict small{color:#9fa9a3;text-transform:uppercase;font-size:.58rem;letter-spacing:.09em}.verdict b{font-family:var(--serif);font-size:1.05rem}.verdict .confidence{margin-left:auto}.no-errors,.empty-filter{font:italic 1.3rem var(--serif);color:#b8c0bb;padding:35px}
    .footer{padding-block:56px 30px}.footer details{border-block:1px solid var(--ink);padding:18px 0}.footer summary{cursor:pointer;font-family:var(--serif);font-size:1.2rem}.method-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:20px 35px;padding:28px 0 8px}.method-grid p{font-size:.78rem;margin:0;color:var(--muted)}.method-grid b{display:block;color:var(--ink);text-transform:uppercase;letter-spacing:.08em;font-size:.65rem;margin-bottom:4px}.artifact-links{display:flex;flex-wrap:wrap;gap:22px;margin:24px 0}.artifact-links a{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em}.footer-mark{font-family:var(--serif);font-size:clamp(1.4rem,3vw,2.4rem);display:flex;justify-content:space-between;margin:50px 0 0}.footer-mark span{font:normal .65rem var(--sans);text-transform:uppercase;letter-spacing:.1em}.empty-state{min-height:80vh;display:grid;align-content:center}.empty-state p{max-width:600px}
    @media(max-width:850px){.hero-grid,.style-grid,.two-up,.confidence-grid,.topic-latency,.errors-head{grid-template-columns:1fr}.hero-score{grid-template-columns:auto 1fr;align-items:center}.score-ring{width:160px}.stat-line{grid-template-columns:repeat(2,1fr)}.stat-line>div:nth-child(2){border-right:0}.style-grid .section-label{margin-bottom:10px}.delta{border-left:0;border-top:1px solid #59655e;padding:25px 0 0}.error-list{grid-template-columns:1fr}.method-grid{grid-template-columns:repeat(2,1fr)}}
    @media(max-width:560px){.wrap{width:min(100% - 24px,1180px)}.hero-grid{padding-top:42px}.hero-grid h1{font-size:3.25rem}.hero-score{grid-template-columns:1fr}.score-ring{width:145px}.runline{display:grid;grid-template-columns:1fr 1fr}.stat-line{grid-template-columns:1fr}.stat-line>div,.stat-line>div+div{padding:22px 0;border-right:0}.style-row{grid-template-columns:88px 1fr 58px}.matrix{grid-template-columns:62px repeat(3,1fr)}.matrix-cell strong{font-size:1.1rem}.topic-row{grid-template-columns:120px 1fr 48px}.filters{align-items:stretch;flex-wrap:wrap}.filters label{width:100%;margin:12px 0 0}.filters input{width:100%}.method-grid{grid-template-columns:1fr}.footer-mark{display:grid;gap:8px}}
    .runline>*,.method-grid>*{min-width:0;overflow-wrap:anywhere}.method-grid code{overflow-wrap:anywhere}
    @media(max-width:560px){.style-row{grid-template-columns:88px minmax(0,1fr) 70px}}
    @media(prefers-reduced-motion:no-preference){.masthead>*{animation:rise .7s cubic-bezier(.22,1,.36,1) both}.masthead>.hero-grid{animation-delay:.08s}.masthead>.runline{animation-delay:.16s}@keyframes rise{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}}
    @media print{body{background:#fff}.errors,.ink-band,.latency-panel{print-color-adjust:exact;-webkit-print-color-adjust:exact}.filters{display:none}.error-card{break-inside:avoid}.section{padding-block:40px}.artifact-links{display:none}}
  </style>
</head>
<body>
{{BODY}}
<script>
  (() => {
    const buttons = [...document.querySelectorAll('[data-filter]')];
    const cards = [...document.querySelectorAll('.error-card')];
    const search = document.querySelector('#error-search');
    const empty = document.querySelector('.empty-filter');
    let style = 'all';
    function apply() {
      const query = (search?.value || '').trim().toLowerCase();
      let visible = 0;
      cards.forEach(card => {
        const show = (style === 'all' || card.dataset.style === style) && (!query || card.dataset.search.includes(query));
        card.hidden = !show;
        if (show) visible++;
      });
      if (empty) empty.hidden = visible !== 0;
    }
    buttons.forEach(button => button.addEventListener('click', () => {
      style = button.dataset.filter;
      buttons.forEach(item => item.classList.toggle('active', item === button));
      apply();
    }));
    search?.addEventListener('input', apply);
  })();
</script>
</body>
</html>
'''
