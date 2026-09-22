# Dataset survey: replacing `ohidaoui/darija-reviews`

Date: 2026-09-21. All counts below were verified by loading the datasets, not
from search snippets. Label mappings were verified by text-join against
original sources where stated.

## Current dataset limits

`ohidaoui/darija-reviews` (test split, 851 rows) is the only 3-class Darija
review set found on Hugging Face, but: eval is 171 rows (neutral n=25,
Arabizi n=44), labels are 53.6% positive, topics skew to IT (33.7%), there is
no documented annotation protocol or adjudication, and the repo has no license.

## Recommendation

**`mteb/AfriSentiClassification`, config `ary`** (mirror of the SemEval-2023
Task 12 AfriSenti Moroccan subset).

- Sizes: train 5583 / validation 494 / test 2048 (mirror test is a subset of
  the original 2961-row test; all 2048 rows text-match the original).
- Labels: native 3-class int, verified 100% by text-join against
  `afrisenti-semeval/afrisent-semeval-2023` TSVs (8125/8125):
  0=positive, 1=neutral, 2=negative.
- Scripts (own audit): train 3259 Arabic / 2276 Latin / 47 mixed;
  test 921 Arabic / 1073 Latin / 54 mixed. Best Arabizi coverage of any
  candidate — the script comparison this benchmark cares about becomes
  well-powered.
- Provenance: peer-reviewed (SemEval-2023 Task 12, 3 annotators per tweet),
  real train/val/test splits, CC-BY-4.0 on both the mirror card and the
  original repo.
- Cost: 12x the current eval (tighter CIs), balanced classes
  (test: pos 798 / neu 660 / neg 590).

Caveats: tweets, not product reviews (`@user` masks, hashtags, emoji), so
schema instructions must drop "reviewer/product or service" wording; no
topic column (adapter must default it, e.g. `social`); `writing_style` must
be derived by script detection instead of read from a column.

## Runner-up

**`MBZUAI-Paris/DarijaBench` split `msda`** (ODC-BY): 5220 rows, balanced
1740/1740/1740, ~10% Latin-script rows. Held-out test slice of MSDA
(Boujou et al. 2021, 52K Darija tweets, semi-automatic labels). Usable, but
prompt-wrapped (`messages` format needs template parsing), weaker label
provenance than AfriSenti, and less Arabizi. Split `mac` (1743 balanced
3-class, manually labeled, Arabic-only) is a quality-over-size alternative.

## Rejected

- `AbderrahmanSkiredj1/MSAC_darija_sentiment_analysis`: 2000 rows but
  binary pos/neg plus one corrupt `,ne` label.
- `arbml/ArSarcasm_v2`: only 45 Maghrebi rows (rest MSA/Egyptian/Gulf/Levant).
- `community-datasets/oclar`: MSA hotel reviews with 1–5 ratings, not Darija.
- `BelhaddadMohamed/darija_sentiment.csv`: repo contains no data files.
- `atlasia/Social_Media_Darija_DS`: no sentiment labels.
- `arbml/Arabic_Sentiment_Twitter_Corpus`: binary, Gulf tweets.
- DarijaBench `myc`/`electro_maroc`/`msac`: binary only.

## Migration sketch (not implemented)

1. `dataset.py`: add source adapter (`text`→`review`, int→label map,
   script-detected `writing_style`, default `topic`), new fingerprint +
   frozen split file; keep the old split file for reproducibility of past runs.
2. `schema.py`: version bump with domain-neutral wording ("message/post"
   instead of "review"); old runs stay readable but incomparable by design.
3. Docs: record the domain change (reviews → tweets) in `docs/evaluation.md`;
   rerun `inspect` + full eval per backend before `compare`.
