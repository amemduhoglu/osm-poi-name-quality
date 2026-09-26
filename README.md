# Point-of-interest name quality in OpenStreetMap: evaluating rule, gazetteer and language-model checks against four gates (released code and data)

This archive holds what a reader needs to re-derive every number in the article
and nothing else. It was built by `scripts/build_archive.py` from the study's
working repository on 2026-09-26.

## Reproducing the article's numbers

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    cp .env.example .env                      # the local server's address, and a key
                                              # only for the hosted references
    python -m pytest                          # the construction rules
    python -m scripts.score_models            # every model score
    python -m scripts.score_gates             # the audit gates and the four views
    python -m scripts.score_matched_subset    # the pairs the construction did not mark
    python -m scripts.score_size_trend        # the pre-registered size rule
    python -m scripts.score_reference         # the four evaluated references
    python -m scripts.score_retrieval_baseline  # the dense-retrieval comparator
    python -m scripts.score_cost              # the efficiency profile
    python -m scripts.score_decoding          # the unconstrained-decoding arm
    python -m scripts.score_pipeline          # the composed pipeline and its cost
    python -m scripts.measure_agreement       # the agreement between the annotators
    python -m scripts.measure_prevalence      # the corpus-weighted rate
    python -m scripts.measure_clean_flags     # the flags on untouched records

Four steps read the corpus or the open references rather than the item sets, and
neither of those is deposited: the corpus is OpenStreetMap and the references are
open gazetteers, both of which the scripts below fetch. Run the fetch first and
the four after it.

    python -m scripts.extract_corpus          # the corpus, re-extracted
    python -m scripts.fetch_authorities       # the open references
    python -m scripts.score_rule_baseline     # the rule baseline
    python -m scripts.score_screen_baseline   # the string baseline
    python -m scripts.measure_item_artefacts  # the construction's marks on the items
    python -m scripts.measure_authority_coverage  # what an open authority matches

One order matters. `score_screen_baseline` writes the baseline-dominance
comparison on the residual question, which `score_gates` reads, so a reader
re-deriving the gate table from nothing runs it again after the fetch:

    python -m scripts.score_gates             # again, once the screen has been scored

Each script reads `data/` and writes into `data/processed/analysis`, which is
also shipped as built, so a reader can compare rather than trust. Every output
in that directory was written by the run of these scripts that the article's
numbers are read from. Nothing here
calls a model: the 157,515 responses the scores are taken from
are included as they arrived, so no inference has to be paid for twice. Running
the models again needs `scripts/run_inference.py`, a local server and the card
the article names.

## What is here

- `config/config.yaml`: every experiment parameter. Nothing is hardcoded elsewhere.
  Each rule threshold carries a comment on where it comes from.
- `rule_specification.md`: the rule baseline in full, check by check, with the
  provenance of every threshold, what each check costs on clean values, and the
  two classes whose recall is near one by construction. The article scores the
  models against this baseline without printing it.
- `poi_audit/`, `scripts/`, `prompts/`, `tests/`: the package, the pipeline, the
  versioned prompt templates, and the tests that enforce the construction rules.
- `data/processed/injected_set/`: the items every model was scored on, with the
  taxonomy and the key from each class's code to the name the article prints.
- `data/processed/natural_set/`: the drawn records, both annotators' independent
  labels, the adjudicated labels and the agreement.
- `data/responses/`: every model answer as it arrived, with the server's own
  timings beside it.
- `data/processed/analysis/`: the tables every printed number is read from,
  including the two baselines the residual class is scored against, the four
  views of the input-dependence check, the construction's marks on the item
  set, the corpus-weighted rate of the residual class, and the flags the rules
  raise on untouched records.

## What is not here, and why

- The working record: the progress log, the specifications and the run logs.
  They are how the study was made rather than what it found.
- The manuscript sources. They are the article rather than its evidence, and so
  is the code that renders its figures and tables, with the style layer and the
  face they use. Every value those items print is in
  `data/processed/analysis/`, which is deposited as built.
- The raw corpus dump. The corpus is OpenStreetMap, extracted on
  2026-08-02 inside a 10 km disc of each city centre, and
  re-extracted by `scripts/extract_corpus.py`. Every record
  any model was scored on is in the item sets above.
- The annotator-facing labelling instrument, which is written in the annotators'
  own language. The labels it produced are included.
- The practice items the annotators worked through before the natural set, and
  their labels. Every analysis script excludes them by name, so no number rests
  on them.
- Two pilots: the city pilot that preceded the mechanical candidate pool, and the
  verification pilot that chose how the natural set would be labelled. The pool
  and the labels they led to are here; the pilots themselves decided a design and
  not a result.

## Licences

Code: MIT, as `LICENSE`. Data derived from OpenStreetMap is under
ODbL; the reference sources and their access dates are recorded in
`data/processed/authorities/authority_manifest.json`, each under the licence
named there. The two evaluated reference models are named in
`data/processed/analysis/reference_provenance.json` with their weight licences.

## Who is named here, and who is not

This archive is published under the name of the article's sole author, which
`LICENSE` and `CITATION.cff` carry and no other file does. It names no
affiliation and no funder, and the two annotators are identified as `reader_a`
and `reader_b` rather than by name. The build fails rather than warns if any of
those reaches the tree.
