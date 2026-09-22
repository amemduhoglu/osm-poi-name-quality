# The rule baseline, in full

This document specifies the rule baseline the article scores the language models
against. It exists because the article reports what the baseline reaches without
printing how it works: the manuscript's word budget is spent on the boundary the
study finds rather than on six format checks, and a reader who wants to judge
whether the baseline is a fair opponent needs the checks themselves. Everything
here is derived from `poi_audit/rules.py` and the thresholds it reads from
`config/config.yaml`; where the two disagree, the code is the specification and
this document is the error.

Nothing here is a result. The numbers quoted are the measured behaviour of each
check on the item set, repeated from the analysis outputs so that a claim about a
threshold can be checked against what that threshold did.

## What the baseline is for

The baseline is the answer a maintainer could write without training data, and it
is the score a model has to beat before any capability is claimed for it. It was
written before any model ran. It is deliberately the obvious implementation
rather than the best possible one: a stronger baseline built by iterating against
the item set would no longer be the thing a practitioner would actually have.

Six checks run, and a seventh is scored and reported without being part of the
baseline.

| Check | Field | Kind | Settles |
|---|---|---|---|
| `phone_syntax` | `phone` | format | malformed phone |
| `opening_hours_syntax` | `opening_hours` | format | malformed opening hours |
| `website_syntax` | `website` | format | malformed web address |
| `value_range` | numeric tags | format | out-of-range value |
| `phone_country_code` | `phone` | reference lookup | wrong calling code |
| `address_city` | `addr:city` | reference lookup | inconsistent address city |
| `category_token` | `name` | reported check, not in the baseline | nothing; see below |

Every check returns one of three outcomes for one field of one record: `pass`,
`flag`, or `abstain`. **An abstention is not a pass.** A lookup that finds no
reference for a record has not cleared that record, and counting it as clean
would let missing coverage read as accuracy. A record is flagged by the baseline
when at least one check flags it.

## The checks

### `phone_syntax`

Tests characters and digit count, not ownership. Whether the number belongs to
the city is the lookup's question.

A value is flagged when it is empty; when it carries a character outside
`+0123456789 ()-./`; when it carries more than one `+`, or a `+` that is not the
first character; when it holds fewer than 5 digits; or when it holds more than
15 digits.

**Thresholds.** The upper bound is the maximum length of an international
telephone number under ITU-T Recommendation E.164, which is fifteen digits: a
published limit, not a choice made here. The lower bound is a choice. Five digits
admits the shortest national and service numbers a mapper may record, and is set
permissively so the rule fails towards passing an unusual real number rather than
towards flagging it.

**What it costs.** 14 of the 421 clean phone values are flagged (3.3%).

### `opening_hours_syntax`

Reads the value as a semicolon-separated list of rules, each an optional selector
followed by comma-separated time spans. `24/7` passes as a whole value and as a
rule. A span passes when it is `off`, `closed`, `open` or `unknown`, or when it
is `HH:MM-HH:MM` with minutes at most 59 and hours at most 48.

**Thresholds.** Forty-eight, not twenty-four, because the OpenStreetMap syntax
writes a span crossing midnight by continuing past 24, so `22:00-26:00` is a
legal way to write a bar closing at two. Rejecting it would flag correct values.

This is deliberately a restrained reading of the opening-hours specification,
which no mapper writes in full and no practitioner implements in full either. The
consequence is not hidden: whatever it rejects among real values is counted as a
false positive and reported.

**What it costs.** 33 of the 385 clean opening-hours values are flagged (8.6%),
the highest false-positive rate of any check, and the reason the baseline's
overall flag rate on clean records is not lower.

### `website_syntax`

A value is flagged when it is empty, contains whitespace, has no `scheme://host`
shape, carries a scheme other than `http` or `https`, carries a second `://`
after the host, or has a host that is not a dotted name with an optional port.

**Thresholds.** The two schemes are the two a browser resolves; a value carrying
another scheme is a link a reader cannot follow from the map, which is what the
tag is for.

**What it costs.** 2 of the 380 clean website values are flagged (0.5%).

### `value_range`

For each numeric tag the record carries and the range table names, the value is
parsed as a number and tested against an inclusive range. A value that does not
parse is flagged. A record carrying no range-checkable tag produces no verdict,
which is an abstention by absence.

The table covers `capacity`, `building:levels`, `level`, `height`, `rooms`,
`beds`, `min_age`, `max_age`, `stars`, `width` and `seats`.

**Thresholds, and an honest note about them.** The ranges are this study's own,
not an external standard. OpenStreetMap documents these keys without publishing a
numeric range for any of them, so no authority could be cited and none is. Each
bound is set deliberately wide, at the point past which a value is not a rare
building but a typing error, and every bound was fixed before the item set was
built. The bounds are listed in `config/config.yaml` under
`references.value_ranges`.

**Shared definition with the corruption.** The same table is read by the
corruption that produces the out-of-range class. Recall on that class is
therefore near one by construction and is reported as such rather than as a
finding. The quantity the table can get wrong is the other one, its flag rate on
clean values.

**What it costs.** 8 of the 203 clean records carrying a range-checkable tag are
flagged (3.9%). A table quietly tightened to fit the corpus would show up there.

### `phone_country_code`

Parses the number and compares its calling code against the calling code expected
for the country the record sits in. The comparison is on calling codes rather
than on regions, because several countries share one code and a number written
under a shared code is not wrong for the city.

**Abstains** when the value carries no international prefix, and when the prefix
does not parse. Most phone values in the corpus are written nationally, so the
abstention rate is substantial and is reported: 53 clean values and 83 corrupted
ones are abstained on.

**Shared definition with the corruption.** The wrong-calling-code class is
injected by replacing the code this lookup expects. Recall is near one by
construction, reported as such, not as a discovery.

**What it costs.** 0 of the 368 clean phone values it decides are flagged.

### `address_city`

Folds case, accents and punctuation from the value, and tests whether it matches
the name of any gazetteer place within 25 km of the record's coordinate. A
gazetteer entry contributes its alternate names as well as its own.

**Abstains** when the gazetteer holds no place within the radius, because it has
then said nothing about the record.

**Thresholds.** The radius is not fitted to the corpus and was fixed before the
lookup was scored. It has to clear the largest distance between a point of
interest inside a city and the gazetteer point standing for that city, which for
a metropolitan area is tens of kilometres, and it has to stay under the spacing
at which a neighbouring settlement would begin answering for a record that is not
in it. Twenty-five kilometres sits inside that band. Places are fetched within a
wider radius than they are matched within, so that the reference is not truncated
at the matching boundary.

**What it costs.** 21 of the 482 clean addr:city values are flagged (4.4%).

### `category_token`, which is not part of the baseline

Tests whether a category-indicative token in the record's name contradicts the
record's category. It is scored against the finished item set and reported, but
it is not part of the baseline and its verdicts do not enter the baseline's
score.

It is reported because the residual class was built with an exclusion that this
check measures, and the exclusion is not trusted. A name that announces its own
category would be settled by a token rule, and the residual class is meant to
survive every rule the baseline could hold, so donor names carrying an indicative
token were excluded when the class was built. A token counts as indicative when
it appears at least twenty times in the corpus and at least half its occurrences
fall in one category; the tokens are derived from the corpus rather than written
by hand, so no translator's judgement enters the construction.

The check therefore abstains on all 300 corrupted residual-class items, which is
what the exclusion guarantees rather than something it discovered. **The number
that carries information is the counterfactual**, and it was measured rather than
assumed: without the exclusion, and with every other eligibility rule held
identical, the token rule would settle 120 of the 300 residual-class items (0.40)
(`scripts/measure_donor_exclusion.py`). The exclusion is doing real work rather
than tidying an edge case. On the clean half of the class the check flags 26 of
300, so applied to this class it yields nothing and costs about nine false
positives in a hundred.

## What the reader should be suspicious of, and where to look

Two questions are worth asking of any baseline reported as competitive, and both
are answerable from the deposited outputs rather than from this document.

**Is the baseline's recall real, or built in?** Partly built in, and the article
says so. Two of the six classes are produced by corrupting exactly what the
corresponding check tests: the out-of-range class against the same range table,
and the wrong-calling-code class against the same expected code. Recall on those
two classes is 100/100 by construction. The remaining classes are not arranged
this way, and the residual class is not touched by any check in the baseline: no
rule in the baseline reads a name.

**Were the thresholds tuned against the data they are scored on?** There is no
held-out split, so this cannot be settled by design, and it should not be taken
on trust. Three things bear on it. Every threshold was fixed before the item set
was built, which the construction log records as it happened. No threshold is a
fitted quantity: each is either a published limit (E.164's fifteen digits), a
syntax convention (the 48-hour wrap), or a deliberately wide bound. And the
quantity a tuned threshold would improve, the flag rate on clean values, is
measured on 900 clean records that carry real values written by mappers, where
the baseline flags 75 of them (8.3%). A baseline fitted to its corpus would be
expected to do better than that, not worse.

The sharpest evidence against overfitting is elsewhere in the article. On the
natural residual set's random stratum, drawn from real records rather than
injected ones and drawn without the baseline playing any part in it, the rule
raises none of the twelve wrong names among the 113 records the readers decided:
a recall of 0.000, against one flag in 101 sound records. The interval on that
recall reaches 0.243, so the honest reading is that the rule is not shown to work
there rather than that it is shown to fail completely. Either way it does not
transfer.

A third point belongs here because it cuts the other way, and the article reports
it rather than leaving it to be found. Part of what the baseline and the models
separate on the injected set is the construction's own: two donor filters
constrain the injected name and nothing constrains the one it is paired against,
so a rule reading only the clean twin's length and category token reaches a
Youden's J of 0.487. That is a property of how the items were built, not of the
task, and it is why the natural set exists.

## Reproducing these numbers

```
python scripts/score_rule_baseline.py
```

writes `data/processed/injected_set/rule_baseline.json`, from which every count
quoted above is taken, together with the per-class and per-check tables beside
it. The checks themselves are in `poi_audit/rules.py`; their thresholds, and a
comment on where each one comes from, are in `config/config.yaml`.
