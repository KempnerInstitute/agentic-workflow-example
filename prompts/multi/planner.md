You plan wave {W} of a pruning study. Plan {K} configurations. Round attempts left: {R}. GPU minutes left: {G}.
Best round so far that keeps the accuracy threshold: {BEST}. Rounds in a row without a smaller model: {STALE} of {PLATEAU}.

Read {EXPERIMENT}, configs/round.schema.json, state/baseline/metrics.json, {STATE}/history.tsv, {STATE}/hypotheses.tsv, and {STATE}/notes.md. Plan {K} configurations that can keep the accuracy threshold with fewer weights than the best round so far. Return them in the required JSON format. Give each one a hypothesis, a predicted validation accuracy, a pass probability, and a short reason.
