# Response to Reviewer — comparison to Celcomen

Single conceptual point. No main-text change; one clarifying sentence added to the
appendix baseline description.

---

## 1. Rebuttal paragraph (response-to-reviewer)

> **On comparing against Celcomen.** We thank the reviewer for raising Celcomen
> \citep{megas2025estimation}. Celcomen targets a different intervention class than our
> tissue-graph counterfactuals: it learns a single, global, tissue-wide gene--gene
> interaction matrix and perturbs **gene values** rather than perturbing the tissue
> graph. Its perturbation primitive is also operationally different from ours: it clamps
> a single gene to a predefined value as an initial condition and relaxes the whole
> expression field to observe propagation within a chosen region, whereas our **node
> perturbation** shifts a coherent programme of many genes (here, 200) in a focal cell's
> *neighbours* by an observed REF$\rightarrow$CRC log-fold change and reads out that
> focal cell under fixed topology. It thus implements neither our node perturbation nor
> our **edge perturbation** (neighbourhood rewiring). A best-effort adaptation to the
> node-perturbation task also did not scale: as released, Celcomen attempts to allocate
> $\sim$42.6\,GiB of VRAM on `crc_232` --- the *smallest* slide in our cohort --- because
> its objective materialises a dense cell-by-cell matrix that grows quadratically in the
> number of cells. We will add a sentence to the appendix baseline description making
> this distinction explicit.

---

## 2. Planned manuscript change (appendix baseline description)

Append to the existing Celcomen text:

```latex
Concretely, Celcomen perturbs \emph{gene values} under a single global, tissue-wide
interaction matrix, clamping a single gene to a predefined value and relaxing the whole
expression field to observe propagation within a chosen region. This differs
operationally from our node perturbation, which shifts a coherent programme of many
genes (here, 200) in a focal cell's neighbours by an observed log-fold change under
fixed topology; Celcomen implements neither this nor our edge perturbation
(neighbourhood rewiring). A best-effort adaptation also did not scale: as released,
Celcomen attempts to allocate $\sim$42.6\,GiB of VRAM on our smallest CRC slide, as its
objective materialises a dense cell-by-cell matrix that grows quadratically in the
number of cells.
```
