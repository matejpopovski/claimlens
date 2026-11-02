"use strict";

const OUTCOME_TITLES = {
  SUPPORTS: "Supported by the retrieved abstracts",
  REFUTES: "Refuted by the retrieved abstracts",
  NOT_ENOUGH_INFO: "Not enough information in the retrieved abstracts",
  ABSTAINED: "Declined to answer",
};
const LABEL_NAMES = {
  SUPPORTS: "Supports",
  REFUTES: "Refutes",
  NOT_ENOUGH_INFO: "Not enough info",
};
const REASON_TITLES = {
  low_confidence: "Low confidence",
  conflicting_evidence: "Conflicting evidence",
  weak_retrieval: "Weak retrieval",
};

const form = document.getElementById("claim-form");
const input = document.getElementById("claim");
const submit = document.getElementById("submit");
const statusLine = document.getElementById("status");
const finding = document.getElementById("finding");
const evidence = document.getElementById("evidence");

function element(tag, options = {}, ...children) {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  if (options.text !== undefined) node.textContent = options.text;
  if (options.title) node.title = options.title;
  node.append(...children);
  return node;
}

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function probabilityBars(probabilities) {
  const bars = element("div", { className: "bars" });
  for (const [label, value] of Object.entries(probabilities)) {
    const fill = element("span");
    fill.style.width = percent(value);
    bars.append(
      element("span", { text: LABEL_NAMES[label] }),
      element("div", { className: `bar ${label}` }, fill),
      element("span", { className: "value", text: percent(value) }),
    );
  }
  return bars;
}

function renderFinding(result) {
  finding.className = `finding ${result.outcome}`;
  const children = [
    element("p", { className: "finding-label", text: "Finding" }),
    element("h2", { text: OUTCOME_TITLES[result.outcome] }),
  ];
  if (result.abstentions.length > 0) {
    const reasons = element("ul", { className: "reasons" });
    for (const abstention of result.abstentions) {
      reasons.append(
        element(
          "li",
          {},
          element("strong", { text: REASON_TITLES[abstention.reason] }),
          element("span", { text: abstention.message }),
        ),
      );
    }
    children.push(
      reasons,
      element("p", {
        className: "guess",
        text: `Leading verdict, not asserted: ${LABEL_NAMES[result.best_guess]}.`,
      }),
    );
  }
  children.push(
    probabilityBars(result.probabilities),
    element("p", {
      className: "bars-caption",
      text:
        `Claim-level probabilities, pooled over the abstracts below. Estimated chance that ` +
        `the leading verdict is correct: ${percent(result.confidence)}.`,
    }),
  );
  finding.replaceChildren(...children);
  finding.hidden = false;
}

function retrievalTrail(retrieval) {
  const parts = [
    retrieval.bm25_rank ? `BM25 #${retrieval.bm25_rank}` : "BM25 unranked",
    retrieval.dense_rank ? `dense #${retrieval.dense_rank}` : "dense unranked",
    `reranker ${retrieval.rerank_score.toFixed(2)}`,
  ];
  return parts.join(" · ");
}

function sentenceTitle(sentence) {
  return `Evidence score ${percent(sentence.evidence_score)}`;
}

function renderAbstract(paper) {
  const body = element("p", { className: "abstract" });
  for (const sentence of paper.sentences) {
    if (sentence.rationale_rank === null) {
      body.append(element("span", { text: sentence.text, title: sentenceTitle(sentence) }));
    } else {
      const mark = element("mark", { className: paper.verdict, title: sentenceTitle(sentence) });
      mark.append(sentence.text, element("sup", { text: `[${sentence.rationale_rank}]` }));
      body.append(mark);
    }
    body.append(" ");
  }
  const meta = element(
    "p",
    { className: "meta" },
    element("span", {
      className: `stance ${paper.verdict}`,
      text: `${LABEL_NAMES[paper.verdict]} ${percent(paper.probabilities[paper.verdict])}`,
    }),
    ` · ${retrievalTrail(paper.retrieval)} · S2 corpus ID ${paper.doc_id}`,
  );
  return element(
    "article",
    { className: "paper" },
    element("h4", { text: `${paper.rank}. ${paper.title}` }),
    meta,
    body,
  );
}

function renderEvidence(result) {
  const legend = element(
    "p",
    { className: "legend" },
    element("span", {
      text: "Marked sentences are the rationale the model read, numbered by strength.",
    }),
    element("span", {
      text: "Per-abstract percentages come before claim-level calibration.",
    }),
  );
  evidence.replaceChildren(
    element("h3", { text: `Abstracts read (${result.evidence.length})` }),
    legend,
    ...result.evidence.map(renderAbstract),
  );
  evidence.hidden = result.evidence.length === 0;
}

async function verify(claim) {
  submit.disabled = true;
  statusLine.textContent = "Retrieving abstracts and reading sentences...";
  try {
    const response = await fetch("/api/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ claim }),
    });
    if (!response.ok) throw new Error(`The server answered ${response.status}.`);
    const result = await response.json();
    renderFinding(result);
    renderEvidence(result);
    statusLine.textContent = `Answered in ${result.seconds.toFixed(1)} s.`;
  } catch (error) {
    statusLine.textContent = `Could not verify the claim. ${error.message}`;
  } finally {
    submit.disabled = false;
  }
}

async function loadPage() {
  const [health, examples] = await Promise.all([
    fetch("/api/health").then((r) => r.json()),
    fetch("/api/examples").then((r) => r.json()),
  ]);
  document.getElementById("corpus-size").textContent =
    `${health.corpus_size.toLocaleString("en-US")}`;
  const policy = health.policy;
  const checks = [`an estimated ${policy.min_confidence.toFixed(2)} chance of being correct`];
  if (policy.min_retrieval_score !== null) {
    checks.push(`a top reranker score of at least ${policy.min_retrieval_score.toFixed(2)}`);
  }
  if (policy.conflict_probability !== null) {
    checks.push("no confident disagreement between abstracts");
  }
  document.getElementById("policy").textContent =
    `An answer requires ${checks.join(", ")}. Each abstract set is the top ${health.top_k} after ` +
    `reranking. Thresholds were fixed on a calibration split. Weights: ${health.weights}.`;
  if (health.weights_note) {
    const note = document.getElementById("weights-note");
    note.textContent = health.weights_note;
    note.hidden = false;
  }
  const holder = document.getElementById("examples");
  for (const example of examples) {
    const button = element("button", { text: example.claim, title: example.source });
    button.type = "button";
    button.addEventListener("click", () => {
      input.value = example.claim;
      verify(example.claim);
    });
    holder.append(button);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const claim = input.value.trim();
  if (claim.length >= 3) verify(claim);
});

loadPage().catch((error) => {
  statusLine.textContent = `Could not reach the service. ${error.message}`;
});
