# Clickstream — IEEE Publication-Grade Implementation Roadmap

This is the master plan. It covers, in order: the publication-integrity framework you need in place before writing a line of code (Part 0), the exact scientific contribution this project is going to defend and why it's real (Part 1), a direct answer to your IDE-vs-Colab question with a week-by-week environment map (Part 2), the 10-week plan at overview level (Part 3 — full code-level detail for each week lives in its own `weekN.md`, starting with the attached `week1.md`), and a final pre-submission checklist (Part 4).

Every claim in Part 0 is sourced directly from IEEE's own author-center policy pages, fetched during this research, not from general recollection — because getting this wrong is precisely the risk you're trying to eliminate.

---

# PART 0 — RESEARCH INTEGRITY & PUBLICATION COMPLIANCE FRAMEWORK

## 0.1 Why this comes before any code

You said you want the theoretical ground "rock solid so that no one gets a chance to point fingers." The single biggest risk to that isn't a weak result — reviewers accept honestly-scoped modest results constantly. The biggest risk is an *undisclosed* process failure: unacknowledged AI-generated text, an uncited source, a dataset claim that doesn't hold up, or a methodology detail that looks like it was hidden rather than omitted by oversight. All of these are avoidable with discipline applied from week 1, and none of them are avoidable if you only think about them in week 10. This section is the discipline.

## 0.2 IEEE authorship rules, and exactly what they mean for AI-assisted work

IEEE (PSPB Operations Manual, Section 8.2.1.A.1, via the IEEE Author Center's Ethical Requirements page) defines authorship as requiring **all three** of the following:

1. Significant intellectual contribution to the theoretical development, system or experimental design, prototype development, and/or analysis and interpretation of the associated results.
2. Contributed to drafting the article or reviewing and/or revising it for intellectual content.
3. Approved the final version of the article as accepted for publication, including references.

Walk this against your situation directly: an AI tool cannot meet criterion 3 — it cannot hold accountability for a published claim, and it does not "approve" anything in the legally/academically meaningful sense the criterion requires. This means an AI is **structurally incapable of being a listed author** under IEEE's own definition, full stop. That's not a gray area you need to argue about — it resolves cleanly. What follows from this is actually good news for your situation: you already intend to do the experimental design, run the analysis, and write and approve the final paper yourself, which means you unambiguously satisfy all three criteria and are unambiguously the sole author. AI assistance in coding and planning is real, substantial help — but it is categorically a *tool*, in the same bucket as a compiler or a statistics package, not a collaborator competing for authorship credit. Say this to yourself now so it isn't a source of anxiety later: using AI extensively and being the sole legitimate author are not in tension. They're just two different questions with two different, already-settled answers.

## 0.3 The AI disclosure requirement — exact policy, and what to write

From IEEE's Submission and Peer Review Policies (fetched directly, current as of this research):

- Authors must **disclose the use of AI-generated content** (including text, figures, images, or code) **in the acknowledgments section** of any submitted article.
- The **AI system used must be identified**.
- **Specific sections of the paper that use AI-generated content must be identified**, along with a **brief explanation of the level at which the AI system was used** to generate the content.
- There is one carve-out: using AI tools solely to **edit or enhance grammar, syntax, and spelling** of author-generated content is generally considered outside the intent of this policy — disclosure isn't strictly required for that narrow case, though IEEE still recommends it. IEEE's own tip here is worth following directly: if you use an AI tool for grammar/editing, exclude your reference list from what you feed it (citation formatting has enough failure modes without an LLM silently "fixing" a correct reference into an incorrect one), and always double-check whatever the tool returns.

**What this means for you concretely.** Everything from this compendium and roadmap that ends up informing your paper — architecture choices, the attack/defense design, code scaffolding — needs to be logged as you go, not reconstructed from memory in week 10. Section 0.8 below gives you the exact log format to start today. Your eventual acknowledgments section will end up reading something like: *"AI language models (Claude, and [Gemini/Copilot if used for implementation]) were used to assist with literature review synthesis, code scaffolding, and editorial feedback during drafting. All experimental design decisions, implementation choices, analysis, and conclusions are the author's own. AI-assisted sections are identified in the supplementary reproducibility log."* — draft this now, refine it as you go, don't invent it retroactively.

## 0.4 Human subjects and data ethics — what applies here, and what doesn't

IEEE's policy is explicit: **any paper reporting on research involving human subjects must include a statement that the research was conducted under the oversight of an Institutional Review Board (IRB) or equivalent local/regional ethics body**, naming that body — **or** must explain why such review was not conducted. Separately, if the research involves identifiable human data, the paper must state that informed consent was obtained, or explain why it wasn't.

This has two distinct implications for Clickstream:

- **The public benchmark datasets (CMU, HMOG, Balabit, Aalto/KVC, etc.) already carry their own ethics approval from when they were originally collected.** Your obligation here is straightforward and low-risk: cite the original dataset papers properly (Section 13/Appendix B of the compendium already has these), and state plainly in your methods section that you are re-using previously-collected, ethically-approved public data, not collecting new data from unconsented subjects.
- **The screen-recording injection-attack demo (Week 6) is a different case, and you need to treat it as one.** The moment you record a real person typing — even yourself — and that recording or data derived from it appears in a published paper as evidence, you are doing human-subjects data collection, and the "or explain why review was not conducted" clause is the one you'll be relying on. The lowest-risk path: restrict this to **yourself only**, as a single-subject proof-of-concept explicitly labeled as such (not a general claim about "users"), with a short, explicit self-consent statement in the paper ("the author recorded themselves as the sole subject for this proof-of-concept; no other individuals' data was collected"), and — this part matters — **ask your department/supervisor now whether Kathmandu University has any student-research ethics process that technically applies**, even to a single-subject self-recording. Don't assume either way. If a lightweight departmental sign-off exists and takes an afternoon, getting it in week 1 costs you nothing and removes this as a review-time question entirely.

## 0.5 Citation, plagiarism, and data-integrity rules

Directly from IEEE's Ethical Requirements page:

- **Any direct quotation** must be in quotation marks with a citation to the original source.
- **Any paraphrase or summary of someone else's work** must still be cited — paraphrasing without citation is plagiarism under IEEE's own definition (PSPB Operations Manual 8.2.1: "the use of someone else's prior ideas, processes, results, or words without explicit acknowledgment").
- This applies to **your own previously published work** too (self-plagiarism / improper "text recycling" is a separate, explicitly named policy — reusing your own prior text as if new requires disclosure and justification of what's actually new).
- **Data, results, tables, and figures reused from another source require citation**, same as text.
- **Fabrication** (inventing data or results) and **falsification** (manipulating or selectively omitting real data to misrepresent a result) are both named research-misconduct categories, independent of plagiarism.

Practical consequence for you: **nothing from this compendium or roadmap should be copy-pasted into your actual paper.** Both documents were built by paraphrasing search results under my own citation discipline, for the purpose of teaching *you* the material — they were never checked for verbatim-overlap risk against the original sources the way a submitted paper's prose needs to be. When you write the paper in week 9–10, write fresh prose from your own understanding and cite the primary sources directly (the bibliography in Part I, Section 13 and Appendix B of the compendium gives you the exact papers to go pull and cite properly) — don't paraphrase *my* paraphrase of them.

## 0.6 Prior publication and multiple submission

If you ever submit an early version to one venue (say, a workshop or a national-level venue) before the target IEEE venue, or submit to more than one place, IEEE requires **full disclosure of all prior publications and all concurrent submissions** covering substantially the same material. If you later want to extend a conference paper into a journal version, IEEE's own "evolutionary publishing" policy explicitly permits this — but only if both are properly peer-reviewed, the journal version contains **substantially more technical content**, and the journal paper **explicitly cites the earlier conference version and states how it differs**. Keep this in mind if your 10-week output ends up submitted first to something smaller before a T-BIOM/IJCB-level target — that's a completely normal, sanctioned path, as long as it's disclosed rather than silent.

## 0.7 Realistic venue targeting — the honest picture

You asked for "the most elite level." Let's be precise about what that means in *this specific subfield*, because the honest answer is more useful to you than an inflated one, and overclaiming venue prestige is its own form of the exact impropriety you're trying to avoid.

| Venue | Type | Fit for this work | Timeline reality (today is July 7, 2026) |
|---|---|---|---|
| **IEEE T-BIOM** (Transactions on Biometrics, Behavior, and Identity Science) | Journal, rolling submission, no fixed deadline | **Best realistic primary target.** Flagship journal of the IEEE Biometrics Council, directly on-topic for continuous behavioral authentication, adversarial robustness of biometric systems is squarely in scope. | Open now — no deadline pressure, submit when the work is genuinely ready |
| **IJCB** (International Joint Conference on Biometrics) | Conference, IEEE Biometrics Council-sponsored, flagship in-person venue for this exact field | Excellent topical fit; genuinely prestigious within biometrics specifically | **IJCB 2026's full-paper deadline (April 10, 2026) has already passed.** IJCB 2027 will be the next opportunity, deadline likely ~April 2027 — a realistic target if this becomes a longer project, not a 10-week one |
| **IEEE TIFS** (Transactions on Information Forensics and Security) | Journal, rolling | Very high prestige, broader security/forensics scope than T-BIOM; your work fits but competes against a wider, deeper pool | Open now, but a harder bar than T-BIOM for a first paper |
| **IEEE S&P / Euro S&P** | Top-tier general security conferences | This is genuinely "most elite level" in computer security broadly — and genuinely a very hard bar: these venues expect deep novel threat models often validated against real deployed systems, not primarily public-benchmark evaluations | Realistic as a *future* target after T-BIOM-level work establishes the result, not as a first submission on a 10-week timeline |
| **IEEE Access** | Mega-journal, faster turnaround, broad scope | Solid, legitimate, indexed IEEE venue — meaningfully less selective/prestigious than the above | Useful fallback if timeline pressure trumps prestige, not your primary target given your stated ambition |

**The honest recommendation:** target **IEEE T-BIOM** as the primary submission. It is a genuinely strong, on-topic, IEEE-flagship venue for exactly this research, it has no deadline forcing rushed work, and a well-executed poisoning-and-injection-attack-with-defenses paper is a realistic, defensible fit there for a strong undergraduate first paper. Treat IJCB 2027 as a natural next step if you want a conference presentation too. Don't target S&P-tier venues for this specific paper — not because the work can't be good, but because those venues' bar is calibrated to a different kind of contribution (deployed-system-scale novelty), and aiming there for a first submission raises real risk of the paper being rejected for scope reasons that have nothing to do with your rigor.

## 0.8 The AI-use log — start this today, in week 1

Create this file in your repo root on day one and update it every session, not retroactively:

```markdown
# AI_USE_LOG.md

## Format: [Date] | [Tool] | [Task] | [Level of use] | [What I verified/changed]

2026-07-08 | Claude | Research synthesis on continuous behavioral biometrics, IEEE policy research | Research + drafting of reference material | N/A — background research, not paper text
2026-07-08 | Claude | 10-week project roadmap structure | Planning assistance | I chose the final scope/contribution framing myself; adjusted week ordering after review

<!-- Continue this for every session, every week. This file becomes the
     backbone of your acknowledgments-section disclosure and your own
     record of exactly what was AI-assisted vs. self-directed, which is
     also just good practice for defending your thesis/viva if asked. -->
```

Levels of AI use worth distinguishing explicitly in each log entry, since "brief explanation of the level at which the AI system was used" is literally what IEEE's policy asks for: (1) background research/synthesis, (2) code scaffolding you then reviewed and modified, (3) code you used largely as-generated, (4) editorial/grammar pass on your own prose, (5) planning/structuring assistance. Being able to say "Section 4.2's algorithm was AI-scaffolded at level 3 and I verified it against [X]" is a categorically stronger position than a vague "AI was used throughout."


---

# PART 1 — THE CORE SCIENTIFIC CONTRIBUTION

## 1.1 Why "we implemented known techniques and evaluated them" gets rejected

Every individual piece of machine learning in the existing Clickstream research (Isolation Forest scoring, Siamese networks, federated learning) is well-established, published technique. A paper whose contribution is "we built a keystroke-authentication system using known methods and measured its EER on a known dataset" will be correctly rejected at any IEEE venue worth submitting to — that's a class project, not a contribution. Reviewers are specifically trained to ask "what's new here that wasn't already known before this paper?" You need a crisp, one-sentence answer to that question, and everything in the 10 weeks below is organized around earning one.

## 1.2 The thesis this project defends

**Continuously-adapting keystroke and mouse authentication systems face two structurally different adversarial threats that the existing literature studies in isolation — gradual baseline poisoning and one-shot injection spoofing — and this project is the first to evaluate both against a common modern system, propose a defense for each, and show how the threat model changes again under federated deployment.**

Unpack why each clause is defensible, not just asserted:

- **"Studies in isolation"** — verified directly during this research thread: Zibo Wang's dissertation develops the Frog-Boiling poisoning attack against classical keystroke authentication specifically, with no evaluation against mouse dynamics, no evaluation against a modern deep-sequence model, and no federated-learning angle. The 2022 screen-recording injection attack paper evaluates spoofing against KeyTrac specifically, with no poisoning angle and no proposed defense in the same paper. No single source found during verification combines both threats against one system.
- **"A common modern system"** — Week 2–3 builds both a classical (Isolation Forest/One-Class SVM) and deep-sequence (Siamese) baseline, so the paper's results aren't tied to one aging model family; this also lets you honestly report whether the deep model is more or less robust than the classical one, which is itself a genuine, citable finding either way.
- **"Propose a defense for each"** — Week 5 (residue-feature/RONI-style gate against poisoning) and Week 7 (a detection countermeasure against injection) are not just re-implementations of what's cited; Zibo Wang's dissertation *proposes* a residue-feature defense conceptually but this project independently implements and empirically validates a version of it against a system Zibo Wang never tested it on (deep sequence models, mouse dynamics) — that reimplementation-and-extension-with-new-evaluation is itself real, citable, defensible novelty, as long as it's framed honestly as an extension rather than a restatement.
- **"How the threat model changes under federated deployment"** — Week 8's stretch goal ties the poisoning attack to the Byzantine-robust-aggregation literature (Fang et al.) in a way that, per the verification work already done, no source specifically applies to continuous behavioral biometrics.

## 1.3 The two threat vectors, and why both are needed for the paper to be complete

A paper studying only the poisoning angle is a solid but narrower paper — reviewers will reasonably ask "what about attacks that don't rely on controlling the retraining loop at all?" A paper studying only the injection angle faces the same problem in reverse — "what about an attacker who's already inside the adaptation loop?" Covering both closes that obvious reviewer question before it's asked, and the two threats are genuinely complementary in the system-security sense: poisoning is a **slow, insider-style attack on the model's memory over time**; injection is a **fast, one-shot attack on the model's input at a single moment**. A complete adversarial-robustness paper for this system class needs both, and having both from week 4 onward is what turns this from "an attack paper" or "a defense paper" into a genuine systems-security contribution.

## 1.4 Explicit scope boundaries — say what you are NOT claiming

This is precisely the kind of thing that protects you from later being told you overclaimed. Write these into the paper's limitations section directly, don't bury them:

- **This is not a claim of production-grade fraud-loss reduction for a real bank.** Every quantitative result comes from public academic benchmark datasets (CMU, HMOG, Balabit, etc.), not real banking telemetry — Section 8.6 of the compendium already establishes this discipline; carry it into the paper verbatim in spirit.
- **This is not a claim that the proposed defenses are unbreakable.** Section 8.4's "state of the art, not solved" framing for Byzantine-robust aggregation applies to your own proposed defenses too — report their *measured* improvement, not a claim of general robustness beyond what you tested.
- **This is not a claim of a fundamentally new attack class.** Both threats are adaptations/extensions of published attacks (Zibo Wang; the 2022 screen-recording paper), applied and evaluated in a new combination and against new model families — say exactly that, don't imply either attack is invented from scratch.
- **The federated-learning result (Week 8) is explicitly a stretch/secondary contribution**, scoped and labeled as such, so that if time runs short it can be cut or reduced to a smaller-scale demonstration without invalidating the paper's primary claims from Weeks 4–7.

## 1.5 The related-work gap, stated as you'll eventually state it in the paper

A rough draft of the sentence that will anchor your introduction and related-work section, to keep the whole 10 weeks pointed at the same target: *"Prior work has independently demonstrated gradual poisoning attacks against classical keystroke authentication [Zibo Wang, 2020] and injection-based spoofing via screen-recorded video against commercial keystroke authentication [2022 Journal of Big Data paper], and separately studied Byzantine-robust aggregation for federated learning in general settings [Fang et al., 2020] and biometric template poisoning in facial-recognition systems [Lovisotto et al., 2020]. No prior work evaluates both threat classes against a common continuous keystroke-and-mouse authentication system spanning classical and deep-sequence model families, proposes and validates defenses for each, or examines how the threat model changes under federated deployment — which is the gap this paper addresses."*

---

# PART 2 — ENVIRONMENT STRATEGY

## 2.1 The general rule

Your MacBook Air M3 (16GB RAM, 10-core GPU, Apple Silicon) does **not** have an NVIDIA GPU, so it cannot run CUDA — the `torch.cuda.is_available()` check will always return `False` locally, and any RAPIDS/CUDA-specific tooling from the compendium's Part III simply does not apply to your local machine at all. What it *does* have is PyTorch's **MPS backend** (Metal Performance Shaders), which gives you real GPU acceleration for small-to-medium models directly on the M3's own GPU — meaningfully faster than CPU-only, though slower than a dedicated NVIDIA T4 for the same job. Given that every dataset in this project (CMU: ~20,400 rows; Balabit: modest session recordings; HMOG: 100 subjects) is small by deep-learning standards — nothing here is ImageNet-scale — the practical rule is:

- **Classical ML (Isolation Forest, One-Class SVM) never needs GPU at all**, on the M3 or anywhere else. These train in seconds to low minutes on CPU regardless of environment.
- **The deep sequence models (Siamese/Transformer) train fine via MPS locally for development and debugging**, but a Colab T4 session will iterate noticeably faster once you're doing real hyperparameter search or multiple full training runs, and Colab's free T4 access removes any risk of your own machine's battery/thermal limits becoming a bottleneck during a long run.
- **Federated-learning simulation (Week 8) multiplies compute by the number of simulated clients** — even a "small" 20-client simulation, each training a copy of the deep model for a few local epochs per round, adds up fast, and this is where local MPS starts to feel genuinely slow. This is the other clear Colab/Kaggle week.

## 2.2 Week-by-week environment map

| Week | Core task | Environment | Rationale |
|---|---|---|---|
| 1 | Data acquisition, feature extraction, ethics docs | **Local (IDE)** | No model training at all |
| 2 | Classical baseline (Isolation Forest / One-Class SVM) + eval harness | **Local (IDE)** | Trains in seconds/minutes on CPU; no benefit from GPU |
| 3 | Deep sequence model (Siamese network) baseline | **Colab T4 (primary)**, local MPS for quick debugging | GPU meaningfully speeds up iteration on the training loop |
| 4 | Poisoning attack implementation + degradation measurement | **Local (IDE)** for the classical-model attack; dip to Colab only if also attacking the Week 3 deep model | Attack logic itself is lightweight; only re-scoring against the deep model needs GPU |
| 5 | Poisoning defense (residue-feature/RONI gate) + evaluation | **Local (IDE)**, same caveat as Week 4 | Defense logic is a validation check, not a training job |
| 6 | Injection attack (screen-recording demo) | **Local (IDE)** | OpenCV video processing is CPU-bound, no GPU needed |
| 7 | Injection defense/detection | **Local (IDE)** | Detection-side statistical logic |
| 8 | Federated-learning stretch extension | **Colab or Kaggle** | Multi-client simulation compute adds up quickly |
| 9 | Results consolidation, figures, paper scaffold | **Local (IDE)** | Writing and plotting, no training |
| 10 | Red-team review, compliance pass, submission prep | **Local (IDE)** | No coding beyond a final reproducibility check |

**Answering your question directly: 8 of the 10 weeks are pure local-IDE work.** Only Week 3 and Week 8 clearly need Colab/Kaggle, and even those have a local fallback via MPS if a GPU session isn't available when you need it — just expect it to run slower, not to fail outright. Kaggle is a reasonable alternate to Colab for either week (also free T4/P100 access, slightly more generous weekly GPU-hour quota than Colab's free tier as of recent history — worth checking current quotas when you get there, since both platforms adjust these periodically). Your friend's GPU laptop as a last resort makes sense only if both Colab and Kaggle quotas are exhausted in the same week — which, given the model sizes here, should be avoidable if you're disciplined about not leaving idle sessions running.


---

# PART 3 — THE 10-WEEK PLAN (overview level)

Each week below has its own dedicated `weekN.md` with full code-level detail — this section gives you the shape of each week, the verification bar it has to clear before you move on, and exactly what to send back to me for review. Do not start week N+1 until week N's verification checklist passes; a broken foundation compounds silently and is far more expensive to find in week 8 than in week 2.

## Week 1 — Foundation: Data, Features, Threat Model, Ethics
**Objective:** a validated data pipeline and a formal, written research framing, before any modeling starts.
**Core coding task:** acquire and validate the CMU keystroke dataset (and begin the Balabit mouse-dynamics acquisition), build and unit-test the feature-extraction module, stand up the repo skeleton.
**Non-coding deliverables:** the threat-model/system-design document (this is your "theoretical grounding," written down, reviewable), the AI-use log (Section 0.8), the ethics statement draft (Section 0.4).
**Verification:** dataset row/subject counts match the published dataset description exactly; feature distributions pass sanity checks against published statistics; all unit tests pass; the threat-model doc names both attacks, both defenses, and the explicit scope boundaries from Part 1.4.
**Environment:** Local.
**Full detail:** see the attached `week1.md`.

## Week 2 — Classical Continuous-Authentication Baseline
**Objective:** a working per-user Isolation Forest and One-Class SVM baseline with a rigorous, reusable evaluation harness.
**Core coding task:** enrollment/scoring pipeline per Section 5.2 of the compendium; EER/FAR/FRR/ROC-AUC/PR-AUC computation per Appendix D; a **temporal split** per subject — enroll on sessions 1–4 (200 reps), genuine-test on sessions 5–8 (200 reps) of the same subject, impostor-test on sessions 5–8 of every other subject — repeated independently for all 51 CMU subjects, with both pooled and per-subject EER reported.
**Verification:** EER should land in a plausible range relative to published classical-method benchmarks on this exact dataset (roughly high-single-digit to low-double-digit percent — if you get 0% or ~50%, that is a bug, not a result: 0% almost always means train/test leakage, ~50% almost always means the scores carry no signal). Run a negative control: shuffle the subject labels and confirm EER collapses to near-50% — if a shuffled-label run *also* gets a good EER, you have a leakage bug that must be fixed before Week 3 builds on top of it.
**Environment:** Local.

## Week 3 — Deep Sequence Model Baseline
**Objective:** a Siamese network (LSTM or small Transformer encoder) baseline trained and evaluated on the identical harness from Week 2, so the two model families are directly comparable.
**Core coding task:** implement the encoder + triplet-loss training loop (extending the sketch in compendium Section 5.3); train on Colab T4.
**Verification:** training/validation loss curves must monotonically trend downward without NaNs or divergence; embedding-space sanity check (same-subject pairs should cluster closer than different-subject pairs, verifiable by plotting a t-SNE or simply comparing average intra-class vs inter-class embedding distance); report EER on the same test split as Week 2 and be honest in your own notes about whether it beats the classical baseline — either outcome is a legitimate, reportable finding.
**Environment:** Colab T4 (primary), local MPS acceptable for short debugging runs.

## Week 4 — Poisoning Attack Implementation
**Objective:** a working, controllable Frog-Boiling-style gradual poisoning attack against the baseline-adaptation mechanism.
**Core coding task:** implement the "absorb low-risk anomalies into the baseline" adaptation loop from compendium Sections 6.7/7.5 first (as the thing being attacked), then the adversarial update-crafter that gradually shifts the baseline toward attacker behavior over a controlled number of adaptation cycles.
**Verification:** the attack must show measurable FAR/EER degradation over the adaptation cycles *relative to a no-attack control run on the same baseline*, and — critically — you must also run the **benign-drift control** (simulate an ordinary behavioral change, not an attack) and confirm the defended metric shift is attack-specific, not just "any drift breaks the system." Without this control, you cannot claim the attack is doing something specific; you can only claim the system is fragile in general, which is a much weaker and less interesting result. **Note from Week 1/2:** the CMU dataset itself was confirmed to contain genuine heavy-tailed outliers (a 2.035s hold time, DD values up to 25.99s, both real subject behavior, not corruption). The benign-drift control needs to specifically include this kind of realistic outlier-like variation, not just smooth typical drift — a defense that simply flags "any statistical outlier" as suspicious isn't actually distinguishing poisoning from the natural variability Week 1 already proved exists in real typing behavior, and reviewers will ask exactly this question if the paper doesn't address it directly.
**Environment:** Local for the classical-model attack; Colab if also attacking the Week 3 deep model.

## Week 5 — Poisoning Defense (Residue-Feature / RONI-Style Gate)
**Objective:** implement and empirically validate a validation gate that catches Week 4's attack before it corrupts the baseline, while still allowing the benign-drift control through.
**Core coding task:** residue-feature correlation check (per Zibo Wang's dissertation framing) or a RONI-style leave-one-out impact test, inserted as a gate before any candidate update is accepted into the baseline.
**Verification:** this is your first core paper result — you need a clean before/after comparison table (undefended vs. defended FAR/EER trajectory under the Week 4 attack) plus a statistical significance test (a paired test across multiple subjects/runs, not a single anecdotal run), and confirmation that the benign-drift control still gets absorbed normally under the defended system (a defense that also blocks legitimate drift isn't a usable defense, it's just a stricter false-positive machine).
**Environment:** Local, with an optional Colab pass if validating against the deep model too.

## Week 6 — Injection Attack (Screen-Recording Demo)
**Objective:** reproduce the video-based keystroke-timing-extraction-and-replay attack against your own trained detector.
**Core coding task:** OpenCV-based frame extraction and keypress-visual-cue timing reconstruction (extending compendium Section 22's sketch), replayed against the Week 2/3 trained models.
**Verification:** report the actual measured evasion rate on your own data and model, and discuss honestly how it compares to the literature's 64% figure (compendium Section 8.5) and why it might differ (different detector, different recording setup, different subject) — a different number than the literature is not a failure, an unexplained different number is.
**Environment:** Local. This is the human-subjects week — make sure Section 0.4's self-consent statement and any departmental check are actually done before recording anything, not after.

## Week 7 — Injection Defense / Detection
**Objective:** propose and validate a countermeasure that reduces the Week 6 attack's evasion rate.
**Core coding task:** a secondary liveness-style signal that's much harder to reconstruct from screen-recorded video than keystroke timing is (mouse micro-jitter is a strong candidate, since it requires sub-pixel visual precision to extract from video that keystroke timing does not), or a statistical "too-clean"/synthetic-timing detector on the injected sequence itself.
**Verification:** measured evasion-rate reduction with the defense active vs. Week 6's undefended baseline, reported with the same rigor (multiple trials, not one anecdote) as Week 5.
**Environment:** Local.

## Week 8 — Federated Learning Extension (Stretch Goal)
**Objective:** show that the poisoning threat model changes under federated deployment, and that Byzantine-robust aggregation (Krum/trimmed mean) mitigates it where naive FedAvg doesn't.
**Core coding task:** Flower-based simulation (compendium Section 24) with N simulated clients, a subset acting maliciously by sending Week-4-style poisoned updates at the client level rather than the centralized-retraining level; compare FedAvg vs. Krum vs. trimmed-mean aggregation under identical attack conditions.
**Verification:** FedAvg should show clear degradation under the simulated attack; Krum/trimmed-mean should show measurably less degradation under the *same* attack — this comparison, not either number alone, is the result.
**Environment:** Colab or Kaggle. **This week is explicitly a stretch goal** (Part 1.4) — if time is short by week 8, a smaller-scale version (fewer clients, one aggregation rule instead of three) that still shows the core comparison is an acceptable reduction in scope; dropping it entirely is a legitimate option that does not invalidate Weeks 1–7's contribution.

## Week 9 — Results Consolidation and Paper Scaffold
**Objective:** every figure and table in the eventual paper generated from real, saved experimental outputs by a single reproducible script — not assembled by hand from memory.
**Core coding task:** a `reproduce_all_results.py`-style script that regenerates every figure/table from saved run artifacts in one command; the IEEE conference/journal LaTeX template populated with a full structural draft (introduction, related work using Part 1.5's framing, methodology, results, limitations per Part 1.4, references).
**Verification:** a fresh clone of the repo plus one command reproduces every figure/table byte-for-byte (or numerically identically, allowing for documented random-seed variation) — this is also exactly the kind of reproducibility rigor that strengthens a paper's review outcome independent of the results themselves.
**Environment:** Local.

## Week 10 — Red-Team Review, Compliance Pass, Submission Prep
**Objective:** find your own paper's weaknesses before a reviewer does.
**Core coding task:** none required beyond fixing anything the review below turns up.
**Verification:** the full checklist in Part 4, run against the actual draft, item by item.
**Environment:** Local.

---

# PART 4 — FINAL SUBMISSION CHECKLIST

Run this against the actual draft in Week 10, not from memory:

- [ ] Every claim in the abstract and conclusion is directly traceable to a specific figure/table in the results section — no claim exists that isn't backed by a shown number.
- [ ] The limitations section explicitly states all four scope boundaries from Part 1.4 (academic datasets not banking data; defenses not claimed unbreakable; attacks are extensions not new classes; FL result scoped as secondary if included).
- [ ] Every dataset used is cited to its original source paper, with an explicit one-line statement that it is a pre-existing, previously-ethics-approved public academic corpus.
- [ ] The Week 6 self-recorded data has its consent/ethics statement in the paper, and you've confirmed whether KU requires anything further.
- [ ] The acknowledgments section discloses every AI tool used, which sections it touched, and at what level (Section 0.3), built from your Section 0.8 log — not reconstructed from memory.
- [ ] Every direct quote (if any) is in quotation marks with a citation; every paraphrase is cited; nothing is copied from this compendium/roadmap's own prose into the paper.
- [ ] No prior submission of substantially the same material exists undisclosed; if one does, it's disclosed per Section 0.6.
- [ ] A plagiarism self-check has been run (a free tool, or your institution's Turnitin access if available) before submission, not after a rejection.
- [ ] The reproducibility script from Week 9 has been re-run fresh, once, right before submission, and matches the draft's numbers exactly.
- [ ] The target venue (Section 0.7 — T-BIOM recommended) and its specific author-guidelines page have been checked directly for any formatting/length/template requirement, since these are checked before review and cause desk rejections independent of content quality.