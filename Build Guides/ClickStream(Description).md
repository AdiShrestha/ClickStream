# Clickstream Research Compendium
## Continuous Behavioral Biometrics, Adversarial Robustness, and Open-Source Dependency Reference

**Compiled for:** Aditya (Adi), Kathmandu University, Department of Computer Science and Engineering
**Project context:** Clickstream — a standalone research and engineering project on continuous behavioral biometrics as a fraud-defense technology: authenticating a banking session throughout its entire duration via keystroke, mouse, and touch dynamics, rather than only at login
**Document type:** Consolidated research reference, verification audit trail, and dependency/licensing manifest
**Status:** All primary claims in Parts I and II were checked against web sources during this research thread; Part III's license claims were checked against each project's own license file, PyPI/GitHub listing, or vendor EULA. Where a claim could not be cleanly verified, this is stated explicitly rather than smoothed over.

---

## How to use this document

This compendium consolidates three rounds of research conducted in this thread into one reference for Clickstream:

1. **Part I** is the full technical deep-dive into continuous behavioral biometrics as a fraud-defense technology — what it is, the threat it defends against, the machine learning underneath it, how it's actually built, who sells it, what regulations shape it, and where the open research problems sit.
2. **Part II** is the verification audit trail — a second AI's ("Gemini's") proposed additions to Part I, independently fact-checked claim-by-claim against primary sources, with corrections applied where the verification itself needed correcting.
3. **Part III** is a complete open-source licensing reference for every library and platform component relevant to building Clickstream, with each license individually confirmed and every "free but not open source" case flagged explicitly.

Appendices hold a glossary, the full reference list, a dataset comparison table, and a metrics reference.

Sections are numbered continuously (1–29) so cross-references work regardless of which Part they fall in.

---

## Table of Contents

**Part I — Continuous Behavioral Biometrics: Complete Technical Reference**
1. Executive Summary
2. Conceptual Foundations
3. The Threat Landscape
4. Signal Taxonomy
5. Machine Learning Approaches
6. Reference Architecture
7. Implementation Strategy
8. The Adversarial Arms Race
9. Standards, Regulation, and Privacy
10. Industry and Market Landscape
11. Ongoing Academic Research Directions
12. Open Problems and Honest Limitations
13. Bibliography for Part I

**Part II — Verification and Fact-Check Audit Trail**
14. Verification Methodology
15. Item-by-Item Verification Findings
16. Final Revised Document Text
17. Lessons on Citation Rigor

**Part III — Dependency and Licensing Reference**
18. Scope and Methodology
19. Core Data and ML Pipeline
20. GPU Acceleration Layer
21. Notebook and Cloud Environment
22. Behavioral Biometrics — Deep Learning and Vision
23. Streaming Architecture
24. Federated Learning
25. Differential Privacy
26. Experiment Tracking
27. Testing and Development Tooling
28. Licensing Compliance Notes
29. Full Environment Setup Guide

**Appendices**
A. Glossary of Terms
B. Complete Bibliography and Reference List
C. Dataset Comparison Table
D. Metrics Reference

---

# PART I — CONTINUOUS BEHAVIORAL BIOMETRICS: COMPLETE TECHNICAL REFERENCE

## 1. Executive Summary

Continuous behavioral biometrics is a fraud-defense technology that authenticates a user throughout an entire digital banking session, rather than only once at login. It works by passively measuring *how* a person interacts with a device — typing rhythm, cursor movement, touchscreen pressure, phone-holding angle — and comparing that behavior in real time against a personal baseline built from that same person's prior sessions. When the live behavior diverges sharply from the baseline, the system lowers a trust score and can trigger a step-up challenge or silently restrict what the session is allowed to do, without ever asking the legitimate user to do anything extra.

The reason this exists is specific and well documented: modern phishing frameworks such as Evilginx, Muraena, and Modlishka do not try to steal a password. They sit as a reverse proxy between the victim and the real bank, let the victim complete a normal login including any SMS or authenticator-app multi-factor step, and then steal the resulting session cookie once the bank has already issued it. That cookie is mathematically valid, so injecting it into an attacker's own browser bypasses every layer of MFA the bank built. This is called an Adversary-in-the-Middle (AitM) attack, and it is explicitly named by the U.S. Federal Reserve as the dominant fraud vector displacing simple password reuse in 2026. Continuous behavioral biometrics is the direct countermeasure: it doesn't try to stop the cookie from being stolen, it notices that the *human holding the cookie* has changed.

This document covers, in order: what the technology precisely is and how it differs from physiological biometrics and point-in-time MFA (Section 2); the fraud statistics and attack mechanics that justify bank investment in it (Section 3); every behavioral signal modality actually used in production and research, with benchmark accuracy figures (Section 4); the machine learning techniques underneath it, from classical Isolation Forest and One-Class SVM through Siamese networks, Transformers, and federated learning (Section 5); a full reference architecture for building such a system end to end, including a working code example (Section 6); a phased implementation roadmap (Section 7); the adversarial research trying to defeat these systems — bot humanization, poisoning attacks on continuously-retrained baselines, and synthetic-data injection attacks — verified independently rather than taken on faith (Section 8); the regulatory constraints (PSD2, GDPR, NIST, and Nepal's own banking cyber-resilience framework) that shape what a compliant system actually looks like (Section 9); the commercial vendor landscape and market size (Section 10); where the live academic research frontier sits (Section 11); and an honest accounting of what remains unsolved (Section 12).

Part II documents a full verification pass on a set of claims proposed by a second AI reviewer (Gemini) as additions to this research — because a well-organized, confident-sounding writeup is not the same thing as an accurate one, and every specific citation was checked rather than assumed correct. Part III is a complete dependency and open-source licensing audit for the actual software stack this project would require, including three components that are free to use but are **not** open source, flagged explicitly because that distinction was a hard requirement.

## 2. Conceptual Foundations

### 2.1 The core distinction: behavioral versus physiological biometrics

**Physiological biometrics** — fingerprint, face, iris — measure a static physical characteristic. They are captured once, matched once, and the match either succeeds or fails at that single moment. Critically, they are also immutable: if a fingerprint template is compromised in a data breach, the person cannot get a new finger. This is a structural weakness that has driven real caution in how physiological biometric templates are stored and regulated (see Section 9.2 on GDPR Article 9).

**Behavioral biometrics** measure *how* a person does something — typing cadence, cursor trajectory, swipe pressure, gait — rather than a fixed physical trait. They are inherently probabilistic and time-varying rather than binary-match, which has two consequences that shape everything downstream: first, a behavioral baseline can and should drift over time (a person's typing rhythm two years from now will not be identical to today, and the system needs to tolerate that); second, because there is no single immutable "correct" value, behavioral templates are commonly argued to sit outside the strictest legal protections that apply to physiological biometric data used for unique identification (a genuinely contested point of law explored fully in Section 9.2).

### 2.2 Continuous versus point-in-time authentication

Traditional authentication — a password, an OTP, a fingerprint scan — is a **point-in-time check**: identity is verified once, at login, and then implicitly trusted for the rest of the session. This is precisely the assumption that AitM phishing exploits. Once the attacker has the session cookie, the point-in-time check is irrelevant; the session is already authenticated and will remain so until it expires or the user logs out.

**Continuous authentication** inverts this assumption. It treats identity not as a single yes/no gate but as a running hypothesis that gets re-evaluated on a rolling basis for the entire duration of the session. Every few seconds (or every N keystrokes, or every mouse gesture), the system asks "does the behavior I am seeing right now still look like the same person who logged in?" This is the specific architectural property that lets it catch a stolen-cookie session: the cookie is valid, the IP might even look plausible, but the *human* behind the keyboard has changed, and that shows up in the behavioral stream within seconds to minutes, not at the next login (which may never come, since the attacker has no need to log in again — they already have a live, valid session).

### 2.3 Verification (1:1) versus identification (1:N)

This distinction matters for both the machine learning design and the legal classification of the system, so it is worth being precise about it early.

- **Verification (1:1):** "Does this session's behavior match *this specific, already-enrolled* user's personal baseline?" The system already knows who is supposed to be logged in (from the session token) and is only checking whether the behavior is consistent with that one person's history.
- **Identification (1:N):** "Whose baseline, out of the entire user population, does this behavior match?" This is a much harder, more computationally expensive, and more privacy-sensitive problem, because it requires comparing incoming behavior against everyone's template, not just one.

Nearly every production banking deployment of continuous behavioral biometrics is verification, not identification, for three converging reasons: it is computationally cheaper (one comparison instead of N), it is more accurate (matching against one well-characterized baseline instead of searching for a best match among millions), and — critically — it is more defensible under data protection law, since regulators have drawn an explicit line between biometric data used to verify a claimed identity versus biometric data used to search for or establish an identity from scratch (see Section 9.2).

### 2.4 Passive/implicit versus active/explicit collection

The entire value proposition of continuous behavioral biometrics rests on it requiring **zero deliberate action** from the legitimate user. This is not a minor UX nicety; it is close to the whole point. A system that asked users to periodically re-type a phrase or re-scan a fingerprint mid-session would reintroduce exactly the friction that point-in-time MFA already causes (and that drives some of the "MFA fatigue" attacks discussed in Section 3), while also giving an attacker a discrete moment to specifically attack rather than a continuous stream that is much harder to game moment-to-moment.

Survey literature covering more than 140 distinct behavioral biometric approaches across six modality groups reports that roughly 90% of users say they prefer this kind of passive verification over explicit credential prompts — an important data point because user acceptance, not just technical accuracy, determines whether banks can deploy a technology like this without a customer backlash.

### 2.5 A worked walkthrough: what actually happens during a session

To make Sections 2.1–2.4 concrete, here is what continuous behavioral biometrics is doing, moment to moment, during an ordinary banking session:

1. **T+0s (login):** User authenticates normally — password plus SMS OTP, say. A trust score initializes, typically seeded high if the device/IP/session context all look familiar (returning device, known location, normal time of day).
2. **T+0s to T+30s (early session):** As the user navigates to their account summary, the client-side listener is already capturing mouse-movement and click-timing telemetry. No behavioral judgment is made yet — there isn't enough data in this short a window.
3. **T+30s to T+2min (steady state):** The user starts a funds transfer. Every keystroke in the recipient/amount fields generates dwell-time and flight-time features (Section 4.2); every field transition generates mouse-trajectory features (Section 4.3). These get aggregated into a rolling window (Section 6.3) and scored against the enrolled baseline (Section 5.1–5.2), producing a continuously updating trust score.
4. **Scenario A — legitimate user:** The behavioral features stay within the statistical envelope of the baseline. Trust score stays high. The transfer proceeds without any additional friction.
5. **Scenario B — stolen session cookie, attacker at the keyboard:** The attacker's typing rhythm, mouse trajectory, and interaction pace are statistically distinct from the enrolled baseline — even if the attacker knows the victim's account details perfectly, because *knowing the details* and *physically interacting like the victim* are unrelated skills. The trust score drops sharply within the observation window. The risk orchestrator (Section 6.5) fuses this drop with device/context signals; if the device fingerprint or IP also looks unfamiliar, the system escalates immediately to a step-up challenge (Section 6.6) — typically a FIDO2/passkey re-verification — which the attacker, lacking the victim's physical authenticator, cannot pass.
6. **Scenario C — legitimate user, but with a broken wrist:** This is the hard case, covered in depth in Section 7.3. The behavioral deviation looks similar in *magnitude* to Scenario B, but the underlying cause is entirely benign. The system's job is to tell these two scenarios apart using auxiliary context (a trusted, unchanged device and IP versus a suddenly unfamiliar one) rather than behavioral deviation alone — this is, empirically, the single hardest engineering problem in the entire field, and it is what makes "just block anything anomalous" an unworkable naive design.


## 3. The Threat Landscape

### 3.1 How an Adversary-in-the-Middle (AitM) attack actually works, step by step

It is worth walking through the mechanics slowly, because the entire justification for continuous behavioral biometrics rests on understanding exactly what point-in-time MFA fails to catch.

**Step 1 — The lure.** The victim receives a phishing email, SMS, or social-media message with a link. Unlike a decade-old phishing page that simply mimics the bank's login page with static HTML, the link points to infrastructure running a reverse-proxy phishing framework — Evilginx, Muraena, and Modlishka are the three most commonly referenced examples in security research. These tools do not build a fake copy of the bank's site; they transparently proxy the *real* site.

**Step 2 — The proxy.** When the victim's browser requests the login page, the phishing framework fetches the actual page from the bank's real servers and relays it back to the victim, rewriting links so that all subsequent traffic continues to route through the attacker's proxy. Visually and functionally, the victim is looking at the genuine bank website — because they are, just through a man-in-the-middle relay.

**Step 3 — Credential capture, transparently.** The victim types their real username and password into what is, from the browser's perspective, a normal form submission. The proxy captures these credentials in transit and forwards them to the real bank.

**Step 4 — MFA capture, transparently.** This is the step that defeats traditional MFA entirely. The bank, believing it is talking to the real user, issues an SMS code or push notification exactly as it would for a legitimate login. The victim receives this on their own phone (because it *is* their own phone — this isn't SIM-swap fraud, it's a completely separate attack class), types the code into the proxied page, and the proxy relays it through to the bank in real time. As far as the bank's authentication system is concerned, a legitimate user, in possession of the correct password and the correct second factor, has just logged in successfully.

**Step 5 — The theft.** The bank, satisfied that authentication succeeded, issues a session cookie/token to what it believes is the victim's browser. That cookie transits through the attacker's proxy on its way back to the victim — and the proxy keeps a copy. The victim is now transparently redirected to the real, authenticated bank session and sees their account exactly as expected, suspecting nothing.

**Step 6 — The reuse.** The attacker now takes the stolen session cookie and injects it directly into their own browser, bypassing the login page entirely. Because the cookie is cryptographically identical to what the bank issued, the bank's session-validation logic has no way to distinguish the attacker's browser from the victim's. The attacker is now inside the account with a fully authenticated session, having never needed to know or guess a password independently, and having successfully passed MFA by simply relaying the victim's own real MFA response.

This is why security researchers describe MFA as "entirely bypassed" by AitM rather than merely weakened: the attack does not try to defeat multi-factor authentication, it lets the legitimate user complete MFA correctly and then steals the resulting proof-of-authentication. No password-strength policy, no additional MFA factor, and no CAPTCHA addresses this, because none of those defenses operate on the axis the attack actually uses (session-token theft after successful authentication, not authentication itself).

### 3.2 Why continuous behavioral biometrics is the correct countermeasure, specifically

Given the mechanics above, the defense has to operate on a different axis than authentication strength: it has to detect that *the entity now using the valid, stolen session* is not the same human who completed the original login. This is precisely a continuous, in-session, identity-consistency problem — not a stronger-login problem — which is why point-in-time technologies (better passwords, hardware security keys used only at login, additional OTP factors) do not close this gap on their own, while continuous behavioral biometrics does, by design.

### 3.3 The fraud numbers that justify the investment

| Metric | Figure | Notes |
|---|---|---|
| US account takeover (ATO) fraud losses, 2024 | Over $15.6 billion | Up from $12.7 billion in 2023 — a year-over-year increase of more than 36% |
| Reports of ATO fraud, YoY change | +36%+ | Federal Reserve fraud-mitigation data |
| SIM-swap fraud growth, 2024 | +1055% | A related but mechanically distinct account-takeover vector (attacker convinces or bribes a carrier to port the victim's number) |
| Credential-stuffing attempts | ~26 billion attempts/month | Akamai measurement, automated login attempts using breached credential lists |
| Organizations targeted for ATO in 2024 | 99% of monitored organizations | Nearly universal exposure across the monitored population |
| Organizations with at least one successful ATO incident | 62% | The gap between "targeted" (99%) and "successfully breached" (62%) is itself evidence that layered defenses, including behavioral biometrics, are meaningfully reducing — not eliminating — successful attacks |
| US banks reportedly already using behavioral AI biometrics | ~65% | This is no longer an emerging or niche technology in the US banking sector; it is closer to a baseline expectation |
| Behavioral biometrics market size, 2025 | $2.72 billion | |
| Behavioral biometrics market size, 2026 | $3.45 billion | |
| Behavioral biometrics market size, 2031 (forecast) | $11.38 billion | Implies a 26.95% compound annual growth rate over the forecast window |

### 3.4 Why traditional bot detection is losing ground

A specific and important detail from Federal Reserve fraud-mitigation guidance: scripted, automated attacks increasingly emulate human behavior directly — mouse movements, typing patterns, and browsing behaviors are deliberately synthesized to look human, specifically to defeat the previous generation of bot-detection heuristics (straight-line mouse movement, uniform keystroke timing, and similar naive signatures). This is the exact reason the field has moved from rule-based bot detection to statistical/ML-based behavioral modeling: the naive version of "check if the behavior looks robotic" stopped being sufficient once attackers started deliberately engineering non-robotic-looking behavior. Section 8.2 covers the specific, well-documented techniques (Bézier-curve mouse paths, Fitts's-Law-derived timing models) that make this arms race concrete rather than abstract.

## 4. Signal Taxonomy

This section catalogs every behavioral modality with production or research deployment, the specific features extracted from each, and the best published accuracy figures, measured in **Equal Error Rate (EER)** — the operating point where the false-acceptance rate equals the false-rejection rate; lower is better. A full explanation of EER, FAR, FRR, and related metrics with worked numeric examples appears in Appendix D; the short version needed here is that an EER of 5% means that, at the system's most balanced operating threshold, about 1 in 20 genuine sessions would be wrongly flagged and about 1 in 20 impostor sessions would be wrongly accepted, if that single threshold were used in isolation (in practice, systems are tuned away from the EER point toward far lower false-accept rates, accepting a higher false-reject rate, because a wrongly-blocked genuine user is a customer-service cost while a wrongly-accepted impostor is a fraud loss — the two error types are not weighted equally in a real deployment).

### 4.1 Full modality comparison table

| Modality | Signals captured | Best published EER | Representative research |
|---|---|---|---|
| Keystroke dynamics | Dwell time, flight time, digraph/trigraph latency, error-correction (backspace) patterns, typing rhythm | 3.33% (desktop), 3.61% (mobile) | KVC-onGoing challenge on the Aalto Desktop/Mobile Keystroke Databases (CodaLab-hosted, ongoing) |
| Mouse dynamics | Curvature, jerk (rate of acceleration change), velocity/acceleration profile, click latency, ~87 engineered features per session | Varies by dataset; no single canonical EER figure across the field | Empirical comparison of decision tree, k-NN, random forest, and CNN classifiers on ~87 extracted features |
| Touch/mobile (HMOG) | Hand micro-tremor, grasp pressure, device tilt (accelerometer/gyroscope/magnetometer), combined with tap timing | 7.16% (walking), 10.05% (sitting) | HMOG (hand movement, orientation, and grasp) feature set, evaluated on 100 subjects under both conditions |
| Device/session ensemble | Power consumption, gesture, touch, and movement fused together on-device | 6.1%–6.9% | Android-native ensemble study, 59 subjects |
| Navigation/app usage | Sequence of screens visited, dwell time per screen, typical transaction flow ordering | No standalone canonical EER — used almost exclusively as a secondary/fusion signal | — |

Notably, the HMOG feature set performs *better* while the subject is walking than while sitting — a counterintuitive result until you consider that walking introduces a second, highly individual biomechanical signal (gait-induced device movement) on top of the tap-timing signal, giving the classifier more discriminating information rather than more noise.

### 4.2 Keystroke dynamics in depth

**Dwell time** is the duration a single key is held down — the interval between a `keydown` event and the matching `keyup` event for the same key. **Flight time** is the interval between releasing one key and pressing the next — the gap between one key's `keyup` and the following key's `keydown`. These two numbers, computed for every keystroke in a typed sequence, are the foundational features of essentially all keystroke-dynamics research; everything more sophisticated (digraph latency, trigraph latency, typing rhythm profiles) is built by aggregating and contextualizing these two base measurements.

**Digraph latency** extends flight time to specific *pairs* of keys — the timing between typing "t" then "h" is measured separately from the timing between typing "h" then "e", because different finger-to-finger transitions have measurably different natural speeds for a given individual (adjacent-finger transitions are typically faster than same-finger transitions, for instance). **Trigraph latency** extends this to three-key sequences, capturing even more individually distinctive rhythm information at the cost of needing much more typing data to build a statistically reliable baseline for each specific trigraph.

**A worked numerical example.** Suppose a user types the string "hi" and the raw browser-captured event stream looks like this:

| Event | Key | Timestamp (ms) |
|---|---|---|
| keydown | h | 1000 |
| keyup | h | 1080 |
| keydown | i | 1150 |
| keyup | i | 1210 |

From this raw stream:
- Dwell time for "h" = 1080 − 1000 = **80 ms**
- Dwell time for "i" = 1210 − 1150 = **60 ms**
- Flight time for the "h→i" digraph = 1150 − 1080 = **70 ms**

A production system accumulates hundreds to thousands of these measurements per user during an enrollment period, builds a statistical distribution (mean, variance, and increasingly a full learned embedding rather than simple summary statistics — see Section 5.2) for each feature, and then scores live sessions against that distribution.

**Error-correction patterns** are a less commonly discussed but genuinely informative feature: how often a user backspaces, how far back they typically correct, and the timing around corrections are all measurably consistent per-individual and are increasingly incorporated as an additional feature dimension in modern keystroke-dynamics systems, though they are less standardized across the literature than dwell/flight time.

### 4.3 Mouse dynamics in depth

Where keystroke dynamics is fundamentally a timing problem, mouse dynamics is fundamentally a *trajectory* problem, and the feature engineering reflects that:

- **Curvature** — how much a cursor path bends versus moving in a straight line between two points. Genuine human mouse movement is measurably non-straight; naive bot scripts historically moved in perfectly straight lines, which was, for years, one of the simplest and most reliable bot-detection signals in existence (see Section 8.2 for how this signal has since been deliberately defeated by more sophisticated automation).
- **Jerk** — the rate of change of acceleration (the derivative of acceleration, or equivalently the third derivative of position). Human movement has a characteristic jerk profile shaped by the biomechanics of arm and wrist control; this is a harder signal for automation to fake convincingly than curvature alone, because it requires modeling not just a bent path but a specific, physiologically-plausible *rate* of directional and speed change along that path.
- **Click latency** — the interval between a cursor arriving at a target and the click being registered, and separately, the physical duration of the click itself (button-down to button-up).
- **Velocity/acceleration profile** — how speed varies across a single mouse movement, which for humans typically follows a characteristic acceleration-then-deceleration curve as the cursor approaches a target (this specific pattern is directly related to Fitts's Law, discussed again in Section 8.2 in the context of bot evasion, since automated tools now deliberately reproduce this exact profile to look human).

An empirical comparison extracting roughly 87 distinct features from raw mouse clickstream data compared decision trees, k-nearest-neighbors, random forest, and CNN classifiers as the downstream anomaly detector; CNNs consistently edged out the simpler classical classifiers when given the full 87-dimensional feature vector, though the gap narrows on smaller datasets where CNNs have less data to learn a useful representation from.

**An unusual but genuinely useful research testbed:** because gameplay naturally generates sustained, high-frequency mouse-movement data across a wide variety of movement types (precision aiming, rapid navigation, idle drift), researchers have used *Minecraft* specifically as a data-collection environment for continuous authentication research based on mouse dynamics — a reminder that behavioral-biometrics data collection does not require a banking context at all, just sustained, naturalistic device interaction.

### 4.4 Touch and mobile sensor dynamics in depth

Mobile continuous authentication draws on a richer sensor set than desktop keystroke/mouse dynamics, because a phone is held, not just operated:

- **Accelerometer** data captures the phone's linear motion — including the subtle, involuntary micro-tremor of a hand holding a device, which is measurably individual.
- **Gyroscope** data captures rotational orientation — the natural tilt angle at which a given person tends to hold their phone while typing or browsing, which tends to be habitual and consistent for a given individual across sessions.
- **Magnetometer** data (compass heading) is a less commonly emphasized third sensor but appears in the fuller HMOG feature set as an auxiliary orientation signal.
- **Touch pressure and contact area** — the surface area of a fingertip against the glass and the force behind a tap or swipe, both of which vary measurably by individual hand size, grip style, and habitual touch force.

The HMOG (Hand Movement, Orientation, and Grasp) feature set, evaluated on 100 subjects under both **sitting** and **walking** conditions, is the most commonly cited benchmark in this specific modality. As noted in 4.1, combining these sensor-derived features with tap timing achieves an EER as low as 7.16% while walking and 10.05% while sitting — the walking condition performing better precisely because gait introduces an additional, highly individual biomechanical signal layered on top of the base touch-timing signal, rather than merely adding noise.

### 4.5 Device and session ensemble signals

Rather than relying on a single modality, an Android-native study combined power consumption patterns, touch dynamics, gesture dynamics, and device-movement signals into a single ensemble classifier, evaluated across 59 users, achieving an EER of 6.1%–6.9%. Power-consumption pattern as a behavioral signal is a genuinely underappreciated modality: different individuals have measurably different usage-intensity patterns (screen brightness habits, app-switching frequency, background-process tolerance) that show up as a distinguishable power-draw signature over time, even though this is a much less direct "behavioral" signal than typing or movement.

### 4.6 Navigation and app-usage patterns

The sequence of screens a user visits, how long they dwell on each screen before proceeding, and the typical order in which they complete a multi-step transaction (for instance, always checking account balance before initiating a transfer, versus going straight to the transfer screen) form a lower-frequency but still measurably individual behavioral signal. This modality is rarely reported with a standalone EER figure in the literature — it functions almost exclusively as a secondary or fusion signal layered on top of a primary modality like keystroke or mouse dynamics, rather than as a standalone authentication mechanism, because the signal is much sparser (a handful of navigation events per session versus hundreds of keystrokes or thousands of mouse-movement samples).

### 4.7 Cross-device consistency — a genuine open challenge

A person's typing rhythm on a physical keyboard does not transfer cleanly to their typing rhythm on a phone's on-screen keyboard; the biomechanics, error-correction patterns, and even the achievable typing speed are fundamentally different across the two input modalities. This means a behavioral baseline built from desktop sessions provides limited direct predictive value for a mobile session by the same person, and vice versa. Section 11.4 covers this as a live, unresolved academic research direction rather than a solved engineering problem — production systems today generally maintain separate baselines per device class rather than attempting a unified cross-device behavioral model.


## 5. Machine Learning Approaches

### 5.1 Why this is fundamentally a one-class problem

Before covering specific algorithms, it is worth being explicit about the framing that shapes every modeling choice in this section. At enrollment, a bank has abundant examples of *one* class — the genuine user's own behavior — and essentially no labeled examples of the other class, because it does not have recordings of an attacker impersonating that specific user (and if it did, that data point would already represent a fraud incident, not a training example collected in advance). This is why the dominant classical approaches are **unsupervised** or **one-class** methods rather than standard binary classifiers: the system is not learning to distinguish "genuine" from "fraudulent," it is learning to model "normal for this specific person" and then flagging statistical distance from that model.

### 5.2 Classical unsupervised methods

**Isolation Forest.** An ensemble method that isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of that feature. Anomalies — points that differ from the bulk of the data — require fewer random splits to isolate than normal points do, because they tend to sit in sparser regions of the feature space. This gives Isolation Forest a natural anomaly score (average path length across the trees in the forest, inverted) without needing any labeled anomaly examples to train against, which is exactly why it fits the one-class framing in 5.1 so well.

Below is an expanded, more complete version of a per-user Isolation Forest pipeline than a bare sketch — showing feature extraction from a raw event stream through to a usable trust score, with inline explanation of each stage:

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def extract_keystroke_features(events):
    """
    events: list of (key, action, timestamp_ms) tuples, action in {'down','up'}
    Returns an (n_samples, 2) array of [dwell_time, flight_time] pairs.
    """
    features = []
    i = 0
    while i < len(events) - 1:
        key, action, down_t = events[i]
        if action != 'down':
            i += 1
            continue
        # find matching keyup for this key
        up_t = None
        for j in range(i + 1, len(events)):
            if events[j][0] == key and events[j][1] == 'up':
                up_t = events[j][2]
                break
        if up_t is None:
            i += 1
            continue
        dwell = up_t - down_t
        # flight time to the next keydown, if one exists
        flight = 0
        for k in range(j + 1, len(events)):
            if events[k][1] == 'down':
                flight = events[k][2] - up_t
                break
        features.append([dwell, flight])
        i = j + 1
    return np.array(features, dtype=float)


class PerUserBehavioralModel:
    """
    One instance of this class is trained per enrolled user, exactly
    matching the verification/1:1 framing in Section 2.3 -- this is
    never trained across users, only on one person's own history.
    """
    def __init__(self, contamination=0.05):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
        )
        self.is_fitted = False

    def enroll(self, historical_sessions):
        """
        historical_sessions: list of raw event-stream lists, one per
        historical session collected during the enrollment window
        (typically the first 1-2 weeks of active use).
        """
        all_features = np.vstack([
            extract_keystroke_features(session)
            for session in historical_sessions
            if len(session) > 2
        ])
        scaled = self.scaler.fit_transform(all_features)
        self.model.fit(scaled)
        self.is_fitted = True

    def score_live_session(self, live_events):
        """
        Returns a trust score in [0, 100]; higher is more consistent
        with the enrolled baseline.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been enrolled yet")
        live_features = extract_keystroke_features(live_events)
        if len(live_features) == 0:
            return 100.0  # not enough data yet to make a judgment
        scaled = self.scaler.transform(live_features)
        # decision_function: higher (more positive) = more normal
        raw_scores = self.model.decision_function(scaled)
        mean_score = raw_scores.mean()
        # squash to a 0-100 trust score via a logistic transform
        trust_score = 100.0 / (1.0 + np.exp(-mean_score * 5))
        return trust_score
```

This is deliberately still a simplified illustration relative to a production system — a real deployment would window this over rolling time periods rather than scoring one session in isolation, would fuse multiple modalities (Section 6.3), and would feed the resulting score into a risk orchestrator rather than acting on it directly (Section 6.5) — but the underlying shape (raw event stream to timing features to per-user unsupervised model to continuous score) is exactly what is running inside every vendor platform named in Section 10.

**One-Class SVM.** Learns a decision boundary that encloses the region of feature space occupied by the training data (the genuine user's behavior), then flags anything falling outside that learned boundary. Conceptually similar in purpose to Isolation Forest but geometrically different in approach — it fits a boundary rather than measuring isolation depth — and tends to be more sensitive to the choice of kernel and hyperparameters, making Isolation Forest the more common default choice in practice for this specific problem due to requiring less tuning per user.

### 5.3 Deep learning architectures

The last several years have seen deep learning substantially overtake classical methods on published benchmark accuracy, though classical methods remain common in production for cold-start and lower-data-volume scenarios where deep models have not yet seen enough of a given user's behavior to train reliably.

**TypeNet (Acien et al., 2021).** The foundational deep-learning keystroke-biometrics paper that most subsequent architectures build on or compare against. Established the viability of learned embeddings (rather than hand-engineered dwell/flight statistics alone) for keystroke-dynamics verification.

**Siamese networks with triplet loss.** A 2025 approach combines keystroke dynamics with a Siamese network architecture and interpolation-based data fusion specifically to standardize feature vector sizes across typing samples of different lengths (a real practical problem: two typing samples of the same password will have identical length, but free-text typing samples from the same user will naturally vary in length session to session). Below is a simplified architectural sketch of what a Siamese verification network looks like conceptually:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class KeystrokeEncoder(nn.Module):
    """
    Encodes a variable-length sequence of [dwell, flight] pairs into
    a fixed-length embedding vector.
    """
    def __init__(self, input_dim=2, hidden_dim=64, embedding_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # concatenate final forward and backward hidden states
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        embedding = F.normalize(self.fc(h_cat), p=2, dim=1)
        return embedding


def triplet_loss(anchor, positive, negative, margin=0.3):
    """
    anchor, positive: embeddings from the SAME user's two different sessions
    negative: embedding from a DIFFERENT user's session
    Pulls anchor and positive together, pushes anchor and negative apart.
    """
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    losses = F.relu(pos_dist - neg_dist + margin)
    return losses.mean()
```

At verification time, the live session's embedding is compared (via cosine or Euclidean distance) against the enrolled user's stored embedding centroid; a distance below a calibrated threshold means "consistent with baseline," exactly analogous in purpose to the Isolation Forest trust score in 5.2, just computed through a learned representation rather than hand-engineered statistics.

**Transformers replacing RNNs.** A 2025 paper proposes replacing the LSTM-style recurrent encoder above with a Transformer encoder using self-attention over the keystroke sequence, reporting that this surpasses traditional RNN-based approaches — following the same general trajectory NLP took several years earlier, where self-attention architectures displaced recurrent ones for sequence modeling once sufficient training data was available.

**CNN-BiLSTM hybrids.** A 2025 approach specifically targets multi-session, uncontrolled-setting keystroke data (i.e., real free-typing behavior collected across many natural sessions rather than a single controlled password-typing task) using a combined convolutional-then-recurrent architecture, aiming to capture both short-range local timing patterns (via the CNN layers) and longer-range sequential dependencies (via the BiLSTM layers).

**Image-encoding tricks — the Gabor Filter Matrix Transformation.** A striking and unusual approach converts numerical keystroke timing values into a 2D image representation via a Gabor filter matrix transformation, then applies standard CNN-based image classification techniques to that image rather than treating the timing data as a 1D sequence at all. This achieved an EER of 0.04545 (4.5%) on combined CMU and GREYC-NISLAB data spanning over 30,000 password samples — treating typing rhythm as a visual texture rather than a time series, and getting strong results doing so.

**Quantum machine learning.** An early-stage, exploratory research frontier applying quantum computing techniques to keystroke-dynamics classification has begun appearing in the literature. This should be read as a genuinely speculative research direction rather than anything with near-term production relevance — it is included here for completeness under "what research is ongoing," not as a recommendation.

### 5.4 Mouse-dynamics-specific modeling

As covered in Section 4.3, comparative studies extracting roughly 87 features from raw mouse clickstream data consistently find CNNs outperforming decision trees, k-nearest-neighbors, and random forest, though the margin narrows on smaller datasets. The Minecraft-based data-collection approach mentioned in 4.3 is worth reiterating here as a modeling note: because gameplay generates a much higher volume and greater diversity of natural mouse movement than typical banking-session telemetry would, it has proven useful as a pretraining or benchmark environment for mouse-dynamics models before fine-tuning on the sparser, more homogeneous movement patterns actually observed in a banking UI.

### 5.5 Federated learning — training without centralizing raw behavioral data

This is arguably the most active current research direction in the entire field (expanded further in Section 11.1), because it resolves a genuine tension the earlier sections gloss over: building an accurate population-level model benefits enormously from pooling behavioral data across many users, but GDPR and PSD2 (Section 9) make centralizing raw biometric-adjacent telemetry from many individuals a serious legal and reputational risk. Federated learning (FL) keeps each user's raw behavioral data physically on their own device and only ever transmits model *updates* (gradients or weight deltas) to a central server for aggregation — the raw dwell-time and mouse-trajectory data itself never leaves the device.

**The non-IID problem, explained concretely.** Standard federated learning (as originally designed for tasks like next-word-prediction keyboards) assumes each client has a locally representative sample of a shared underlying task. Continuous behavioral biometrics breaks this assumption in an unusually severe way: each client, by construction, only ever has genuine examples of *one specific person's* behavior, and no two clients' local data distributions are alike at all, since each is a different person's typing rhythm. This is a much harder non-IID (non-independent, identically distributed) setting than most federated learning research originally targeted, and it is the specific reason several of the architectures below build in additional machinery beyond vanilla federated averaging.

**FL-RBA² and the cold-start problem.** One proposed architecture addresses the cold-start problem (a brand-new user has no baseline yet — see Section 7.1) via clustering-based risk labeling, grouping new users provisionally with behaviorally similar existing users until enough of their own data has accumulated to train an individual model, while using differential privacy and message authentication codes to protect the resulting aggregate updates in transit.

**A minimal illustrative Flower-based federated setup**, showing the basic client/server shape (not the full FL-RBA² clustering logic, which is considerably more involved, but enough to make the architecture concrete):

```python
import flwr as fl
import torch

class BehavioralFLClient(fl.client.NumPyClient):
    """
    Runs on each user's own device. Trains only on that device's
    local behavioral data; never transmits the raw data itself.
    """
    def __init__(self, model, local_dataloader):
        self.model = model
        self.local_dataloader = local_dataloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # local_train() trains only on this device's own sessions
        local_train(self.model, self.local_dataloader, epochs=1)
        return self.get_parameters(config={}), len(self.local_dataloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = local_evaluate(self.model, self.local_dataloader)
        return loss, len(self.local_dataloader.dataset), {"accuracy": accuracy}


# Server-side aggregation (conceptually -- see Section 8.4 for why naive
# averaging here is itself a documented attack surface, and what the
# robust-aggregation alternatives are)
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.1,          # sample 10% of enrolled devices per round
    min_fit_clients=50,
    min_available_clients=200,
)
```

Recent proposed architectures extend this basic shape with a hybrid CNN-RNN model architecture to capture spatio-temporal behavior patterns, layer in differential privacy (Section 5.6) for an additional formal privacy guarantee beyond FL's structural data-locality alone, and add a blockchain-based audit trail for update provenance; one such architecture reports sub-150-millisecond latency validated as production-feasible across three regulated-industry settings, with over 92% accuracy across modalities including keystroke dynamics.

### 5.6 Differential privacy as a complementary layer

Federated learning alone provides *structural* privacy (raw data never leaves the device) but does not provide a *formal mathematical guarantee* against a determined adversary who might, in principle, attempt to reconstruct information about individual training examples from the aggregated model updates themselves. Differential privacy (DP) closes this gap by adding calibrated statistical noise during training, providing a provable bound on how much any single training example could have influenced the final model — typically implemented via DP-SGD (differentially private stochastic gradient descent), which clips and noises per-example gradients before they are aggregated.

### 5.7 Metrics that actually matter — explained with worked examples

A full formulaic reference for every metric below appears in Appendix D; the version here focuses on what each metric means practically for a continuous-authentication system and why the choice of metric materially changes how a system's performance should be interpreted.

- **FAR (False Acceptance Rate)** — the proportion of impostor sessions incorrectly accepted as genuine. This is the number that directly maps to fraud losses: a FAR of 1% means roughly 1 in 100 attacker sessions get through undetected.
- **FRR (False Rejection Rate)** — the proportion of genuine sessions incorrectly flagged as anomalous. This maps directly to customer friction and support costs: a FRR of 5% means roughly 1 in 20 legitimate sessions get an unnecessary step-up challenge.
- **EER (Equal Error Rate)** — the single operating point where FAR equals FRR, commonly used as a headline comparison number across research papers because it collapses the FAR/FRR tradeoff curve into one number, but it is *not* necessarily where a production system should actually operate (see below).
- **ROC-AUC** — the area under the receiver-operating-characteristic curve (true-positive rate versus false-positive rate across all thresholds), summarizing overall discriminative ability independent of any one threshold choice.
- **PR-AUC (Precision-Recall AUC)** — the area under the precision-recall curve, which is far more informative than ROC-AUC specifically when the two classes are heavily imbalanced (which, in fraud detection generally and in behavioral-biometrics scoring specifically, they always are — genuine sessions vastly outnumber fraudulent ones in any real deployment).

**Why the ROC-AUC/PR-AUC distinction genuinely matters here, with a worked illustration.** Suppose a deployment sees 10,000 genuine sessions for every 10 fraudulent ones (a realistic ratio). A model that reports 99.20% ROC-AUC on a near-balanced *evaluation* split can look excellent, but ROC-AUC is computed against the false-positive rate as a fraction of all negatives, and with 10,000 negatives for every 10 positives, even a very good false-positive rate in absolute terms translates to a large *absolute number* of false alarms relative to the tiny number of true frauds being caught — a fact ROC-AUC's shape can obscure. Recomputing the same model's performance under this realistic imbalance, PR-AUC drops to around 0.842 with a Matthews Correlation Coefficient (MCC) of 0.781 — both meaningfully more conservative, and more operationally honest, readings of the same underlying model. This is directly worth internalizing for any evaluation Clickstream reports: a high ROC-AUC on a curated or rebalanced evaluation set is not the same claim as strong performance under the true, heavily-imbalanced class distribution a deployed system would actually face — reporting PR-AUC and MCC alongside ROC-AUC, on the true class distribution rather than a rebalanced one, is the more defensible practice, and is exactly the kind of methodological point worth stating explicitly in a writeup rather than leaving implicit.

- **Time-to-detect** — how many seconds (or how many keystrokes/mouse events) elapse between a session actually becoming fraudulent and the system's trust score crossing an action threshold. This is a metric unique to the continuous-authentication setting (a point-in-time system has no equivalent concept) and is, in practice, at least as operationally important as raw EER: a system with a mediocre EER but a very fast time-to-detect can still meaningfully limit fraud losses (the attacker gets a narrow window to act before being challenged), while a system with an excellent EER but a slow time-to-detect gives an attacker a long undetected window regardless of how accurate the eventual verdict is.


## 6. Reference Architecture

### 6.1 The five-stage pipeline, restated with implementation detail

The high-level shape — client capture, stream processing, per-user model, risk orchestrator, enforcement, with a feedback loop back into baseline adaptation — was presented as a diagram earlier in this research thread. This section goes one level deeper into how each stage is actually implemented.

### 6.2 Stage 1 — Client capture layer

A lightweight JavaScript listener (or native mobile SDK equivalent) attaches to standard DOM events. The critical, non-negotiable design constraint — required for GDPR/GLBA compliance, not merely good practice — is that this layer records **timing and coordinates only, never content**. It must be structurally impossible for this layer to log what a user typed, only how.

```javascript
// Illustrative client-side capture layer.
// Note: only timestamps and key IDENTITY (not character content in
// the sense of what was typed as a message) are ever captured, and
// even key identity is typically hashed/bucketed in production
// systems handling anything resembling password fields.

const behavioralBuffer = {
  keyEvents: [],
  mouseEvents: [],
};

function onKeyDown(e) {
  behavioralBuffer.keyEvents.push({
    keyCode: e.keyCode,      // NOT e.key / character value in sensitive fields
    action: 'down',
    t: performance.now(),
  });
}

function onKeyUp(e) {
  behavioralBuffer.keyEvents.push({
    keyCode: e.keyCode,
    action: 'up',
    t: performance.now(),
  });
}

function onMouseMove(e) {
  behavioralBuffer.mouseEvents.push({
    x: e.clientX,
    y: e.clientY,
    t: performance.now(),
  });
}

document.addEventListener('keydown', onKeyDown, { passive: true });
document.addEventListener('keyup', onKeyUp, { passive: true });
document.addEventListener('mousemove', onMouseMove, { passive: true });

// Flush the buffer over a WebSocket every N seconds, rather than on
// every single event, to bound network chatter.
setInterval(() => {
  if (behavioralBuffer.keyEvents.length || behavioralBuffer.mouseEvents.length) {
    socket.send(JSON.stringify(behavioralBuffer));
    behavioralBuffer.keyEvents = [];
    behavioralBuffer.mouseEvents = [];
  }
}, 2000);
```

### 6.3 Stage 2 — Stream processing

Every interaction event becomes a message published to a Kafka topic. Kafka's distributed, partitioned log architecture is specifically chosen (over, say, a simple message queue) because it durably retains the event stream in order and lets multiple independent consumers (the behavioral model, an audit/logging pipeline, a real-time dashboard) read the same stream without interfering with each other.

```python
# Producer side: the backend service receiving the client's WebSocket
# payload republishes it onto a per-session Kafka topic.
from confluent_kafka import Producer
import json

producer = Producer({'bootstrap.servers': 'kafka-broker:9092'})

def publish_behavioral_event(session_id, payload):
    producer.produce(
        topic='behavioral-events',
        key=session_id.encode('utf-8'),
        value=json.dumps(payload).encode('utf-8'),
    )
    producer.poll(0)  # trigger delivery callbacks without blocking
```

Apache Flink then consumes this stream and maintains **stateful, keyed windows** per session — this is the specific capability that distinguishes Flink from a simpler stream consumer: it can maintain a rolling window of "the last 10 seconds of this specific session's keystroke events" as continuously-updated state, re-evaluating the window's aggregate features every time a new event arrives, rather than requiring a batch job to periodically recompute from scratch.

```python
# Illustrative PyFlink windowed feature aggregation (conceptual --
# a production job would be considerably more involved, including
# watermarking for out-of-order events and exactly-once checkpointing
# configuration).
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.window import SlidingEventTimeWindows
from pyflink.common.time import Time

env = StreamExecutionEnvironment.get_execution_environment()
behavioral_stream = env.add_source(kafka_consumer)

windowed = (
    behavioral_stream
    .key_by(lambda event: event['session_id'])
    .window(SlidingEventTimeWindows.of(Time.seconds(10), Time.seconds(2)))
    .aggregate(BehavioralFeatureAggregator())
)

windowed.add_sink(model_scoring_sink)
env.execute("behavioral-feature-windowing")
```

Real-world validation of this general Kafka-plus-stream-processing pattern exists well outside behavioral biometrics specifically: Grab's GrabDefence fraud-detection platform, built on this kind of streaming architecture, reduced fraud rates to 0.2% — well below industry average — and Uber's Project RADAR combines machine learning with human reviewers on a similar real-time streaming foundation to adapt to evolving fraud patterns.

A production-grade reference pattern worth adopting explicitly: keep the **rules engine** (explainable, fast-to-ship business logic — "always challenge transfers over X amount from a new device") architecturally separate from the **ML model** (which catches subtler anomalies rules cannot articulate) from day one, rather than trying to fold both into a single scoring function. Flink handles the keyed, windowed aggregation and feeds both the rules engine and the model in parallel, with state checkpointed at a regular interval (commonly every 10-15 seconds) for fault tolerance, so that a node failure does not lose in-flight session state.

### 6.4 Stage 3 — Per-user anomaly model

Covered in full in Section 5 (Isolation Forest, One-Class SVM, Siamese networks, Transformers). The architectural point worth adding here is that this stage should be designed as a pluggable scoring interface, since the actual algorithm underneath is one of the most likely components to be swapped or upgraded as the system matures — a system built around "call `score(features) -> trust_delta`" can migrate from Isolation Forest to a deep sequence model without touching the streaming or orchestration layers at all.

### 6.5 Stage 4 — Risk orchestrator

This is the fusion point where the behavioral score stops being evaluated in isolation and gets combined with device fingerprint, IP/geolocation reputation, and transaction-specific context (amount, recipient novelty, time of day relative to the user's typical pattern). This fusion is precisely what turns a single noisy modality into something reliable enough to act on — a behavioral anomaly alone, acted on in isolation, produces too many false positives (Section 7.3); a behavioral anomaly *combined with* an unfamiliar device and an unusual transaction amount is a much stronger, more actionable signal.

**Score-level versus feature-level fusion.** Two broad strategies exist for combining multiple modalities:

- **Feature-level fusion** concatenates raw features from every modality (keystroke timing, mouse trajectory, device signals) into one combined feature vector, then trains a single model on the combined representation.
- **Score-level fusion** scores each modality independently with its own model, then combines the resulting scores (via weighted averaging, a small secondary meta-model, or simple rule-based combination) into a final trust score.

Score-level fusion is more common in production specifically because it **degrades gracefully** when one modality's data is missing or unreliable — a desktop session simply has no gyroscope data at all, and a score-level system can proceed with the modalities it does have, while a feature-level system trained expecting a fixed-size combined vector handles missing modalities far more awkwardly.

### 6.6 Stage 5 — Enforcement

Three broad enforcement actions, in increasing order of friction:

1. **Allow** — trust score remains within normal bounds; no action taken, session proceeds invisibly.
2. **Invisible restriction** — trust score is moderately depressed; the orchestrator silently reduces the session's privileges (e.g., allowing balance viewing but blocking high-value transfers) without alerting the user at all, buying time for either the score to recover (false alarm) or further evidence to accumulate (genuine attack).
3. **Step-up challenge** — trust score drops sharply, especially combined with unfamiliar device/context signals; the session is paused and a cryptographic FIDO2/passkey challenge (or a biometric facial re-verification on the registered mobile device) is required before the session can continue.

### 6.7 The feedback loop — baseline adaptation

The dashed feedback path in the architecture diagram represents the mechanism by which a user's baseline evolves over time rather than staying frozen at its initial enrollment state. This is necessary because behavior genuinely drifts (new devices, minor injuries, simply getting faster at using the app over months of habitual use) but is also, as covered in depth in Section 8.3, the specific mechanism that a documented class of poisoning attacks exploits — so this feedback loop cannot simply retrain on every low-risk anomaly unconditionally; it needs a validation gate, covered fully in Section 8.3.

## 7. Implementation Strategy

### 7.1 Phase 1 — Enrollment and the cold-start problem

A brand-new user has, by definition, no behavioral history to compare against. Two complementary strategies address this:

- **Population-level fallback model.** Before an individual baseline exists, score new sessions against a model trained across the *entire* user population's general behavioral statistics (which can still catch grossly anomalous behavior, like clearly bot-like uniform timing, even without knowing anything about this specific person) rather than leaving the session entirely unscored during the enrollment window.
- **Gradual handoff.** As enrollment data accumulates (typically over the first 1-2 weeks of active use), the system progressively shifts weight from the population model toward the individual model, rather than making an abrupt cutover at some arbitrary data-volume threshold. The FL-RBA² clustering approach from Section 5.5 formalizes this via provisional clustering with behaviorally similar existing users during the handoff window.

### 7.2 Phase 2 — Multimodal fusion strategy

Decide feature-level versus score-level fusion (Section 6.5) early, since this is an architectural choice that is expensive to reverse later. Score-level fusion is the more common production default specifically for its graceful degradation property when a modality is unavailable.

### 7.3 Phase 3 — Thresholding and environmental correlation (the hardest problem in the field)

This deserves to be stated plainly rather than softened: tuning the sensitivity threshold of a continuous behavioral biometrics system is, empirically, the single hardest ongoing engineering problem in this entire field — harder than any individual modeling choice covered in Section 5.

**The concrete failure case.** A legitimate user breaks their wrist. Their keystroke dwell/flight timing changes dramatically and immediately — slower, more erratic, with a completely different rhythm than their established baseline. Considered purely as a behavioral-distance measurement, this looks statistically similar in *magnitude* to what a stolen-session attacker's behavior would look like. A system that treats behavioral anomaly alone as sufficient grounds for enforcement action will lock this legitimate user out of their own account — and if this happens routinely enough (new mouse, new keyboard, minor injury, simply being tired), customer support volume becomes overwhelming and user trust in the system collapses.

**The standard mitigation.** Correlate the behavioral anomaly against environmental/contextual trust signals *before* acting: is this a known, previously-trusted device? Is the IP address and general location consistent with this user's history? Is the requested transaction itself low-risk (small amount, familiar recipient)? If the behavior is anomalous but the device/context is trusted and the transaction is low-risk, the system should generally **absorb the new behavioral data into the baseline** (treating it as legitimate drift) rather than escalate to a step-up challenge — but, per Section 8.3, this exact absorption mechanism needs a validation gate before accepting new data into the baseline, because unconditionally trusting "device is familiar, therefore new behavior is legitimate drift" is precisely the assumption a poisoning attack can exploit over a longer time horizon.

### 7.4 Phase 4 — Step-up integration

Wire the orchestrator's high-risk enforcement path into existing FIDO2/passkey infrastructure rather than building new step-up challenge infrastructure from scratch — this is both an engineering-effort argument and a regulatory one, since FIDO2/passkey challenges already satisfy the inherence/possession requirements of frameworks like PSD2's Strong Customer Authentication (Section 9.1).

### 7.5 Phase 5 — Continuous retraining and drift adaptation

Schedule periodic, validated baseline refresh (weekly is a common cadence in production streaming-fraud systems, per the Grab/Uber examples in Section 6.3) rather than relying solely on reactive retraining triggered after a specific false-positive incident. This keeps the baseline current with genuine, gradual behavioral drift without requiring a human-in-the-loop review of every minor adaptation.

## 8. The Adversarial Arms Race

Every claim in this section was independently verified against primary or near-primary sources during this research thread — not accepted on the strength of how confidently it was written up by an earlier reviewer. Where verification produced a correction, or found a claim more nuanced than initially presented, that correction is preserved here rather than smoothed over, because an inaccurate but confident-sounding citation is a worse outcome than an accurate but hedged one, especially for anything that might end up in a citation-formal appendix.

### 8.1 Mimicry, puppet, and hill-climbing attacks

**Mimicry attacks.** An adversary goes beyond merely observing a victim's authentication and actively imitates the victim's interaction rhythm, timing, and pressure — in extreme cases constructing physical replicas or rehearsing physically to reproduce a specific victim's behavioral traits closely enough to pass verification.

**Puppet attacks.** A more extreme variant in which the adversary forcibly uses the victim's own genuine input — for instance, coercing the victim, or acting while the victim is asleep or otherwise not in control of the device. This defeats behavioral detection precisely *because* the biometric trait being measured really is the genuine victim's own behavior; it is a coercion/physical-security problem rather than a data-spoofing problem, and no purely behavioral-analytics countermeasure addresses it directly.

**Hill-climbing attacks.** An adversary who can observe match scores (even just an accept/reject signal, repeated many times) can iteratively refine a synthetic template — adjusting it slightly, checking whether the match score improved, and repeating — until the synthetic template crosses the acceptance threshold. This is a black-box evasion strategy that generalizes across biometric modalities generically, not something specific to keystroke or mouse dynamics, which is exactly why it is grouped here as a general adversarial pattern rather than under a specific modality.

### 8.2 Bot humanization — the currently active front of the arms race

This is the practical, currently-deployed front of the arms race, and unlike some of the more academic attacks in this section, it is extensively documented in public security research, open-source tooling, and even the marketing material of bot-detection vendors themselves (who discuss it openly as part of explaining why their product is necessary).

Modern browser-automation frameworks generate mouse trajectories using **cubic Bézier curves** for the path shape itself, layer in velocity variation and timing derived from **Fitts's Law** (the well-validated psychological model describing how human pointing-movement time relates to target distance and size), and add stochastic micro-adjustments deliberately engineered to mimic natural overshoot and correction jitter. This level of sophistication is common enough to exist as ready-made, publicly available libraries built directly on top of standard browser-automation tools — this is not a theoretical attack, it is commodity tooling.

Bot-detection vendors themselves are candid about the resulting cat-and-mouse dynamic: Bézier-curve-generated mouse paths are measurably harder to distinguish from genuine human movement, both by eye and by naive rule-based detection, than the straight-line movement of older-generation bots — but combining deeper feature engineering (jerk profiles, click-timing distributions, and other signals beyond raw path shape) with machine learning can still catch a meaningful fraction of these more sophisticated bots.

**A quantified measurement of this exact dynamic.** A measurement study found that simple, unsophisticated bots are caught with roughly 95% precision and 97% recall by ML-based detectors — a strong result — but bots that additionally morph their browser/device fingerprint alongside their humanized mouse movement drop detector accuracy to just 55%, close to a coin flip. The same body of research documents dedicated bot frameworks that exist specifically to generate keystrokes and mouse clicks closely resembling human actions, for the express purpose of evading behavioral detection.

**The honest, load-bearing takeaway.** Naive, rule-based behavioral checks — "is the mouse path a perfectly straight line," "are keystroke intervals suspiciously uniform" — are already obsolete against a moderately resourced adversary. The entire reason deep feature engineering (Section 4) and statistical/ML-based anomaly scoring (Section 5), rather than simple heuristic rules, are the standard approach in this field is that the simple heuristic version of bot detection stopped working years ago. Any system design that implicitly assumes naive bot behavior as the threat model is already building against an outdated adversary.

### 8.3 Poisoning continuously-retrained baselines

This subsection addresses a real design tension surfaced directly by the implementation strategy in Section 7.5: the recommendation to let low-risk behavioral anomalies slowly retrain the baseline (rather than triggering enforcement every time) is itself the attack surface for a documented poisoning attack class, not merely a hypothetical concern.

**The specific, exact-match citation.** Zibo Wang's 2020 Louisiana Tech University PhD dissertation, "Poisoning Attacks on Learning-Based Keystroke Authentication and a Residue Feature Based Defense" (advisor: Vir Phoha), develops attacks the dissertation itself names **Frog-Boiling attacks** — update samples crafted with slow changes and random perturbations specifically engineered to bypass classifier detection during template retraining, gradually shifting a keystroke-authentication baseline toward an attacker's behavior rather than attempting one large, easily-detected jump. This is the single cleanest citation available for this attack class, because it was purpose-built for keystroke dynamics specifically, rather than adapted from an adjacent biometric modality. The same dissertation proposes a defense using correlation between successive training samples to catch a spurious, gradually-drifting input pattern before it fully corrupts the retrained model.

**The biometrics-general lineage, with an important caveat.** Lovisotto, Eberz, and Martinovic's "Biometric Backdoors: A Poisoning Attack Against Unsupervised Template Updating" (IEEE EuroS&P 2020, pages 184-197, University of Oxford) generalizes the same underlying idea to biometric template poisoning broadly, under the more recent, biometrics-native term "biometric backdoors" — demonstrating the attack against unsupervised face-recognition template updating, notably even in scenarios where the attacker has no digital access to the sensor and no knowledge of the classifier's decision boundary. The caveat worth preserving explicitly: this paper's own primary evaluation is on **face recognition**, not keystroke or mouse dynamics. It is legitimately cited here as evidence that the underlying poisoning mechanism is not specific to any one modality — but it should not be presented as a keystroke-specific result, since it isn't one.

**The architecture-class precedent — corrected.** Kravchik and Shabtai's "Can't Boil This Frog: Robustness of Online-Trained Autoencoder-Based Anomaly Detectors to Adversarial Poisoning Attacks" (2020) tested the same general class of poisoning attack against online-retrained autoencoder anomaly detectors in an industrial-control-systems context — architecturally the same "retrain continuously on new incoming data" pattern as an Isolation Forest baseline that absorbs low-risk anomalies, just applied to a different domain and detector type. It would be a mistake to cite this paper as clean confirmation that this architecture class is vulnerable: the paper's own actual finding, verified directly from its abstract, is closer to the **opposite** — testing an interpolation-based poisoning algorithm and a back-gradient-optimization poisoning algorithm against the online-trained autoencoder detector, the authors found the detector **resilient** across all ten relevant attacks in their evaluation dataset, with poisoning success limited to only a narrow, specific set of attack types and magnitudes. The honest, defensible framing is that this is a **complicating data point**, not a second confirmation: the same broad architecture class (continuously retrained, unsupervised anomaly detection) was tested for exactly this vulnerability in a different domain and came back more robust than a naive reading of "retraining is inherently exploitable" would predict. Read together, Zibo Wang's result (keystroke dynamics: vulnerable) and Kravchik and Shabtai's result (industrial-control autoencoders: largely resilient in their tested scenarios) support a more precise and more defensible claim than either alone: poisoning of continuously-retrained baselines is a real, demonstrated risk in at least one directly relevant modality, but it is not a universal property of every online-retrained anomaly detector, and which systems are vulnerable likely depends on specifics of the feature space and retraining cadence that have not been exhaustively mapped across modalities.

**On the origin of the "boiling frog" terminology — hedged appropriately.** One line of citing literature explicitly attributes the "boiling frog attacks" terminology to Kloft and Laskov's work on incremental poisoning of centroid anomaly detection via retraining (published as "Online Anomaly Detection under Adversarial Impact," AISTATS 2010, and "Security Analysis of Online Centroid Anomaly Detection," JMLR 2012). However, an earlier and more literal use of the term exists: Kim et al.'s 2009 paper is literally titled "The Frog-Boiling Attack: Limitations of Anomaly Detection for Secure Network Coordinate Systems," applied to an entirely different problem domain (network coordinate systems, used for network-distance estimation, not biometrics or intrusion detection at all). A clean, single-origin claim for this terminology could not be verified with confidence. The more defensible framing: the boiling-frog metaphor for this style of gradual-poisoning-via-retraining attack traces back to at least Kim et al. (2009) in the network-coordinate-systems context, with Kloft and Laskov's centroid-anomaly-detection analysis (2010/2012) providing a rigorous security-bound treatment of the same underlying incremental-contamination mechanism in a security-detection context, and Zibo Wang's 2020 dissertation applying and renaming the same family of attack specifically for keystroke-dynamics authentication. Presenting any one of these three as *the* singular origin of the term would overstate what could actually be confirmed.

**Required mitigation: a validation gate before baseline updates.** Given the above, Section 7.5's recommendation to slowly absorb low-risk anomalies into the baseline cannot be an unconditional rule. The required addition is a gate — a residue-feature-based defense (as Zibo Wang's own dissertation proposes) or a Reject-on-Negative-Impact (RONI)-style defense — that validates any candidate baseline update against the recent sample history *before* accepting it, specifically checking for the slow, incremental, statistically-smooth drift signature that characterizes a Frog-Boiling-style attack, rather than treating "this anomaly is low-risk by the device/context correlation in Section 7.3" as sufficient justification on its own to fold new data into the baseline unconditionally.

### 8.4 Byzantine-robust aggregation for federated learning

Section 5.5 described an FL-RBA² architecture using differential privacy plus message authentication codes (MACs) to protect federated model updates. It is important to be precise about exactly what that combination does and does not defend against: differential privacy and MACs jointly protect data **confidentiality** and update **integrity in transit** — a MAC guarantees that an update was not tampered with *after* a legitimate device produced it, on its way to the aggregation server. Neither mechanism does anything to address a **legitimately-authenticated device deliberately sending a maliciously-crafted update** — which is precisely the actual Byzantine threat model in federated learning: not an outside attacker intercepting traffic, but a compromised or malicious participant that is a fully valid, authenticated member of the federation, sending intentionally corrupted gradients.

**The real defenses: robust aggregation rules.** Rather than naively averaging every client's update (as vanilla Federated Averaging does), robust aggregation rules down-weight or discard outlier updates before aggregating:

- **Krum / Multi-Krum** — selects the client update that is closest to its neighbors in the update-vector space (i.e., most consistent with what other clients are reporting), explicitly designed to be robust against a bounded number of Byzantine (malicious) participants.
- **Coordinate-wise median** — takes the median value, coordinate by coordinate, across all client updates, rather than the mean, which is inherently less sensitive to a small number of extreme outlier values than averaging is.
- **Trimmed mean** — discards the highest and lowest some-percent of values per coordinate before averaging the remainder, a middle ground between full averaging and taking the median.

**A minimal Krum implementation**, to make the "closest to its neighbors" selection rule concrete rather than purely descriptive:

```python
import numpy as np

def krum(client_updates, num_byzantine, num_to_select=1):
    """
    client_updates: list of flattened numpy arrays, one per client's
                    model update for this round.
    num_byzantine:  the number of malicious clients the rule should
                    tolerate (must be known or conservatively estimated
                    in advance -- this is itself a real limitation).
    Returns the update(s) selected as most representative/trustworthy.
    """
    n = len(client_updates)
    updates = np.stack(client_updates)  # shape: (n, param_dim)

    # pairwise squared Euclidean distances between all client updates
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dists[i, j] = np.sum((updates[i] - updates[j]) ** 2)

    # for each client, sum the distances to its (n - num_byzantine - 2)
    # closest neighbors -- excluding itself and the most distant clients,
    # which are the ones most likely to be Byzantine
    num_neighbors = n - num_byzantine - 2
    scores = []
    for i in range(n):
        sorted_dists = np.sort(dists[i])
        score = np.sum(sorted_dists[1:num_neighbors + 1])  # skip index 0 (self, distance 0)
        scores.append(score)

    # Krum selects the client(s) with the LOWEST score -- i.e., the
    # update most consistent with the majority of other clients
    selected_indices = np.argsort(scores)[:num_to_select]
    return [client_updates[i] for i in selected_indices]
```

This is the plain, textbook version of the rule — a production Flower deployment would use the framework's built-in `Krum` strategy rather than this hand-rolled version, but the code above is what that strategy is doing underneath, and makes explicit the real limitation worth flagging: `num_byzantine` has to be supplied or conservatively estimated in advance, and Fang et al.'s attack (below) specifically exploits situations where this assumption is violated or where a sophisticated attacker crafts updates that look statistically unremarkable by this exact distance-based criterion while still being poisoned.

**These are not unbreakable either — stated with the same epistemic honesty as the EER fairness caveat in Section 4.1 and the PSD2/GDPR nuances in Section 9.** Fang, Cao, Jia, and Gong's "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning" (USENIX Security Symposium, 2020) demonstrates a local model poisoning attack that, under full or partial knowledge of benign client updates, specifically targets and degrades Krum, trimmed mean, and coordinate-wise median aggregation. In one reported case, a Krum-targeting attack raises the defended model's test error from 0.14 to 0.58 — a 314% relative increase — directly refuting any claim that these robust-aggregation rules make federated learning immune to malicious participants. The correct framing is "state of the art, not solved": robust aggregation is a substantial, real improvement over naive averaging, and should be used, but should not be described or relied upon as a complete Byzantine-robustness guarantee.

### 8.5 Injection attacks — a separate threat class from bot humanization, correctly scoped

Injection attacks are frequently conflated with either presentation attacks (Section 8.2's bot humanization) or synthetic-media/deepfake attacks in casual discussion, but they are a distinct and separately standardized threat category, and it matters to keep the boundaries precise.

**The three-layer framing, and where it actually comes from.** Presentation Attack Detection (governed by the ISO/IEC 30107 standard family) defends the **sensor** against physical fakes presented directly to it — a photo, a mask, a video replayed in front of a camera. Deepfake and liveness detection defends the **image/media stream itself** against synthetic or manipulated content. Injection attack detection — governed by the newer CEN/TS 18099 technical specification (2024) — defends the **data pipeline**, addressing an attacker who bypasses the physical sensor entirely and feeds synthetic data directly into the software pipeline downstream of it (for instance, injecting a fabricated video stream through a virtual camera driver rather than presenting anything to a physical camera at all).

**An important, independently-verified caveat that strengthens rather than weakens the original concern about this framing.** Every source checked for both ISO/IEC 30107 and CEN/TS 18099 — including CEN/TS 18099's own explicit scope language — frames these standards entirely around **face, iris, voice, and fingerprint** biometrics: photographs, masks, screen-replay attacks, deepfakes, and virtual-camera injection into identity-verification and KYC pipelines. Neither standard has a keystroke- or mouse-dynamics-specific counterpart at present. The three-layer framing above is therefore accurate as a **conceptual scaffold** borrowed from the physiological-biometrics world, and it is genuinely useful for organizing the problem space — but it should not be presented as though a keystroke/mouse-dynamics-specific standard exists and certifies a solution for this modality, because none currently does. Any deepfake or injection-attack *statistic* circulating in industry material (rates of deepfake-driven fraud attempts, documented attack volumes against liveness checks, and similar figures) should be read as applying to **facial biometrics and KYC onboarding video specifically**, not to typing or mouse-movement biometrics, unless a source explicitly states otherwise — importing those statistics uncaveated into a keystroke/mouse-dynamics discussion would be a real defensibility problem if scrutinized closely.

**What injection attacks against keystroke dynamics specifically actually look like, with verified citations:**

- Monaco, Ali, and Tappert's "Spoofing key-press latencies with a generative keystroke dynamics model" (IEEE BTAS, 2015) demonstrated a generative model — the Linguistic Buffer and Motor Control (LBMC) model — that directly spoofs key-press latency sequences to defeat keystroke-dynamics authentication, without any need for a keylogger or direct observation of the victim's actual typing.
- Migdal and Rosenberger's follow-up work, "Statistical modeling of keystroke dynamics samples for the generation of synthetic datasets" (Future Generation Computer Systems, volume 100, 2019, pages 907-920), found that heavy-tailed statistical distributions — specifically the **Gumbel distribution** — best approximate real human typing-timing behavior closely enough to synthesize convincing forgeries at scale, going beyond a single spoofed sample to a full synthetic dataset generation approach.
- The most striking and directly demonstrable attack in this category requires **no keylogger and no direct data access at all**: a 2022 paper (Journal of Big Data, volume 9, article 1) extracts a victim's typing rhythm purely from a **screen-recorded video** of them typing, then replays the extracted timing pattern to authenticate. Tested specifically against KeyTrac's commercial keystroke-dynamics authentication product, this attack achieved a **64% evasion rate** — meaning the majority of attempted forgeries successfully passed authentication despite the attacker never having touched the victim's device, network traffic, or keyboard at all.
- A body of research applies generative adversarial networks (GANs) to keystroke-injection-style attacks, with reported performance figures in the 93-96% range for fixed-text scenarios and 75-92% for free-text scenarios appearing in the literature. **This figure is flagged with appropriate hedging rather than stated as settled**: verification could not cleanly confirm whether these specific percentages represent the *attacker's* success rate at fooling an authentication system, or the *defender's* detection accuracy at catching injected/synthetic samples — both framings appear attached to closely related papers in this space, and the two are not the same claim. Given that ambiguity, these numbers should be understood as "reported performance figures in this range appear in both attack-generation and detection literature for GAN-based keystroke injection" rather than asserted in a specific direction that has not been independently confirmed from a primary source.

**Worth building as an actual demonstration, not just cited.** The screen-recording attack above is cheap enough to reproduce directly rather than merely cite: record a subject typing, extract keystroke timing purely from the video footage (frame-by-frame key-press visual detection, or even simpler proxies like visible finger/hand movement timing), replay the extracted timing pattern against a locally trained keystroke-dynamics detector, and report the actual evasion rate achieved against your own model. Doing this converts this subsection from a literature summary into an original, reportable result, and is a natural fit for a system that is already collecting and modeling exactly this kind of timing data.

### 8.6 A dataset-honesty note that applies across this entire section

Every public dataset referenced anywhere in this research (CMU, Aalto/KVC, GREYC-NISLAB, Balabit, HMOG — full details in Appendix C) is a generic academic benchmark corpus, collected from research-study volunteers under controlled or semi-controlled conditions. None of them are real banking session data, and none of the accuracy figures reported throughout Sections 4, 5, and this section should be read as a direct prediction of performance on actual banking telemetry, which differs in volume, session length, typing context (form fields versus free text), and population characteristics from any of these academic corpora. Any evaluation built on these datasets should be stated plainly as having been "evaluated against public academic behavioral-biometrics benchmarks as a proof of concept," rather than in any phrasing that could be read as implying validation against real banking data — this is a distinction worth having a ready, explicit answer for before it is asked, rather than after.


## 9. Standards, Regulation, and Privacy

### 9.1 PSD2, Strong Customer Authentication, and Transaction Risk Analysis

A specific, counterintuitive regulatory nuance sits at the center of this section: behavioral biometrics is **not** one of PSD2's three recognized SCA factors. The European Banking Authority has stated this explicitly — behavior-based transaction characteristics cannot be uniquely attributed to a specific payment service user in the way a password (knowledge), a registered device (possession), or a fingerprint (inherence) can be, so they are not appropriate for *completing* Strong Customer Authentication on their own.

**What behavioral biometrics actually does under PSD2, then: Transaction Risk Analysis (TRA) exemption scoring.**

| SCA element | Category | Can behavioral biometrics fill this role? |
|---|---|---|
| Knowledge | Something the user knows (password, PIN) | No |
| Possession | Something the user has (registered device, hardware key) | No |
| Inherence | Something the user is (fingerprint, face, behavioral pattern arguably closest here, but not formally recognized as sufficient alone) | Not formally, on its own |
| Transaction Risk Analysis exemption | A risk-scoring pathway that can exempt a transaction from SCA entirely if fraud rates for that payment type stay below a regulatory threshold | **Yes — this is its actual regulatory role** |

| PSD2 rule | Detail |
|---|---|
| Hard SCA floor | Transactions above €500 always require SCA, regardless of how low-risk the behavioral signal looks |
| TRA exemption logic | Lower-value transactions can be exempted from SCA if the payment service provider's own reference fraud rate for that transaction-value band stays under a regulatory threshold |
| Where behavioral biometrics fits | It feeds the risk score used to decide whether a given transaction qualifies for the TRA exemption pathway, rather than acting as a recognized authentication factor in its own right |
| Practical implication | A well-tuned behavioral-biometrics layer can let a bank offer a frictionless (SCA-exempted) experience on more low-value transactions, by keeping the provider's measured fraud rate low enough to qualify for the exemption — this is a real commercial incentive distinct from its fraud-catching value alone |
| Passkeys / FIDO2 and PSD2 | Passkey-based step-up challenges satisfy SCA cleanly because a passkey combines possession (the registered authenticator) and inherence (the biometric or PIN unlocking it on-device) — this is why Section 6.6 and 7.4 specifically recommend wiring high-risk enforcement into FIDO2/passkey infrastructure rather than inventing a new step-up mechanism |

### 9.2 GDPR Article 9 — the special-category-data question

**The specific legal trigger, stated precisely.** GDPR Article 9's strictest protections for biometric data apply specifically when biometric data is processed "for the purpose of uniquely identifying a natural person." This is a narrower trigger than "any processing of biometric-adjacent data" — general analytics, verification, or anomaly-detection use of biometric-adjacent signals may fall outside this narrowest, most heavily regulated category, depending on exactly how the processing is structured and what purpose it serves.

**Why this shapes system architecture, not just legal paperwork.** This is precisely why most behavioral-biometrics vendors architect their systems around **verification** (1:1 — "does this match this one already-claimed, enrolled identity") rather than **identification** (1:N — "search across everyone to find out who this is"), as established in Section 2.3. A verification-only design has a more defensible argument for sitting outside Article 9's strictest special-category rules than an identification-capable design would.

**An important qualifier on this whole subsection.** This is a genuinely contested area of law, not a settled one, and any vendor's confident claim about exactly which side of this line their product sits on should be treated with some skepticism rather than accepted as settled fact. This is exactly the kind of question a bank's own compliance and legal team should get a real legal opinion on before shipping a production system, not something to take as resolved from a technical research document such as this one (nor is this document, or any AI system producing it, a substitute for qualified legal advice).

### 9.3 US regulatory landscape — NIST and FFIEC

| Standard | Status | Key content |
|---|---|---|
| NIST SP 800-63-3 | Superseded | The prior Digital Identity Guidelines edition, built around static, checklist-based authenticator assurance levels |
| NIST SP 800-63-4 | Current — finalized July 2025 | Replaces the checklist-based model with a continuous, risk-based Digital Identity Risk Management (DIRM) framework — a substantially closer conceptual match to what a behavioral-biometrics layer actually does (continuous, risk-scored evaluation) than the prior static-assurance-level model |
| NIST SP 800-207 | General Zero Trust Architecture guidance | Real and relevant to the broader system design, but more generic than 800-63-4 specifically for identity/authentication; 800-63-4 is the sharper, more current citation for this particular technology |
| FFIEC authentication guidance (US) | Ongoing, risk-based | Takes a layered, risk-based approach rather than mandating any specific technology, and explicitly recognizes behavioral analytics as one acceptable layer alongside device fingerprinting and transaction-velocity checks |

### 9.4 Nepal — Nepal Rastra Bank's regulatory framework

Directly relevant given Clickstream's Nepal-based research context, and to any real deployment of this technology at a Nepali bank.

| NRB instrument | Year / status | Key content |
|---|---|---|
| NRB IT Guidelines | 2012, still in force | Requires banks to implement more than one authentication factor for higher-risk activities such as fund transfers, with the authentication methodology required to scale with the risk level of the transaction — a risk-commensurate framing that maps directly onto what continuous behavioral biometrics is architecturally built to provide |
| NRB Cyber Resilience Guidelines | In force since August 2023 | Structures bank cybersecurity obligations around five categories: **Governance, Identification, Protection, Detection, and Response & Recovery** — closely mirroring the structure of the NIST Cybersecurity Framework |

**Where a behavioral-biometrics layer maps onto NRB's five-category structure:**

- **Governance** — policy and oversight of how the behavioral-biometrics system's decisions are reviewed and escalated
- **Identification** — asset and risk inventory; behavioral biometrics itself doesn't directly satisfy this category, but the risk assessment justifying its deployment does
- **Protection** — the preventive control layer; this is where a continuous-authentication system's *design* (client capture, encryption, access control around the behavioral data store) lives
- **Detection** — this is where continuous behavioral biometrics most directly and cleanly slots in: it is, functionally, a detection control, continuously evaluating whether the current session is consistent with the expected identity
- **Response & Recovery** — the step-up challenge and enforcement actions in Section 6.6 are the response half of this category; the baseline-adaptation feedback loop (Section 6.7, with the poisoning-defense gate from Section 8.3) is closer to the recovery half

This mapping is worth stating explicitly in any compliance-facing narrative for a Nepal-deployed system, since it directly ties a specific technology choice to a specific, already-mandated regulatory framework category rather than requiring a bespoke justification from first principles.

## 10. Industry and Market Landscape

### 10.1 Market sizing

| Year | Market size | Notes |
|---|---|---|
| 2025 | $2.72 billion | |
| 2026 | $3.45 billion | Current year |
| 2031 (forecast) | $11.38 billion | Implies a 26.95% compound annual growth rate across the forecast window |

### 10.2 Vendor comparison table

| Vendor | Founded / structure | Notable facts |
|---|---|---|
| BioCatch | Israel-based, most widely deployed pure-play behavioral biometrics platform | Over 16 billion sessions analyzed; 500+ million digital banking customers protected; raised $35M in a September 2025 Series E to expand into healthcare and government; product suite includes DeviceIQ, Connect, Align, Link (mule-network mapping), and Scout (money-mule detection) |
| BehavioSec | Started in Stockholm, keystroke/pointer analytics focus; now part of LexisNexis Risk Solutions | Integrated with the ThreatMetrix Digital Identity Network; recognized in Forrester's 2024 Enterprise Fraud Management Solutions Wave |
| Callsign | UK-based | Bundles behavioral biometrics, device intelligence, and location signals into a single "Intelligence Engine" |
| Mastercard NuData | Acquired by Mastercard | Upgraded June 2025 with real-time gait analysis for mobile payments across Europe, enabling PSD2-compliant continuous authentication via smartphone sensors |
| TypingDNA | Keystroke-dynamics pure play | Ranked among Top 10 Zero Trust security solutions in independent assessments; approved by both the New York State DMV and the European Banking Authority as a compliant identity-authentication method; granted a patent in February 2024 for its typing-biometrics 2FA method; integrates with Okta, Microsoft Entra, and Ping identity platforms |
| Sardine | Founded 2020 by former Revolut and Coinbase security leads | Raised $75.6M from Andreessen Horowitz, Visa, and Google Ventures; serves 250+ companies |
| ThreatMark | Omnichannel fraud and social-engineering detection | "Cognitive Security Platform" branding |
| Plurilock DEFEND | Enterprise endpoint continuous authentication | Targets general enterprise endpoints rather than banking specifically |

### 10.3 Money-mule detection statistics (BioCatch Scout)

| Metric | Figure |
|---|---|
| Active money-mule accounts identified | 98% |
| New mule accounts identified before their first transfer | 70% |

### 10.4 Fraud-loss context that motivates vendor adoption

Restating the Section 3.3 statistics here in market-adoption context: with US ATO fraud losses exceeding $15.6 billion in 2024 and roughly 65% of US banks already deploying some form of behavioral AI biometrics, the vendor landscape above should be read as a mature, competitive market responding to an already-quantified and growing loss category — not an emerging or speculative technology space.


## 11. Ongoing Academic Research Directions

This section covers where the live research frontier actually sits, as opposed to what is already settled and productized (Sections 5 and 10).

### 11.1 Federated learning, differential privacy, and Zero Trust convergence

By a clear margin, this is the most active research theme in 2025–2026 publications on this topic, for a structural reason rather than a fashion one: it is the only architectural approach that simultaneously satisfies the ML need for population-scale training data (Section 5.1's one-class problem is much easier to solve with more genuine-user examples to learn from) and the GDPR/PSD2 constraint against centralizing that data across users (Section 9.2). Expect continued work specifically on: better non-IID handling for the single-genuine-class-per-client setting described in Section 5.5 (this remains harder than the non-IID settings most federated learning research was originally designed around); tighter integration of differential privacy budgets with Byzantine-robust aggregation (Section 8.4) rather than treating the two as separate bolt-on layers; and continued Zero-Trust-Architecture framing (Section 9.3) of federated behavioral biometrics as one input into a broader continuous risk evaluation rather than a standalone authentication mechanism.

### 11.2 Explainable AI (XAI) for biometrics

The first systematic literature review specifically applying XAI techniques to biometrics only appeared in 2024 — meaning this is a genuinely nascent subfield relative to the maturity of the underlying detection techniques it is trying to explain. The driving force is regulatory, not purely technical: when a step-up challenge or an outright block is triggered, both the bank's dispute-resolution process and (depending on jurisdiction) the affected customer may have a legitimate interest in understanding *why* — "your typing rhythm didn't match your usual pattern" is a very different, and much more actionable, explanation than an opaque anomaly score. Expect continued work on producing domain-native explanations (e.g., which specific behavioral features drove a given trust-score drop) rather than generic model-agnostic explanation techniques borrowed wholesale from other ML domains.

### 11.3 Concept and behavioral drift adaptation

Distinguishing "this person got a new phone" from "this isn't the same person" without requiring manual retraining after every minor, benign behavioral shift remains genuinely unsolved in general, even though Section 7.3's device/context correlation heuristic and Section 8.3's validation-gated baseline update handle the common cases reasonably well in practice. The open research question is closer to: can a system learn *which kinds* of behavioral drift are typically benign (new device, minor injury, general skill improvement over months of use) versus which kinds are typically attack-indicative, from the shape of the drift itself, rather than relying primarily on external device/context correlation as a proxy?

### 11.4 Cross-device and cross-context generalization

As established in Section 4.7, a behavioral baseline built from desktop keyboard-and-mouse sessions transfers poorly to a mobile touchscreen session by the same person, because the underlying biomechanics are genuinely different, not just measured differently. Unifying these into a single cross-device behavioral model — rather than maintaining separate per-device-class baselines, which is the current production norm — remains an open research problem, and one with real practical stakes, since a banking customer routinely switches between a mobile app and a desktop browser within the same account.

### 11.5 Transformer and attention-based architectures displacing recurrent models

Following the same broad trajectory NLP took several years earlier, self-attention-based sequence models are displacing LSTM/RNN-based approaches for keystroke and mouse sequence modeling specifically because they handle longer-range dependencies in a typing or movement sequence more effectively once sufficient training data is available to fit the larger parameter count these architectures typically require (Section 5.3).

### 11.6 Synthetic data and generative models for training augmentation

Given the dataset-honesty concern raised in Section 8.6 — every available public dataset is a small, controlled-condition academic corpus, not real banking telemetry at scale — there is active research interest in generative models (the same broad technique family covered adversarially in Section 8.5, but here applied constructively) to synthesize additional, realistic training data, specifically to reduce dependence on collecting ever more real behavioral data from actual users under privacy constraints. This is a double-edged research direction worth being aware of: the same generative techniques that can help augment legitimate training data (constructive use) are mechanically related to the techniques covered in Section 8.5 as attack vectors (adversarial use) — the difference is entirely about who controls the generative model and what it is used for, not about the underlying technique.

### 11.7 Demographic fairness

The age- and gender-related score disparities flagged even in the state-of-the-art KVC-onGoing benchmark results (Section 4.1) represent a genuinely open, unresolved problem rather than a solved footnote. A system that is measurably more accurate, or has a measurably different false-reject rate, for one demographic group than another raises both a technical performance question and a fair-lending/discrimination-adjacent regulatory question for a bank deploying it — this is worth flagging explicitly in any implementation writeup rather than leaving implicit, given the regulatory sensitivity of differential treatment by protected characteristic in financial services specifically.

### 11.8 Public, ongoing benchmark challenges

The KVC-onGoing challenge (Section 4.1), hosted on CodaLab against the Aalto Desktop/Mobile Keystroke Databases, continues to push measured state-of-the-art EER downward in a way that is directly comparable across research groups using a shared, standardized evaluation protocol — this is a genuinely useful resource if a rigorous, citable, independently-verifiable EER baseline is ever needed for a paper or competition writeup, since the alternative (comparing EER figures across papers using different private train/test splits) is considerably less rigorous.

## 12. Open Problems and Honest Limitations

Stated plainly, because an accurate accounting of what remains unsolved is more useful than an uncritically positive one: multiple vendors in this space are themselves explicit that behavioral biometrics alone cannot stop fraud — believing otherwise would be a real operational mistake. The concrete open problems, consolidated from throughout this document:

- **Cold start** (Section 7.1) — a brand-new user has no baseline; mitigated by population-level fallback and gradual handoff, not eliminated.
- **Behavioral drift** (Section 7.3, Section 11.3) — genuine, benign behavioral change (injury, new device, skill improvement) is statistically hard to distinguish from an attack in isolation; mitigated by environmental correlation, not solved by behavioral signal alone.
- **Cross-device inconsistency** (Section 4.7, Section 11.4) — baselines do not transfer cleanly across device classes; current practice is separate per-device baselines, not a unified model.
- **The bot-humanization arms race** (Section 8.2) — naive rule-based detection is already obsolete against a moderately resourced adversary; this is a genuinely live, ongoing contest, not a one-time solved problem.
- **Poisoning of continuously-retrained baselines** (Section 8.3) — a documented, demonstrated attack class against exactly the kind of gradual-baseline-adaptation mechanism this field relies on to avoid false-positive fatigue; requires an explicit validation gate, not just device/context correlation.
- **Byzantine-robust federated aggregation is "state of the art, not solved"** (Section 8.4) — the best current defenses (Krum, trimmed mean, coordinate-wise median) are demonstrably defeatable under a documented attack, not a closed problem.
- **Injection attacks** (Section 8.5) — a separately standardized, actively researched threat class with real, demonstrated attacks specific to keystroke dynamics (the 64% screen-recording evasion result is not a hypothetical).
- **The explainability gap** (Section 11.2) — still a nascent research area, and a real regulatory exposure until it matures.
- **Demographic fairness** (Section 11.7) — an acknowledged, unresolved disparity even in state-of-the-art published benchmarks.
- **The privacy-utility tradeoff under GDPR/PSD2** (Section 9.1, Section 9.2) — genuinely unresolved as a matter of settled law, not merely a matter of engineering effort.

Every credible deployment in this space treats behavioral biometrics as one signal fused with device intelligence and transaction context (Section 6.5) — never as a standalone gate. That fusion requirement is not a minor implementation detail; it is the central design principle the entire rest of this document is organized around.

## 13. Bibliography for Part I

Full bibliographic detail for every source cited in Sections 1–12, organized by topic area. Author names, venues, and years are as verified during this research thread; where a source is a company/vendor page rather than an academic paper, it is marked accordingly.

**Foundational concepts and threat landscape**
- Federal Reserve fraud-mitigation guidance on account takeover fraud, session hijacking, and MFA-fatigue attacks as the dominant 2026 fraud vector (industry/regulatory source, not an academic paper).
- Mordor Intelligence, behavioral biometrics market sizing report (2025–2031 forecast).

**Survey literature**
- Abuhamad et al., survey covering 140+ behavioral biometric approaches across six modality groups, including the ~90% passive-authentication user-preference finding.
- Liang et al., survey on behavioral biometrics from an IoT-wide AI perspective.
- Murmuria et al., Android-native ensemble study (power, touch, gesture, movement fusion), 59 subjects, EER 6.1–6.9%.

**Keystroke dynamics**
- Acien et al. (2021), TypeNet — foundational deep-learning keystroke-biometrics paper.
- Budžys et al. (2025), Siamese neural network with triplet loss and interpolation-based data fusion for keystroke dynamics, validated on fused CMU/KeyRecs/GREYC-NISLAB data.
- 2025 Soft Computing paper on Transformer self-attention architecture for keystroke sequence modeling, surpassing RNN-based approaches.
- Putra & Chowanda (2025), CNN-BiLSTM hybrid for multi-session, uncontrolled-setting keystroke data.
- Gabor Filter Matrix Transformation paper — numerical-to-image encoding of keystroke timing for CNN classification, EER 0.04545 on combined CMU/GREYC-NISLAB data (30,000+ samples).
- Bhasin et al. (2025), early-stage quantum machine learning applied to keystroke dynamics.
- Killourhy & Maxion (2009), the original CMU keystroke-dynamics benchmark paper and dataset (51 subjects, password ".tie5Roanl", 400 entries/user).
- KVC-onGoing challenge (CodaLab), current state-of-the-art benchmark on Aalto Desktop/Mobile Keystroke Databases, EER 3.33%/3.61%.

**Mouse dynamics**
- Almalki, Assery, and Roy, ~87-feature mouse clickstream study comparing decision tree/k-NN/random forest/CNN classifiers.
- Balabit Mouse Dynamics Challenge dataset paper (Fülöp, Kovács, Kurics, Windhager-Pokol, 2016).
- BEACON dataset paper — Minecraft-based multimodal behavioral fingerprinting from gameplay data.

**Mobile and touch dynamics**
- Sitová, Šeděnka, Yang, Peng, Zhou, Gasti, and Balagani (2016), HMOG — IEEE Transactions on Information Forensics and Security, volume 11, issue 5, pages 877–892.
- HuMIdb (Human Mobile Interaction Database) paper — unsupervised mobile sensor data collection.

**Federated learning and differential privacy**
- Flower (flwr) federated learning framework documentation.
- FL-RBA² architecture paper — clustering-based cold-start mitigation, differential privacy, and MAC-based update protection.
- Fang, Cao, Jia, and Gong (2020), "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning," USENIX Security Symposium.

**Poisoning and adversarial attacks**
- Zibo Wang (2020), "Poisoning Attacks on Learning-Based Keystroke Authentication and a Residue Feature Based Defense," PhD dissertation, Louisiana Tech University, advisor Vir Phoha.
- Lovisotto, Eberz, and Martinovic (2020), "Biometric Backdoors: A Poisoning Attack Against Unsupervised Template Updating," IEEE EuroS&P, pages 184–197.
- Kravchik and Shabtai (2020), "Can't Boil This Frog: Robustness of Online-Trained Autoencoder-Based Anomaly Detectors to Adversarial Poisoning Attacks."
- Kim et al. (2009), "The Frog-Boiling Attack: Limitations of Anomaly Detection for Secure Network Coordinate Systems."
- Kloft and Laskov, "Online Anomaly Detection under Adversarial Impact" (AISTATS 2010) and "Security Analysis of Online Centroid Anomaly Detection" (JMLR 2012).
- Monaco, Ali, and Tappert (2015), "Spoofing key-press latencies with a generative keystroke dynamics model," IEEE BTAS.
- Migdal and Rosenberger (2019), "Statistical modeling of keystroke dynamics samples for the generation of synthetic datasets," Future Generation Computer Systems, volume 100, pages 907–920.
- Screen-recording keystroke-timing extraction attack paper (2022), Journal of Big Data, volume 9, article 1 — 64% evasion rate against KeyTrac.

**Standards**
- ISO/IEC 30107 (parts 1–3), Presentation Attack Detection.
- CEN/TS 18099 (2024), Biometric Data Injection Attack Detection.
- NIST SP 800-63-4 (finalized July 2025), Digital Identity Guidelines.
- NIST SP 800-207, Zero Trust Architecture (general reference, superseded in relevance by 800-63-4 for this specific application).

**Regulation**
- European Banking Authority (EBA) statements on behavior-based transaction characteristics and Transaction Risk Analysis exemptions under PSD2.
- GDPR Article 9, special-category personal data.
- Nepal Rastra Bank, IT Guidelines (2012) and Cyber Resilience Guidelines (in force since August 2023).
- FFIEC authentication guidance (US).

**Industry/vendor sources** (company material, not peer-reviewed research)
- BioCatch, BehavioSec/LexisNexis Risk Solutions, Callsign, Mastercard NuData, TypingDNA, Sardine, ThreatMark, Plurilock DEFEND — vendor documentation and press material as cited in Section 10.


# PART II — VERIFICATION AND FACT-CHECK AUDIT TRAIL

## 14. Verification Methodology

A second AI system (Gemini) reviewed an earlier version of the Part I research and proposed five specific additions, framed as "gaps." Rather than accepting either AI's framing at face value, each of the five claims was independently checked through a three-step process: first, searching for the actual primary or academic source rather than accepting the proposing AI's characterization of it; second, checking whether the "gap" was in fact already covered in the existing document (this caught one claim that was simply false — see Section 15); third, checking whether the evidence cited actually applied to behavioral biometrics specifically, or was borrowed uncaveated from an adjacent literature (this caught a real, load-bearing issue around facial-biometrics/deepfake statistics being applied to keystroke/mouse dynamics without flagging the difference — see Section 15.3). Separately from grading the proposed additions, an independent search was conducted for gaps neither AI had raised, on the reasoning that the goal is a defensible document, not simply a rebuttal of one AI's review by another.

This document (the compendium you are reading) goes one step further: every citation surfaced during that verification pass — including the ones that were already validated — was independently re-checked again during compilation, because a well-organized, confidently-written verification writeup can still carry forward subtle errors, and a competition-adjacent research document is exactly the wrong place for that to happen quietly. This second pass is what surfaced the Kravchik-and-Shabtai correction and the boiling-frog-terminology-origin hedge documented in Section 15.2 below — neither of which were flagged as issues in the original verification writeup, but both of which needed correction upon independent re-verification.

## 15. Item-by-Item Verification Findings

### 15.1 Confirmed clean — citable as originally given

| Claim | Verification outcome |
|---|---|
| Zibo Wang's Louisiana Tech PhD dissertation on Frog-Boiling attacks against keystroke authentication | Exact match. Title, advisor (Vir Phoha), year (2020), and the "Frog-Boiling attacks" terminology all confirmed directly from the dissertation's own abstract. This is the single cleanest, most precise citation in the entire adversarial-attacks section, because it was purpose-built for keystroke dynamics specifically. |
| Fang, Cao, Jia, and Gong's local model poisoning attacks against Byzantine-robust federated learning (USENIX Security 2020) | Exact match, and more concretely useful than the original summary indicated: the paper reports specific numbers, including a Krum-targeting attack that raises testing error from 0.14 to 0.58 (a 314% relative increase), with separate documented attacks against trimmed-mean and coordinate-wise-median aggregation as well. |
| Monaco, Ali, and Tappert's generative keystroke-latency spoofing model (IEEE BTAS 2015) | Exact match — the Linguistic Buffer and Motor Control (LBMC) model. |
| Migdal and Rosenberger's statistical modeling of synthetic keystroke-dynamics datasets (2019) | Exact match, including the specific finding that Gumbel-distributed (heavy-tailed) models best approximate real typing-timing behavior. |
| The 2022 screen-recording keystroke-timing extraction attack (Journal of Big Data) | Exact match, and the 64% figure is precisely right — it is specifically the evasion rate achieved against KeyTrac's commercial keystroke-dynamics authentication product, not a general or approximate figure. |
| NIST SP 800-63-4 | Confirmed finalized July 31, 2025, superseding SP 800-63-3, built around a continuous, risk-based Digital Identity Risk Management framework rather than the prior static, checklist-based assurance-level model. |

### 15.2 Confirmed real, but requiring a correction on top of the original framing

**Kravchik and Shabtai, "Can't Boil This Frog" (2020).** The paper itself is real, but it was originally proposed as "architecture-class precedent" for the vulnerability of online-retrained anomaly detectors to poisoning. Independent verification of the paper's own abstract found the actual result cuts the other way: testing an interpolation-based poisoning algorithm and a back-gradient-optimization poisoning algorithm against an online-trained autoencoder anomaly detector on the SWaT industrial-control-system dataset, the authors found the detector **resilient** across all ten relevant attacks tested, with poisoning success limited to a narrow, specific set of attack types and magnitudes. The corrected, defensible framing (as written into Section 8.3 above): this paper is a complicating data point demonstrating that the same broad architecture class (continuously retrained, unsupervised anomaly detection) is not universally vulnerable to this attack family — a more precise and more honest claim than treating it as a second confirmation of vulnerability alongside Zibo Wang's keystroke-specific result.

**The origin of the "boiling frog" terminology.** Originally attributed cleanly to Kloft and Laskov's centroid-anomaly-detection work. Independent verification found a competing, earlier, more literal source: Kim et al.'s 2009 paper, literally titled "The Frog-Boiling Attack," applied to an entirely different problem domain (secure network coordinate systems, not biometrics or intrusion detection). A clean single-origin claim could not be confirmed with confidence from available sources, so the corrected framing (Section 8.3) attributes the metaphor's traceable origin to at least Kim et al. (2009), credits Kloft and Laskov (2010/2012) with the rigorous security-bound analysis of the same underlying incremental-contamination-via-retraining mechanism in a security-detection context, and credits Zibo Wang (2020) with applying and renaming the attack family specifically for keystroke dynamics — rather than asserting any single one of the three as the definitive origin.

**GAN-based keystroke injection accuracy figures (93–96% fixed-text, 75–92% free-text).** These specific numbers are real and do appear in the literature, but independent verification could not confidently determine whether they represent the *attacker's* success rate at fooling an authentication system or the *defender's* detection accuracy at catching injected synthetic samples — both framings appeared attached to closely related papers in this specific research area, and the two claims are not equivalent. The corrected, hedged framing (Section 8.5): these figures should be understood as "reported performance figures in this range appear in both attack-generation and detection literature for GAN-based keystroke injection," rather than being asserted in one specific direction that was not independently confirmed from a primary source.

### 15.3 Confirmed real, with a nuance that strengthens rather than resolves the original caution

**ISO/IEC 30107 versus CEN/TS 18099 scoping.** The original proposal correctly distinguished Presentation Attack Detection (sensor-level, ISO/IEC 30107) from injection attack detection (pipeline-level, CEN/TS 18099), and correctly flagged that deepfake/injection statistics circulating in industry material (rates of deepfake-driven fraud, documented attack volumes against liveness checks) are drawn from facial-biometrics and KYC-onboarding contexts, not from keystroke or mouse dynamics. Independent verification of both standards' actual scope language found this caution to be, if anything, *understated* rather than overstated: every source checked for both standards frames them entirely around face, iris, voice, and fingerprint biometrics — photographs, masks, screen replay, deepfakes, virtual-camera injection into identity-verification pipelines. Neither standard has a keystroke- or mouse-dynamics-specific counterpart at present. This means the three-layer framing (presentation attack detection defends the sensor, deepfake/liveness detection defends the image, injection attack detection defends the pipeline) is accurate and useful as a conceptual organizing scaffold, but it is imported wholesale from the physiological-biometrics world rather than reflecting an existing behavioral-biometrics-native standard — a distinction worth stating explicitly rather than implying a keystroke/mouse-specific standard exists when it does not.

### 15.4 Correctly identified as false or already covered, and properly discarded

- **The feature-engineering critique** (that digraph/trigraph timing and jerk/derivative-based mouse features were "vastly oversimplified" in the original document) — false on inspection. The signal-taxonomy material (Section 4 of this compendium) already covers digraph/trigraph latency explicitly, and the adversarial-arms-race material (Section 8.2) already covers velocity/acceleration/jerk profiles in the specific context of Bézier-curve and Fitts's-Law-based bot evasion. This critique was evaluating content that was already present in the document it was reviewing.
- **The claim that jerk-based detection is "mathematically impossible for standard bot scripts to mimic smoothly"** — directly contradicted by the document's own adversarial-arms-race section, which already states plainly that sophisticated bots deliberately generate smooth, human-like jerk profiles specifically to evade this kind of detection (Section 8.2). This claim was not carried forward anywhere in this compendium.
- **Any deepfake or injection-attack statistic presented without the facial-versus-behavioral-biometrics caveat** — not wrong in isolation, but incomplete and potentially misleading if applied to this project's actual modality (keystroke/mouse dynamics) without the explicit caveat documented in Section 15.3 above.

### 15.5 Two additional findings that neither AI reviewer raised

**A dataset-honesty requirement.** Every public dataset available for this kind of research — CMU, Aalto/KVC, GREYC-NISLAB, GREYC-Keystroke, KeyRecs, Balabit, HMOG, HuMIdb, Clarkson I/II, and the others catalogued in Appendix C — is a generic academic research corpus collected from study volunteers under controlled or semi-controlled conditions, not real banking session telemetry. This should be stated plainly in any writeup ("evaluated against public academic behavioral-biometrics benchmarks as a proof of concept") before it is asked about, not after.

**Building the screen-recording attack as an actual demonstration, not just a citation.** This is comparatively cheap to build directly: record a subject typing, extract keystroke timing purely from the video footage, replay the extracted timing pattern against a locally trained keystroke-dynamics detector, and report the real evasion rate achieved. Doing this converts Section 8.5 from a literature summary into an original, reportable result — plausibly the single highest-leverage addition available given the amount of remaining implementation time on the underlying project, since the data-collection and model-training pipeline this project already requires (Section 6) is most of what such a demonstration would need anyway.

## 16. Final Revised Document Text

The following is the actual corrected/added prose, as integrated into Part I of this compendium, presented here as a standalone record of exactly what changed and why — useful if this document is ever compared against an earlier draft.

**Change 1 (Section 5.5 / 8.4 boundary — Byzantine-robust aggregation).** Original text stated that differential privacy plus message authentication codes "guarantee model integrity" for the FL-RBA² architecture. This is corrected: MACs guarantee an update was not altered *in transit*; they do nothing to address a legitimately-authenticated device sending a maliciously-crafted update, which is the actual Byzantine threat model in federated learning. The correction adds Krum/Multi-Krum, coordinate-wise median, and trimmed mean as the real defenses against this threat, with the Fang et al. counter-attack findings included so the defense is not overstated as unbreakable.

**Change 2 (Section 8.6 — dataset honesty, new material).** Added in full as Section 8.6, stating plainly that every available dataset is an academic benchmark corpus, not banking telemetry, and specifying the exact non-misleading phrasing to use when describing any evaluation built on these datasets.

**Change 3 (Section 8.3 — poisoning of continuously-retrained baselines, new subsection).** Added in full, directly naming the tension between Section 7.5's baseline-adaptation recommendation and the documented poisoning-attack class that exploits exactly that mechanism, with the Zibo Wang, Lovisotto/Eberz/Martinovic, and (corrected) Kravchik/Shabtai citations, and specifying a residue-feature or RONI-style validation gate as the required mitigation.

**Change 4 (Section 8.5 — injection attacks, rewritten).** The original "Generative AI Spoofing" material is replaced with a version that correctly scopes ISO/IEC 30107 versus CEN/TS 18099, flags the facial-versus-behavioral modality caveat explicitly, cites the keystroke-specific injection literature (Monaco/Ali/Tappert, Migdal/Rosenberger, the screen-recording attack) with the GAN-figure ambiguity hedged rather than asserted, and ends with the screen-recording demo build recommendation.

**Change 5 (Section 9.3 — standards citation swap).** NIST SP 800-207 (general Zero Trust Architecture) is replaced as the primary citation with NIST SP 800-63-4 (finalized 2025), which is both more current and a sharper conceptual match to a continuously risk-scored authentication system specifically, while 800-207 is retained as a secondary, more general reference.

**Change 6 (Section 6, corollary note on edge computing).** A brief note is retained on-device/TinyML deployment of keystroke, gait, and touchscreen models as a bandwidth-driven corollary of the federated-learning approach already covered in depth — relevant specifically because Nepal Rastra Bank's regulatory jurisdiction covers substantial low-bandwidth terrain outside Kathmandu, making this more than a purely theoretical consideration for this specific project's deployment context.

## 17. Lessons on Citation Rigor

Three general lessons from this verification process are worth carrying forward into any future citation-formal work on this project (the IEEE-adjacent appendix mentioned as a possible future need, in particular):

1. **A paper being real does not mean its finding supports the claim it is being cited for.** The Kravchik-and-Shabtai case is the clearest example in this entire audit: the paper exists, is correctly titled, and is topically relevant — but its actual finding is closer to the opposite of how it was originally being used. Checking that a citation exists is a necessary but not sufficient verification step; checking what the citation's source actually concludes is the step that is easy to skip and expensive to get wrong.
2. **Terminology origin claims are especially easy to get subtly wrong, because later literature often retroactively attributes a term to an earlier work that didn't use that exact phrase.** The "boiling frog" terminology case shows this precisely: a citing paper attributes the term to Kloft and Laskov, but an earlier, more literal use of the specific phrase exists in an unrelated problem domain (Kim et al., network coordinate systems). When a document needs to state where a term "comes from," the honest move is very often to name multiple contributing sources with their actual individual contributions, rather than asserting a single clean origin.
3. **Statistics from an adjacent modality need an explicit caveat, not silent omission, when reused.** The ISO/IEC 30107/CEN-TS-18099 case shows that the correct response to "this evidence is from a different modality" is not to drop the framing entirely, but to keep the useful conceptual structure while stating plainly, in the actual prose, that the standards and statistics originate from facial biometrics rather than implying — through simple silence on the point — that they were developed for or validated against keystroke/mouse dynamics.

One honesty note carried forward from the original verification process and worth restating here: the original verification pass was itself conducted through web search rather than full-paper reading. The citations in this compendium are solid enough to build on for a research-informed engineering decision, but if any of this material is used in something citation-formal — the IEEE-adjacent appendix, a competition report requiring exact page numbers — the primary sources should be pulled directly for exact author lists, page numbers, and DOIs before finalizing, rather than relying on this compendium's citations as camera-ready.


# PART III — DEPENDENCY AND LICENSING REFERENCE

## 18. Scope and Methodology

This part covers every library and platform component relevant to building Clickstream: the client-side capture layer, the streaming pipeline, the classical and deep-learning models from Section 5, and the federated-learning/differential-privacy extensions from Sections 5.5–5.6. Every license claim below was checked against the project's own license file, its PyPI/GitHub listing, or (for platform-level components) the vendor's own terms/EULA — not stated from general recollection. Legend: ✅ = OSI-recognized open-source license, free at any scale. ⚠️ = free to use, but not open source by any standard definition — flagged explicitly because the project's requirement was both properties simultaneously, not just absence of cost.

## 19. Core Data and ML Pipeline

| Library | License | Status | Role in Clickstream |
|---|---|---|---|
| pandas | BSD-3-Clause | ✅ | Loading and windowing raw keystroke/mouse event streams into per-session feature tables |
| numpy | BSD-3-Clause | ✅ | Underlying array operations for every other library in this table |
| scipy | BSD-3-Clause | ✅ | Statistical distance metrics for anomaly scoring; distribution fitting for synthetic-data work (Section 8.5) |
| scikit-learn | BSD-3-Clause | ✅ | Isolation Forest and One-Class SVM implementations (Section 5.2), preprocessing (`StandardScaler`), evaluation metrics (`roc_auc_score`, `precision_recall_curve`) |
| matplotlib | Matplotlib License (BSD-style) | ✅ | EER/ROC/PR curve plots for model evaluation (Section 5.7, Appendix D) |
| seaborn | BSD-3-Clause | ✅ | Feature-distribution visualization during exploratory analysis of dwell/flight-time and mouse-trajectory features |

**Minimal install:**
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

**Illustrative integration — computing EER from a trained Isolation Forest's scores, made concrete (extending the `PerUserBehavioralModel` class from Section 5.2):**
```python
import numpy as np
from sklearn.metrics import roc_curve

def compute_eer(genuine_scores, impostor_scores):
    """
    genuine_scores: trust scores from held-out GENUINE sessions
    impostor_scores: trust scores from held-out IMPOSTOR sessions
                      (a different enrolled user's sessions, scored
                      against this user's model)
    """
    labels = np.concatenate([
        np.ones(len(genuine_scores)),   # 1 = genuine
        np.zeros(len(impostor_scores)), # 0 = impostor
    ])
    scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr  # false negative rate = FRR
    # EER is where FPR (≈ FAR) and FNR (≈ FRR) curves cross
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]
    return eer, eer_threshold
```

## 20. GPU Acceleration Layer

| Component | License | Status | Notes |
|---|---|---|---|
| NVIDIA CUDA Toolkit | Proprietary NVIDIA EULA | ⚠️ Free, **not** open source | Required for GPU-accelerated training of the deep-learning models in Section 22 (PyTorch/TensorFlow). NVIDIA's own EULA explicitly states the SDK may not be used "in any manner that would cause it to become subject to an open source software license" — an unambiguous non-OSS flag by NVIDIA's own design, not an oversight. No open-source substitute exists that provides CUDA-specific acceleration; ROCm (AMD) and oneAPI (Intel) are open alternatives, but only for their own hardware. |

**Checking GPU/CUDA availability before committing to GPU-specific code paths (a defensive pattern worth building in, since Colab's free-tier GPU allocation is not guaranteed):**
```python
import torch

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()
model = KeystrokeEncoder().to(device)  # from Section 5.3
```

## 21. Notebook and Cloud Environment

| Component | License | Status | Notes |
|---|---|---|---|
| Google Colab | Proprietary Google service | ⚠️ Free, **not** open source | Free of charge, but a closed, hosted product — not something that can be inspected or self-hosted. |
| Jupyter / IPython | BSD-3-Clause | ✅ | The actual open-source notebook engine underneath; relevant if the project ever moves off Colab onto self-hosted infrastructure. |
| PyDrive2 | Apache 2.0 | ✅ | Google Drive persistence, actively maintained by the Iterative/DVC team. |
| google-api-python-client | Apache 2.0 | ✅ | Underlying Google API client that PyDrive2 wraps. |

**Illustrative Drive persistence pattern used in this project's existing pipeline:**
```python
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # or gauth.CommandLineAuth() in a headless Colab cell
drive = GoogleDrive(gauth)

def save_model_to_drive(local_path, drive_folder_id):
    f = drive.CreateFile({'parents': [{'id': drive_folder_id}]})
    f.SetContentFile(local_path)
    f.Upload()
    return f['id']
```

## 22. Behavioral Biometrics — Deep Learning and Vision

| Library | License | Status | Purpose |
|---|---|---|---|
| PyTorch | Modified BSD-3-Clause | ✅ | Siamese networks, Transformer encoders, CNN-BiLSTM (Section 5.3) |
| TensorFlow / Keras | Apache 2.0 | ✅ | Alternative deep-learning backend, and required if using TensorFlow Federated or TensorFlow Privacy (Sections 24–25) |
| OpenCV | Apache 2.0 (v4.5+) / BSD-3-Clause (v4.4 and earlier) | ✅ | Screen-recording demo (Section 8.5): frame extraction and video processing |

**Minimal install:**
```bash
pip install torch torchvision opencv-python
# or, for the TensorFlow-based path:
pip install tensorflow opencv-python
```

**Illustrative frame-extraction starting point for the screen-recording demo (Section 8.5):**
```python
import cv2

def extract_frame_timestamps(video_path, target_fps=30):
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval_ms = 1000.0 / source_fps
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_ms = frame_idx * frame_interval_ms
        frames.append((frame, timestamp_ms))
        frame_idx += 1
    cap.release()
    return frames
    # Next step: detect key-press visual cues (finger contact events,
    # or on-screen keyboard highlight changes if applicable) per frame
    # to reconstruct approximate keydown/keyup timestamps, then feed
    # those into the same extract_keystroke_features() function from
    # Section 6.2 -- this is exactly why the demo is cheap to build:
    # it reuses the project's existing feature-extraction code.
```

## 23. Streaming Architecture

| Component | License | Status | Notes |
|---|---|---|---|
| Apache Kafka (core) | Apache 2.0 | ✅ | The broker itself |
| Apache Flink | Apache 2.0 | ✅ | Stream-processing engine |
| kafka-python | Apache 2.0 | ✅ | Pure-Python client, simplest to set up |
| confluent-kafka-python | Apache 2.0 | ✅ | Faster client (wraps librdkafka); free/open despite Confluent maintaining it |
| Confluent Platform extras (ksqlDB, Schema Registry, REST Proxy, Confluent Server) | Confluent Community License or Confluent Enterprise License | ⚠️ Different license than Kafka itself | Not needed unless Confluent's specific managed tooling is wanted. The Community License carries a "no competing SaaS" clause and is not OSI-approved. Sticking to vanilla Apache Kafka plus either client library above avoids this row entirely. |

**A minimal `docker-compose.yml` for local development (vanilla, Apache-2.0-only components):**
```yaml
version: '3.8'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.6.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
  kafka:
    image: confluentinc/cp-kafka:7.6.0
    depends_on: [zookeeper]
    ports: ["9092:9092"]
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
  flink-jobmanager:
    image: flink:1.18
    ports: ["8081:8081"]
    command: jobmanager
    environment:
      FLINK_PROPERTIES: "jobmanager.rpc.address: flink-jobmanager"
  flink-taskmanager:
    image: flink:1.18
    depends_on: [flink-jobmanager]
    command: taskmanager
    environment:
      FLINK_PROPERTIES: "jobmanager.rpc.address: flink-jobmanager"
```
Note: the `confluentinc/cp-kafka` Docker *images* are convenience packaging of the Apache-2.0 Kafka broker; they are not the same thing as the Confluent Platform *enterprise features* flagged in the table above (ksqlDB, Schema Registry, etc.), which are not included in this compose file at all.

**Minimal install for the Python client side:**
```bash
pip install confluent-kafka
# or the pure-Python alternative:
pip install kafka-python
```

## 24. Federated Learning

| Library | License | Status |
|---|---|---|
| Flower (flwr) | Apache 2.0 | ✅ |
| PySyft (OpenMined) | Apache 2.0 | ✅ |
| TensorFlow Federated | Apache 2.0 | ✅ |

**Minimal install:**
```bash
pip install flwr
```

Flower is the recommended starting point of the three for this project specifically: it is framework-agnostic (works with either PyTorch or TensorFlow models, unlike TensorFlow Federated which requires TensorFlow specifically), has the most active current development and community support of the three, and its `Strategy` abstraction (see the `FedAvg` example in Section 5.5) is where a custom Byzantine-robust aggregation rule (Section 8.4) would actually be implemented — Flower ships `fl.server.strategy.Krum` and can be extended with a custom trimmed-mean or coordinate-wise-median strategy if the built-in options don't fit exactly.

## 25. Differential Privacy

| Library | License | Status |
|---|---|---|
| Opacus (PyTorch) | Apache 2.0 | ✅ |
| TensorFlow Privacy | Apache 2.0 | ✅ |
| IBM diffprivlib | MIT | ✅ |

**Minimal install (choosing the PyTorch path, to pair with the PyTorch-based models in Section 22):**
```bash
pip install opacus
```

**Illustrative DP-SGD wrapping of a standard PyTorch training loop:**
```python
from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)
# training loop proceeds as normal from here; privacy accounting
# (the epsilon/delta budget) is tracked automatically by the engine
```

## 26. Experiment Tracking

| Tool | License | Status | Notes |
|---|---|---|---|
| MLflow | Apache 2.0 | ✅ | Fully self-hostable, zero licensing cost at any scale |
| Weights & Biases | Proprietary | ⚠️ Generous free tier, **not** open source | Commercial SaaS; self-hosting ("W&B Server") requires a separate enterprise contract. MLflow is the option that actually satisfies "open source and free" without caveats. |

**Minimal install and local tracking setup:**
```bash
pip install mlflow
```
```python
import mlflow

mlflow.set_tracking_uri("file:./mlruns")  # fully local, no external service at all
mlflow.set_experiment("clickstream")

with mlflow.start_run():
    mlflow.log_param("model", "isolation_forest")
    mlflow.log_metric("eer", eer_score)
    mlflow.log_metric("pr_auc", pr_auc_score)  # per Section 5.7's imbalance caveat
    mlflow.sklearn.log_model(model, "model")
```

## 27. Testing and Development Tooling

| Tool | License | Status |
|---|---|---|
| pytest | MIT | ✅ |

**Minimal install:**
```bash
pip install pytest
```

## 28. Licensing Compliance Notes

Restating plainly, since this was an explicit project requirement rather than a nice-to-have: **NVIDIA CUDA Toolkit**, **Google Colab**, and **Weights & Biases** are all free of charge but not open source under any standard definition. CUDA's own license explicitly forbids being open-sourced; Colab is a closed Google-run service with no self-hosting option; W&B is commercial SaaS with an optional paid self-hosted tier. None has a drop-in open-source replacement that does exactly the same job — a local GPU plus open-source Jupyter plus MLflow gets closest, minus Colab's free hosted compute specifically. Every other library in Sections 19–27 is confirmed OSI-recognized open source at no cost.

## 29. Full Environment Setup Guide

**`requirements.txt`** (CPU-portable; the PyTorch CUDA build is a separate, environment-specific install step since it needs to match the local CUDA driver version):

```text
# Core data & ML pipeline
pandas>=2.2
numpy>=1.26
scipy>=1.13
scikit-learn>=1.5
matplotlib>=3.9
seaborn>=0.13

# Notebook / cloud persistence
pydrive2>=1.19
google-api-python-client>=2.140

# Behavioral biometrics -- deep learning & vision
torch>=2.3
torchvision>=0.18
opencv-python>=4.10

# Streaming
confluent-kafka>=2.5

# Federated learning
flwr>=1.9

# Differential privacy
opacus>=1.5

# Experiment tracking
mlflow>=2.15

# Testing
pytest>=8.2
```

```bash
pip install -r requirements.txt --break-system-packages  # if running outside a venv, e.g. inside Colab
```

**GPU-specific step (run only in an environment with an NVIDIA GPU and a matching CUDA driver already installed):**
```bash
# Verify the installed PyTorch build actually has CUDA support before
# relying on it -- Colab occasionally serves a CPU-only build after
# an environment reset.
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

**Apache Kafka + Flink local development stack:** use the `docker-compose.yml` in Section 23 — `docker compose up -d` brings up a local, vanilla, fully-Apache-2.0-licensed broker plus stream-processing engine for development and testing before anything touches production infrastructure.


# APPENDICES

## Appendix A — Glossary of Terms

**AitM (Adversary-in-the-Middle)** — An attack where the adversary sits as a reverse proxy between a victim and a legitimate service, relaying real traffic (including MFA challenges) while stealing the resulting authenticated session token. Distinct from classic phishing, which steals only credentials. See Section 3.1.

**ATO (Account Takeover)** — Fraud in which an attacker gains unauthorized control of a legitimate user's account, whether via AitM, credential stuffing, SIM swap, or other vectors. See Section 3.3 for current loss figures.

**Behavioral biometrics** — Authentication based on *how* a person interacts with a device (typing rhythm, cursor movement, touch pressure) rather than a static physical trait. See Section 2.1.

**Biometric backdoor** — The biometrics-native term for a poisoning attack against unsupervised template updating, per Lovisotto, Eberz, and Martinovic (2020). See Section 8.3.

**Byzantine-robust aggregation** — Federated-learning aggregation rules (Krum, trimmed mean, coordinate-wise median) designed to tolerate a bounded number of malicious participants sending deliberately corrupted updates. See Section 8.4.

**CEN/TS 18099** — A 2024 European technical specification for Biometric Data Injection Attack Detection, scoped to face/iris/voice/fingerprint biometrics. See Section 8.5.

**Cold start** — The problem of scoring a brand-new user's session before enough historical data exists to build a personal behavioral baseline. See Section 7.1.

**Concept drift / behavioral drift** — Gradual, legitimate change in a user's behavioral pattern over time (new device, minor injury, skill improvement) that a system must tolerate without either ignoring genuine attacks or locking out genuine users. See Section 7.3 and Section 11.3.

**Continuous authentication** — Authentication re-evaluated on a rolling basis throughout an entire session, rather than once at login. See Section 2.2.

**Digraph / trigraph latency** — The timing between two (digraph) or three (trigraph) specific consecutive keys in a keystroke sequence, more individually distinctive than raw dwell/flight time alone. See Section 4.2.

**Differential privacy (DP)** — A formal mathematical framework providing a provable bound on how much any single training example could have influenced a trained model's output, typically implemented via DP-SGD (noised, clipped per-example gradients). See Section 5.6.

**DIRM (Digital Identity Risk Management)** — The continuous, risk-based framework introduced by NIST SP 800-63-4 (2025), replacing the prior static, checklist-based authenticator assurance model. See Section 9.3.

**Dwell time / flight time** — Dwell time is how long a single key is held down; flight time is the gap between releasing one key and pressing the next. The two foundational features of keystroke-dynamics research. See Section 4.2.

**EER (Equal Error Rate)** — The operating threshold at which false-acceptance rate equals false-rejection rate; the standard headline accuracy figure in biometrics research, though not necessarily where a production system should actually operate. See Section 5.7 and Appendix D.

**FAR / FRR (False Acceptance Rate / False Rejection Rate)** — FAR is the proportion of impostor sessions wrongly accepted; FRR is the proportion of genuine sessions wrongly rejected. See Section 5.7 and Appendix D.

**Federated learning (FL)** — A distributed training approach where raw data stays on each client device and only model updates are shared with a central aggregator, structurally avoiding centralization of sensitive data. See Section 5.5.

**FIDO2 / Passkey** — A cryptographic authentication standard combining possession (a registered authenticator) and inherence (a biometric or PIN unlocking it on-device), commonly used as the step-up challenge triggered by a continuous-authentication system. See Section 6.6 and Section 9.1.

**Frog-Boiling attack** — A poisoning attack (named by Zibo Wang, 2020, for keystroke authentication specifically) that crafts slow, incrementally-perturbed update samples designed to bypass classifier detection during baseline retraining. See Section 8.3.

**HMOG (Hand Movement, Orientation, and Grasp)** — A mobile behavioral-biometrics feature set combining accelerometer, gyroscope, and magnetometer data with tap timing. See Section 4.4 and Appendix C.

**Hill-climbing attack** — A black-box evasion technique where an adversary iteratively refines a synthetic biometric sample using repeated match-score feedback until it crosses the acceptance threshold. See Section 8.1.

**Isolation Forest** — An unsupervised anomaly-detection algorithm that isolates data points via random feature splits; anomalies require fewer splits to isolate than normal points. The default classical choice for one-class behavioral-biometrics scoring. See Section 5.2.

**ISO/IEC 30107** — The international standard family for Presentation Attack Detection (PAD), scoped to attacks at the physical sensor during live presentation (photos, masks, video replay). See Section 8.5.

**MCC (Matthews Correlation Coefficient)** — A single-number classification-quality metric that remains informative under class imbalance, unlike raw accuracy. See Section 5.7 and Appendix D.

**Non-IID (non-independent, identically distributed)** — A federated-learning challenge where each client's local data distribution differs substantially from every other client's; especially severe in behavioral biometrics because each client only ever has one person's genuine-behavior examples. See Section 5.5.

**One-Class SVM** — An unsupervised classifier that learns a boundary enclosing the training data's region of feature space, flagging anything outside it as anomalous. See Section 5.2.

**PAD (Presentation Attack Detection)** — See ISO/IEC 30107.

**Puppet attack** — An attack where the adversary forcibly uses the victim's own genuine input (coercion, or acting while the victim is unaware), defeating behavioral detection because the biometric trait really is genuine. See Section 8.1.

**PR-AUC (Precision-Recall Area Under Curve)** — The area under the precision-recall curve; more informative than ROC-AUC specifically under class imbalance, which is the normal condition in fraud detection. See Section 5.7 and Appendix D.

**RONI (Reject on Negative Impact)** — A defense strategy that validates a candidate training update against its effect on model performance before accepting it, used as a mitigation against poisoning attacks on continuously-retrained models. See Section 8.3.

**SCA (Strong Customer Authentication) / TRA (Transaction Risk Analysis)** — PSD2's mandated two-of-three-factor authentication requirement (SCA), and the risk-scoring pathway (TRA) that can exempt low-risk transactions from requiring it. Behavioral biometrics feeds TRA scoring rather than serving as a recognized SCA factor itself. See Section 9.1.

**Siamese network / triplet loss** — A neural-network architecture that learns to embed inputs such that same-identity examples land close together and different-identity examples land far apart, trained via triplet loss (anchor, positive, negative). See Section 5.3.

**Trust score** — The continuously-updating output of a behavioral-biometrics model, summarizing how consistent recent session behavior is with a user's enrolled baseline.

**Zero Trust Architecture (ZTA)** — A security model that treats no session or device as inherently trusted, continuously re-evaluating risk throughout access rather than granting persistent trust after a single login. See Section 9.3.

## Appendix B — Additional Reference Resources

Section 13 is the canonical bibliography for Part I; this appendix is not a duplicate of it. It instead lists standards documents and dataset-access points as direct resources, since "citation" and "where do I actually go get this" are different needs.

**Standards documents (official bodies, not academic citations):**
- ISO/IEC 30107 (Presentation Attack Detection) — published by ISO/IEC JTC 1/SC 37.
- CEN/TS 18099 (2024, Biometric Data Injection Attack Detection) — published by CEN (European Committee for Standardization).
- NIST SP 800-63-4 — published by the U.S. National Institute of Standards and Technology, publicly available at NIST's own Special Publications archive.
- Nepal Rastra Bank IT Guidelines (2012) and Cyber Resilience Guidelines (2023) — published directly by Nepal Rastra Bank.

**Framework/library documentation (for the Part III dependency stack):**
- Flower: flower.ai and its GitHub repository.
- Opacus: opacus.ai and its GitHub repository.
- MLflow: mlflow.org.
- Apache Kafka and Apache Flink: kafka.apache.org and flink.apache.org, both under the Apache Software Foundation.

## Appendix C — Dataset Comparison Table

Consolidating every dataset referenced anywhere in this compendium, including several surfaced only during the additional research pass for this appendix, with access details where confirmed.

| Dataset | Modality | Subjects | Samples/scale | Conditions | Access |
|---|---|---|---|---|---|
| CMU Keystroke Benchmark (Killourhy & Maxion, 2009) | Keystroke, fixed-text | 51 | 400 entries/subject, single password (".tie5Roanl") | Controlled, same-text | Publicly available; also mirrored in ready CSV form on keystroke.fr |
| GREYC-Keystroke | Keystroke, fixed-text | 133 | 5–107 entries/subject | Controlled, same-text | Available via Christophe Rosenberger's lab site; mirrored on keystroke.fr |
| GREYC-NISLAB | Keystroke, fixed-text | 110 | 20 entries/subject | Same-text, includes handedness/age/gender metadata; last 10 entries per subject typed one-handed | Available to the international research community; mirrored on keystroke.fr |
| KeyRecs | Keystroke | Combined with CMU/GREYC-NISLAB in fusion studies | ~54,000 password samples across the combined corpus | Fixed-text | Referenced in 2024–2025 fusion-methodology papers |
| Aalto Desktop/Mobile Keystroke Databases (KVC-onGoing) | Keystroke, free-text | Large-scale, exact N not restated here | Current SOTA benchmark, EER 3.33% (desktop) / 3.61% (mobile) | Free-text, both device classes | Hosted as an ongoing challenge on CodaLab |
| Balabit Mouse Dynamics Challenge | Mouse | 10 | Remote-desktop session recordings, split into training/testing folders | Naturalistic, remote-desktop context | Publicly available on GitHub (balabit/Mouse-Dynamics-Challenge) |
| DFL Dataset | Mouse | 21 | Multi-device (desktop, laptop, external mouse, touchpad) | Naturalistic, daily-activity background monitoring | Referenced in 2019 comparative studies |
| Cho/Shen Dataset | Mouse | 28 | Background-monitoring application capture | Naturalistic | Referenced in 2012-origin comparative studies |
| HMOG (Sitová et al., 2016) | Touch + accelerometer/gyroscope/magnetometer | 100 | 8 keystroke-typing sessions/subject, minimum 250 characters/question, 3 questions/session | Both sitting and walking conditions | Published alongside the IEEE TIFS paper; widely used as a mobile-continuous-auth benchmark |
| HuMIdb | Multi-sensor mobile | Not restated here | 5 GB total | Unsupervised, naturalistic mobile interaction | Publicly accessible |
| Clarkson I | Keystroke + mouse + video | 39 | 21,533 samples | Free & fixed text; uniquely includes video of subjects' facial expressions and hand movements | Available upon request from the original authors |
| Clarkson II (Murphy et al.) | Keystroke + mouse + active-window context | 103 | 12.9 million total keystrokes, ~125,000/subject average | Free-text, ~2.5 years of real personal-computer use per subject — the most naturalistic, longest-duration dataset in this table | Referenced in keystroke-dynamics survey literature |
| BEACON | Multimodal gameplay-derived | Not restated here | Minecraft-gameplay-derived behavioral fingerprinting | Naturalistic, high-volume mouse movement from gameplay | Referenced in a 2026 arXiv paper |

**The dataset-honesty point from Section 8.6 applies to every row in this table without exception: all are academic research corpora, none are banking session telemetry.**

## Appendix D — Metrics Reference

**FAR (False Acceptance Rate):**
```
FAR = (number of impostor sessions incorrectly accepted) / (total impostor sessions evaluated)
```

**FRR (False Rejection Rate):**
```
FRR = (number of genuine sessions incorrectly rejected) / (total genuine sessions evaluated)
```

**EER (Equal Error Rate):** the threshold value at which FAR(threshold) = FRR(threshold), read directly off the point where the FAR curve and FRR curve intersect when both are plotted against decision threshold.

**Worked EER illustration.** Suppose a system's decision threshold sweeps from 0 to 100, and the following FAR/FRR pairs are measured at several thresholds:

| Threshold | FAR | FRR |
|---|---|---|
| 40 | 12% | 2% |
| 50 | 6% | 5% |
| 55 | 5% | 5.4% |
| 60 | 3% | 9% |

The EER sits at approximately threshold 52-53, where the FAR and FRR curves cross around 5.2-5.4% — this single crossing value is what gets reported as "EER = ~5.3%" in a paper, even though a production deployment would very likely not operate at this exact threshold (see below).

**Why production systems deliberately operate away from the EER point.** A wrongly-accepted impostor session (a false accept) is a fraud loss; a wrongly-rejected genuine session (a false reject) is a customer-friction and support cost. These two error types are essentially never weighted equally by a bank's actual risk tolerance — fraud losses are typically far more costly per-incident than the cost of one extra step-up challenge for a legitimate user. This means a production system typically moves the operating threshold well past the EER point, deliberately accepting a higher FRR in exchange for a much lower FAR, rather than using the "balanced" EER threshold a research paper would report.

**ROC-AUC:** the area under the curve plotting True Positive Rate (= 1 − FRR) against False Positive Rate (= FAR) across every possible threshold. Ranges from 0.5 (no better than random) to 1.0 (perfect separation).

**PR-AUC:** the area under the curve plotting Precision against Recall across every threshold, where:
```
Precision = True Positives / (True Positives + False Positives)
Recall = True Positives / (True Positives + False Negatives)
```
PR-AUC is the more operationally honest metric under class imbalance because, unlike ROC-AUC, it is directly sensitive to how rare the positive class actually is — see the worked illustration in Section 5.7 (99.20% ROC-AUC dropping to 0.842 PR-AUC under a realistic 10,000-genuine-to-10-fraud ratio).

**MCC (Matthews Correlation Coefficient):**
```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```
Ranges from −1 (total disagreement) to +1 (perfect prediction), with 0 representing performance no better than random guessing given the actual class balance — unlike raw accuracy, MCC does not give an inflated-looking score simply because the negative class is huge, which is exactly why it is reported alongside PR-AUC as a more conservative companion metric under imbalance (Section 5.7).

**Time-to-detect:** the elapsed time (or elapsed keystroke/mouse-event count) between a session actually becoming fraudulent and the trust score crossing an enforcement threshold — a metric unique to continuous authentication with no equivalent in point-in-time systems, and, in practice, often at least as operationally important as EER itself (Section 5.7).

---

*End of compendium. This document consolidates the continuous-behavioral-biometrics research, the independent verification audit, and the open-source dependency reference produced across this research thread. Every citation and license claim was checked against a primary or near-primary source at the time of writing; where verification could not produce a confident answer, that uncertainty is stated explicitly rather than smoothed over.*
