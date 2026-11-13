# relay-lane: frontend prompt pack

Paste these into your IDE assistant in order. Each prompt builds on the one before it. Keep the real conversation, mistakes included, because the brief asks for it.

## Read this first

**Is a frontend required?** No. The brief grades your measurements, your reasoning and your write-up. A frontend is a showcase, and it helps most in the live conversation, where you will be asked to change your system and explain results. Time-box it to about half a day, and only start it once the pipeline, stand-in and `results.csv` work.

**Three rules that keep it from hurting you**
1. **No invented numbers.** Every figure the frontend shows about performance must come from `results/results.csv` or the trace log. Reviewers recompute from your raw file, so a pretty chart that disagrees with Table 2 is worse than no chart.
2. **Default to the stand-in.** Live demos must not burn your Groq daily quota. Real mode is behind an explicit toggle with a visible request counter and a hard cap.
3. **Nothing in the frontend is a second pipeline.** It only displays what the backend reports. If the UI shows a step, the trace recorded it.

## The idea

relay-lane is a pipeline that passes one question through five steps, under limits. The frontend treats that literally: **an indoor athletics track**. Each pipeline step is a lane. Each request is a baton moving through the lanes. Limits are where runners get held at the line. A finished request leaves a **photo-finish strip**, a single image of where its time went.

This is not decoration. Lanes numbered 1 to 5 are legitimately a sequence (understand, retrieve, decide, draft, verify), and the photo-finish strip is exactly what Section 4 of the write-up asks you to show: waiting, working, retrying, before or after the first token.

### Design tokens (give these to the assistant, do not let it pick its own)

**Color**

| Name | Hex | Role |
|---|---|---|
| Mondo blue | `#1B3FA0` | Main ground, like an indoor track surface |
| Infield | `#0C1A47` | Deep surfaces, the answer panel |
| Lane white | `#EEF2F8` | Lane lines, text on blue |
| Baton yellow | `#FFD21F` | The live request, first token, primary action |
| False-start red | `#E03A3E` | Errors, retries, rate limits |
| Split green | `#3DAA7D` | Verified citations, success |

Not used on purpose: cream and terracotta, near-black with acid green, grey SaaS cards.

**Type**
- **Anybody** (Google Fonts, variable width 50 to 150 and weight 100 to 900) for headlines, lane names and numerals. Its width axis is the signature: headline width stretches as queue depth grows, so load is visible in the typography itself.
- **Source Serif 4** for answers and passages from the papers. Serif body gets line-height around 1.6 and a measure under 70 characters. It makes the cited text feel like the papers, not the interface.
- No monospace for labels. Tabular numerals from Anybody (`font-variant-numeric: tabular-nums`) for timings.

**Layout principles**
- Left-aligned everywhere. Wide horizontal lanes, not centered cards.
- One memorable element: the track. Everything else stays quiet.
- Structure carries information: lane lines are real lane lines, numbering is the real step order, the vertical "first token" line on every strip means something.
- No eyebrow labels, no all-caps tracked labels, no arrows on buttons, no gradient washes, no identical rounded-card grids.

---

## Prompt 0: Design plan before any code

```
We are building the frontend for "relay-lane", a Python asyncio pipeline that
answers questions from three arXiv papers under rate limits. Before writing
code, write a short design plan and show it to me.

Concept: an indoor athletics track. Pipeline steps are lanes (1 understand,
2 retrieve, 3 decide, 4 draft, 5 verify). A request is a baton moving through
lanes. Rate limits hold the baton at a line. Each finished request leaves a
"photo-finish strip": its total time split into waiting, working and retrying,
with a vertical line marking first token.

Use exactly these tokens:
Color: Mondo blue #1B3FA0 (ground), Infield #0C1A47 (deep surface),
Lane white #EEF2F8 (lines, text), Baton yellow #FFD21F (live request, first
token, primary action), False-start red #E03A3E (errors, retries, limits),
Split green #3DAA7D (verified citations).
Type: Anybody (variable width axis, headlines/lane names/numerals), Source
Serif 4 (answers and passages from papers). No monospace labels.

Avoid: cream background with terracotta accent, near-black with acid green,
broadsheet hairline columns, grids of identical rounded cards with soft grey
shadows, gradient washes, tracked ALL-CAPS eyebrow labels, "A · B · C" meta
strings, arrows appended to buttons, accenting one word of a headline.

Deliver: (1) the token system as CSS custom properties, (2) an ASCII
wireframe of each of the four views below, (3) a one-paragraph critique of the
plan asking "would I get the same design for any other pipeline project?" and
revise anything that is generic. Do not write code until I approve.

Views: Track (live ask), Heats (load-test results), Limits (budgets and the
degradation ladder), Papers (source passages).
```

*Check: if the plan comes back with cards and gradients anyway, push back before approving.*

## Prompt 1: Scaffold and tooling

```
Set up the frontend in frontend/ inside the existing relay-lane repo
(Windows, Python 3.11 backend). Use Vite + React + TypeScript. Dependencies:
react, react-router-dom, nothing else for UI. No Tailwind, no component
library, no chart library: all charts and the track are hand-written SVG.
Plain CSS files with the custom properties from the approved plan.

Add:
- frontend/index.html loading Anybody (wght and wdth axes) and Source Serif 4
  from Google Fonts with sensible fallback stacks
- src/styles/tokens.css, base.css, and one CSS file per view
- vite.config.ts proxying /api to http://localhost:8000
- a root README section "Frontend" with: npm install, npm run dev,
  npm run build, and how FastAPI serves frontend/dist in production
- package.json scripts for dev, build, typecheck

Routes: / (Track), /heats, /limits, /papers. A persistent left rail holds the
wordmark "relay-lane" and the four links. Ask me before adding any dependency.
```

## Prompt 2: Backend API the frontend needs

```
Add relay/api.py (FastAPI) exposing the pipeline to the frontend. Do not
change pipeline logic; read from the existing trace object.

Endpoints:
- POST /api/ask  body {question, source: "stand-in"|"real", retrieval_setting}
  Streams Server-Sent Events. Event types, each with a timestamp in ms from
  request start:
    accepted {request_id, queue_position}
    step_start {step, lane}
    step_end {step, lane, model_ms, tokens_in, tokens_out}
    limiter_wait {ms, reason: "rpm"|"tpm"|"concurrency"|"daily"}
    retry {attempt, backoff_ms, reason}
    degrade {rung, reason}      (e.g. fewer chunks, skip verify LLM pass)
    first_token {}
    token {text}
    final {status, answer, citations[], refusal_reason, grade_hint}
- GET /api/limits       current limiter state: RPM/TPM/RPD used and remaining,
                        in-flight, queue length, resets_at
- GET /api/results/summary   per-scenario and per-retrieval-setting stats
                        computed from results/results.csv using the same
                        functions as analysis/make_tables.py (import them,
                        do not duplicate)
- GET /api/results/requests?scenario=   rows for the per-request view
- GET /api/trace/{request_id}           full trace for one request
- GET /api/chunks/{chunk_id}            chunk text, paper_id, version, page
- GET/POST /api/standin/config          read and update the stand-in config
- Mount frontend/dist as static files at / when it exists.

Rules: real-source asks must respect a server-side hard cap and return 429 with
a clear JSON body when the cap or the daily budget is reached. CORS only for
localhost dev. Add tests for the SSE event order and the cap behavior.
```

## Prompt 3: Typed API client and event store

```
In frontend/src/api/, write:
- types.ts matching the SSE events and endpoint payloads exactly
- sse.ts: a fetch-based streaming reader for POST /api/ask (EventSource only
  supports GET), with abort support and parse errors surfaced, not swallowed
- useAsk.ts: a hook that turns the event stream into state: current lane,
  per-step timings, a list of limiter_wait/retry/degrade events, streamed
  answer text, first_token_ms, total_ms, and final result
- a replay mode: the hook can take a saved trace JSON from /api/trace/{id}
  and play it back at 1x, 2x or 4x speed with the same state shape

Strict TypeScript, no `any`. Write unit tests for the event reducer using a
recorded fixture.
```

## Prompt 4: Typography system with the load-driven width

```
Build the type system in tokens.css and base.css.

- A type scale on a modular ratio, defined once as custom properties.
- Headlines use Anybody. Add a CSS variable --stretch (50 to 150) applied via
  font-variation-settings: "wdth" var(--stretch). Write a small React context,
  LoadProvider, that polls /api/limits every 2 seconds and maps queue length
  (0 to 20+) to --stretch: empty queue is narrow and quick (about 70), heavy
  queue is wide and strained (about 140). Transition the change smoothly over
  600ms. Respect prefers-reduced-motion by holding --stretch at 100.
- Source Serif 4 for body and answers, line-height 1.6, max 66ch.
- Numerals use tabular figures.
- Sentence case everywhere. No all-caps labels.

Show a type specimen route at /specimen (dev only) rendering the scale at
three --stretch values side by side so I can check legibility.
```

## Prompt 5: Track view (the hero)

```
Build the Track view at / as the memorable element of the app.

Layout: five horizontal lanes spanning the full width, left-aligned, drawn in
SVG with lane-white lines on Mondo blue. Lane labels (1 understand, 2 retrieve,
3 decide, 4 draft, 5 verify) sit at the left end of each lane in Anybody.
Above the track: a single text field for the question, a source switch
(stand-in default, real), a retrieval-setting select, and one yellow button
"Run the question". For real source, show a visible counter "real requests
used today: n of cap" next to the switch.

Behavior, driven only by useAsk events:
- On accepted: the baton (a small yellow bar) waits at the start line; if
  queue_position > 0, show that many ghost batons behind it.
- step_start/step_end: the baton moves into that lane and travels along it
  for as long as the step takes in real time. Do not invent progress; if the
  step is unknown-length, show it as a pulsing bar.
- limiter_wait: the baton stops at a red hold line across the lane with the
  wait duration counting down and the reason in plain words ("per-minute
  limit, 4.2 s").
- retry: the baton steps back to the lane start; red tick on the lane for each
  attempt.
- degrade: a small bracket appears over the affected lane stating what was
  reduced, for example "fetched 3 passages instead of 6".
- first_token: a yellow vertical line drops across all lanes at that moment.

Answer panel below the track (see next prompt). Include an aria-live region
that announces step changes for screen readers. Provide a "replay a recorded
run" control that uses the replay mode, so the demo works with no backend
quota.
```

## Prompt 6: Answer panel, citations, refusals

```
Build the answer panel under the track.

- Streamed answer in Source Serif 4 on an Infield surface. Left-aligned, 66ch.
- Citations render inline as small bracketed markers [paper, page]. Clicking
  one opens a side drawer showing the exact chunk text from
  GET /api/chunks/{chunk_id}, with the cited sentence highlighted in baton
  yellow, plus paper id, version and page. A marker whose citation passed the
  verify step's code check gets a Split green tick; a failed one is red and
  struck through. Never render a citation as verified unless the final event
  says it passed.
- Refusal state: when final.status is a correct refusal, write plainly what the
  papers do not cover and which passages were closest, with their citations.
  No apology language, no "I'm just an AI". Example shape: "The papers don't
  report this. The closest passages are below."
- Under the answer: the photo-finish strip for this request, one horizontal bar
  for total_ms split into waiting (lane white), working (baton yellow) and
  retrying (false-start red), with a vertical first-token line and a plain
  caption: "first token at 812 ms, before/after the wait". Compute segments
  from the trace, not from the SSE timestamps alone.
- Tokens in and out, and the retrieval setting used, in one quiet line.
```

## Prompt 7: Heats view (load-test results)

```
Build /heats. It reads /api/results/summary and /api/results/requests, which
come from results.csv, and must reproduce Tables 2a and 2b exactly.

- Top: one row per scenario (steady, burst, sustained overload, rpm halved,
  daily exhausted), each row a horizontal lane. Left side: scenario name and
  "real" or "stand-in" in words. Along the lane, every request in that
  scenario is a thin vertical photo-finish strip placed by start_time, colored
  by status: ok in lane white, rate_limited in red, timeout in a hatched red,
  error in dark red. Height encodes total_ms on a shared scale across lanes.
- Click a scenario to expand: p50/p95/p99 for first token and total time,
  success %, rate-limited %, timed out %, retries per request, peak queue,
  tokens per answer, cost per 1,000 answers. Show the numbers in a plain table
  in Anybody tabular figures. Percentages to one decimal, times in ms.
- A toggle to overlay two scenarios (e.g. steady vs burst) on the same time
  scale.
- A "recompute" note stating the percentile method and how many requests each
  figure is based on. Rows with missing values are shown as excluded, not
  silently dropped.
- Retry comparison: if results exist with retries disabled, show both side by
  side so the "retrying made things worse" case is visible.

No chart library. If a figure cannot be derived from the CSV or the trace log,
render "not measured".
```

## Prompt 8: Limits view and the degradation ladder

```
Build /limits from GET /api/limits.

- Three horizontal bars: per-minute requests, per-minute tokens, daily
  requests. Each shows used, remaining and when it resets, with the bar
  drawn like a lane filling from the start line. Turn the leading edge red
  when under 15 percent remains.
- Below, the degradation ladder drawn as a vertical set of steps (this is a
  true ordered sequence, so numbering is fine). Highlight the rung currently
  active and show each rung's trigger in plain words, for example:
  1 full retrieval, 2 fewer passages, 3 skip the model verify pass (keep the
  code check), 4 smaller model, 5 honest "try later" reply. Read the rung
  definitions from the backend config, do not hardcode them.
- A control panel to edit the stand-in config (latency scale, failure rate,
  RPM, hang probability) via POST /api/standin/config, with a "reset" action,
  and a plain warning when the stand-in is being used instead of the real
  service.
- A short log of the last 20 scheduler decisions (who was delayed, dropped or
  degraded, and why) from the trace log, in sentence form.
```

## Prompt 9: Papers view and the 3pm screen

```
Two more views.

/papers: the three papers with exact versions (1706.03762 v7, 2410.00037 v1,
2507.07955 v2). A passage browser: choose a chunking setting, scroll chunks
with page and section, and a search box that runs the retriever and shows the
top-k results with scores so I can show how a naive retriever mislead
(terminology collisions like "attention" or "chunking", results buried in
tables, similar sections across papers). Mark chunks the extractor flagged as
garbled.

The 3pm screen: when the backend returns the daily-budget-exhausted state,
the Track view switches to a dedicated state. It must contain:
- one plain sentence saying today's model budget is used up
- the time it resets, in the person's local time zone
- a retrieval-only result: the passages most relevant to the question, with
  citations, clearly labeled as passages and not as an answer, with no model
  call
- an option to be told when it is back (stub is fine; store nothing)
Never show a generated answer in this state. Add a dev switch at
/limits to trigger this state from the stand-in.
```

## Prompt 10: Motion, accessibility, responsive

```
Polish pass. Motion rules: no entrance animations on sections, no hover
transitions on every element. Allowed motion: the baton moving through lanes,
the hold line during a limiter wait, the first-token line dropping, the
headline width easing, and one page-load moment on the Track view where the
five lanes draw in left to right. Everything else is instant or responds to
a click. Honor prefers-reduced-motion by replacing all of the above with
instant state changes and text.

Accessibility: visible keyboard focus in baton yellow with 3:1 contrast on
Mondo blue, all colors checked for 4.5:1 text contrast (lane white on Mondo
blue and on Infield; fix any pairing that fails), baton and status colors
never the only signal (add text or pattern), full keyboard operation of the
drawer, aria-live announcements for step changes and final results.

Responsive: on screens under 720px the lane rail collapses to a stacked
list of five lanes (still horizontal bars), the left rail becomes a top bar,
and the Heats table scrolls inside its own container.
```

## Prompt 11: Wire-up, run, and critique

```
Do the final integration and verify it end to end.

1. Add a root-level script (PowerShell on Windows) dev.ps1 that starts the
   stand-in server on 8000... (pick free ports, document them), the relay API,
   and `npm run dev`, and a prod.ps1 that builds the frontend and serves
   everything from FastAPI.
2. Run typecheck, the frontend tests, and pytest. Fix failures.
3. Walk through these flows against the stand-in and report what happened:
   a normal question; a question the papers cannot answer; a burst that
   triggers limiter_wait and retry; daily exhausted; replay of a saved trace.
4. Compare /heats numbers with python -m analysis.make_tables output. They
   must match exactly. If they do not, fix the frontend, not the tables.
5. Critique your own result against the design plan: where did it drift toward
   generic defaults (cards, gradients, all-caps labels, arrows, monospace
   labels)? List what you removed or changed.
```

---

## Using this in the submission

- **Keep it out of the two pages.** Mention it in one line under heading 1 or in the assistant section. The writing budget belongs to the measurements.
- **Screenshots help in the live conversation.** Save one of the Track view mid-retry, one of Heats, and one of the 3pm screen. They answer "what does the person see?" faster than prose.
- **Your overrule moment can come from here.** If the assistant proposes charts or numbers that do not match `results.csv`, or suggests a chart library, a gradient look, or a different palette, that is a real decision to note in your assistant section.
- **Defend only what you understand.** If you cannot explain how the first-token line or the photo-finish strip is computed, simplify it before you submit.

## Honest costs of this design

- The width-driven headline is distinctive but can hurt legibility at extreme widths. Check the specimen view, and clamp the range if needed.
- Hand-written SVG takes longer than a chart library. It is worth it because the lanes and strips are the idea; if time runs short, cut the Papers view first, then the overlay toggle, and keep Track and Heats.
- Replay mode exists because live demos are fragile and your Groq quota is shared. Do not skip it.
