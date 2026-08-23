# DiacriticS Research Experience — Redesign Plan

**Audience:** the implementing agent (Claude Sonnet). Follow this literally. Every change below
names an exact file, an exact selector, and the exact replacement. Where a value is given, use that
value — do not substitute your own judgement.

**Scope:** visual/structural only. Do not change copy in `src/content/site.ts`, do not change the
API route `src/pages/api/diacritize.ts`, do not change any `aria-*`, `role`, `lang`, or `dir`
attribute except where this document explicitly says to.

**Verification target:** dev server is already running at `http://localhost:4321`. Routes to check
after every phase: `/`, `/ar/`, `/demo/`, `/ar/demo/` — each in **light and dark** theme.

---

## Context: what is wrong, measured

Three complaints, each confirmed by inspecting the running site at 1440×900.

### 1. The section background images are misplaced

| Section | Image | Intrinsic size | What actually happens |
|---|---|---|---|
| `.idea` | `andalusian-carved-detail.webp` | 960×1709 (portrait) | Stretched full-bleed across a ~1440px-wide section. `object-fit:cover` on a 0.56-ratio image in a ~0.9-ratio box crops to the middle third and upscales ~1.5×. Reads as unidentifiable mush. |
| `.pipeline` | `andalusian-glass-roof-arches.webp` | 960×1440 (portrait) | `.pipeline__veil` composites to ~96% opaque canvas. **The image is not visible at all.** 233 KB downloaded and painted for zero visual effect. |
| `.results` | `andalusian-courtyard-analysis.webp` | 1600×1067 | Height clamped to 30–43rem with a veil fading to solid. Only a faint band survives at the very top. |

The root cause is not the crop — it is the **layer stack**. Each of these sections paints
image → ~90 % light veil → a 94 %-opaque white `.environment-heading` card with a `--shadow`. That
is three light translucent layers on top of each other. Apple's materials guidance is explicit that
you never stack a light translucent surface on another, because legibility collapses and the top
surface stops reading as a material and starts reading as a sticker. That is exactly the "another
black/white layout on top" feeling in the complaint.

The hero and the demo page work because both have a **tall** container that matches a portrait
image, and both put text on a genuinely dark/quiet part of the frame.

### 2. The pipeline steps cannot be followed

Five stacked cards, each ~230 px tall, `h3` at `var(--step-2)` = up to **39 px**. The section runs
~1600 px. At 900 px viewport you see **1.5 steps at a time**, and there is no device anywhere that
tells you the sequence has five items or which one you are on. The step number — the one element
that carries the sequence — is `0.8rem` and the weakest thing on screen, while the title is the
loudest. The information architecture is inverted.

### 3. The boxes read as dated / AI-generated

The recurring tell in AI-generated interfaces is *rounded card + soft shadow + light fill, repeated
for every content type regardless of what the content is* — the statistical average of a million
landing pages rather than a decision. This file has that pattern six times: `.environment-heading`,
`.pipeline-steps li`, `.finding-card`, `.final-cta__copy`, `.demo-panel`, `.result-panel`.

Nielsen Norman Group's framing is the useful test: **cards are for browsing, lists are for
scanning, tables are for comparing.** By that test almost every card here is the wrong container —
the pipeline steps are a *sequence*, the findings are *two related claims*, neither is a browsable
set of snapshots.

Two further tells present in the file:
- **Nested cards.** `.demo-panel` (1.4rem radius + border + shadow) contains `.result-panel`
  (1rem radius + border + own background). A rounded box inside a rounded box.
- **Ornament with no meaning.** `.demo-panel__accent` (three macOS-window "traffic light" pills
  hanging half off the panel edge), `.signature::before/::after` (two floating outline circles),
  `.team-card__number` (gold `01`–`04` in the corner), `.image-plate::before` (corner brackets).

**The good news, and the organising idea for this whole plan:** the site already contains its own
better answer. `.team-grid`, `.scoretable` and `.formula-block` are square, hairline-ruled,
shadowless, and they look considered and research-appropriate. The redesign is not an invention —
it is **extending the table/grid language the site already has, and deleting the card language it
does not need.**

---

## The design system to converge on

Three devices, and nothing else.

1. **Hairline rules** (`1px solid var(--rule)` / `--rule-strong`) to divide.
2. **A surface change** (`--canvas` ↔ `--surface`) to group. If the fill already separates the
   region, it does not also get a border.
3. **Elevation reserved for genuinely floating chrome** — the sticky header, and the demo result
   panel while it holds a result. Nothing else gets a `box-shadow`, ever.

Consequences applied globally:
- `--radius` goes `0.45rem` → `0`. Pills (`999px`) stay for buttons and chips only.
- `--shadow` survives only on `.site-header` and `.demo-panel`.
- Type: step/finding titles come down from `--step-2` to `--step-1`. Apple's typography rule is
  that hierarchy is built from **weight + size + leading as a set**, not size alone — so the
  demoted titles gain weight (500 → 600) rather than staying huge.

---

## Phase 0 — Delete dead code first

`AnalysisSection.astro` is imported by nothing. `ImagePlate.astro` is imported only by
`AnalysisSection.astro`. Both are dead, and so is the CSS serving them. Removing this first makes
every later phase easier to verify.

```bash
cd DiacriticS-Website-With-Model
rm src/components/AnalysisSection.astro src/components/ImagePlate.astro
```

Then delete these rule blocks from `src/styles/global.css` (they have **no** markup anywhere —
verified with grep across `src/components`, `src/pages`, `src/layouts`):

`.intro`, `.intro__grid`, `.intro__copy`, `.intro__visual`, `.image-plate` and all
`.image-plate--*` variants, `.method`, `.method__grid`, `.method__content`, `.limitations` and
`.limitations__*` (including the two dark-theme overrides near the end of the file), `.fact-grid`
and its `dt`/`dd`, `.boundary-note`, `.chapters`, `.chapter` and all `.chapter__*`, `.status-chip`,
`.metric-figure`, `.metric-figure__title`, `.metric-bars`, `.metric-row` and `.metric-row__*`,
`.metric-table`, `.pending-figure`, `.context-figure`, `.provenance-details`.

Keep `.intro__marks` — it is still used by `HomePage.astro` (see Phase 3.4). Keep
`.section-heading--wide` — still used by `HomePage.astro`.

Also remove the now-orphaned references to those selectors inside the `@media (max-width: 56rem)`
and `@media (max-width: 43rem)` blocks (`.intro__grid`, `.method__grid`, `.limitations__grid`,
`.chapter`, `.fact-grid`, `.image-plate*`, `.metric-row`, `.metric-table`, `.context-figure`).

Expected: ~400 lines removed. `npm run build` must still succeed.

---

## Phase 1 — Fix the background images

**Strategy: stop using photographs as section backgrounds. Turn them into full-bleed bands that sit
*between* sections.** This is the "transition period before the points" instinct from the brief, and
it is the right call for three reasons: the band's aspect ratio can match the photo instead of
fighting it; the band needs no heavy veil because no body text sits on it; and once the sections
underneath are plain surfaces, every heading card and content card can lose its opaque background
and shadow, which is most of Phase 3 solved for free.

### 1.1 Create `src/components/SectionBreak.astro`

```astro
---
interface Props {
  src: string;
  credit: string;
  width: number;
  height: number;
  position?: string;   /* object-position */
  from?: string;       /* CSS colour the band fades FROM at its top edge */
  to?: string;         /* CSS colour the band fades TO at its bottom edge */
}
const { src, credit, width, height, position = "center 50%", from = "var(--surface)", to = "var(--canvas)" } = Astro.props;
---

<div class="section-break" style={`--break-from:${from};--break-to:${to};--break-pos:${position}`}>
  <img src={src} alt="" width={width} height={height} loading="lazy" decoding="async" aria-hidden="true" />
  <div class="section-break__blend" aria-hidden="true"></div>
  <small class="section-break__credit">{credit}</small>
</div>
```

### 1.2 Add to `global.css` (place it where the deleted `.image-plate` block was)

```css
.section-break {
  position: relative;
  isolation: isolate;
  height: clamp(13rem, 24vh, 20rem);
  overflow: clip;
  background: var(--sand);
}

.section-break img {
  position: absolute;
  inset: 0;
  z-index: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: var(--break-pos);
  filter: saturate(0.8) contrast(0.96);
}

/* Only the EDGES are veiled, so the middle of the photograph is seen at full fidelity. */
.section-break__blend {
  position: absolute;
  inset: 0;
  z-index: 1;
  pointer-events: none;
  background: linear-gradient(
    180deg,
    var(--break-from) 0,
    color-mix(in srgb, var(--break-from) 55%, transparent) 14%,
    transparent 34%,
    transparent 66%,
    color-mix(in srgb, var(--break-to) 55%, transparent) 86%,
    var(--break-to) 100%
  );
}

.section-break__credit {
  position: absolute;
  z-index: 2;
  inset-block-end: 0.5rem;
  inset-inline-end: 0.75rem;
  color: color-mix(in srgb, var(--ink) 62%, transparent);
  font-size: 0.66rem;
  line-height: 1.4;
}

:root[data-theme="dark"] .section-break img {
  filter: saturate(0.76) contrast(0.98) brightness(0.68);
}
```

### 1.3 Rewire `HomePage.astro`

Delete from `.idea`, `.pipeline` and `.results` these three lines each: the
`<img class="section-environment__media …">`, the `<div class="section-environment__veil">`, and the
`section-environment__content` class on the `.shell` (keep the `shell` class itself). Also remove
`environment-heading` from the three `<header>` elements — the headings now sit on the plain
section surface.

Then insert `<SectionBreak />` **between** sections, choosing each image by aspect ratio:

| Position | Image | Why |
|---|---|---|
| after `.idea`, before `.team` | `andalusian-glass-roof-arches.webp` (960×1440) | `object-position: center 30%` — the arch tops are the readable part |
| after `.team`, before `.pipeline` | `andalusian-courtyard-analysis.webp` (1600×1067) | the only true landscape image; best band candidate |
| after `.results`, before `.references` | `andalusian-carved-detail.webp` (960×1709) | `object-position: center 42%`; a tight ornament crop is fine in a short band because it is *meant* to read as texture |

Set `from`/`to` on each to match the neighbouring sections' backgrounds (`.idea` is `--surface`,
`.team` is `--canvas`, `.pipeline` is `--canvas`, `.results` is `--surface`, `.references` is
`--canvas`) so the band dissolves into both neighbours instead of butting against them.

Pass `credit` from `t.media.*.credit` in `site.ts` — the credits must survive, they are a listed
publication blocker in `README.md`.

### 1.4 Delete the now-unused background machinery

Remove `.section-environment__media`, `.section-environment__veil`, `.section-environment__content`,
`.environment-heading`, `.idea__media`, `.idea__veil`, `.pipeline__media`, `.pipeline__veil`,
`.results__media`, `.results__veil`, and the matching `:root[data-theme="dark"]` overrides for
`.idea__veil`, `.pipeline__veil`, `.results__veil`. In the shared
`:root[data-theme="dark"] .signature__media, … .section-environment__media, …` rule, drop the
`.section-environment__media` selector but keep the others.

Remove `isolation: isolate; overflow: clip;` from `.idea, .pipeline, .results` (keep them on
`.final-cta`, which still has a background image and still works).

**Leave `.signature` (hero), `.demo-page` and `.final-cta` exactly as they are.** They were
explicitly excluded from the complaint and they are the two places where a full-bleed image is
correct.

---

## Phase 2 — Rebuild the pipeline as a followable sequence

**Target:** all five steps legible in ~1.2 viewports instead of ~4, with the full sequence visible
at all times and the current position always indicated. Apple's wayfinding rule is that every screen
answers *where am I / where can I go / what's there*; a stack of five identical cards answers none
of them.

### 2.1 Markup — replace the `<ol class="pipeline-steps">` block in `HomePage.astro`

```astro
<div class="pipeline-layout">
  <nav class="pipeline-rail" aria-label={t.pipeline.label}>
    <ol>
      {t.pipeline.stages.map((stage, i) => (
        <li><a href={`#stage-${i + 1}`} data-rail-link={i + 1}>
          <span class="pipeline-rail__num" aria-hidden="true">{stage.number}</span>
          <span class="pipeline-rail__title">{stage.title}</span>
        </a></li>
      ))}
    </ol>
  </nav>

  <ol class="pipeline-steps">
    {t.pipeline.stages.map((stage, i) => (
      <li id={`stage-${i + 1}`} data-stage={i + 1}>
        <span class="pipeline-step__number" aria-hidden="true">{stage.number}</span>
        <div>
          <h3>{stage.title}</h3>
          <p>{stage.body}</p>
        </div>
      </li>
    ))}
  </ol>
</div>
```

### 2.2 CSS — replace the existing `.pipeline-steps` / `.pipeline-steps li` / `.pipeline-step__number` / `.pipeline-steps h3` / `.pipeline-steps p` rules entirely

```css
.pipeline-layout {
  display: grid;
  grid-template-columns: minmax(13rem, 0.32fr) minmax(0, 1fr);
  gap: clamp(2rem, 6vw, 5rem);
  align-items: start;
}

/* --- the rail: always shows all five, and which one you are on --- */
.pipeline-rail {
  position: sticky;
  inset-block-start: 6.5rem;
}

.pipeline-rail ol {
  display: grid;
  gap: 0;
  margin: 0;
  padding: 0;
  list-style: none;
  border-inline-start: 1px solid var(--rule);
}

.pipeline-rail a {
  display: grid;
  grid-template-columns: 2.1rem 1fr;
  gap: 0.5rem;
  align-items: baseline;
  margin-inline-start: -1px;
  border-inline-start: 1px solid transparent;
  padding: 0.7rem 0 0.7rem 1rem;
  color: var(--muted);
  font-size: 0.84rem;
  line-height: 1.35;
  text-decoration: none;
  transition: color 180ms ease, border-color 180ms ease;
}

html[dir="rtl"] .pipeline-rail a { padding: 0.7rem 1rem 0.7rem 0; }

.pipeline-rail a:hover { color: var(--ink); }

.pipeline-rail__num { color: var(--gold); font-size: 0.72rem; font-weight: 600; }

.pipeline-rail a.is-active {
  border-inline-start-color: var(--red);
  color: var(--ink);
  font-weight: 600;
}

.pipeline-rail a.is-active .pipeline-rail__num { color: var(--red); }

/* --- the steps: one continuous spine, no cards --- */
.pipeline-steps {
  display: grid;
  gap: 0;
  margin: 0;
  padding: 0;
  list-style: none;
}

.pipeline-steps li {
  position: relative;
  display: grid;
  grid-template-columns: 2.75rem minmax(0, 1fr);
  gap: clamp(0.75rem, 2vw, 1.5rem);
  padding-block: var(--space-3);
  scroll-margin-block-start: 7rem;
}

/* the spine, drawn behind the numbers */
.pipeline-steps li::before {
  position: absolute;
  content: "";
  inset-block: 0;
  inset-inline-start: 0.85rem;
  width: 1px;
  background: var(--rule);
}

.pipeline-steps li:first-child::before { inset-block-start: 1.9rem; }
.pipeline-steps li:last-child::before  { inset-block-end: calc(100% - 1.9rem); }

.pipeline-step__number {
  position: relative;
  z-index: 1;
  display: grid;
  width: 1.7rem;
  height: 1.7rem;
  place-items: center;
  margin-block-start: 0.28rem;
  border: 1px solid var(--rule-strong);
  border-radius: 50%;
  color: var(--red);
  background: var(--canvas);
  font-size: 0.7rem;
  font-weight: 600;
  line-height: 1;
}

.pipeline-steps h3 {
  margin-block-end: 0.4rem;
  font-size: var(--step-1);
  font-weight: 600;
  letter-spacing: -0.01em;
}

html[dir="rtl"] .pipeline-steps h3 { letter-spacing: 0; }

.pipeline-steps p {
  max-width: 46rem;
  color: var(--muted);
  font-size: 0.95rem;
}
```

Note the two structural wins: **`h3` drops from `--step-2` (39 px) to `--step-1` (24 px)** but gains
weight 500 → 600, and **the number becomes a circular node on a continuous line** — so the sequence
is now carried by the strongest graphic element instead of the weakest.

### 2.3 Active-state script

Add at the bottom of `HomePage.astro` (Astro will bundle it; no framework needed):

```astro
<script>
  const links = document.querySelectorAll<HTMLAnchorElement>("[data-rail-link]");
  const stages = document.querySelectorAll<HTMLElement>("[data-stage]");
  if (links.length && stages.length && "IntersectionObserver" in window) {
    const setActive = (n: string | undefined) => {
      for (const link of links) link.classList.toggle("is-active", link.dataset.railLink === n);
    };
    const seen = new Map<string, number>();
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) seen.set(entry.target.dataset.stage!, entry.intersectionRatio);
      let best: string | undefined; let bestRatio = 0;
      for (const [stage, ratio] of seen) if (ratio > bestRatio) { bestRatio = ratio; best = stage; }
      if (best) setActive(best);
    }, { rootMargin: "-25% 0px -55% 0px", threshold: [0, 0.25, 0.5, 1] });
    for (const stage of stages) observer.observe(stage);
  }
</script>
```

No-JS behaviour: the rail still renders as five working anchor links. That is an acceptable
degradation and must not be "fixed" by making the rail JS-only.

### 2.4 Responsive

In `@media (max-width: 56rem)` add:

```css
.pipeline-layout { grid-template-columns: 1fr; gap: var(--space-3); }

.pipeline-rail {
  inset-block-start: 4.5rem;
  z-index: 4;
  margin-inline: calc(50% - 50vw);
  padding-inline: max(1rem, calc(50vw - 50%));
  background: color-mix(in srgb, var(--canvas) 92%, transparent);
  backdrop-filter: blur(12px);
  border-block-end: 1px solid var(--rule);
}

.pipeline-rail ol {
  grid-auto-flow: column;
  grid-auto-columns: max-content;
  overflow-x: auto;
  border-inline-start: 0;
  scrollbar-width: none;
}

.pipeline-rail a {
  margin-inline-start: 0;
  border-inline-start: 0;
  border-block-end: 2px solid transparent;
  padding: 0.6rem 0.9rem;
}

.pipeline-rail a.is-active { border-block-end-color: var(--red); }
.pipeline-rail__title { display: none; }   /* numbers only on narrow screens */
```

---

## Phase 3 — De-card everything

### 3.1 Global tokens (`:root` in `global.css`)

```css
--radius: 0;                                   /* was 0.45rem */
--shadow: 0 18px 44px rgb(39 27 18 / 0.10);    /* keep, but it is now used twice, not seven times */
```

### 3.2 Findings — two claims, not two cards

Replace `.findings` / `.finding-card` / `.finding-card h3` / `.finding-card > p:not(.eyebrow)`:

```css
.findings {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
  margin-block-start: var(--space-5);
  border-block-start: 1px solid var(--rule-strong);
}

.finding-card {
  display: grid;
  align-content: start;
  padding: var(--space-4) 0;
  padding-inline-end: clamp(1.5rem, 4vw, 3.5rem);
}

.finding-card + .finding-card {
  border-inline-start: 1px solid var(--rule);
  padding-inline: clamp(1.5rem, 4vw, 3.5rem) 0;
}

html[dir="rtl"] .finding-card { padding-inline: 0 clamp(1.5rem, 4vw, 3.5rem); }
html[dir="rtl"] .finding-card + .finding-card { padding-inline: 0 clamp(1.5rem, 4vw, 3.5rem); }

.finding-card h3 {
  margin-block: var(--space-2);
  font-size: var(--step-1);
  font-weight: 600;
}

.finding-card > p:not(.eyebrow) { color: var(--muted); font-size: 0.95rem; }
```

### 3.3 Remove the remaining shadows and radii

- `.scoreboard` — delete `box-shadow: var(--shadow);`
- `.final-cta__copy` — delete `box-shadow: var(--shadow);` and `border-radius: var(--radius);`;
  change `background` to `var(--surface)` (solid, not `color-mix(… 96%)`) so it is one clean
  surface on the photo rather than a translucent light layer on a light veil.
- `.idea__marks` — delete `box-shadow: var(--shadow);`
- `.signature::before` and `.signature::after` — **delete both rules entirely.** Two floating
  outline circles that mean nothing.
- `.team-card__number` — **delete the rule and the `<span class="team-card__number">` from
  `HomePage.astro`.** The grid position already communicates order.
- `.team-card h3 a` — add `text-decoration: none;` and a `:hover { text-decoration: underline; }`
  so four permanently-underlined names stop reading as a link dump.

### 3.4 Fix the broken `.intro__marks` block

The four diacritic cells are a 4-column grid, but the block also paints a `4rem × 4rem` graph-paper
`background-size` that does not align with the cells, so the marks look like specks scattered in a
ruled box. Remove the two `linear-gradient(...)` layers from `background`, keeping only
`var(--canvas)`, and delete the `background-size` line and the `background-size: 3rem 3rem` override
in the `43rem` media query. The 1px cell borders already provide the grid.

### 3.5 Press feedback instead of hover-lift

`.button:hover { transform: translateY(-2px) }` is a generic tell, and Apple's first rule is that
feedback belongs on **pointer-down**, not on hover or release. Replace:

```css
.button:hover { transform: none; }           /* delete the translateY rule */

.button:active { transform: scale(0.97); transition-duration: 100ms; }

.example-chip:active, .copy-button:active:not(:disabled), .theme-toggle:active {
  transform: scale(0.97);
}
```

Also delete `transform: rotate(8deg);` from `.theme-toggle:hover` — keep the colour change.

### 3.6 Fix the sticky-header anchor collision

Confirmed bug: jumping to `#team` or `#results` puts the `h2` underneath the sticky header. Add near
`.section`:

```css
:where(#idea, #team, #pipeline, #results, #references) { scroll-margin-block-start: 5.5rem; }

@media (max-width: 43rem) {
  :where(#idea, #team, #pipeline, #results, #references) { scroll-margin-block-start: 8rem; }
}
```

### 3.7 Fix the `.section-heading--wide` void

`align-items: end` pushes the `h2` to the bottom of a tall row while the lede sits in a narrow
right-hand column, producing a large empty region top-left (clearly visible on `#results`). Change:

```css
.section-heading--wide {
  grid-template-columns: minmax(0, 1.1fr) minmax(18rem, 0.75fr);
  align-items: start;            /* was: end */
  margin-block-end: var(--space-5);   /* was: var(--space-6) */
}
```

---

## Phase 4 — The demo page

`/demo/` and `/ar/demo/`. The background image here is fine and must not be touched.

### 4.1 Collapse the nested cards

Currently `.demo-panel` (radius 1.4rem, border, shadow) wraps `.demo-form` and `.result-panel`
(radius 1rem, own border, own gradient background). One card inside another. Make `.demo-panel` the
**single** surface and separate the two regions with a rule.

```css
.demo-panel {
  position: relative;
  max-width: 62rem;
  margin-inline: auto;
  border: 1px solid var(--rule-strong);
  border-radius: 0;
  padding: 0;                                  /* was clamp(1rem, 2.5vw, 1.6rem) */
  background: var(--surface);
  box-shadow: var(--shadow);
}

.demo-form,
.result-panel { min-width: 0; padding: clamp(1.25rem, 3vw, 2.25rem); }

.demo-form { padding-block-end: clamp(1.25rem, 3vw, 2.25rem); }

.result-panel {
  position: relative;
  display: grid;
  border: 0;                                   /* was a 1px gold border + radius */
  border-block-start: 1px solid var(--rule-strong);
  border-radius: 0;
  background: color-mix(in srgb, var(--sand) 18%, var(--surface));
}
```

Delete the `radial-gradient(circle at 14% 16%, …)` from `.result-panel`'s background.

### 4.2 Delete the fake window chrome

Remove `.demo-panel__accent`, `.demo-panel__accent span`, `.demo-panel__accent span:nth-child(2)`,
`.demo-panel__accent span:nth-child(3)` from `global.css`, and remove the
`<div class="demo-panel__accent" aria-hidden="true">…</div>` line from `DemoInterface.astro`.

### 4.3 Calm the input

```css
.demo-form textarea {
  min-height: 7.5rem;                          /* was 11rem — it was mostly void */
  border-radius: 0;                            /* was 1rem */
  font-size: clamp(1.15rem, 2vw, 1.5rem);      /* was clamp(1.35rem, 2.6vw, 1.8rem) */
  line-height: 1.9;
}
```

Also set `border-radius: 0` in the `43rem` media query where `.demo-panel` currently gets `1.1rem`,
and change that block's `min-height: 11rem` on the textarea to `7.5rem`.

Keep the `:focus` treatment exactly as it is — the ring + background change is correct, instant
feedback and should not be weakened.

### 4.4 Fix the illegible empty state

`.result-panel[data-state="empty"] .result-text::before` renders `"◌َ   ◌ّ   ◌ُ"` at `1.3rem` with
`0.35em` tracking; at that size the dotted circles collapse into specks. Replace the whole rule with
a skeleton that reads as "a line of Arabic will appear here":

```css
.result-panel[data-state="empty"] .result-text::before {
  display: block;
  width: min(22rem, 70%);
  height: 0.85rem;
  margin: 0 auto 0.9rem;
  content: "";
  background: repeating-linear-gradient(
    90deg,
    color-mix(in srgb, var(--gold) 30%, transparent) 0 3.5rem,
    transparent 3.5rem 4.25rem
  );
  opacity: 0.65;
}
```

### 4.5 Reduce the result min-height

`.result-text { min-height: 10rem }` reserves a large void before any result exists. Set `7rem`, and
`8rem` in the `43rem` media query.

---

## Do not change

- Any `aria-*`, `role`, `dir`, `lang`, `for`/`id` pairing, or the `data-*` hooks that
  `DemoInterface.astro`'s script reads (`data-demo`, `data-input`, `data-result-panel`,
  `data-state`, `data-count`, `data-submit`, `data-clear`, `data-copy`, `data-meta`, `data-model`,
  `data-duration`, `data-status`, `data-example`, `data-errors`, `data-default-error`,
  `data-starting`, `data-locale`).
- `src/pages/api/*`, `astro.config.mjs`, `.env*`, `src/content/site.ts`.
- The `@media (prefers-reduced-motion: reduce)` block — and add nothing that would need a new
  exception in it. The pipeline rail uses no motion by design.
- The hero (`.signature*`) and its scroll script.
- The `.demo-page` background image treatment.

---

## Verification

Run after **each** phase, not once at the end.

```bash
cd DiacriticS-Website-With-Model
npm run build          # must exit 0 — this is the only automated gate in the repo
```

Then, with the dev server on `http://localhost:4321`, check this matrix by eye. There is no test
suite and no linter in this repo, so visual verification is the gate.

| Route | Light | Dark |
|---|---|---|
| `/` | ☐ | ☐ |
| `/ar/` | ☐ | ☐ |
| `/demo/` | ☐ | ☐ |
| `/ar/demo/` | ☐ | ☐ |

Widths: **1440**, **900**, **390**.

Specific things to confirm:

1. **Phase 1** — no section has a photograph behind its body text any more; the three bands appear
   *between* sections; each band's photo is actually visible (not washed out); credits still render.
2. **Phase 2** — at 1440×900 the rail shows all five step titles at once and the active one is
   marked while you scroll; clicking a rail link lands the step below the header, not under it. At
   390px the rail is a sticky numbered strip. **Check RTL specifically** — the rail border and
   padding are logical properties and must flip.
3. **Phase 3** — search `global.css` for `box-shadow`: it must appear only on `.site-header`,
   `.demo-panel`, and inside the dark-theme override for `.demo-panel`. Search for
   `border-radius`: only `999px` pills, `50%` circles, and `0`.
4. **Phase 4** — the demo page has exactly **one** bordered container, not two nested; no traffic
   lights; the empty result panel shows the skeleton bar, not specks.
5. **Regression check** — submit an example on `/demo/`, confirm the result renders, the copy button
   enables, the meta row (model + latency) appears, and `Clear` resets. The API path is untouched
   but the CSS changes touch `[data-state]` styling.
6. **Reduced motion** — with "Reduce motion" enabled in macOS System Settings, reload `/`; the hero
   should still render fully revealed and nothing should animate.

## Suggested commit sequence

```
1. chore: remove unused AnalysisSection/ImagePlate and their dead CSS
2. feat: replace section background images with between-section bands
3. feat: rebuild pipeline as sticky-rail sequence with a spine
4. style: remove card chrome sitewide; hairline rules and surface changes only
5. style: collapse nested demo cards, remove fake window chrome
```
