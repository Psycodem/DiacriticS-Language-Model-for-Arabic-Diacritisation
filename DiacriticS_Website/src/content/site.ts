export type Locale = "en" | "ar";

type ResourceLink = { label: string; meta: string; href?: string; external?: boolean; disabled?: boolean };
type TeamMember = { role: string; name: string; bio: string; href: string };
type PipelineStage = { number: string; title: string; body: string };
type Finding = { tag: string; title: string; body: string; example: string; gloss: string };

export const shared = {
  brand: "DiacriticS",
  githubUrl: "https://github.com/Psycodem/DiacriticS-Language-Model-for-Arabic-Diacritisation",
  heroArabic: "تَشْكِيل",
  heroBase: "تشكيل",
  heroSubArabic: "اللُّغَة العَرَبِيَّة",
  heroSubBase: "اللغة العربية",
  provisionalResults: [
    { value: "8.5", label: "LoRA", tone: "gold" },
    { value: "2.6", label: "Full FT", tone: "blue" },
    { value: "2.1", label: "Tagger", tone: "green" },
  ],
  baselineResults: [
    { model: "Flan-T5-Tashkeel-Small", derCe: "4.10", werCe: "11.86", derNoCe: "3.61", werNoCe: "7.88", best: true },
    { model: "gemma-4-E4B-it", derCe: "4.18", werCe: "9.96", derNoCe: "3.80", werNoCe: "6.78" },
    { model: "Tashkeel-350M-v2", derCe: "6.33", werCe: "15.28", derNoCe: "5.64", werNoCe: "10.40" },
    { model: "Fine-Tashkeel", derCe: "17.91", werCe: "22.61", derNoCe: "17.73", werNoCe: "20.35" },
    { model: "Fanar-1-9B-Instruct", derCe: "26.14", werCe: "31.48", derNoCe: "26.35", werNoCe: "28.81" },
    { model: "Glonor-ByT5-Arabic", derCe: "33.62", werCe: "50.68", derNoCe: "33.22", werNoCe: "44.77" },
    { model: "Qwen3.5-9B", derCe: "39.78", werCe: "59.44", derNoCe: "41.92", werNoCe: "55.69" },
    { model: "gemma-3-1b-pt-10k-diacritization", derCe: "50.25", werCe: "55.81", derNoCe: "50.66", werNoCe: "52.95" },
    { model: "Qwen3.5-4B", derCe: "58.29", werCe: "79.00", derNoCe: "59.43", werNoCe: "75.22" },
    { model: "aya-expanse-8b", derCe: "66.55", werCe: "85.78", derNoCe: "67.08", werNoCe: "84.04" },
    { model: "Moonlight-16B-A3B-Instruct", derCe: "86.60", werCe: "99.01", derNoCe: "87.10", werNoCe: "98.84" },
    { model: "Qwen3.5-0.8B", derCe: "92.18", werCe: "99.67", derNoCe: "92.41", werNoCe: "99.62" },
  ],
  fineTuningResults: [
    { model: "gemma-4-E4B-it", data: "zero-shot", derCe: "4.18", werCe: "9.96", derNoCe: "3.80", werNoCe: "6.78" },
    { model: "gemma-4-E4B-it", data: "10%", derCe: "3.15", werCe: "7.28", derNoCe: "2.68", werNoCe: "4.47" },
    { model: "gemma-4-E4B-it", data: "30%", derCe: "2.98", werCe: "6.82", derNoCe: "2.54", werNoCe: "4.12" },
    { model: "gemma-4-E4B-it", data: "50%", derCe: "2.81", werCe: "6.54", derNoCe: "2.38", werNoCe: "3.96", best: true },
    { model: "Qwen3.5-4B", data: "zero-shot", derCe: "58.29", werCe: "79.00", derNoCe: "59.43", werNoCe: "75.22" },
    { model: "Qwen3.5-4B", data: "10%", derCe: "7.07", werCe: "11.74", derNoCe: "6.94", werNoCe: "8.74" },
    { model: "Qwen3.5-4B", data: "30%", derCe: "5.27", werCe: "9.28", derNoCe: "5.03", werNoCe: "6.54" },
    { model: "Qwen3.5-4B", data: "50%", derCe: "4.38", werCe: "8.18", derNoCe: "4.07", werNoCe: "5.57" },
  ],
  regimeResults: [
    { model: "Zero-shot", data: "no training", derCe: "92.18", werCe: "99.67", derNoCe: "92.41", werNoCe: "99.62" },
    { model: "LoRA", data: "rank 16, full corpus", derCe: "3.60", werCe: "8.28", derNoCe: "3.15", werNoCe: "4.98" },
    { model: "Full fine-tuning", data: "all weights, full corpus", derCe: "3.06", werCe: "7.34", derNoCe: "2.58", werNoCe: "4.12", best: true },
  ],
  references: [
    { title: "Character-based Arabic Tashkeel Transformer (CATT)", meta: "A. Abbad et al. · github.com/abjadai/catt · 2023", href: "https://arxiv.org/abs/2407.03236" },
    { title: "Arabic Text Diacritization In The Age Of Transfer Learning: Token Classification Is All You Need", meta: "F. Author et al. · 2024", href: "https://arxiv.org/abs/2401.04848" },
    { title: "Sadeed: Advancing Arabic Diacritization Through Small Language Model", meta: "S. Author et al. · 2025", href: "https://arxiv.org/abs/2504.21635" },
    { title: "AraXLM: Evaluating Arabic Diacritization Tools for Cross-Language Plagiarism Detection", meta: "M. Author et al. · 2024", href: "https://annals-csis.org/Volume_43/drp/4862.html" },
    { title: "More Data, Fewer Diacritics: Scaling Arabic TTS", meta: "T. Author et al. · 2024", href: "https://arxiv.org/abs/2603.01622" },
    { title: "Arabic Text Diacritization Using Deep Neural Networks and Transformer-Based Architectures", meta: "M. Cherradi & H. El Mahajer · Knowledge and Decision Systems with Applications · 2025", href: "https://doi.org/10.59543/kadsa.v1i.15077" },
  ],
} as const;

export const copy = {
  en: {
    lang: "en", dir: "ltr",
    meta: {
      homeTitle: "DiacriticS: Contamination-Controlled Arabic Diacritisation",
      homeDescription: "DiacriticS: fine-tuning open language models for robust Arabic text diacritisation, benchmarked on the contamination-controlled SadeedDiac-25 test set.",
      demoTitle: "Try DiacriticS: Arabic diacritisation demo",
      demoDescription: "Test the DiacriticS interface with Arabic text and inspect its before-and-after output.",
    },
    nav: { idea: "Idea", team: "Team", pipeline: "Pipeline", results: "Results", fineTuning: "Fine-tuning", references: "References", demo: "Try the model", language: "العربية", languageLabel: "Open the Arabic version" },
    hero: {
      eyebrow: "Arabic language research · DiacriticS", brandLabel: "DiacriticS", caption: "tashkīl: the marks that carry the meaning",
      support: "Fine-tuning a language model to restore Arabic's missing vowels, evaluated where it counts: on a benchmark built to resist memorisation.",
      note: "Scroll to restore the marks that written Arabic often leaves out.", primary: "Read the project", try: "Try the model",
      resources: [
        { label: "Project report", meta: "Final report · PDF", href: "/DiacriticS_Final_Report.pdf", external: true },
        { label: "Code on GitHub", meta: "Scripts, configs, results", href: shared.githubUrl, external: true },
      ] satisfies ResourceLink[],
    },
    media: {
      hero: { credit: "Artwork: diacritised Arabic set in Amiri" },
      intro: { alt: "Arabic words shown with their full diacritics.", credit: "Artwork: diacritised Arabic set in Amiri" },
      method: { alt: "An ornately carved archway frames a flowering courtyard garden, with a reclining statue resting on the windowsill.", credit: "Artwork: diacritised Arabic set in Amiri" },
      analysis: { alt: "Arabic words shown with their full diacritics.", credit: "Artwork: diacritised Arabic set in Amiri" },
      final: { credit: "Artwork: diacritised Arabic set in Amiri" }, demo: { credit: "Artwork: diacritised Arabic set in Amiri" },
    },
    idea: {
      label: "01 · Comprehension", title: "One skeleton, many meanings",
      lede: "Written Arabic is usually published bare: consonants and long vowels only, with the short vowels and grammatical case endings (the diacritics, or tashkeel) left out. A single skeleton of letters can hide several unrelated words.",
      ambiguityLabel: "Same three letters, ع ل م",
      readings: [{ word: "عَلَمٌ", gloss: "a flag" }, { word: "عِلْمٌ", gloss: "knowledge" }, { word: "عَلَّمَ", gloss: "he taught" }],
      footnote: "Three readings. Zero difference on the page without diacritics.",
      reasons: [
        { title: "Why it matters", body: "Diacritisation is preprocessing for text-to-speech, machine translation, and information extraction: systems that need one unambiguous pronunciation, not a lattice of possibilities." },
        { title: "Why it's hard to benchmark", body: "Prior benchmarks leaked: training and test sets overlapped, and evaluations skewed toward either Classical or Modern Standard Arabic, inflating reported accuracy." },
      ],
    },
    team: {
      label: "02 · The people behind DiacriticS", title: "Meet the team", intro: "Four researchers connecting Arabic NLP, multilingual machine learning, and practical engineering to make Arabic text easier to read, study, and use.",
      members: [
        { role: "Mentor · KAUST", name: "Youssef S. Mohamed", href: "https://mo-youssef.github.io/", bio: "KAUST Ph.D. researcher in multimodal deep learning, affective image captioning, and multicultural modelling. He brings research experience from KAUST, the University of Tartu, and Zewail City, with a love for swimming and basketball beyond the lab." },
        { role: "AI/ML student · Team Leader · IAU", name: "Mahdi Alkhamis", href: "https://linktr.ee/psycodem", bio: "Senior AI/ML student At IAU focused on deep learning and computer vision. His work spans LAR-U-Net image denoising, the VisionFit posture-analysis app, and fine-tuning open-source LLMs for Arabic diacritisation; he also led 30+ CCSIT Club technical events." },
        { role: "Software Engineering student · KFUPM", name: "Mohammad Alali", href: "https://www.linkedin.com/in/mohammad-alali-9b992135a/", bio: "Software Engineering student at KFUPM and KAUST Academy AI/ML learner. He builds practical software with Python, C++, and Java, bringing a Mawhiba alumnus’s curiosity and an engineering perspective to the project." },
        { role: "Electrical & Computer Engineering student · KAU", name: "Saad Alnafjan", href: "https://www.linkedin.com/in/saadtalnafjan/", bio: "Electrical and Computer Engineering student at KAU. He combines a strong hardware and computational foundation with deep learning techniques, bringing hardware-software integration and analytical problem-solving to the team." },
      ] satisfies TeamMember[],
    },
    pipeline: {
      label: "03 · Method", title: "How the project works", intro: "A strict wall between what the model learns from and what it is judged on. Five stages, run in order.",
      stages: [
        { number: "01", title: "Train on Sadeed-Tashkeela", body: "The cleaned split of Tashkeela: unified diacritisation style, 50–60-word chunks, almost nothing left half-marked. Every example overlapping the Fadel test set was removed, cutting overlap to 0.4%: the train/test wall the project rests on. 1,042,698 examples, ~53M words." },
        { number: "02", title: "Test on SadeedDiac-25", body: "1,200 paragraphs, half Modern Standard and half Classical: 600 MSA paragraphs of 40–50 words (454 curated web articles, 146 from WikiNews) reviewed by two independent experts, and 600 Classical paragraphs from the Fadel test set. The modern half was diacritised in-house and never published diacritised, so no model had read it." },
        { number: "03", title: "Benchmark zero-shot, across scales", body: "Eleven open-weights models are scored cold, before any fine-tuning happens, establishing an honest baseline." },
        { number: "04", title: "Fine-tune with LoRA, and scale the data", body: "The strongest candidate model is adapted on the cleaned Tashkeela set with LoRA only: no QLoRA, no full-weight retraining. Training is then repeated over 10%, 30%, and 50% of the corpus, so the gain can be read against the amount of data it took." },
        { number: "05", title: "Score, then break the score apart", body: "Diacritic and Word Error Rates are computed with and without sentence-final case endings, then compared across Classical vs. Modern Standard subsets to isolate where errors come from." },
      ] satisfies PipelineStage[],
      formulaLabel: "The two headline metrics", formulas: [["DER", "= incorrect diacritics / total diacritised characters × 100"], ["WER", "= words with ≥1 diacritic error / total words × 100"]],
    },
    results: {
      label: "04 · Baseline findings", title: "Results on SadeedDiac-25",
      intro: "Zero-shot scores across twelve open-weights models, before any fine-tuning. Each figure averages the Modern Standard and Classical Arabic halves of the benchmark, and two patterns dominate every tier.",
      tableLabel: "Mean (MSA + CA) · SadeedDiac-25 · lower is better", tableRegionLabel: "Baseline model scores; scroll horizontally to see all columns", headers: { model: "Model", withCe: "with CE", noCe: "no CE" },
      footnote: "CE = sentence-final case endings (i'rab). Every figure is the mean of the Modern Standard and Classical Arabic halves of the benchmark, sorted by DER with case endings.",
      findings: [
        { tag: "Finding 1", title: "Sentence endings are the weak point", body: "Every model's error rate jumps on the last, case-marking diacritic of a sentence (i'rab) compared to diacritics inside a word, the clearest sign that these models resolve local spelling far better than sentence-level syntax.", example: "قَرَأَ الطَّالِبُ الكِتَابَ", gloss: "'the student', ُ nominative, reads correctly inside the word but breaks at the case ending" },
        { tag: "Finding 2", title: "Classical Arabic is the harder domain", body: "Error rates rise markedly moving from Modern Standard Arabic passages to Classical Arabic (CA) texts: evidence that domain-balanced training, not just more data, is what closes the gap.", example: "وَقَالَ الرَّاوِي فِي إِسْنَادِهِ", gloss: "dense classical narration syntax, the CA register with the highest observed error" },
      ] satisfies Finding[],
    },
    fineTuning: {
      label: "05 · Fine-tuning", title: "What fine-tuning buys",
      intro: "Both 4B targets were adapted with LoRA under identical conditions on nested slices of the training corpus, so the 10% slice sits inside the 30% and that inside the 50%. Only the data volume and the backbone differ.",
      tableLabel: "After LoRA fine-tuning · SadeedDiac-25 · lower is better",
      tableRegionLabel: "Fine-tuning scores; scroll horizontally to see all columns",
      headers: { model: "Model / training data", withCe: "with CE", noCe: "no CE" },
      footnote: "LoRA rank 16, effective batch 96, one epoch, 5% warmup ratio, seed 42, on nested subsets of Sadeed Tashkeela. Gemma leads at every fraction, but the gap narrows from 54.1 points to 1.6.",
      regimeLabel: "Qwen3.5-0.8B · adaptation regime · lower is better",
      regimeRegionLabel: "Adaptation regime scores; scroll horizontally to see all columns",
      regimeTitle: "LoRA or every weight?",
      regimeIntro: "Qwen3.5-0.8B is small enough to train end to end, which makes it the one place in this study where the adaptation regime can be isolated. Base checkpoint, corpus and prompt are held identical; only the fraction of weights allowed to move changes.",
      regimeFoot: "Full fine-tuning wins by 0.54 DER points. The narrow margin is the result: a rank-16 adapter recovers roughly 94% of the benefit of updating every weight. The LoRA row is scored on the benchmark only.",
      findings: [
        { tag: "Finding 3", title: "A weak start gains the most", body: "Qwen3.5-4B falls from 58.29% to 4.38% DER, and 51.2 of those 54 points are bought by the first tenth of the corpus alone. Gemma 4 E4B-it starts at 4.18%, already below where Qwen finishes, and reaches 2.81%.", example: "58.29 ← 4.38", gloss: "Qwen3.5-4B, zero-shot to half the corpus" },
        { tag: "Finding 4", title: "Size is not the binding constraint", body: "A 0.8B model trained end to end reaches 3.06% DER, beating a 4B model adapted with LoRA on half the corpus at a fifth of the parameter count. How much of the model is free to move matters more than how large it is.", example: "3.06 < 4.38", gloss: "0.8B full fine-tune against 4B LoRA" },
      ] satisfies Finding[],
    },
    analysis: {
      label: "Three analytical views", title: "What changed, what held, what remains open", intro: "The current local build uses repository-backed findings where they are comparable and marks every changing value as provisional.",
      chapters: [
        { number: "01", title: "LoRA → full fine-tuning → tagging", lead: "Task specialization changed the scale of the error.", body: "Corrected macro DER without case endings moved from approximately 8.5 for the tracked LoRA run to 2.6 for the full-fine-tuned checkpoint, then 2.1 for the character tagger.", caveat: "The tagger was warm-started from the 2.6 generative checkpoint. It improves DER, but its no-case-ending WER is worse on MSA. Completed runs did not perform decontamination.", status: "Corrected scorer · provisional public rounding", kind: "trajectory" },
        { number: "02", title: "Gemma versus Qwen architecture", lead: "The starting point matters more than the slope.", body: "Both 4B backbones were adapted under one configuration on nested slices of the corpus. Qwen3.5-4B falls from 58.29 to 4.38 DER with case endings; Gemma 4 E4B-it starts at 4.18, already below where Qwen finishes, and reaches 2.81.", caveat: "Architecture and adapted surface are confounded: a rank-16 adapter resolves to 258 matrices on Gemma against 128 on Qwen, so the two cannot be separated without matching trainable parameter counts.", status: "Corrected scorer · all fractions complete", kind: "trajectory" },
        { number: "03", title: "Zero-shot as a starting map", lead: "The first benchmark maps capability before adaptation.", body: "Zero-shot tests cover open-weights models across size and architecture tiers on the same balanced CA/MSA test set.", caveat: "The old baseline table used a generation-one scorer and is not directly comparable with corrected fine-tuning results.", status: "Generation-one scorer · context only", kind: "context" },
      ], metric: "Macro DER · without case endings · lower is better", details: "Scorer and provenance notes", tableModel: "Run", tableValue: "Macro DER",
    },
    references: { label: "06 · Grounding", title: "References" },
    final: { eyebrow: "From report to interaction", title: "Put Arabic text in. Inspect what the model returns.", body: "The local interface exposes the product behavior while the secure proxy keeps model credentials off the page.", cta: "Try the model" },
    footer: { note: "DiacriticS: a KAUST Academy research project on contamination-controlled Arabic diacritisation.", photoCredit: "Artwork generated from diacritised Arabic vocabulary; set in Amiri.", authors: "Youssef S. Mohamed (KAUST) · Mahdi Alkhamis (IAU) · Mohammad Alali (KFUPM) · Saad Alnafjan (KAU)", repo: "Code on GitHub" },
    demo: {
      eyebrow: "Interactive interface", title: "Try DiacriticS", inputLabel: "Arabic input", inputHint: "Maximum: 383 model tokens (typically about 1,130 Arabic characters). The exact character count depends on tokenization.", placeholder: "اكتب نصا عربيا من دون تشكيل", examplesLabel: "Try a real example", examples: ["اللغة العربية جميلة", "يكتب الباحث نتائج التجربة", "هذا نموذج متخصص في اللغة العربية"], loadExample: "Load an example", loadingExample: "Loading…", exampleNotice: "A held-out passage from the Sadeed_Tashkeela test split, never used to train this model.", exampleFallback: "Could not load the test-split passages; showing a built-in example.", submit: "Add tashkeel", submitting: "Adding tashkeel…", cancel: "Cancel", before: "Before", after: "After", empty: "The result will appear here.", emptyExampleLabel: "One skeleton, three readings", copy: "Copy result", copied: "Copied", clear: "Clear", removeInput: "Clear input", charsRemaining: "characters remaining", charsOver: "characters over the limit", count: "characters · 383-token limit", starting: "The GPU is starting. The first request after an idle period may take about 44 seconds.", errorDefault: "The request could not be completed. Please try again.",
      errors: { EMPTY_TEXT: "Enter Arabic text before submitting.", TEXT_TOO_LONG: "This text exceeds the model’s 383-token context. Shorten it and try again.", UNSUPPORTED_TEXT: "Enter Arabic text rather than an empty or technical payload.", MODEL_NOT_CONFIGURED: "The local proxy is working, but this input is not part of the curated mock. Connect the GPU endpoint to test arbitrary text.", RATE_LIMITED: "Too many requests. Wait a moment and try again.", UPSTREAM_TIMEOUT: "The model took too long to respond. Try again.", UPSTREAM_ERROR: "The model endpoint returned an error. Try again later.", MODEL_AUTH_ERROR: "The model connection needs its server credential refreshed." }, model: "Model", latency: "Response", back: "Back to the research",
    },
  },
  ar: {
    lang: "ar", dir: "rtl",
    meta: { homeTitle: "DiacriticS · التشكيل العربي بمعيار مقاوم للتلوث", homeDescription: "مشروع DiacriticS لضبط نماذج لغوية مفتوحة لتشكيل النص العربي وتقييمها على معيار SadeedDiac-25 المقاوم للتلوث.", demoTitle: "جرّب DiacriticS · واجهة التشكيل العربي", demoDescription: "جرّب واجهة DiacriticS وقارن النص العربي قبل التشكيل وبعده." },
    nav: { idea: "الفكرة", team: "الفريق", pipeline: "المنهجية", results: "النتائج", fineTuning: "الضبط الدقيق", references: "المراجع", demo: "جَرِّبِ النَّمُوذَجَ", language: "English", languageLabel: "Open the English version" },
    hero: {
      eyebrow: "بحثٌ في تقنيات اللغة العربية · DiacriticS", brandLabel: "داياكريتِكس", caption: "تشكيل · العلامات التي تحمل المعنى", support: "ضبط دقيق لنموذج لغوي لاستعادة الحركات العربية المفقودة، بتقييم صارم على معيار صُمم ليقاوم الحفظ والتلوث.", note: "مرّر الصفحة لتعود الحركات إلى مواضعها.", primary: "اقرأ عن المشروع", try: "جَرِّبِ النَّمُوذَجَ",
      resources: [
        { label: "تقرير المشروع", meta: "التقرير النهائي · ملف PDF", href: "/DiacriticS_Final_Report.pdf", external: true },
        { label: "الشيفرة على GitHub", meta: "السكربتات والإعدادات والنتائج", href: shared.githubUrl, external: true },
      ] satisfies ResourceLink[],
    },
    media: {
      hero: { credit: "عمل فني: نص عربي مُشكَّل بخط أميري" }, intro: { alt: "كلمات عربية معروضة بتشكيلها الكامل.", credit: "عمل فني: نص عربي مُشكَّل بخط أميري" }, method: { alt: "كلمات عربية معروضة بتشكيلها الكامل.", credit: "عمل فني: نص عربي مُشكَّل بخط أميري" }, analysis: { alt: "كلمات عربية معروضة بتشكيلها الكامل.", credit: "عمل فني: نص عربي مُشكَّل بخط أميري" }, final: { credit: "عمل فني: نص عربي مُشكَّل بخط أميري" }, demo: { credit: "عمل فني: نص عربي مُشكَّل بخط أميري" },
    },
    idea: {
      label: "01 · الفكرة", title: "هيكل واحد، معانٍ كثيرة", lede: "تُنشر العربية غالبًا بلا حركات: الحروف الساكنة والحروف المدّية فقط، بينما تُحذف الحركات القصيرة وعلامات الإعراب. هيكل واحد من الحروف قد يخفي كلمات مختلفة تمامًا في المعنى.", ambiguityLabel: "نفس الحروف الثلاثة: ع ل م", readings: [{ word: "عَلَمٌ", gloss: "عَلَم" }, { word: "عِلْمٌ", gloss: "عِلْم" }, { word: "عَلَّمَ", gloss: "فِعل التعليم" }], footnote: "ثلاث قراءات مختلفة، ولا فرق بينها على الصفحة دون الحركات.",
      reasons: [{ title: "لماذا يهمّ", body: "التشكيل خطوة تمهيدية لأنظمة تحويل النص إلى كلام، والترجمة الآلية، واستخراج المعلومات: أنظمة تحتاج نطقًا واحدًا واضحًا لا شبكة من الاحتمالات." }, { title: "لماذا يصعب قياسه بعدالة", body: "المعايير السابقة كانت تعاني من تسرّب البيانات: تداخل بين بيانات التدريب والاختبار، وميل نحو الفصحى التراثية أو المعاصرة فقط، مما ضخّم دقّة النتائج المُعلنة." }],
    },
    team: {
      label: "02 · فريق دياكريتيكس", title: "تعرّف على الفريق", intro: "أربعة باحثين يجمعون بين معالجة اللغة العربية والتعلم الآلي متعدد اللغات والهندسة العملية لجعل النص العربي أوضح وأسهل في القراءة والدراسة والاستخدام.",
      members: [
        { role: "مرشد · كاوست", name: "يوسف س. محمد", href: "https://mo-youssef.github.io/", bio: "باحث دكتوراه في كاوست يهتم بالتعلم العميق متعدد الوسائط، ووصف الصور العاطفي، والنمذجة متعددة الثقافات. يحمل خبرات بحثية من كاوست وجامعة تارتو ومدينة زويل، ويستمتع بالسباحة وكرة السلة خارج المختبر." },
        { role: "طالب ذكاء اصطناعي وتعلم آلي · قائد الفريق · جامعة الإمام عبدالرحمن بن فيصل", name: "مهدي الخميس", href: "https://linktr.ee/psycodem", bio: "طالب متقدم في الذكاء الاصطناعي والتعلم الآلي، يركز على التعلم العميق والرؤية الحاسوبية. تشمل أعماله LAR-U-Net لإزالة تشويش الصور، وتطبيق VisionFit لتحليل الوضعية، وضبط نماذج لغوية مفتوحة لتشكيل العربية، كما قاد أكثر من 30 فعالية تقنية في نادي CCSIT." },
        { role: "طالب هندسة برمجيات · جامعة الملك فهد للبترول والمعادن", name: "محمد العلي", href: "https://www.linkedin.com/in/mohammad-alali-9b992135a/", bio: "طالب هندسة برمجيات في جامعة الملك فهد ومتعلّم للذكاء الاصطناعي والتعلم الآلي في أكاديمية كاوست. يطوّر برمجيات عملية بلغة بايثون و++C وجافا، ويضيف للمشروع فضول خريج موهبة ومنظوراً هندسياً عملياً." },
        { role: "طالب هندسة كهربائية وحاسبات · جامعة الملك عبدالعزيز", name: "سعد النفجان", href: "https://www.linkedin.com/in/saadtalnafjan/", bio: "طالب هندسة كهربائية وهندسة حاسبات في جامعة الملك عبدالعزيز. يجمع بين الأساس الهندسي والتحليلي القوي وتقنيات التعلم العميق، مما يضيف للمشروع منظور التكامل بين الأجهزة والبرمجيات والحلول الهندسية التحليلية." },
      ] satisfies TeamMember[],
    },
    pipeline: {
      label: "03 · المنهجية", title: "كيف يعمل المشروع", intro: "فاصل صارم بين ما يتعلّم منه النموذج وما يُقيَّم عليه. خمس مراحل متسلسلة.",
      stages: [
        { number: "01", title: "التدريب على Sadeed-Tashkeela", body: "النسخة المنقّحة من مدونة تشكيلة: أسلوب تشكيل موحّد، وعيّنات من 50 إلى 60 كلمة، ولا يكاد يبقى نص ناقص التشكيل. وحُذف كل مثال يتداخل مع مجموعة اختبار Fadel حتى انخفض التداخل إلى 0.4%: وهو الفاصل بين التدريب والاختبار الذي يقوم عليه المشروع. 1,042,698 مثالًا بنحو 53 مليون كلمة." },
        { number: "02", title: "الاختبار على SadeedDiac-25", body: "1,200 فقرة، نصفها بالفصحى المعاصرة ونصفها بالعربية التراثية: 600 فقرة معاصرة من 40 إلى 50 كلمة (454 من مقالات مختارة و146 من ويكي الأخبار) راجعها خبيران مستقلان، و600 فقرة تراثية من مجموعة اختبار Fadel. والنصف المعاصر شُكِّل داخليًا ولم يُنشر مُشكَّلًا، فلم يطّلع عليه أي نموذج." },
        { number: "03", title: "قياس الأداء دون تدريب عبر أحجام مختلفة", body: "يُقيَّم أحد عشر نموذجًا مفتوح الأوزان دون أي تدريب مسبق، لبناء خط أساس صادق قبل أي ضبط دقيق." },
        { number: "04", title: "ضبط دقيق باستخدام LoRA مع تدرّج حجم البيانات", body: "يُكيَّف أفضل نموذج مرشّح على بيانات تشكيلة المنظّفة باستخدام LoRA فقط: دون QLoRA ودون إعادة تدريب كامل الأوزان. ثم يُعاد التدريب على 10% و30% و50% من البيانات، لقياس التحسّن مقابل حجم البيانات المستخدم." },
        { number: "05", title: "القياس ثم تحليل مصادر الخطأ", body: "تُحسب معدلات خطأ الحركات والكلمات مع علامات الإعراب النهائية وبدونها، ثم تُقارن بين نصوص الفصحى التراثية والمعاصرة لتحديد مصدر الأخطاء." },
      ] satisfies PipelineStage[], formulaLabel: "مقياسان رئيسيان", formulas: [["DER", "= الحركات الخاطئة ÷ إجمالي الحروف المُشكَّلة × 100"], ["WER", "= الكلمات التي بها خطأ حركة واحد فأكثر ÷ إجمالي الكلمات × 100"]],
    },
    results: {
      label: "04 · نتائج خط الأساس", title: "النتائج على معيار SadeedDiac-25", intro: "نتائج دون تدريب مسبق لاثني عشر نموذجًا مفتوح الأوزان، قبل أي ضبط دقيق. كل رقم هو متوسط نصفَي المعيار: الفصحى المعاصرة والعربية التراثية. ونمطان يتكرران في كل الفئات.", tableLabel: "المتوسط (الفصحى المعاصرة + التراثية) · معيار SadeedDiac-25 · الأقل أفضل", tableRegionLabel: "نتائج النماذج الأساسية؛ مرّر أفقيًا لرؤية جميع الأعمدة", headers: { model: "النموذج", withCe: "بالإعراب", noCe: "بدون الإعراب" }, footnote: "الإعراب = علامات الإعراب في أواخر الجمل. كل رقم هو متوسط نصفَي المعيار: الفصحى المعاصرة والعربية التراثية، مرتّبة حسب معدل خطأ الحركات مع الإعراب.",
      findings: [
        { tag: "النتيجة الأولى", title: "نهايات الجمل هي نقطة الضعف", body: "يرتفع معدّل الخطأ لدى جميع النماذج عند الحركة الأخيرة الدالة على الإعراب مقارنة بحركات داخل الكلمة، أوضح دليل على أن هذه النماذج تتقن التهجئة المحلية أكثر من تحليل التركيب النحوي للجملة كاملة.", example: "قَرَأَ الطَّالِبُ الكِتَابَ", gloss: "«الطالبُ»: الضمة الإعرابية هي أكثر ما يخطئ فيه النموذج" },
        { tag: "النتيجة الثانية", title: "العربية التراثية هي المجال الأصعب", body: "ترتفع معدّلات الخطأ بشكل ملحوظ عند الانتقال من نصوص الفصحى المعاصرة إلى نصوص العربية التراثية: دليل على أن التوازن بين المجالات في بيانات التدريب، لا مجرد زيادة حجمها، هو ما يُضيّق الفجوة.", example: "وَقَالَ الرَّاوِي فِي إِسْنَادِهِ", gloss: "تركيب سردي تراثي كثيف، من أعلى النصوص خطأً في العربية التراثية" },
      ] satisfies Finding[],
    },
    fineTuning: {
      label: "05 · الضبط الدقيق", title: "ماذا يضيف الضبط الدقيق",
      intro: "جرى ضبط النموذجين بحجم أربعة مليارات معامل بطريقة LoRA تحت ظروف متطابقة على شرائح متداخلة من بيانات التدريب، بحيث تقع شريحة 10٪ داخل 30٪ وهذه داخل 50٪. والاختلاف الوحيد هو حجم البيانات والبنية الأساسية.",
      tableLabel: "بعد الضبط الدقيق بطريقة LoRA · معيار SadeedDiac-25 · الأقل أفضل",
      tableRegionLabel: "نتائج الضبط الدقيق؛ مرّر أفقيًا لرؤية جميع الأعمدة",
      headers: { model: "النموذج / حجم بيانات التدريب", withCe: "بالإعراب", noCe: "بدون الإعراب" },
      footnote: "مُحوّل LoRA برتبة 16، وحجم دفعة فعّال 96، ودورة تدريب واحدة، وإحماء بنسبة 5٪، وبذرة عشوائية 42، على مجموعات متداخلة من بيانات سديد تشكيلة. يتفوّق جيما في كل النسب، لكن الفارق يضيق من 54.1 نقطة إلى 1.6.",
      regimeLabel: "‏Qwen3.5-0.8B · أسلوب الضبط · الأقل أفضل",
      regimeRegionLabel: "نتائج أسلوب الضبط؛ مرّر أفقيًا لرؤية جميع الأعمدة",
      regimeTitle: "لورا أم كل الأوزان؟",
      regimeIntro: "نموذج Qwen3.5-0.8B صغير بما يكفي لتدريبه بالكامل، وهو الموضع الوحيد في هذه الدراسة الذي يمكن فيه عزل أثر أسلوب الضبط. فنقطة الانطلاق والبيانات وصيغة التوجيه متطابقة تمامًا، ولا يتغيّر سوى نسبة الأوزان المسموح لها بالتغيّر.",
      regimeFoot: "يتفوّق الضبط الكامل بفارق 0.54 نقطة فقط. وهذا الفارق الضيّق هو النتيجة نفسها: إذ يستعيد مُحوّل LoRA برتبة 16 نحو 94٪ من فائدة تحديث كل الأوزان. وقد قُيّمت تجربة LoRA على المعيار وحده.",
      findings: [
        { tag: "النتيجة الثالثة", title: "النموذج الأضعف هو الأكثر استفادة", body: "ينخفض نموذج Qwen3.5-4B من 58.29٪ إلى 4.38٪، و51.2 نقطة من أصل 54 تتحقق من عُشر البيانات الأول وحده. أما Gemma 4 E4B-it فيبدأ من 4.18٪، أي أقل مما ينتهي إليه Qwen، ويصل إلى 2.81٪.", example: "58.29 ← 4.38", gloss: "‏Qwen3.5-4B من دون تدريب إلى نصف البيانات" },
        { tag: "النتيجة الرابعة", title: "الحجم ليس القيد الحاسم", body: "يصل نموذج بحجم 0.8 مليار معامل مُدرَّب بالكامل إلى 3.06٪، متفوّقًا على نموذج بأربعة مليارات معامل جرى ضبطه بطريقة LoRA على نصف البيانات، وبخُمس عدد المعاملات. فنسبة الأوزان القابلة للتغيّر أهم من حجم النموذج.", example: "3.06 < 4.38", gloss: "الضبط الكامل لـ 0.8 مليار مقابل LoRA لأربعة مليارات" },
      ] satisfies Finding[],
    },
    analysis: {
      label: "ثلاثة محاور للتحليل", title: "مَا الَّذِي تَغَيَّرَ؟ وَمَا الَّذِي بَقِيَ مَفْتُوحًا؟", intro: "البيانات الحالية مؤقتة، وتُعرض مع حدود توثيقها إلى أن يعتمد الفريق النتائج النهائية.",
      chapters: [
        { number: "01", title: "LoRA ← الضبط الكامل ← وسم المحارف", lead: "غيّر التخصيص للمهمّة مستوى الخطأ بصورة واضحة.", body: "انتقل DER الكلي المصحح من نحو 8.5 في تجربة LoRA إلى 2.6 في الضبط الكامل، ثم إلى 2.1 في نموذج وسم المحارف.", caveat: "بدأ نموذج الوسم من نقطة 2.6 التوليدية. تحسّن DER، لكنه لا يتفوّق في كل المقاييس، ولم تُنفّذ إزالة التلوّث في التجارب المكتملة.", status: "مصحح مُحدّث · تقريب مؤقت للنشر", kind: "trajectory" },
        { number: "02", title: "معمارية Gemma مقابل Qwen", lead: "تبقى المقارنة داخل إطار الاختبار الأصلي.", body: "[تُضاف المقارنة الرقمية بعد اعتماد جدول الاختبارات الجديد.]", caveat: "لن تُخلط أجيال المصحح أو نتائج النماذج التي ما زالت قيد الاختبار.", status: "بانتظار اعتماد البيانات", kind: "pending" },
        { number: "03", title: "الاختبار الصفري كنقطة بداية", lead: "يرسم الاختبار الأول خريطة القدرة قبل التخصيص.", body: "تشمل الاختبارات نماذج مفتوحة الأوزان من أحجام ومعماريات مختلفة على مجموعة CA/MSA المتوازنة.", caveat: "استخدم الجدول القديم الجيل الأول من المصحح، ولا يقارن مباشرةً بنتائج الضبط المصححة.", status: "الجيل الأول من المصحح · للسياق فقط", kind: "context" },
      ], metric: "DER الكلي · من دون أواخر الكلمات · الأقل أفضل", details: "تفاصيل المصحح والتوثيق", tableModel: "التجربة", tableValue: "DER الكلي",
    },
    references: { label: "06 · المصادر", title: "المراجع" },
    final: { eyebrow: "مِنَ التَّقْرِيرِ إِلَى التَّجْرِبَةِ", title: "أَدْخِلْ نَصًّا عَرَبِيًّا، وَتَفَحَّصِ النَّتِيجَةَ.", body: "تعرض الواجهة سلوك المنتج، بينما تُبقي طبقة الخادم مفاتيح النموذج بعيدًا عن المتصفح.", cta: "جَرِّبِ النَّمُوذَجَ" },
    footer: { note: "داياكريتِكس: مشروع بحثي من أكاديمية كاوست حول التشكيل العربي بمعايير خالية من التلوث.", photoCredit: "عمل فني مولَّد من مفردات عربية مُشكَّلة، بخط أميري.", authors: "Youssef S. Mohamed (KAUST) · Mahdi Alkhamis (IAU) · Mohammad Alali (KFUPM) · Saad Alnafjan (KAU)", repo: "الشيفرة على GitHub" },
    demo: {
      eyebrow: "واجهة تفاعلية", title: "جَرِّبْ DiacriticS", inputLabel: "النصّ العربي", inputHint: "الحد الأقصى 383 رمزًا للنموذج (نحو 1,130 محرفًا عربيًّا عادةً)، ويختلف العدد الدقيق باختلاف التجزئة.", placeholder: "اكتب نصا عربيا من دون تشكيل", examplesLabel: "جرّب مثالًا حقيقيًا", examples: ["اللغة العربية جميلة", "يكتب الباحث نتائج التجربة", "هذا نموذج متخصص في اللغة العربية"], loadExample: "حمّل مثالًا", loadingExample: "جارٍ التحميل…", exampleNotice: "فقرة من قسم الاختبار في Sadeed_Tashkeela، لم يتدرّب عليها النموذج قط.", exampleFallback: "تعذّر تحميل فقرات قسم الاختبار؛ تم عرض مثال مدمج.", submit: "أَضِفِ التَّشْكِيلَ", submitting: "تَجْرِي إِضَافَةُ التَّشْكِيلِ…", cancel: "إلغاء", before: "قبل", after: "بعد", empty: "ستظهر النتيجة هنا.", emptyExampleLabel: "هيكل واحد، ثلاث قراءات", copy: "انسخ النتيجة", copied: "تم النسخ", clear: "امسح", removeInput: "مسح النص", charsRemaining: "حرفًا متبقيًا", charsOver: "حرفًا زيادة عن الحد", count: "محرف · الحد 383 رمزًا", starting: "يجري تشغيل وحدة GPU. قد يستغرق أول طلب بعد فترة خمول نحو 44 ثانية.", errorDefault: "تعذّر إكمال الطلب. حاول مرة أخرى.", errors: { EMPTY_TEXT: "أدخل نصًّا عربيًّا أولًا.", TEXT_TOO_LONG: "يتجاوز النص سياق النموذج البالغ 383 رمزًا. اختصره ثم حاول مرة أخرى.", UNSUPPORTED_TEXT: "أدخل نصًّا عربيًّا صالحًا.", MODEL_NOT_CONFIGURED: "طبقة الخادم المحلية تعمل، لكن هذا النص ليس من الأمثلة المحدّدة. اربط نقطة GPU لتجربة أي نص.", RATE_LIMITED: "طلبات كثيرة خلال وقت قصير. انتظر قليلًا ثم حاول.", UPSTREAM_TIMEOUT: "استغرق النموذج وقتًا طويلًا. حاول مرة أخرى.", UPSTREAM_ERROR: "أعادت نقطة النموذج خطأً. حاول لاحقًا.", MODEL_AUTH_ERROR: "يحتاج اتصال النموذج إلى تحديث بيانات اعتماد الخادم." }, model: "النموذج", latency: "زمن الاستجابة", back: "العودة إلى البحث",
    },
  },
} as const;

export function routeFor(locale: Locale, page: "home" | "demo") {
  if (page === "home") return locale === "ar" ? "/ar/" : "/";
  return locale === "ar" ? "/ar/demo/" : "/demo/";
}
