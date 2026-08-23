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
    { model: "gemma-4-E4B-it", derCe: "5.74", werCe: "10.17", derNoCe: "5.15", werNoCe: "6.86", best: true },
    { model: "Tashkeel-350M-v2", derCe: "7.56", werCe: "14.53", derNoCe: "6.49", werNoCe: "9.65" },
    { model: "Flan-T5-Tashkeel-Small", derCe: "11.52", werCe: "19.94", derNoCe: "10.56", werNoCe: "14.12" },
    { model: "Fine-Tashkeel", derCe: "19.06", werCe: "22.52", derNoCe: "18.80", werNoCe: "20.25" },
    { model: "Fanar-1-9B-Instruct", derCe: "24.98", werCe: "29.58", derNoCe: "24.72", werNoCe: "26.78" },
    { model: "Glonor-ByT5-Arabic", derCe: "41.35", werCe: "48.76", derNoCe: "39.48", werNoCe: "42.58" },
    { model: "Gemma-3-1B-pt-10k-diacritization", derCe: "41.58", werCe: "47.98", derNoCe: "42.26", werNoCe: "44.26" },
    { model: "Qwen3.5-9B", derCe: "55.19", werCe: "65.93", derNoCe: "55.44", werNoCe: "60.49" },
    { model: "Qwen3.5-4B", derCe: "73.74", werCe: "81.86", derNoCe: "73.49", werNoCe: "77.25" },
    { model: "aya-expanse-8b", derCe: "92.35", werCe: "92.78", derNoCe: "92.02", werNoCe: "91.85" },
    { model: "Moonlight-16B-A3B-Instruct", derCe: "99.38", werCe: "98.68", derNoCe: "99.31", werNoCe: "98.50" },
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
    nav: { idea: "Idea", team: "Team", pipeline: "Pipeline", results: "Results", references: "References", demo: "Try the model", language: "العربية", languageLabel: "Open the Arabic version" },
    hero: {
      eyebrow: "Arabic language research · DiacriticS", brandLabel: "DiacriticS", caption: "tashkīl: the marks that carry the meaning",
      support: "Fine-tuning a language model to restore Arabic's missing vowels, evaluated where it counts: on a benchmark built to resist memorisation.",
      note: "Scroll to restore the marks that written Arabic often leaves out.", primary: "Read the project", try: "Try the model",
      resources: [
        { label: "Project report", meta: "PDF · coming soon", disabled: true },
        { label: "Code on GitHub", meta: "Scripts, configs, results", href: shared.githubUrl, external: true },
      ] satisfies ResourceLink[],
    },
    media: {
      hero: { credit: "Photo: Pixels Elements / Pexels" },
      intro: { alt: "Close-up of flowing carved ornament and Arabic calligraphy on an Andalusian arch.", credit: "Photo: Daka / Pexels" },
      method: { alt: "An ornately carved archway frames a flowering courtyard garden, with a reclining statue resting on the windowsill.", credit: "Photo: AXP Photography / Pexels" },
      analysis: { alt: "A long reflecting pool leads through an Andalusian courtyard lined with arches and orange trees.", credit: "Photo: Igor Passchier / Pexels" },
      final: { credit: "Photo: Daka / Pexels" }, demo: { credit: "Photo: Daka / Pexels" },
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
      intro: "Zero-shot scores across eleven open-weights models, before fine-tuning. Each figure averages the Modern Standard and Classical Arabic halves of the benchmark, and two patterns dominate every tier.",
      tableLabel: "Mean (MSA + CA) · SadeedDiac-25 · lower is better", tableRegionLabel: "Baseline model scores; scroll horizontally to see all columns", headers: { model: "Model", withCe: "with CE", noCe: "no CE" },
      footnote: "CE = sentence-final case endings (i'rab). Every figure is the mean of the Modern Standard and Classical Arabic halves of the benchmark, sorted by DER with case endings.",
      findings: [
        { tag: "Finding 1", title: "Sentence endings are the weak point", body: "Every model's error rate jumps on the last, case-marking diacritic of a sentence (i'rab) compared to diacritics inside a word, the clearest sign that these models resolve local spelling far better than sentence-level syntax.", example: "قَرَأَ الطَّالِبُ الكِتَابَ", gloss: "'the student', ُ nominative, reads correctly inside the word but breaks at the case ending" },
        { tag: "Finding 2", title: "Classical Arabic is the harder domain", body: "Error rates rise markedly moving from Modern Standard Arabic passages to Classical Arabic (CA) texts: evidence that domain-balanced training, not just more data, is what closes the gap.", example: "وَقَالَ الرَّاوِي فِي إِسْنَادِهِ", gloss: "dense classical narration syntax, the CA register with the highest observed error" },
      ] satisfies Finding[],
    },
    analysis: {
      label: "Three analytical views", title: "What changed, what held, what remains open", intro: "The current local build uses repository-backed findings where they are comparable and marks every changing value as provisional.",
      chapters: [
        { number: "01", title: "LoRA → full fine-tuning → tagging", lead: "Task specialization changed the scale of the error.", body: "Corrected macro DER without case endings moved from approximately 8.5 for the tracked LoRA run to 2.6 for the full-fine-tuned checkpoint, then 2.1 for the character tagger.", caveat: "The tagger was warm-started from the 2.6 generative checkpoint. It improves DER, but its no-case-ending WER is worse on MSA. Completed runs did not perform decontamination.", status: "Corrected scorer · provisional public rounding", kind: "trajectory" },
        { number: "02", title: "Gemma versus Qwen architecture", lead: "Architecture comparisons stay inside their original benchmark frame.", body: "The site will present like-for-like zero-shot comparisons and the selection logic for task-specific training once the refreshed run table is authorized.", caveat: "Values are intentionally withheld here rather than mixing scorer generations or presenting unfinished model tests as final evidence.", status: "Authorized data refresh pending", kind: "pending" },
        { number: "03", title: "Zero-shot as a starting map", lead: "The first benchmark maps capability before adaptation.", body: "Zero-shot tests cover open-weights models across size and architecture tiers on the same balanced CA/MSA test set.", caveat: "The old baseline table used a generation-one scorer and is not directly comparable with corrected fine-tuning results.", status: "Generation-one scorer · context only", kind: "context" },
      ], metric: "Macro DER · without case endings · lower is better", details: "Scorer and provenance notes", tableModel: "Run", tableValue: "Macro DER",
    },
    references: { label: "05 · Grounding", title: "References" },
    final: { eyebrow: "From report to interaction", title: "Put Arabic text in. Inspect what the model returns.", body: "The local interface exposes the product behavior while the secure proxy keeps model credentials off the page.", cta: "Try the model" },
    footer: { note: "DiacriticS: a KAUST Academy research project on contamination-controlled Arabic diacritisation.", photoCredit: "Photos: Pixels Elements, Dark Astraal, Igor Passchier, AXP Photography, and Daka / Pexels.", authors: "Youssef S. Mohamed (KAUST) · Mahdi Alkhamis (IAU) · Mohammad Alali (KFUPM) · Saad Alnafjan (KAU)", repo: "Code on GitHub" },
    demo: {
      eyebrow: "Interactive interface", title: "Try DiacriticS", inputLabel: "Arabic input", inputHint: "Maximum: 383 model tokens (typically about 1,130 Arabic characters). The exact character count depends on tokenization.", placeholder: "اكتب نصا عربيا من دون تشكيل", examplesLabel: "Try a real example", examples: ["اللغة العربية جميلة", "يكتب الباحث نتائج التجربة", "هذا نموذج متخصص في اللغة العربية"], loadExample: "Load an example", loadingExample: "Loading…", exampleNotice: "A held-out passage from the Sadeed_Tashkeela test split, never used to train this model.", exampleFallback: "Could not load the test-split passages; showing a built-in example.", submit: "Add tashkeel", submitting: "Adding tashkeel…", cancel: "Cancel", before: "Before", after: "After", empty: "The result will appear here.", emptyExampleLabel: "One skeleton, three readings", copy: "Copy result", copied: "Copied", clear: "Clear", removeInput: "Clear input", charsRemaining: "characters remaining", charsOver: "characters over the limit", count: "characters · 383-token limit", starting: "The GPU is starting. The first request after an idle period may take about 44 seconds.", errorDefault: "The request could not be completed. Please try again.",
      errors: { EMPTY_TEXT: "Enter Arabic text before submitting.", TEXT_TOO_LONG: "This text exceeds the model’s 383-token context. Shorten it and try again.", UNSUPPORTED_TEXT: "Enter Arabic text rather than an empty or technical payload.", MODEL_NOT_CONFIGURED: "The local proxy is working, but this input is not part of the curated mock. Connect the GPU endpoint to test arbitrary text.", RATE_LIMITED: "Too many requests. Wait a moment and try again.", UPSTREAM_TIMEOUT: "The model took too long to respond. Try again.", UPSTREAM_ERROR: "The model endpoint returned an error. Try again later.", MODEL_AUTH_ERROR: "The model connection needs its server credential refreshed." }, model: "Model", latency: "Response", back: "Back to the research",
    },
  },
  ar: {
    lang: "ar", dir: "rtl",
    meta: { homeTitle: "DiacriticS · التشكيل العربي بمعيار مقاوم للتلوث", homeDescription: "مشروع DiacriticS لضبط نماذج لغوية مفتوحة لتشكيل النص العربي وتقييمها على معيار SadeedDiac-25 المقاوم للتلوث.", demoTitle: "جرّب DiacriticS · واجهة التشكيل العربي", demoDescription: "جرّب واجهة DiacriticS وقارن النص العربي قبل التشكيل وبعده." },
    nav: { idea: "الفكرة", team: "الفريق", pipeline: "المنهجية", results: "النتائج", references: "المراجع", demo: "جَرِّبِ النَّمُوذَجَ", language: "English", languageLabel: "Open the English version" },
    hero: {
      eyebrow: "بحثٌ في تقنيات اللغة العربية · DiacriticS", brandLabel: "داياكريتِكس", caption: "تشكيل · العلامات التي تحمل المعنى", support: "ضبط دقيق لنموذج لغوي لاستعادة الحركات العربية المفقودة، بتقييم صارم على معيار صُمم ليقاوم الحفظ والتلوث.", note: "مرّر الصفحة لتعود الحركات إلى مواضعها.", primary: "اقرأ عن المشروع", try: "جَرِّبِ النَّمُوذَجَ",
      resources: [
        { label: "تقرير المشروع", meta: "ملف PDF · قريبًا", disabled: true },
        { label: "الشيفرة على GitHub", meta: "السكربتات والإعدادات والنتائج", href: shared.githubUrl, external: true },
      ] satisfies ResourceLink[],
    },
    media: {
      hero: { credit: "تصوير: Pixels Elements / Pexels" }, intro: { alt: "تفصيل قريب لزخارف نباتية وكتابة عربية منحوتة على قوس أندلسي.", credit: "تصوير: Daka / Pexels" }, method: { alt: "قوس منحوت بزخارف دقيقة يؤطر حديقة أندلسية مزهرة، وتمثال متكئ على حافة النافذة.", credit: "تصوير: AXP Photography / Pexels" }, analysis: { alt: "حوض عاكس طويل يمتد عبر فناء أندلسي تحيط به الأقواس وأشجار النارنج.", credit: "تصوير: Igor Passchier / Pexels" }, final: { credit: "تصوير: Daka / Pexels" }, demo: { credit: "تصوير: Daka / Pexels" },
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
        { number: "٠١", title: "التدريب على Sadeed-Tashkeela", body: "النسخة المنقّحة من مدونة تشكيلة: أسلوب تشكيل موحّد، وعيّنات من 50 إلى 60 كلمة، ولا يكاد يبقى نص ناقص التشكيل. وحُذف كل مثال يتداخل مع مجموعة اختبار Fadel حتى انخفض التداخل إلى 0.4%: وهو الفاصل بين التدريب والاختبار الذي يقوم عليه المشروع. 1,042,698 مثالًا بنحو 53 مليون كلمة." },
        { number: "٠٢", title: "الاختبار على SadeedDiac-25", body: "1,200 فقرة، نصفها بالفصحى المعاصرة ونصفها بالعربية التراثية: 600 فقرة معاصرة من 40 إلى 50 كلمة (454 من مقالات مختارة و146 من ويكي الأخبار) راجعها خبيران مستقلان، و600 فقرة تراثية من مجموعة اختبار Fadel. والنصف المعاصر شُكِّل داخليًا ولم يُنشر مُشكَّلًا، فلم يطّلع عليه أي نموذج." },
        { number: "٠٣", title: "قياس الأداء دون تدريب عبر أحجام مختلفة", body: "يُقيَّم أحد عشر نموذجًا مفتوح الأوزان دون أي تدريب مسبق، لبناء خط أساس صادق قبل أي ضبط دقيق." },
        { number: "٠٤", title: "ضبط دقيق باستخدام LoRA مع تدرّج حجم البيانات", body: "يُكيَّف أفضل نموذج مرشّح على بيانات تشكيلة المنظّفة باستخدام LoRA فقط: دون QLoRA ودون إعادة تدريب كامل الأوزان. ثم يُعاد التدريب على 10% و30% و50% من البيانات، لقياس التحسّن مقابل حجم البيانات المستخدم." },
        { number: "٠٥", title: "القياس ثم تحليل مصادر الخطأ", body: "تُحسب معدلات خطأ الحركات والكلمات مع علامات الإعراب النهائية وبدونها، ثم تُقارن بين نصوص الفصحى التراثية والمعاصرة لتحديد مصدر الأخطاء." },
      ] satisfies PipelineStage[], formulaLabel: "مقياسان رئيسيان", formulas: [["DER", "= الحركات الخاطئة ÷ إجمالي الحروف المُشكَّلة × 100"], ["WER", "= الكلمات التي بها خطأ حركة واحد فأكثر ÷ إجمالي الكلمات × 100"]],
    },
    results: {
      label: "04 · نتائج خط الأساس", title: "النتائج على معيار SadeedDiac-25", intro: "نتائج دون تدريب مسبق لأحد عشر نموذجًا مفتوح الأوزان، قبل الضبط الدقيق. كل رقم هو متوسط نصفَي المعيار: الفصحى المعاصرة والعربية التراثية. ونمطان يتكرران في كل الفئات.", tableLabel: "المتوسط (الفصحى المعاصرة + التراثية) · معيار SadeedDiac-25 · الأقل أفضل", tableRegionLabel: "نتائج النماذج الأساسية؛ مرّر أفقيًا لرؤية جميع الأعمدة", headers: { model: "النموذج", withCe: "بالإعراب", noCe: "بدون الإعراب" }, footnote: "الإعراب = علامات الإعراب في أواخر الجمل. كل رقم هو متوسط نصفَي المعيار: الفصحى المعاصرة والعربية التراثية، مرتّبة حسب معدل خطأ الحركات مع الإعراب.",
      findings: [
        { tag: "النتيجة الأولى", title: "نهايات الجمل هي نقطة الضعف", body: "يرتفع معدّل الخطأ لدى جميع النماذج عند الحركة الأخيرة الدالة على الإعراب مقارنة بحركات داخل الكلمة، أوضح دليل على أن هذه النماذج تتقن التهجئة المحلية أكثر من تحليل التركيب النحوي للجملة كاملة.", example: "قَرَأَ الطَّالِبُ الكِتَابَ", gloss: "«الطالبُ»: الضمة الإعرابية هي أكثر ما يخطئ فيه النموذج" },
        { tag: "النتيجة الثانية", title: "العربية التراثية هي المجال الأصعب", body: "ترتفع معدّلات الخطأ بشكل ملحوظ عند الانتقال من نصوص الفصحى المعاصرة إلى نصوص العربية التراثية: دليل على أن التوازن بين المجالات في بيانات التدريب، لا مجرد زيادة حجمها، هو ما يُضيّق الفجوة.", example: "وَقَالَ الرَّاوِي فِي إِسْنَادِهِ", gloss: "تركيب سردي تراثي كثيف، من أعلى النصوص خطأً في العربية التراثية" },
      ] satisfies Finding[],
    },
    analysis: {
      label: "ثلاثة محاور للتحليل", title: "مَا الَّذِي تَغَيَّرَ؟ وَمَا الَّذِي بَقِيَ مَفْتُوحًا؟", intro: "البيانات الحالية مؤقتة، وتُعرض مع حدود توثيقها إلى أن يعتمد الفريق النتائج النهائية.",
      chapters: [
        { number: "٠١", title: "LoRA ← الضبط الكامل ← وسم المحارف", lead: "غيّر التخصيص للمهمّة مستوى الخطأ بصورة واضحة.", body: "انتقل DER الكلي المصحح من نحو 8.5 في تجربة LoRA إلى 2.6 في الضبط الكامل، ثم إلى 2.1 في نموذج وسم المحارف.", caveat: "بدأ نموذج الوسم من نقطة 2.6 التوليدية. تحسّن DER، لكنه لا يتفوّق في كل المقاييس، ولم تُنفّذ إزالة التلوّث في التجارب المكتملة.", status: "مصحح مُحدّث · تقريب مؤقت للنشر", kind: "trajectory" },
        { number: "٠٢", title: "معمارية Gemma مقابل Qwen", lead: "تبقى المقارنة داخل إطار الاختبار الأصلي.", body: "[تُضاف المقارنة الرقمية بعد اعتماد جدول الاختبارات الجديد.]", caveat: "لن تُخلط أجيال المصحح أو نتائج النماذج التي ما زالت قيد الاختبار.", status: "بانتظار اعتماد البيانات", kind: "pending" },
        { number: "٠٣", title: "الاختبار الصفري كنقطة بداية", lead: "يرسم الاختبار الأول خريطة القدرة قبل التخصيص.", body: "تشمل الاختبارات نماذج مفتوحة الأوزان من أحجام ومعماريات مختلفة على مجموعة CA/MSA المتوازنة.", caveat: "استخدم الجدول القديم الجيل الأول من المصحح، ولا يقارن مباشرةً بنتائج الضبط المصححة.", status: "الجيل الأول من المصحح · للسياق فقط", kind: "context" },
      ], metric: "DER الكلي · من دون أواخر الكلمات · الأقل أفضل", details: "تفاصيل المصحح والتوثيق", tableModel: "التجربة", tableValue: "DER الكلي",
    },
    references: { label: "05 · المصادر", title: "المراجع" },
    final: { eyebrow: "مِنَ التَّقْرِيرِ إِلَى التَّجْرِبَةِ", title: "أَدْخِلْ نَصًّا عَرَبِيًّا، وَتَفَحَّصِ النَّتِيجَةَ.", body: "تعرض الواجهة سلوك المنتج، بينما تُبقي طبقة الخادم مفاتيح النموذج بعيدًا عن المتصفح.", cta: "جَرِّبِ النَّمُوذَجَ" },
    footer: { note: "داياكريتِكس: مشروع بحثي من أكاديمية كاوست حول التشكيل العربي بمعايير خالية من التلوث.", photoCredit: "الصور: Pixels Elements وDark Astraal وIgor Passchier وAXP Photography وDaka / Pexels.", authors: "Youssef S. Mohamed (KAUST) · Mahdi Alkhamis (IAU) · Mohammad Alali (KFUPM) · Saad Alnafjan (KAU)", repo: "الشيفرة على GitHub" },
    demo: {
      eyebrow: "واجهة تفاعلية", title: "جَرِّبْ DiacriticS", inputLabel: "النصّ العربي", inputHint: "الحد الأقصى 383 رمزًا للنموذج (نحو 1,130 محرفًا عربيًّا عادةً)، ويختلف العدد الدقيق باختلاف التجزئة.", placeholder: "اكتب نصا عربيا من دون تشكيل", examplesLabel: "جرّب مثالًا حقيقيًا", examples: ["اللغة العربية جميلة", "يكتب الباحث نتائج التجربة", "هذا نموذج متخصص في اللغة العربية"], loadExample: "حمّل مثالًا", loadingExample: "جارٍ التحميل…", exampleNotice: "فقرة من قسم الاختبار في Sadeed_Tashkeela، لم يتدرّب عليها النموذج قط.", exampleFallback: "تعذّر تحميل فقرات قسم الاختبار؛ تم عرض مثال مدمج.", submit: "أَضِفِ التَّشْكِيلَ", submitting: "تَجْرِي إِضَافَةُ التَّشْكِيلِ…", cancel: "إلغاء", before: "قبل", after: "بعد", empty: "ستظهر النتيجة هنا.", emptyExampleLabel: "هيكل واحد، ثلاث قراءات", copy: "انسخ النتيجة", copied: "تم النسخ", clear: "امسح", removeInput: "مسح النص", charsRemaining: "حرفًا متبقيًا", charsOver: "حرفًا زيادة عن الحد", count: "محرف · الحد 383 رمزًا", starting: "يجري تشغيل وحدة GPU. قد يستغرق أول طلب بعد فترة خمول نحو 44 ثانية.", errorDefault: "تعذّر إكمال الطلب. حاول مرة أخرى.", errors: { EMPTY_TEXT: "أدخل نصًّا عربيًّا أولًا.", TEXT_TOO_LONG: "يتجاوز النص سياق النموذج البالغ 383 رمزًا. اختصره ثم حاول مرة أخرى.", UNSUPPORTED_TEXT: "أدخل نصًّا عربيًّا صالحًا.", MODEL_NOT_CONFIGURED: "طبقة الخادم المحلية تعمل، لكن هذا النص ليس من الأمثلة المحدّدة. اربط نقطة GPU لتجربة أي نص.", RATE_LIMITED: "طلبات كثيرة خلال وقت قصير. انتظر قليلًا ثم حاول.", UPSTREAM_TIMEOUT: "استغرق النموذج وقتًا طويلًا. حاول مرة أخرى.", UPSTREAM_ERROR: "أعادت نقطة النموذج خطأً. حاول لاحقًا.", MODEL_AUTH_ERROR: "يحتاج اتصال النموذج إلى تحديث بيانات اعتماد الخادم." }, model: "النموذج", latency: "زمن الاستجابة", back: "العودة إلى البحث",
    },
  },
} as const;

export function routeFor(locale: Locale, page: "home" | "demo") {
  if (page === "home") return locale === "ar" ? "/ar/" : "/";
  return locale === "ar" ? "/ar/demo/" : "/demo/";
}
