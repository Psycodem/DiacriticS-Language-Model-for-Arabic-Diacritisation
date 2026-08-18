// DiacriticS — language toggle only.
// This site ships no diacritisation model in the browser: it presents
// the project's idea, method, and results. Switching EN/AR just swaps
// the interface copy (data-en / data-ar attributes) and page direction.

(function () {
  const root = document.documentElement;
  const toggle = document.getElementById('langToggle');
  const nodes = document.querySelectorAll('[data-en][data-ar]');

  function applyLang(lang) {
    root.setAttribute('data-lang', lang);
    root.setAttribute('lang', lang);
    root.setAttribute('dir', lang === 'ar' ? 'rtl' : 'ltr');

    nodes.forEach((el) => {
      const text = el.getAttribute(lang === 'ar' ? 'data-ar' : 'data-en');
      if (text != null) el.textContent = text;
    });

    toggle.classList.toggle('is-ar', lang === 'ar');
    localStorage.setItem('diacritics-lang', lang);
  }

  toggle.addEventListener('click', () => {
    const current = root.getAttribute('data-lang') || 'en';
    applyLang(current === 'en' ? 'ar' : 'en');
  });

  // Restore prior choice, default to English.
  const saved = localStorage.getItem('diacritics-lang');
  if (saved === 'ar') applyLang('ar');
})();

// Presentation only: the nav gains a hairline once the page scrolls, and
// blocks fade in as they enter view. Both are added from JS so the page
// stays fully readable when scripting is off.
(function () {
  const nav = document.querySelector('.nav');
  if (nav) {
    const setStuck = () => nav.classList.toggle('is-stuck', window.scrollY > 8);
    setStuck();
    window.addEventListener('scroll', setStuck, { passive: true });
  }

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduced || !('IntersectionObserver' in window)) return;

  const blocks = document.querySelectorAll(
    '.section__head, .idea__lede, .ambiguity-card, .why-item, .team-card, ' +
    '.step, .formula-block, .scoreboard, .finding-card, .ref-card'
  );

  const io = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('is-in');
      io.unobserve(entry.target);
    });
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.06 });

  blocks.forEach((el, i) => {
    el.classList.add('reveal');
    el.style.transitionDelay = (i % 4) * 60 + 'ms';
    io.observe(el);
  });
})();
