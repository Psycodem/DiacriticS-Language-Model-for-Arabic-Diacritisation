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
