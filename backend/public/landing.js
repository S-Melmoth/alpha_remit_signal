// Vanilla adaptation of the pointer tilt in ruixen.ui's Credit Card Hero,
// retrieved via 21st MCP. No React or animation library is needed here.
const art = document.querySelector('.hero-art');
const scene = document.querySelector('#art-scene');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const finePointer = matchMedia('(pointer: fine)');

art.addEventListener('pointermove', event => {
  if (reducedMotion.matches || !finePointer.matches) return;
  const box = art.getBoundingClientRect();
  const x = (event.clientX - box.left) / box.width - .5;
  const y = (event.clientY - box.top) / box.height - .5;
  scene.style.setProperty('--tilt-x', `${-y * 7}deg`);
  scene.style.setProperty('--tilt-y', `${x * 7}deg`);
});
const resetTilt = () => {
  scene.style.setProperty('--tilt-x', '0deg');
  scene.style.setProperty('--tilt-y', '0deg');
};
art.addEventListener('pointerleave', resetTilt);
reducedMotion.addEventListener('change', resetTilt);

const cities = { tj: 'Душанбе', uz: 'Ташкент', kg: 'Бишкек', by: 'Минск' };
document.querySelectorAll('[data-destination]').forEach(button => {
  button.addEventListener('click', () => {
    const destination = button.dataset.destination;
    document.querySelectorAll('[data-destination]').forEach(option => option.setAttribute('aria-pressed', String(option === button)));
    document.querySelector('#destination-city').textContent = cities[destination];
    document.querySelector('#destination-flag').className = `flag flag-${destination}`;
    document.querySelector('#destination-status').textContent = `Направление на иллюстрации: Москва — ${cities[destination]}`;
  });
});

const tryDialog = document.querySelector('#try-dialog');
document.querySelectorAll('.try-trigger').forEach(button => button.addEventListener('click', () => tryDialog.showModal()));
document.querySelector('#try-close').addEventListener('click', () => tryDialog.close());
tryDialog.addEventListener('click', event => {
  if (event.target === tryDialog) tryDialog.close();
});
