import { INITIAL_BALANCE, COUNTRIES, PURPOSES, formatMoney, formatPhone, validPhone, calculateTransfer } from './transfer.js';
import { createExperience, featureKey, hasFeature, demoBalance, seedTransfers } from './payment-experience.js';

if (new URLSearchParams(location.search).get('embed') === '1' && parent !== window) document.documentElement.classList.add('embedded');
let experience = null;

const $ = (selector) => document.querySelector(selector);
const escapeHTML = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
const asset = (name, cls = '', alt = '') => `<img src="/assets/${name}.png" class="${cls}" alt="${alt}" draggable="false">`;
const paths = {
  back: '<path d="m14 5-7 7 7 7"/>', right: '<path d="m9 5 7 7-7 7"/>', down: '<path d="m5 9 7 7 7-7"/>',
  arrow: '<path d="M5 12h14m-6-6 6 6-6 6"/>', search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
  check: '<path d="m5 12 4 4L19 6"/>', phone: '<rect x="7" y="2" width="10" height="20" rx="1"/><path d="M10 18h4"/>',
  qr: '<path d="M3 3h6v6H3zm12 0h6v6h-6zM3 15h6v6H3zm12 0h2v2h-2zm4 4h2v2h-2zm-4 2v-2m6-6v2M3 12h4m5-9v4m0 7v3"/>',
  wifi: '<path d="M2 8a16 16 0 0 1 20 0M5 12a11 11 0 0 1 14 0M8 16a6 6 0 0 1 8 0m-5 4h2"/>',
  house: '<path d="m2 11 10-9 10 9M5 9v12h14V9M10 21v-7h4v7"/>', transport: '<rect x="5" y="5" width="14" height="14" rx="3"/><path d="M5 12h14M8 2h8M8 19l-2 3m10-3 2 3M8 15h1m6 0h1"/>',
  history: '<circle cx="12" cy="12" r="9"/><path d="M12 6v7H7"/>', delete: '<path d="M9 5h12v14H9l-7-7 7-7zm3 4 6 6m0-6-6 6"/>',
  calculator: '<rect x="6" y="2" width="12" height="20" rx="1"/><path d="M9 5h6M9 10h1m4 0h1m-6 4h1m4 0h1m-6 4h1m4 0h1"/>',
};
const icon = (name, cls = '') => `<svg class="icon ${cls}" viewBox="0 0 24 24" aria-hidden="true">${paths[name] || paths.right}</svg>`;
const flag = (country) => country.name === 'Абхазия' ? '<span class="flag flag-abkhazia" aria-hidden="true"></span>' : `<span class="flag" aria-hidden="true">${country.flag}</span>`;
const bankIcon = (country) => country.icon ? asset(country.icon, 'bank-icon') : `<span class="bank-icon bank-letter">${escapeHTML(country.bank[0])}</span>`;
const STORAGE_KEY = hasFeature ? `alfa-feature-transfers-v2-${featureKey}` : 'alfa-transfer-prototype-v1';
let transactions = hasFeature ? seedTransfers() : [];
try { const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); if (Array.isArray(stored)) transactions = stored.filter(t => t && typeof t.id === 'string' && Number.isFinite(t.total) && t.total >= 100 && COUNTRIES.some(c => c.name === t.country)); } catch { /* A blocked or empty store starts a fresh demo. */ }
if (hasFeature && !transactions.length) transactions = seedTransfers();
let balance = Math.max(0, Math.round(((hasFeature ? demoBalance : INITIAL_BALANCE) - transactions.filter(t => !t.synthetic_history).reduce((sum, t) => sum + t.total, 0)) * 100) / 100);
const freshDraft = () => ({ country: COUNTRIES.find(c => c.name === 'Таджикистан'), phone: '', recipient: false, purpose: null, amount: '', message: '', keyboard: false });
let draft = freshDraft();
let screen = 'payments';
let modal = null;
let amountError = '';
let sending = false;
let activeTransaction = null;
let info = { title: '', body: '' };
let toastTimer;
let lastFocus = null;
const steps = [
  ['Платежи', 'Начните с платежей', 'В разделе «Переводы» выберите «Переводы за рубеж».', 'payments'],
  ['Страна получателя', 'Выберите страну', 'В демосценарии доступны Беларусь, Кыргызстан, Таджикистан и Узбекистан.', 'countries'],
  ['Способ перевода', 'Достаточно номера', 'Выберите «По номеру телефона через СБП», как в примере из PDF.', 'methods'],
  ['Номер телефона', 'Кому переводим?', 'Укажите номер получателя или используйте пример +7 926 577-51-28.', 'recipient'],
  ['Банк получателя', 'Выберите банк', 'Для сценария из PDF выберите «Душанбе Сити Банк».', 'banks'],
  ['Назначение перевода', 'Укажите назначение', 'В примере — «Безвозмездный перевод на текущие расходы».', 'purpose'],
  ['Сумма перевода', 'Введите сумму', 'Попробуйте 500 ₽. Комиссия и сумма зачисления появятся перед отправкой.', 'amount'],
  ['Подтверждение', 'Проверьте и отправьте', 'Сверьте реквизиты и нажмите «Перевести». Затем можно сохранить демоквитанцию.', 'confirmation'],
];
function stepIndex() {
  if (modal === 'countries') return 1;
  if (modal === 'phone') return 3;
  if (modal === 'banks') return 4;
  if (modal === 'purpose') return 5;
  if (['confirmation', 'success'].includes(screen)) return 7;
  if (screen === 'methods') return 2;
  if (screen === 'transfer') return draft.recipient ? 6 : 3;
  return 0;
}
function updateGuide() {
  const current = stepIndex();
  $('#steps').innerHTML = steps.map((step, i) => `<li><button class="step-button ${i === current ? 'active' : i < current ? 'complete' : ''}" data-action="jump" data-step="${i}" ${i === current ? 'aria-current="step"' : ''}><span class="step-dot">${i < current ? '✓' : String(i + 1).padStart(2, '0')}</span>${step[0]}</button></li>`).join('');
  $('#context-number').textContent = `${String(current + 1).padStart(2, '0')} / 08`;
  $('#context-title').textContent = screen === 'success' ? 'Перевод выполнен' : steps[current][1];
  $('#context-description').textContent = screen === 'success' ? 'Демооперация сохранена в истории. Можно скачать квитанцию или повторить сценарий.' : steps[current][2];
}
function nav(active = 'payments') {
  return `<nav class="bottom-nav" aria-label="Основная навигация">${[['home', 'Главный'], ['payments', 'Платежи'], ['benefits', 'Выгода'], ['history', 'История'], ['chats', 'Чаты']].map(([key, label]) => `<button data-action="nav" data-page="${key}" aria-label="${label}" ${active === key ? 'aria-current="page"' : ''}>${asset('nav-' + key, '', label)}</button>`).join('')}</nav>`;
}
function topbar(title = '', action = 'back') { return `<header class="topbar"><button class="back" data-action="${action}" aria-label="Назад">${icon('back')}</button><span>${title}</span><span></span></header>`; }
function payments() {
  const transfers = [['accounts', 'Между<br>счетами'], ['phone', 'По номеру<br>телефона'], ['card', 'По номеру<br>карты'], ['details', 'По<br>реквизитам'], ['abroad', 'Переводы<br>за рубеж']];
  return `<div class="screen payments-screen"><div class="screen-body"><div class="payments-header"><div class="payments-heading"><button data-action="nav" data-page="home" aria-label="Профиль" style="padding:0">${asset('avatar', 'avatar')}</button><h2>Платежи</h2><button class="search-button" data-action="search" aria-label="Поиск платежей">${icon('search')}</button></div><div class="favorite-grid"><button class="favorite-card" data-action="my-payments">${asset('my-payments')}<span>Мои<br>платежи</span></button><button class="favorite-card" data-action="service" data-name="Дом">${asset('home')}<span><small>пр-кт.Олим...</small>Дом</span></button><button class="favorite-card" data-action="service" data-name="Штрафы ГАИ">${asset('car')}<span><small>Штрафы ГАИ</small>Авто</span></button></div></div><div class="payments-content"><h3 class="section-title">Переводы</h3><div class="transfer-grid">${transfers.map(([name, label]) => `<button class="transfer-tile" data-action="transfer-type" data-type="${name}">${asset('transfer-' + name)}<span>${label}</span></button>`).join('')}</div><div class="payment-list"><h3 class="section-title">Платежи</h3><button class="commission-banner" data-action="service" data-name="Все платежи без комиссии"><span>😍</span>Все платежи без комиссии</button>${[['qr', 'Оплата по QR'], ['phone', 'Мобильная связь'], ['wifi', 'Интернет, телефон, ТВ'], ['house', 'Коммунальные услуги'], ['transport', 'Транспорт']].map(([i, label]) => `<button class="service-row" data-action="service" data-name="${label}">${icon(i)}${label}</button>`).join('')}</div></div></div>${nav()}</div>`;
}
function methods() {
  return `<div class="screen">${topbar('', 'countries')}<div class="screen-body page-pad"><h2 class="page-title methods-title">Выберите способ перевода<br>в ${escapeHTML(draft.country.to)}</h2>${['По номеру карты', ...(draft.country.name === 'Таджикистан' ? ['По номеру карты (Korti Milli и Express Pay)'] : []), 'По номеру телефона через СБП'].map((label, i, all) => `<button class="method-row" data-action="${i === all.length - 1 ? 'phone-method' : 'other-method'}">${label}${icon('right', 'chevron')}</button>`).join('')}</div>${nav()}</div>`;
}
function accountCard() {
  return `<div class="account-card"><button class="account-row" data-action="accounts"><span class="account-symbol">₽</span><span class="account-info"><strong>${formatMoney(balance)} ₽</strong><small>Текущий счёт</small></span><span class="account-ending">··7238</span>${icon('down', 'chevron')}</button><button class="account-row" data-action="recipient">${draft.recipient ? `${bankIcon(draft.country)}<span class="account-info">Ш. А. Д.<small>${escapeHTML(formatPhone(draft.phone))}</small></span>` : '<span class="add-recipient">+</span><span class="recipient-placeholder">Укажите номер получателя</span>'}${icon('down', 'chevron')}</button></div>`;
}
function keypad() {
  return `<div class="keyboard" aria-label="Клавиатура для ввода суммы"><div class="quick-amounts"><button data-action="hide-keyboard" aria-label="Скрыть клавиатуру">${icon('calculator')}</button>${[500,1000,2000].map(n => `<button data-action="quick-amount" data-amount="${n}">${formatMoney(n, 0)} ₽</button>`).join('')}</div><div class="keys">${[['1',''],['2','ABC'],['3','DEF'],['4','GHI'],['5','JKL'],['6','MNO'],['7','PQRS'],['8','TUV'],['9','WXYZ'],[',',''],['0',''],['delete','']].map(([key, letters]) => `<button class="key ${[',','delete'].includes(key) ? 'special' : ''}" data-action="key" data-key="${key}" aria-label="${key === 'delete' ? 'Удалить цифру' : key === ',' ? 'Запятая' : key}">${key === 'delete' ? icon('delete') : key}${letters ? `<small>${letters}</small>` : ''}</button>`).join('')}</div></div>`;
}
function transfer() {
  return `<div class="screen">${topbar('По номеру телефона')}<div class="screen-body">${accountCard()}<p class="arrival-hint">${draft.recipient ? 'Зачисление в течение одной минуты через СБП' : 'Зачисление происходит моментально.'}</p>${draft.recipient ? `<button class="purpose-field" data-action="purpose"><span>${draft.purpose === null ? 'Выберите назначение перевода' : PURPOSES[draft.purpose][0]}</span>${icon('down', 'chevron')}</button>` : `<div class="message-field"><input id="message" maxlength="140" aria-label="Ваше сообщение" placeholder="Ваше сообщение" value="${escapeHTML(draft.message)}"><small>Максимум 140 символов</small></div>`}</div><div class="amount-area"><div class="amount-line"><button class="amount-value ${draft.amount ? '' : 'empty'}" data-action="amount" aria-label="Введите сумму перевода"><span id="amount-display">${draft.amount ? displayAmount(draft.amount) : '0'}</span>${draft.keyboard ? '<i class="cursor"></i>' : ''} ₽${draft.recipient ? icon('down','chevron') : ''}</button>${draft.recipient ? `<button class="continue-amount" data-action="review" aria-label="Продолжить перевод" ${draft.amount ? '' : 'disabled'}>${icon('arrow')}</button>` : ''}</div><button class="amount-caption" data-action="tariffs">${draft.recipient ? 'Расчёт комиссии — далее' : 'Тарифы и Лимиты'} <span class="info-circle">i</span></button><p class="amount-error" id="amount-error" role="alert">${escapeHTML(amountError)}</p></div>${draft.keyboard ? keypad() : nav()}</div>`;
}
function displayAmount(raw) { const [whole, fraction] = raw.split(','); return formatMoney(Number(whole || 0), 0) + (fraction !== undefined ? ',' + fraction : ''); }
function detail(label, value, extra = '') { return `<div class="detail"><dt>${label}</dt><dd>${escapeHTML(value)}</dd>${extra}</div>`; }
function confirmation() {
  const quote = calculateTransfer(draft.amount, draft.country, balance);
  if (quote.error) return transfer();
  return `<div class="screen">${topbar()}<div class="screen-body confirmation-content"><h2 class="page-title">Подтверждение перевода</h2><dl>${detail('Получатель', 'Ш. А. Д.')}${detail('Номер телефона получателя', formatPhone(draft.phone), asset('bank-sbp','sbp-mini'))}${detail('Банк получателя',draft.country.bank)}${detail('Страна банка получателя',draft.country.name)}${detail('Назначение перевода',PURPOSES[draft.purpose][0])}${detail('Сумма списания, включая все комиссии',formatMoney(quote.total, Number.isInteger(quote.total) ? 0 : 2) + ' ₽')}${detail('Сумма зачисления', formatMoney(quote.credited) + ' ' + quote.currency)}${detail('Курс конвертации', `1 ${quote.currency} = ${formatMoney(quote.rate, draft.country.name === 'Узбекистан' ? 4 : 2).replace(/0$/, '')} ₽`)}${detail('Комиссия Альфа-Банка', '30 ₽')}${detail('Комиссия банков-корреспондентов', '0 ₽')}${draft.message ? detail('Сообщение', draft.message) : ''}</dl><p class="demo-inline">Демонстрационный расчёт. ${draft.country.name === 'Таджикистан' ? 'Курс и комиссия воспроизводят пример из PDF.' : 'Для этой страны используется условный курс.'}</p></div><div class="confirm-action"><button class="primary" data-action="send" ${sending ? 'disabled' : ''}>${sending ? '<span class="spinner"></span>Переводим…' : 'Перевести'}</button></div>${nav()}</div>`;
}
function success() {
  const t = activeTransaction;
  return `<div class="screen success-screen">${topbar('', 'done')}<div class="success-content"><div class="success-check">${icon('check')}</div><h2>Перевод выполнен</h2><p class="success-subtitle">Деньги зачислены получателю</p><div class="success-amount">${formatMoney(t.total, Number.isInteger(t.total) ? 0 : 2)} ₽</div><p class="success-subtitle">${formatMoney(t.credited)} ${t.currency} получателю</p><dl class="success-summary">${detail('Получатель', 'Ш. А. Д.')}${detail('Номер телефона',t.phone)}${detail('Банк получателя',t.bank)}${detail('Комиссия включена в сумму','30 ₽')}</dl><button class="schedule-offer" data-action="schedule-demo"><span>↻</span><div><strong>Сделать платёж постоянным?</strong><small>Выберите дату и время следующего перевода</small></div>${icon('right','chevron')}</button><button class="secondary" data-action="receipt">Сохранить квитанцию ↓</button><p class="demo-inline">Демооперация. Реальные деньги не отправлялись.</p></div><div class="confirm-action"><button class="primary" data-action="done">Готово</button></div>${nav()}</div>`;
}
function historyScreen() {
  return `<div class="screen">${topbar('', 'done')}<div class="screen-body page-pad"><h2 class="page-title">История</h2>${transactions.length ? `<p class="demo-inline">Демонстрационные переводы</p>${transactions.slice().reverse().map(t => `<button class="history-item" data-action="transaction" data-id="${escapeHTML(t.id)}">${bankIcon(COUNTRIES.find(c => c.name === t.country))}<span class="history-item-text">${escapeHTML(t.bank)}<small>${escapeHTML(t.phone)}<br>${new Date(t.date).toLocaleString('ru-RU', { day: 'numeric', month: 'long', hour:'2-digit',minute:'2-digit' })} · Выполнен</small></span><span class="history-amount">−${formatMoney(t.total, Number.isInteger(t.total) ? 0 : 2)} ₽</span></button>`).join('')}` : `<div class="empty-state">${icon('history')}<h3>Здесь будут ваши переводы</h3><p>Отправьте первый демоперевод<br>по номеру телефона.</p><button class="secondary" data-action="countries">Перевести за рубеж</button></div>`}</div>${nav('history')}</div>`;
}
function homeScreen() { return `<div class="screen">${topbar('Главный', 'done')}<div class="screen-body page-pad"><h2 class="page-title">Добрый день, Дмитрий</h2><div class="balance-card"><small>Текущий счёт ··7238</small><h2>${formatMoney(balance)} ₽</h2><button class="primary" data-action="countries">Перевести за рубеж</button></div><p class="inline-note">Это демосчёт прототипа. ${hasFeature ? 'Начальный баланс — 83 500,35 ₽. Исторические переводы синтетические.' : 'Начальный баланс — 3 500,35 ₽, как в PDF.'}</p><button class="secondary" data-action="nav" data-page="history">История переводов</button><button class="secondary" data-action="reset-balance">Восстановить демобаланс</button></div>${nav('home')}</div>`; }
function sheet(title, body, footer = '', close = false) {
  return `<div class="sheet-overlay" data-action="backdrop"><section class="sheet" role="dialog" aria-modal="true" aria-label="${escapeHTML(title)}"><span class="sheet-handle"></span><header class="sheet-header"><h2>${title}</h2><button class="${close ? 'close-button' : 'cancel'}" data-action="close-modal" aria-label="Закрыть">${close ? '×' : 'Отмена'}</button></header><div class="sheet-body">${body}</div>${footer ? `<div class="sheet-bottom">${footer}</div>` : ''}</section></div>`;
}
function countryRows(query = '') {
  const list = COUNTRIES.filter(c => c.name.toLocaleLowerCase('ru').includes(query.trim().toLocaleLowerCase('ru')));
  return list.length ? list.map(c => `<button class="country-row" data-action="select-country" data-country="${c.name}">${flag(c)}<span>${c.name}</span></button>`).join('') : '<p class="no-results">Страна не найдена. Попробуйте другое название.</p>';
}
function phoneField() { return `<div class="phone-field"><span class="flag">🇷🇺</span><input id="recipient-phone" inputmode="tel" type="tel" autocomplete="off" maxlength="20" aria-label="Номер телефона получателя" placeholder="+7 000 000-00-00" value="${escapeHTML(draft.phone)}"><button class="close-button" data-action="clear-phone" aria-label="Очистить номер">×</button></div>`; }
function modalContent() {
  if (modal === 'countries') return sheet('Выберите страну', '<input class="search-field" id="country-search" aria-label="Поиск страны" placeholder="Страна" autocomplete="off"><div class="country-list" id="country-list">' + countryRows() + '</div>');
  if (modal === 'phone') return sheet('Кому', phoneField() + '<p class="phone-entry-help">Введите номер телефона получателя, привязанный к банковскому счёту.</p><button class="sample-phone" data-action="sample-phone">Использовать номер из PDF</button><p class="amount-error" id="phone-error" role="alert"></p>', '<button class="primary" data-action="find-banks">Продолжить</button>', true);
  if (modal === 'banks') return sheet('Кому', phoneField() + `<h3 class="sheet-subtitle">Выберите банк получателя</h3><div id="bank-list">${bankRows()}</div>`, '', true);
  if (modal === 'purpose') return sheet('Выберите назначение<br>перевода', '<p class="sheet-description">Его будет знать только банк получателя. Выбор назначения не влияет на размер комиссии</p>' + PURPOSES.map(([ru,en], i) => `<button class="purpose-row" data-action="select-purpose" data-purpose="${i}">${ru}<small>${en}</small></button>`).join(''));
  if (modal === 'accounts') return sheet('Откуда списать', `<button class="bank-row" data-action="close-modal"><span class="account-symbol">₽</span><span class="bank-row-text">${formatMoney(balance)} ₽<small>Текущий счёт ··7238</small></span><span style="margin-left:auto;color:#16a13d">${icon('check')}</span></button><p class="inline-note">Доступен один демонстрационный счёт.</p>`);
  if (modal === 'search') return sheet('Поиск платежей', '<input class="search-field" id="payment-search" aria-label="Поиск платежей" placeholder="Название платежа или перевода"><div id="search-results"><button class="service-row" data-action="countries">🌐 Переводы за рубеж</button></div>');
  if (modal === 'info') return sheet(info.title, `<div class="info-content">${info.body}</div>`, '<button class="primary" data-action="close-modal">Понятно</button>');
  return '';
}
function bankRows() {
  return `${draft.country.name !== 'Беларусь' ? `<button class="bank-row" data-action="domestic-bank">${asset('bank-alfa', 'bank-icon')}<span class="bank-row-text">Альфа-Банк<small>Дмитрий Александрович Ш.</small><small class="green-text">Основной банк получателя</small></span></button>` : ''}<button class="bank-row" data-action="select-bank">${bankIcon(draft.country)}<span class="bank-row-text">${draft.country.bank}<small>${draft.country.name}</small></span></button><button class="bank-row" data-action="other-bank">${asset('bank-sbp', 'bank-icon')}<span class="bank-row-text">В другой банк<small>Через СБП</small></span></button>`;
}
function render() {
  $('#app').innerHTML = (experience?.customScreen(screen) ?? ({ payments, methods, transfer, confirmation, success, history: historyScreen, home: homeScreen }[screen] || payments)()) + modalContent();
  if (modal) { $('#app > .screen').inert = true; requestAnimationFrame(() => { const focusable = $('.sheet button'); focusable?.focus({ preventScroll: true }); }); }
  updateGuide();
  experience?.afterRender();
}
function route(nextScreen, nextModal = null, replace = false) {
  if (sending) return;
  screen = nextScreen; modal = nextModal;
  history[replace ? 'replaceState' : 'pushState']({ screen, modal }, '', '#'+ (modal || screen));
  render();
}
function openModal(name) { lastFocus = document.activeElement; route(screen, name); }
function closeModal() { route(screen); if (lastFocus?.isConnected) lastFocus.focus(); }
function showInfo(title, body) { info = { title, body }; openModal('info'); }
function toast(message) { $('#toast').textContent = message; $('#toast').classList.add('visible'); clearTimeout(toastTimer); toastTimer = setTimeout(() => $('#toast').classList.remove('visible'), 3500); }
function resetDraft() { draft = freshDraft(); amountError = ''; activeTransaction = null; }
function jump(index) {
  if (sending) return;
  amountError = '';
  if (index === 0) return route('payments');
  if (index === 1) return route('payments','countries');
  if (index === 2) return route('methods');
  if (index === 3) { draft.recipient = false; draft.keyboard = false; return route('transfer'); }
  if (!draft.phone || !validPhone(draft.phone)) draft.phone = '+7 926 577-51-28';
  if (index === 4) return route('transfer','banks');
  draft.recipient = true;
  if (index === 5) return route('transfer','purpose');
  draft.purpose ??= 1;
  if (!draft.amount) draft.amount = '500';
  draft.keyboard = true;
  if (index === 6) return route('transfer');
  review();
}
function updateAmount() {
  amountError = '';
  const display = $('#amount-display'); if (display) display.textContent = draft.amount ? displayAmount(draft.amount) : '0';
  $('.amount-value')?.classList.toggle('empty', !draft.amount);
  const next = $('.continue-amount'); if (next) next.disabled = !draft.amount;
  if ($('#amount-error')) $('#amount-error').textContent = '';
  experience?.notify();
}
function typeAmount(key) {
  if (key === 'delete') draft.amount = draft.amount.slice(0,-1);
  else if (key === ',') { if (!draft.amount.includes(',')) draft.amount = (draft.amount || '0') + ','; }
  else if (/^\d$/.test(key)) {
    if (draft.amount.includes(',') && draft.amount.split(',')[1].length >= 2) return;
    if (!draft.amount.includes(',') && draft.amount.length >= 6) return;
    draft.amount = (draft.amount === '0' ? '' : draft.amount) + key;
  }
  updateAmount();
}
function review() {
  if (!draft.recipient || !validPhone(draft.phone)) return openModal('phone');
  if (draft.purpose === null) return openModal('purpose');
  const quote = calculateTransfer(draft.amount, draft.country, balance);
  if (quote.error) { amountError = quote.error; route('transfer'); return; }
  route('confirmation');
}
function findBanks() {
  if (!validPhone(draft.phone)) { $('#phone-error').textContent = 'Проверьте номер: например, +7 926 577-51-28'; $('#recipient-phone').setAttribute('aria-invalid','true'); $('#recipient-phone').focus(); return; }
  draft.phone = formatPhone(draft.phone);
  route('transfer','banks');
}
async function sendTransfer() {
  if (sending || screen !== 'confirmation') return;
  const quote = calculateTransfer(draft.amount, draft.country, balance);
  if (quote.error) { amountError = quote.error; route('transfer'); return; }
  if (!validPhone(draft.phone) || draft.purpose === null || !draft.recipient) return review();
  sending = true; render();
  await new Promise(resolve => setTimeout(resolve, 1000));
  activeTransaction = { id: crypto.randomUUID?.() || `demo-${Date.now()}-${Math.random().toString(16).slice(2)}`, date: experience.now(), recipientName: draft.recipientName || 'Ш. А. Д.', country: draft.country.name, bank: draft.country.bank, phone: formatPhone(draft.phone), purpose: PURPOSES[draft.purpose][0], message: draft.message, ...quote };
  transactions.push(activeTransaction);
  balance = Math.round((balance - quote.total) * 100) / 100;
  let saved = true;
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(transactions)); } catch { saved = false; }
  sending = false; draft.amount = ''; draft.keyboard = false;
  route('success', null, true);
  experience.onTransfer();
  if (!saved) toast('Перевод выполнен. История доступна только до закрытия страницы.');
}
function downloadReceipt() {
  const t = activeTransaction;
  const body = ['ДЕМОНСТРАЦИОННАЯ КВИТАНЦИЯ', 'Прототип по макету Альфа-Банка', 'Не является банковским документом. Деньги не переводились.', '', `Номер: ${t.id}`, `Дата: ${new Date(t.date).toLocaleString('ru-RU')}`, 'Статус: Выполнен (демо)', 'Получатель: Ш. А. Д.', `Телефон: ${t.phone}`, `Банк: ${t.bank}`, `Страна: ${t.country}`, `Назначение: ${t.purpose}`, `Сумма списания: ${formatMoney(t.total)} ₽`, `Комиссия: ${formatMoney(t.fee)} ₽`, `Зачислено: ${formatMoney(t.credited)} ${t.currency}`, `Демонстрационный курс: 1 ${t.currency} = ${t.rate} ₽`, ...(t.message ? [`Сообщение: ${t.message}`] : [])].join('\n');
  const url = URL.createObjectURL(new Blob(['\uFEFF' + body], { type: 'text/plain;charset=utf-8' }));
  const link = document.createElement('a'); link.href = url; link.download = `demo-transfer-${t.id.slice(0,8)}.txt`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); toast('Демоквитанция сохранена');
}
const actions = {
  'jump': b => jump(Number(b.dataset.step)),
  'countries': () => openModal('countries'),
  'transfer-type': b => b.dataset.type === 'abroad' || b.dataset.type === 'phone' ? openModal('countries') : showInfo('Способ перевода', '<p>В этом прототипе воспроизведён перевод за рубеж по номеру телефона.</p><button class="secondary" data-action="countries">Перейти к выбору страны</button>'),
  'select-country': b => { const c = COUNTRIES.find(c => c.name === b.dataset.country); if (!c.available) return toast('В демо доступны Беларусь, Кыргызстан, Таджикистан и Узбекистан.'); if (draft.country.name !== c.name) { draft.recipient = false; draft.purpose = null; draft.amount = ''; } draft.country = c; draft.keyboard = false; route('methods'); },
  'phone-method': () => { draft.keyboard = false; route('transfer'); },
  'other-method': () => showInfo('Перевод по номеру карты', '<p>В PDF показан сценарий по номеру телефона. Для продолжения выберите этот способ перевода.</p><button class="secondary" data-action="phone-method">По номеру телефона через СБП</button>'),
  'recipient': () => openModal(draft.recipient ? 'banks' : 'phone'),
  'sample-phone': () => { draft.phone = '+7 926 577-51-28'; $('#recipient-phone').value = draft.phone; $('#phone-error').textContent = ''; $('#recipient-phone').removeAttribute('aria-invalid'); },
  'clear-phone': () => { draft.phone = ''; draft.recipient = false; route('transfer','phone'); requestAnimationFrame(() => $('#recipient-phone').focus()); },
  'find-banks': findBanks,
  'select-bank': () => { if (!validPhone(draft.phone)) { draft.recipient = false; return route('transfer','phone'); } draft.phone = formatPhone(draft.phone); draft.recipient = true; draft.keyboard = false; route('transfer','purpose'); },
  'domestic-bank': () => toast('Для перевода за рубеж выберите банк в стране получателя: ' + draft.country.bank + '.'),
  'other-bank': () => toast('В демосценарии доступен ' + draft.country.bank + '. Выберите его в списке.'),
  'purpose': () => openModal('purpose'),
  'select-purpose': b => { draft.purpose = Number(b.dataset.purpose); draft.keyboard = true; route('transfer'); },
  'amount': () => { if (!draft.recipient) return openModal('phone'); if (draft.purpose === null) return openModal('purpose'); draft.keyboard = true; render(); },
  'key': b => typeAmount(b.dataset.key),
  'quick-amount': b => { draft.amount = b.dataset.amount; updateAmount(); },
  'hide-keyboard': () => { draft.keyboard = false; render(); },
  'review': review,
  'send': sendTransfer,
  'receipt': downloadReceipt,
  'done': () => { resetDraft(); route('payments'); },
  'close-modal': closeModal,
  'backdrop': (b,e) => { if (e.target === b) closeModal(); },
  'back': () => { if (modal) return closeModal(); if (screen === 'confirmation') { draft.keyboard = true; return route('transfer'); } if (screen === 'transfer') return route('methods'); route('payments'); },
  'accounts': () => openModal('accounts'),
  'tariffs': () => showInfo('Тарифы и лимиты', `<h3>Демонстрационный расчёт</h3><p>Комиссия — 30 ₽, включена в сумму списания. Перевод от 100 до 100 000 ₽ в пределах остатка на демосчёте.</p><h3>Пример из PDF</h3><p>При переводе 500 ₽ получатель получает 51,65 TJS. Курс: 1 TJS = 9,1 ₽.</p><p class="inline-note">Условия нужны для работы прототипа и не являются актуальными тарифами банка. Для других стран используются условные курсы.</p>`),
  'nav': b => { if (['payments','history','home'].includes(b.dataset.page)) return route(b.dataset.page); showInfo(b.dataset.page === 'benefits' ? 'Выгода' : 'Чаты', `<p>${b.dataset.page === 'benefits' ? 'Предложения и кешбэк' : 'Обращения в поддержку'} не входят в сценарий перевода из PDF.</p><button class="secondary" data-action="countries">Перевести за рубеж</button>`); },
  'service': b => showInfo(escapeHTML(b.dataset.name), '<p>Этот раздел показан на главном экране PDF. Интерактивный сценарий прототипа — перевод за рубеж по номеру телефона.</p><button class="secondary" data-action="countries">Перевести за рубеж</button>'),
  'search': () => openModal('search'),
  'transaction': b => { activeTransaction = transactions.find(t => t.id === b.dataset.id); route('success'); },
  'reset-balance': () => showInfo('Восстановить демобаланс', `<p>${hasFeature ? 'Восстановятся начальный демобаланс и два исторических перевода.' : 'На счёте снова будет 3 500,35 ₽. История демонстрационных переводов будет очищена.'}</p><button class="secondary" data-action="confirm-reset">Восстановить и очистить историю</button>`),
  'confirm-reset': () => { if (hasFeature) { experience.resetTransfers(); route('home'); return; } transactions = []; balance = INITIAL_BALANCE; try { localStorage.removeItem(STORAGE_KEY); } catch {} resetDraft(); route('home'); toast('Демобаланс восстановлен'); },
};
document.addEventListener('click', event => {
  const button = event.target.closest('[data-action]');
  if (!button || button.disabled || sending) return;
  if (experience?.action(button.dataset.action, button)) return;
  actions[button.dataset.action]?.(button, event);
});
document.addEventListener('input', event => {
  const input = event.target;
  if (input.id === 'country-search') $('#country-list').innerHTML = countryRows(input.value);
  if (input.id === 'recipient-phone') {
    draft.phone = input.value;
    if ($('#phone-error')) $('#phone-error').textContent = '';
    input.removeAttribute('aria-invalid');
    if ($('#bank-list')) $('#bank-list').innerHTML = validPhone(draft.phone) ? bankRows() : '<p class="no-results">Введите полный номер телефона для выбора банка.</p>';
  }
  if (input.id === 'message') draft.message = input.value;
  if (input.id === 'payment-search') $('#search-results').innerHTML = 'переводы за рубеж по номеру телефона'.includes(input.value.toLowerCase().trim()) ? '<button class="service-row" data-action="countries">🌐 Переводы за рубеж</button>' : '<p class="no-results">Ничего не найдено. Попробуйте «переводы».</p>';
});
document.addEventListener('keydown', event => {
  if (sending) { event.preventDefault(); return; }
  if ($('#reference-dialog').open) return;
  if (document.querySelector('.phone-confirm[open]')) return;
  if (event.key === 'Escape') { if (modal) closeModal(); else if (draft.keyboard) { draft.keyboard = false; render(); } return; }
  if (event.key === 'Tab' && modal) {
    const focusable = [...document.querySelectorAll('.sheet button:not(:disabled), .sheet input')];
    const first = focusable[0], last = focusable.at(-1);
    if (event.shiftKey && document.activeElement === first) { last?.focus(); event.preventDefault(); }
    else if (!event.shiftKey && document.activeElement === last) { first?.focus(); event.preventDefault(); }
  }
  if (event.key === 'Enter' && modal === 'phone') { event.preventDefault(); findBanks(); return; }
  if (event.target.matches('input, textarea')) return;
  if (screen === 'transfer' && draft.keyboard && !modal) {
    if (/^[0-9.,]$/.test(event.key) || event.key === 'Backspace') { event.preventDefault(); typeAmount(event.key === 'Backspace' ? 'delete' : event.key === '.' ? ',' : event.key); }
    if (event.key === 'Enter') { event.preventDefault(); review(); }
  }
});
$('#reset').addEventListener('click', () => { if (!sending) { resetDraft(); route('payments'); } });
$('#start-demo').addEventListener('click', () => { if (!sending) { resetDraft(); route('payments','countries'); } });
$('#reference-toggle').addEventListener('click', () => { const index = stepIndex(); $('#reference-title').textContent = `${index+1}. ${steps[index][0]}`; $('#reference-image').src = `/assets/reference-${steps[index][3]}.png`; $('#reference-dialog').showModal(); });
$('#reference-close').addEventListener('click', () => $('#reference-dialog').close());
$('#reference-dialog').addEventListener('click', event => { if (event.target === $('#reference-dialog')) $('#reference-dialog').close(); });
window.addEventListener('popstate', event => {
  if (sending) return;
  const state = event.state;
  screen = state?.screen || 'payments'; modal = state?.modal || null;
  if (screen === 'success' && !activeTransaction) screen = 'payments';
  if (screen === 'confirmation' && (!draft.amount || !draft.recipient || draft.purpose === null)) screen = 'transfer';
  render();
});
experience = createExperience({
  get: () => ({ screen, modal, draft, transactions, balance, activeTransaction, sending }),
  render, route, toast, escapeHTML, nav, topbar, detail,
  prefill: t => {
    if (!t) return route('payments');
    draft = { ...freshDraft(), country: COUNTRIES.find(c => c.name === t.country), phone: t.phone, recipientName: t.recipientName, recipient: true, purpose: Math.max(0, PURPOSES.findIndex(p => p[0] === t.purpose)), amount: String(t.total).replace('.', ','), message: t.message || '', keyboard: true };
    amountError = ''; route('transfer');
  },
  reset: () => {
    transactions = hasFeature ? seedTransfers() : []; balance = hasFeature ? demoBalance : INITIAL_BALANCE;
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(transactions)); } catch {}
    resetDraft(); route('payments');
  },
});
history.replaceState({ screen:'payments',modal:null }, '', '#payments');
render();
