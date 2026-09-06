export const INITIAL_BALANCE = 3500.35;
export const COUNTRIES = [
  { name: 'Абхазия', flag: '🏳️', available: false },
  { name: 'Азербайджан', flag: '🇦🇿', available: false },
  { name: 'Армения', flag: '🇦🇲', available: false },
  { name: 'Беларусь', to: 'Беларусь', flag: '🇧🇾', available: true, currency: 'BYN', rate: 28.6, bank: 'Альфа-Банк', icon: 'bank-alfa' },
  { name: 'Вьетнам', flag: '🇻🇳', available: false },
  { name: 'Грузия', flag: '🇬🇪', available: false },
  { name: 'Индия', flag: '🇮🇳', available: false },
  { name: 'Индонезия', flag: '🇮🇩', available: false },
  { name: 'Казахстан', flag: '🇰🇿', available: false },
  { name: 'Кыргызстан', to: 'Кыргызстан', flag: '🇰🇬', available: true, currency: 'KGS', rate: 1.05, bank: 'МБанк', icon: null },
  { name: 'Таджикистан', to: 'Таджикистан', flag: '🇹🇯', available: true, currency: 'TJS', rate: 9.1, bank: 'Душанбе Сити Банк', icon: 'bank-dushanbe' },
  { name: 'Узбекистан', to: 'Узбекистан', flag: '🇺🇿', available: true, currency: 'UZS', rate: 0.0072, bank: 'Капиталбанк', icon: null },
];
export const PURPOSES = [
  ['Себе на свой счёт', 'Transfer on my own account'],
  ['Безвозмездный перевод на текущие расходы', 'Non-repayable transfer for current expenses'],
  ['Перевод за услуги', 'Payment for services'],
  ['Оплата за товары для личных нужд', 'Payment for goods for personal use'],
];
export function formatMoney(amount, digits = 2) {
  return Number(amount).toLocaleString('ru-RU', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
export function formatPhone(raw) {
  const digits = raw.replace(/\D/g, '');
  if (digits.length === 11 && (digits[0] === '7' || digits[0] === '8')) return `+7 ${digits.slice(1,4)} ${digits.slice(4,7)}-${digits.slice(7,9)}-${digits.slice(9,11)}`;
  return '+' + digits;
}
export function validPhone(raw) {
  const digits = raw.replace(/\D/g, '');
  return /^(7|8)\d{10}$/.test(digits) || /^(375|992|996|998)\d{9}$/.test(digits);
}
export function calculateTransfer(amount, country, balance) {
  const value = Number(String(amount).replace(',', '.'));
  if (!Number.isFinite(value) || value < 100) return { error: 'Минимальная сумма перевода — 100 ₽' };
  if (value > 100000) return { error: 'Максимальная сумма перевода — 100 000 ₽' };
  if (Math.round(value * 100) > Math.round(balance * 100)) return { error: 'Недостаточно денег на счёте' };
  // Fixed demonstration tariff reproduces the PDF: 500 ₽ total, 30 ₽ fee, 51.65 TJS.
  const fee = 30;
  return { total: Math.round(value * 100) / 100, fee, credited: Math.round(((value - fee) / country.rate) * 100) / 100, rate: country.rate, currency: country.currency };
}
