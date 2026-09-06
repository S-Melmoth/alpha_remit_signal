import test from 'node:test';
import assert from 'node:assert/strict';
import { COUNTRIES, INITIAL_BALANCE, validPhone, formatPhone, calculateTransfer } from '../public/transfer.js';
const tajikistan = COUNTRIES.find(c => c.name === 'Таджикистан');

test('PDF example: total includes fee and recipient receives 51.65 TJS', () => {
  assert.deepEqual(calculateTransfer('500', tajikistan, INITIAL_BALANCE), { total:500, fee:30, credited:51.65, rate:9.1, currency:'TJS' });
});
test('Reject invalid amounts and insufficient balance', () => {
  for (const amount of ['', 'abc', '-500', '99', 'Infinity', '100001', '3500,36']) assert.ok(calculateTransfer(amount, tajikistan, INITIAL_BALANCE).error, amount);
  assert.equal(calculateTransfer('3500,35',tajikistan,INITIAL_BALANCE).total,INITIAL_BALANCE);
  assert.ok(calculateTransfer('500',tajikistan,499.99).error);
});
test('Phone validation and normalization', () => {
  for (const number of ['+7 926 577-51-28', '89265775128', '+992901234567', '+375291234567', '+996700123456', '+998901234567']) assert.equal(validPhone(number),true,number);
  for (const number of ['', '+7 926', '+792657751281', '+00000000000']) assert.equal(validPhone(number),false,number);
  assert.equal(formatPhone('89265775128'), '+7 926 577-51-28');
});
test('All four supported countries have working demonstration quotes', () => {
  const supported = COUNTRIES.filter(c => c.available);
  assert.equal(supported.length, 4);
  for (const country of supported) {
    const quote = calculateTransfer('500,50',country,INITIAL_BALANCE);
    assert.equal(quote.total,500.5);
    assert.ok(quote.credited > 0);
    assert.equal(quote.currency,country.currency);
  }
});
