import { createRequire } from 'node:module';
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';

const require = createRequire(import.meta.url);
const { chromium } = require(process.argv[2] || 'playwright');
const browser = await chromium.launch({ headless: true, ...(process.argv[3] ? { executablePath: process.argv[3] } : {}) });
const errors = [];
await mkdir('reference/verification', { recursive: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
  await page.goto('http://localhost:5173');
  assert.match(await page.locator('h1').innerText(), /Ближе,\s*чем кажется/);
  await page.screenshot({ path: 'reference/verification/landing-desktop.png', fullPage: true });
  for (const [code, city] of Object.entries({ uz: 'Ташкент', kg: 'Бишкек', by: 'Минск', tj: 'Душанбе' })) {
    await page.locator(`[data-destination="${code}"]`).click();
    assert.equal(await page.locator('#destination-city').innerText(), city);
    assert.equal(await page.locator('[data-destination][aria-pressed="true"]').count(), 1);
  }
  await page.getByRole('link', { name: 'Как это работает', exact: true }).click();
  assert.equal(new URL(page.url()).hash, '#how-it-works');
  await page.locator('.nav-try').click();
  assert.equal(new URL(page.url()).pathname, '/payments');
  await page.locator('[data-type="abroad"]').waitFor();
  await page.reload();
  await page.locator('[data-type="abroad"]').waitFor();
  await page.locator('.brand').click();
  await page.locator('#hero-title').waitFor();
  for (const width of [1024, 768, 390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    await page.goto('http://localhost:5173');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `overflow at ${width}px`);
    assert.equal(await page.locator('.nav-try').isVisible(), true);
    if (width === 390 || width === 320) await page.screenshot({ path: `reference/verification/landing-${width}.png`, fullPage: true });
  }
  await page.locator('.nav-try').click();
  await page.locator('[data-type="abroad"]').waitFor();
  const response = await page.goto('http://localhost:5173/payments/');
  assert.equal(response.status(), 200);
  await page.locator('[data-type="abroad"]').waitFor();
  assert.deepEqual(errors, []);
  console.log('PASS: Landing, four destinations, navigation, payments reload, 320–1440px layouts, no browser errors.');
} finally {
  await browser.close();
}
