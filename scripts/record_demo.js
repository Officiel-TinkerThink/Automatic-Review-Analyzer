const { chromium } = require('playwright');
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 }, recordVideo: { dir: 'assets/raw-video', size: { width: 1280, height: 800 } } });
  const page = await ctx.newPage();
  await page.addInitScript(() => { document.addEventListener('DOMContentLoaded', () => {
    const c = document.createElement('div'); c.style.cssText = 'position:fixed;z-index:9999;width:22px;height:22px;border-radius:50%;background:rgba(15,118,110,.4);border:2px solid #fff;pointer-events:none;transform:translate(-50%,-50%);transition:transform .08s;left:-100px;top:-100px;box-shadow:0 2px 8px rgba(0,0,0,.4)';
    document.body.appendChild(c); document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; });
    document.addEventListener('mousedown', () => { c.style.transform = 'translate(-50%,-50%) scale(.7)'; }); document.addEventListener('mouseup', () => { c.style.transform = 'translate(-50%,-50%) scale(1)'; }); }); });
  await page.goto('http://127.0.0.1:8000/index.html'); await sleep(4000);
  async function glideClick(sel, pause = 400) { const b = await page.locator(sel).first().boundingBox(); await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 14 }); await sleep(pause); await page.mouse.down(); await sleep(60); await page.mouse.up(); }
  await glideClick('#reviewText', 300); await page.fill('#reviewText', ''); await page.type('#reviewText', 'The cookies were stale and the box arrived crushed. Really disappointed for the price.', { delay: 35 }); await sleep(1600);
  await page.fill('#reviewText', ''); await page.type('#reviewText', 'Wow — crunchy, fresh and perfectly salted. My kids loved these chips, will buy again!', { delay: 30 }); await sleep(1600);
  await page.selectOption('#anModel', 'perceptron'); await sleep(1200);
  await page.selectOption('#anModel', 'pegasos'); await sleep(600);
  await glideClick('.tab[data-tab="train"]', 400); await sleep(600);
  await glideClick('#btnTrain', 400); await sleep(3500);
  await glideClick('#btnSweep', 400); await sleep(9000);
  await page.evaluate(() => document.querySelector('#sweep').scrollIntoView({ behavior: 'smooth', block: 'center' })); await sleep(1800);
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'smooth' })); await sleep(400);
  await glideClick('.tab[data-tab="words"]', 400); await sleep(2200);
  await glideClick('.tab[data-tab="reviews"]', 400); await page.selectOption('#rvFilter', 'wrong'); await sleep(2000);
  await ctx.close(); await browser.close(); console.log('recorded');
})();
