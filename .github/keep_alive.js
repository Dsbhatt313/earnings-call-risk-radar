// Keeps a Streamlit Community Cloud app awake.
//
// A plain HTTP ping (e.g. UptimeRobot) does NOT work: the sleeping app still
// returns HTTP 200, and Streamlit's inactivity timer counts real app sessions
// (websocket connections), not page loads. So we open the app in a real
// headless browser, and if the "Yes, get this app back up!" button is showing,
// we click it to wake the app.

const { chromium } = require('playwright');

const APP_URL = process.env.APP_URL || 'https://earningscallriskradar.streamlit.app/';

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  let woke = false;

  try {
    console.log(`Visiting ${APP_URL}`);
    await page.goto(APP_URL, { waitUntil: 'domcontentloaded', timeout: 60000 });

    // The wake button text can vary slightly across Streamlit versions.
    const wakeButton = page.getByRole('button', {
      name: /get this app back up|Yes, get this app/i,
    });

    // Give the sleeping-page (or the app) a moment to render.
    try {
      await wakeButton.waitFor({ state: 'visible', timeout: 15000 });
      console.log('App was asleep — clicking the wake button.');
      await wakeButton.click();
      woke = true;
      // Wait for the app to actually spin back up.
      await page.waitForLoadState('networkidle', { timeout: 120000 }).catch(() => {});
      await page.waitForTimeout(20000);
    } catch {
      console.log('No wake button found — app is already awake.');
    }

    // Keep a session open briefly so it registers as genuine activity.
    await page.waitForTimeout(10000);

    const title = await page.title();
    console.log(`Done. Page title: "${title}". Woke app: ${woke}`);
  } catch (err) {
    console.error('Keep-alive run failed:', err);
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
})();
