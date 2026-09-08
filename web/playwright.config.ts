import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1, // All cases deliberately exercise one local PeopleOS runtime.
  retries: 0, // A flaky first attempt is a release failure, not hidden by retry.
  timeout: 90_000,
  expect: { timeout: 15_000 },
  forbidOnly: Boolean(process.env.CI),
  outputDir: 'browser-artifacts/test-results',
  reporter: [['list'], ['html', { outputFolder: 'browser-artifacts/report', open: 'never' }], ['json', { outputFile: 'browser-artifacts/results.json' }]],
  use: {
    baseURL: 'http://127.0.0.1:3000',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    { name: 'chromium-desktop', use: { ...devices['Desktop Chrome'] } },
    { name: 'chromium-mobile', use: { ...devices['Pixel 7'] } },
  ],
  webServer: {
    command: `${process.env.PEOPLEOS_TEST_PYTHON || 'python'} ../scripts/run_browser_acceptance.py`,
    url: 'http://127.0.0.1:3000',
    reuseExistingServer: false,
    timeout: 180_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
})
