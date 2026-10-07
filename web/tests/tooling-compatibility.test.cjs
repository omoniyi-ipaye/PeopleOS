const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { getRootDirs } = require('@next/eslint-plugin-next/dist/utils/get-root-dirs')

test('Next lint root discovery retains directory, brace and array glob support', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'peopleos-lint-roots-'))
  try {
    const first = path.join(root, 'first')
    const second = path.join(root, 'second')
    fs.mkdirSync(first)
    fs.mkdirSync(second)
    fs.writeFileSync(path.join(root, 'not-a-directory'), '')
    const discover = (rootDir) => getRootDirs({ cwd: root, settings: { next: { rootDir } } }).map(dir => path.resolve(dir)).sort()
    assert.deepEqual(discover(undefined), [root])
    assert.deepEqual(discover(`${root}/*`), [first, second])
    assert.deepEqual(discover(`${root}/{first,second}`), [first, second])
    assert.deepEqual(discover([first, second]), [first, second])
    assert.deepEqual(discover(`${root}/missing-*`), [])
    // Exercise the consuming rule, not only path matching: root discovery
    // must still find pages and reject plain anchors for internal navigation.
    fs.mkdirSync(path.join(first, 'pages'))
    fs.writeFileSync(path.join(first, 'pages', 'about.js'), 'export default function Page() {}')
    const { Linter } = require('eslint')
    const next = require('@next/eslint-plugin-next')
    const messages = new Linter().verify('<a href="/about">About</a>', [{
      languageOptions: { parserOptions: { ecmaFeatures: { jsx: true } } },
      plugins: { '@next/next': next },
      settings: { next: { rootDir: `${root}/{first,second}` } },
      rules: { '@next/next/no-html-link-for-pages': 'error' },
    }])
    assert.ok(messages.some(message => message.ruleId === '@next/next/no-html-link-for-pages'))
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('Tailwind 4 class conflicts retain explicit caller overrides', () => {
  const { twMerge } = require('tailwind-merge')
  assert.equal(twMerge('shadow-xs outline-hidden', 'shadow-lg outline-none'), 'shadow-lg outline-none')
  assert.equal(twMerge('max-w-4xl', 'max-w-6xl'), 'max-w-6xl')
})
