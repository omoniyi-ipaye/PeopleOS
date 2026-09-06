import fs from 'node:fs'
import path from 'node:path'

const root = process.cwd()
const debt = JSON.parse(fs.readFileSync(path.join(root, 'design-system/debt.json'), 'utf8'))
const sourceRoots = ['app', 'components']
const allowedFiles = new Set(debt.allowedFiles ?? [])
const allowedPrefixes = debt.allowedPrefixes ?? []

function walk(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true })
  return entries.flatMap((entry) => {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) return walk(full)
    return /\.(tsx|ts|jsx|js)$/.test(entry.name) ? [full] : []
  })
}

function isDebt(relative) {
  return allowedFiles.has(relative) || allowedPrefixes.some((prefix) => relative.startsWith(prefix))
}

const violations = []
for (const sourceRoot of sourceRoots) {
  const absolute = path.join(root, sourceRoot)
  if (!fs.existsSync(absolute)) continue
  for (const file of walk(absolute)) {
    const relative = path.relative(root, file).replaceAll('\\', '/')
    if (isDebt(relative)) continue
    const content = fs.readFileSync(file, 'utf8')

    if (/rounded-\[[^\]]+\]/.test(content)) {
      violations.push(`${relative}: arbitrary radius; use design-system radius primitives/tokens`)
    }
    if (/#[0-9a-fA-F]{3,8}\b/.test(content)) {
      violations.push(`${relative}: hardcoded hex color; use semantic design tokens`)
    }
    if (/from ['"]@\/components\/ui\/glass-card['"]/.test(content)) {
      violations.push(`${relative}: GlassCard is deprecated for new work`)
    }
  }
}

const required = [
  'design-system/tokens.ts',
  'components/ui/surface.tsx',
  'components/ui/page.tsx',
  'components/ui/status.tsx',
  'components/ui/field.tsx',
  'components/ui/data-display.tsx',
]
for (const requiredPath of required) {
  if (!fs.existsSync(path.join(root, requiredPath))) violations.push(`${requiredPath}: required enterprise primitive missing`)
}

if (violations.length) {
  console.error('DESIGN SYSTEM GOVERNANCE: FAIL')
  for (const violation of violations) console.error(`- ${violation}`)
  process.exit(1)
}

console.log('DESIGN SYSTEM GOVERNANCE: PASS')
console.log(`Explicit transition debt: ${(debt.allowedFiles?.length ?? 0) + (debt.allowedPrefixes?.length ?? 0)} scoped exceptions`)
