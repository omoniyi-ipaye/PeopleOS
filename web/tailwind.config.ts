import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './design-system/**/*.{js,ts,jsx,tsx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        accent: {
          DEFAULT: '#7c3aed',
          50: '#f5f3ff', 100: '#ede9fe', 200: '#ddd6fe', 300: '#c4b5fd',
          400: '#a78bfa', 500: '#8b5cf6', 600: '#7c3aed', 700: '#6d28d9',
          800: '#5b21b6', 900: '#4c1d95',
        },
        success: { DEFAULT: '#059669', 50: '#ecfdf5', 100: '#d1fae5', 500: '#10b981', 600: '#059669' },
        warning: { DEFAULT: '#d97706', 50: '#fffbeb', 100: '#fef3c7', 500: '#f59e0b', 600: '#d97706' },
        danger: { DEFAULT: '#dc2626', 50: '#fef2f2', 100: '#fee2e2', 500: '#ef4444', 600: '#dc2626' },
        background: {
          DEFAULT: '#f8fafc', secondary: '#f1f5f9', tertiary: '#e2e8f0',
          dark: '#020617', 'dark-secondary': '#0f172a', 'dark-tertiary': '#1e293b',
        },
        surface: { DEFAULT: '#ffffff', hover: '#f8fafc', dark: '#0f172a', 'dark-hover': '#1e293b' },
        border: { DEFAULT: '#e2e8f0', dark: '#334155' },
        text: {
          primary: '#0f172a', secondary: '#475569', muted: '#94a3b8',
          'dark-primary': '#f8fafc', 'dark-secondary': '#cbd5e1', 'dark-muted': '#64748b',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
      },
      borderRadius: {
        xl: '0.75rem',
        '2xl': '1rem',
        '3xl': '1.5rem',
      },
      boxShadow: {
        card: '0 1px 2px rgba(15, 23, 42, 0.05), 0 1px 3px rgba(15, 23, 42, 0.08)',
        'card-hover': '0 12px 28px -12px rgba(15, 23, 42, 0.24)',
        glass: '0 8px 24px -10px rgba(15, 23, 42, 0.24)',
        glow: '0 0 0 3px rgba(124, 58, 237, 0.14)',
      },
      animation: {
        float: 'float 6s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-6px)' },
        },
      },
    },
  },
  plugins: [],
}

export default config
