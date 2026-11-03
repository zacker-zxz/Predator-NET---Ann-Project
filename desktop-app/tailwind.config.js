/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cyber: {
          dark: '#07090e',
          card: '#0d111a',
          cardHover: '#131926',
          border: '#1e293b',
          accent: '#00ff9d',       // Predator Green
          danger: '#ff0055',       // Attack Crimson
          warning: '#ffb703',      // Warning Amber
          cyan: '#00e5ff',         // Flow Cyan
          dim: '#4a5568',          // Weak / Prey Grey
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'neon-green': '0 0 20px rgba(0, 255, 157, 0.4)',
        'neon-red': '0 0 25px rgba(255, 0, 85, 0.5)',
        'neon-cyan': '0 0 20px rgba(0, 229, 255, 0.35)',
      }
    },
  },
  plugins: [],
}
