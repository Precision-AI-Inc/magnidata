/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Epilogue', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        pai: {
          navy:    '#013755',
          teal:    '#006A7C',
          teal2:   '#008899',
          fresh:   '#6DF2A3',
          green:   '#29CCA5',
          plant:   '#02AA52',
        },
      },
    },
  },
  plugins: [],
}
