/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        serif: ['Cormorant Garamond', 'serif'],
        sans: ['Inter', 'sans-serif'],
      },
      colors: {
        rose: {
          primary: '#D4537E',
          light: '#FBEAF0',
          mid: '#ED93B1',
          border: '#F4C0D1',
          dark: '#72243E',
          deeper: '#993556',
        },
        cream: {
          DEFAULT: '#FFFCFB',
          warm: '#FFF8F5',
        },
        ink: '#2C2C2A',
        muted: '#888780',
        soft: '#B4B2A9',
      },
    },
  },
  plugins: [],
}