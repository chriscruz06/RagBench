/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        serif: ['"EB Garamond"', 'Georgia', 'Cambria', 'serif'],
      },
      colors: {
        // Warm, bookish canvas — slightly warmer than before
        cream: '#faf7f1',
        parchment: '#f3ecdf',
        ink: '#1f1a17',
        // Restrained ink-red — barely a color, more an aged-paper underline.
        // Used sparingly: citation locators, focus, the Ask link.
        accent: {
          DEFAULT: '#5c2a2a',
          light: '#f1e9dd',
          hover: '#3d1a1a',
        },
        // Subtle rule color — between background and text, never pure gray
        rule: '#e6dccb',
      },
      maxWidth: {
        content: '38rem',
      },
    },
  },
  plugins: [],
}
