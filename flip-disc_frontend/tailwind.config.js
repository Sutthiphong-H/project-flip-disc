/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // Behind the display. Flipdot.jsx clears its canvas to the same grey (0.1).
        page: "#1a1a1a",
      },
    },
  },
  plugins: [],
};
