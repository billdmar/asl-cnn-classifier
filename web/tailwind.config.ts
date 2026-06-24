import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Dark "AI product" palette. Background near-black, single accent gradient.
        bg: {
          DEFAULT: "#0a0a0f",
          subtle: "#101018",
          card: "#13131c",
        },
        border: { DEFAULT: "#23232f", subtle: "#1a1a24" },
        fg: { DEFAULT: "#f5f5fa", muted: "#a0a0b0", subtle: "#6a6a7a" },
        accent: {
          DEFAULT: "#7c5cff",
          from: "#7c5cff",
          to: "#2dd4bf",
        },
      },
      backgroundImage: {
        "accent-gradient": "linear-gradient(135deg, #7c5cff 0%, #2dd4bf 100%)",
        "accent-radial":
          "radial-gradient(60% 60% at 50% 0%, rgba(124,92,255,0.18) 0%, rgba(10,10,15,0) 100%)",
      },
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-geist-mono)", "ui-monospace", "monospace"],
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.4s ease-out both",
      },
    },
  },
  plugins: [],
};

export default config;
