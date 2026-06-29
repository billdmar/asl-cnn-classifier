import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Themeable palette: tokens resolve through CSS vars defined in
        // app/globals.css (:root = dark default, [data-theme="light"] = light).
        // The `<alpha-value>` slot keeps opacity utilities (bg-bg/50, …) working.
        bg: {
          DEFAULT: "rgb(var(--bg) / <alpha-value>)",
          subtle: "rgb(var(--bg-subtle) / <alpha-value>)",
          card: "rgb(var(--bg-card) / <alpha-value>)",
        },
        border: {
          DEFAULT: "rgb(var(--border) / <alpha-value>)",
          subtle: "rgb(var(--border-subtle) / <alpha-value>)",
        },
        fg: {
          DEFAULT: "rgb(var(--fg) / <alpha-value>)",
          muted: "rgb(var(--fg-muted) / <alpha-value>)",
          subtle: "rgb(var(--fg-subtle) / <alpha-value>)",
        },
        accent: {
          DEFAULT: "rgb(var(--accent) / <alpha-value>)",
          from: "rgb(var(--accent-from) / <alpha-value>)",
          to: "rgb(var(--accent-to) / <alpha-value>)",
        },
      },
      backgroundImage: {
        // Gradients can't use the <alpha-value> slot (it's only substituted by
        // Tailwind's color-utility machinery), so alpha is pinned literally.
        "accent-gradient":
          "linear-gradient(135deg, rgb(var(--accent-from)) 0%, rgb(var(--accent-to)) 100%)",
        "accent-radial":
          "radial-gradient(60% 60% at 50% 0%, rgb(var(--accent) / 0.18) 0%, rgb(var(--bg) / 0) 100%)",
      },
      fontFamily: {
        // System font stack — no web-font fetch, so text paints immediately
        // (no FOIT/FOUT, better LCP). Matches the modern "AI product" look.
        sans: [
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        // Transform-only entrance for LCP-critical content: the element is
        // painted fully opaque from the first frame (so Largest Contentful
        // Paint fires immediately) while still getting a subtle slide-up.
        "rise-up": {
          "0%": { transform: "translateY(8px)" },
          "100%": { transform: "translateY(0)" },
        },
        // Loading shimmer sweep — animates background-position (compositor-only,
        // no layout). Pair with a gradient background on the skeleton element.
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        // Slow accent-gradient pan for decorative gradient text. Also
        // background-position only; pair with `.bg-pan` (background-size: 200%).
        "gradient-pan": {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.4s ease-out both",
        "rise-up": "rise-up 0.4s ease-out both",
        shimmer: "shimmer 1.6s linear infinite",
        "gradient-pan": "gradient-pan 8s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
