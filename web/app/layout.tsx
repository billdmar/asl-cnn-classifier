import type { Metadata, Viewport } from "next";
import "./globals.css";

import { ServiceWorkerRegister } from "@/components/service-worker-register";
import { buildJsonLd } from "@/lib/structured-data";

const TITLE = "ASL Classifier — in-browser, real-time";
const DESCRIPTION =
  "Real-time American Sign Language alphabet recognition running 100% in your browser. A MobileNetV2 model (59.8% honest cross-dataset accuracy, 96.9% same-dataset) via onnxruntime-web — webcam frames never leave your device.";

export const metadata: Metadata = {
  metadataBase: new URL("https://asl-cnn-classifier.vercel.app"),
  title: TITLE,
  description: DESCRIPTION,
  openGraph: {
    title: TITLE,
    description: DESCRIPTION,
    url: "/",
    siteName: "ASL Classifier",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: TITLE,
    description: DESCRIPTION,
  },
};

// Next 15 moved themeColor to the viewport export. Matches the dark default
// (and the manifest's theme_color) so the browser chrome tints to the brand.
export const viewport: Viewport = { themeColor: "#0a0a0f" };

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* No-FOUC theme init: set data-theme before first paint from the saved
            preference (default dark = current look). suppressHydrationWarning on
            <html> covers the attribute this adds before React hydrates. */}
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem('theme');if(t!=='light'&&t!=='dark'){t='dark';}document.documentElement.setAttribute('data-theme',t);}catch(e){document.documentElement.setAttribute('data-theme','dark');}})();`,
          }}
        />
        {/* schema.org WebApplication structured data (static inline JSON-LD). */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(buildJsonLd()) }}
        />
      </head>
      <body>
        {/* Keyboard/screen-reader users can jump straight to the content. */}
        <a
          href="#top"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-bg-card focus:px-4 focus:py-2 focus:text-fg focus:ring-2 focus:ring-accent"
        >
          Skip to main content
        </a>
        {children}
        {/* Registers the offline service worker (prod hosts only; skips localhost
            so e2e/Lighthouse run on a clean SW-free origin). */}
        <ServiceWorkerRegister />
      </body>
    </html>
  );
}
