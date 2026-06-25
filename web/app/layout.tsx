import type { Metadata } from "next";
import "./globals.css";

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

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        {/* Keyboard/screen-reader users can jump straight to the content. */}
        <a
          href="#top"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-bg-card focus:px-4 focus:py-2 focus:text-fg focus:ring-2 focus:ring-accent"
        >
          Skip to main content
        </a>
        {children}
      </body>
    </html>
  );
}
