import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ASL Classifier — in-browser, real-time",
  description:
    "Real-time American Sign Language alphabet recognition running 100% in your browser. A MobileNetV2 model (59.8% honest cross-dataset accuracy, 96.9% same-dataset) via onnxruntime-web — webcam frames never leave your device.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
