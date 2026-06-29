/**
 * schema.org JSON-LD builder for the ASL Classifier web app.
 *
 * The returned object is serialized with JSON.stringify and injected into an
 * inline <script type="application/ld+json"> by the layout, so it must contain
 * only JSON-safe values (no functions, no `undefined`).
 *
 * Accuracy figures mirror the live copy in app/layout.tsx: 59.8% honest
 * cross-dataset accuracy, 96.9% same-dataset.
 */
export function buildJsonLd(): Record<string, unknown> {
  return {
    "@context": "https://schema.org",
    "@type": "WebApplication",
    name: "ASL Classifier",
    url: "https://asl-cnn-classifier.vercel.app",
    description:
      "Real-time American Sign Language alphabet recognition running 100% in your browser. A MobileNetV2 model (59.8% honest cross-dataset accuracy, 96.9% same-dataset) via onnxruntime-web — webcam frames never leave your device.",
    applicationCategory: "EducationalApplication",
    operatingSystem: "Any (modern web browser with WebAssembly)",
    browserRequirements: "Requires JavaScript, WebAssembly, and a webcam",
    isAccessibleForFree: true,
    offers: {
      "@type": "Offer",
      price: "0",
      priceCurrency: "USD",
    },
    license: "https://opensource.org/licenses/MIT",
    codeRepository: "https://github.com/billdmar/asl-cnn-classifier",
    softwareHelp: "https://asl-cnn-classifier.vercel.app/about",
    featureList: [
      "In-browser real-time ASL alphabet (A–Y) recognition",
      "MobileNetV2 inference via onnxruntime-web",
      "On-device privacy — no frames leave the browser",
    ],
    author: {
      "@type": "Person",
      name: "Bill Dmar",
      url: "https://github.com/billdmar",
    },
  };
}
