import { ImageResponse } from "next/og";

// Static-export friendly: force-static makes Next pre-render this to a single
// static og image file at build time (required under `output: export`).
export const dynamic = "force-static";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const alt =
  "ASL Classifier — in-browser American Sign Language recognition";

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          padding: "80px",
          background:
            "linear-gradient(135deg, #0a0a0f 0%, #16121f 55%, #0c1a1a 100%)",
          color: "#f5f5fa",
          fontFamily: "sans-serif",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "28px",
            marginBottom: "32px",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: "120px",
              height: "120px",
              borderRadius: "28px",
              background: "linear-gradient(135deg, #7c5cff 0%, #2dd4bf 100%)",
              fontSize: "76px",
              fontWeight: 800,
              color: "#0a0a0f",
            }}
          >
            A
          </div>
          <div style={{ fontSize: "60px", fontWeight: 800 }}>ASL Classifier</div>
        </div>
        <div style={{ fontSize: "34px", color: "#a0a0b0", maxWidth: "1000px" }}>
          Real-time American Sign Language recognition — 100% in your browser.
        </div>
        <div
          style={{
            display: "flex",
            gap: "16px",
            marginTop: "44px",
            fontSize: "26px",
          }}
        >
          <span
            style={{
              padding: "10px 22px",
              borderRadius: "999px",
              background: "rgba(45,212,191,0.15)",
              color: "#2dd4bf",
            }}
          >
            59.8% honest cross-dataset (A–Y)
          </span>
          <span
            style={{
              padding: "10px 22px",
              borderRadius: "999px",
              background: "rgba(124,92,255,0.15)",
              color: "#b3a0ff",
            }}
          >
            MobileNetV2 · onnxruntime-web
          </span>
        </div>
      </div>
    ),
    size,
  );
}
