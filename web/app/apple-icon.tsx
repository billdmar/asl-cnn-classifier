import { ImageResponse } from "next/og";

// Static-export friendly: force-static pre-renders the apple-touch-icon to a
// static file at build time (required under `output: export`). Next auto-links
// it as <link rel="apple-touch-icon">.
export const dynamic = "force-static";
export const size = { width: 180, height: 180 };
export const contentType = "image/png";

export default function AppleIcon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#0a0a0f",
          // satori has no gradient text — use solid accent for the glyph.
          color: "#7c5cff",
          fontSize: 132,
          fontWeight: 800,
          fontFamily: "sans-serif",
        }}
      >
        A
      </div>
    ),
    size,
  );
}
