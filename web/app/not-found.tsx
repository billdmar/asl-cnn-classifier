import Link from "next/link";

/** Custom 404 — keeps users on-brand and gives a clear way back. */
export default function NotFound() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-4 px-6 text-center">
      <p className="font-mono text-5xl font-bold text-fg-subtle">404</p>
      <h1 className="text-2xl font-semibold text-fg">Page not found</h1>
      <p className="max-w-md text-sm text-fg-muted">
        That page doesn&apos;t exist. Try the live ASL classifier on the home page.
      </p>
      <Link
        href="/"
        className="rounded-md bg-accent-gradient px-4 py-2 text-sm font-medium text-bg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
      >
        Back to home
      </Link>
    </main>
  );
}
