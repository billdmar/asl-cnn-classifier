import { Github } from "lucide-react";

import { Reveal } from "@/components/ui/reveal";
import { formatBuildInfo } from "@/lib/build-info";

const REPO_URL = "https://github.com/billdmar/asl-cnn-classifier";

export function Footer() {
  // Baked in at build time by next.config.mjs; "dev" locally.
  const build = formatBuildInfo(
    process.env.NEXT_PUBLIC_BUILD_SHA,
    process.env.NEXT_PUBLIC_BUILD_DATE,
  );

  return (
    <footer className="border-t border-border">
      <Reveal className="mx-auto flex max-w-6xl flex-col gap-4 px-6 py-10 text-sm text-fg-muted sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-col gap-2">
          <a
            href={REPO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex w-fit items-center gap-2 rounded-md text-fg transition-colors hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
          >
            <Github className="h-4 w-4" aria-hidden="true" />
            asl-cnn-classifier on GitHub
          </a>
          <p className="text-fg-subtle">MIT © William Mar</p>
        </div>
        <div className="flex flex-col gap-2 sm:items-end">
          <p className="max-w-md text-pretty text-fg-subtle sm:text-right">
            Every metric shown here is produced by reproducible code in the repo — nothing
            is hand-edited.
          </p>
          {/* Deploy provenance: which commit is live + when it was built. */}
          <p className="font-mono text-xs text-fg-subtle">
            {build.commitUrl ? (
              <a
                href={build.commitUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded transition-colors hover:text-fg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
              >
                build {build.shortSha}
              </a>
            ) : (
              <span>build {build.shortSha}</span>
            )}
            {build.date ? ` · ${build.date}` : null}
          </p>
        </div>
      </Reveal>
    </footer>
  );
}
