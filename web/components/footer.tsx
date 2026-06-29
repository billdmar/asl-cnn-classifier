import { Github } from "lucide-react";

import { Reveal } from "@/components/ui/reveal";

const REPO_URL = "https://github.com/billdmar/asl-cnn-classifier";

export function Footer() {
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
        <p className="max-w-md text-pretty text-fg-subtle sm:text-right">
          Every metric shown here is produced by reproducible code in the repo — nothing
          is hand-edited.
        </p>
      </Reveal>
    </footer>
  );
}
