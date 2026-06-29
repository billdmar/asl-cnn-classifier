"use client";

import { useEffect, useState } from "react";
import { Github } from "lucide-react";

import { cn } from "@/lib/utils";

const NAV_LINKS = [
  { href: "#live", label: "Live Demo" },
  { href: "#upload", label: "Upload" },
  { href: "#metrics", label: "Metrics" },
  { href: "#how", label: "How it works" },
] as const;

const REPO_URL = "https://github.com/billdmar/asl-cnn-classifier";

export function SiteHeader() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={cn(
        "sticky top-0 z-50 w-full border-b backdrop-blur transition-colors",
        scrolled ? "border-border bg-bg/80" : "border-transparent bg-bg/40",
      )}
    >
      <nav
        aria-label="Primary"
        className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6"
      >
        <a
          href="#top"
          className="group flex items-center gap-2 rounded-md font-semibold text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
        >
          <span
            aria-hidden="true"
            className="h-6 w-6 rounded-md bg-accent-gradient transition-transform group-hover:scale-110"
          />
          ASL Classifier
        </a>

        <div className="flex items-center gap-1 sm:gap-2">
          <ul className="hidden items-center gap-1 md:flex">
            {NAV_LINKS.map((link) => (
              <li key={link.href}>
                <a
                  href={link.href}
                  className="rounded-md px-3 py-2 text-sm text-fg-muted underline decoration-accent decoration-2 underline-offset-8 decoration-transparent transition-colors hover:text-fg hover:decoration-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
                >
                  {link.label}
                </a>
              </li>
            ))}
          </ul>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="View source on GitHub"
            className="flex h-9 w-9 items-center justify-center rounded-md text-fg-muted transition-colors hover:bg-bg-card hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
          >
            <Github className="h-5 w-5" aria-hidden="true" />
          </a>
        </div>
      </nav>
    </header>
  );
}
