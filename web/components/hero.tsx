"use client";

import { motion, useReducedMotion, type Variants } from "framer-motion";
import { ArrowRight, Upload } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

export function Hero() {
  const reduceMotion = useReducedMotion();

  // When the user prefers reduced motion, render static (no offset, no stagger).
  const container: Variants = {
    hidden: {},
    show: {
      transition: reduceMotion ? {} : { staggerChildren: 0.08 },
    },
  };
  const item: Variants = reduceMotion
    ? { hidden: { opacity: 1 }, show: { opacity: 1 } }
    : {
        hidden: { opacity: 0, y: 12 },
        show: {
          opacity: 1,
          y: 0,
          transition: { duration: 0.4, ease: "easeOut" },
        },
      };

  return (
    <section
      id="top"
      className="relative overflow-hidden bg-accent-radial"
      aria-labelledby="hero-heading"
    >
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="mx-auto flex max-w-3xl flex-col items-center px-6 py-24 text-center sm:py-32"
      >
        <motion.div variants={item}>
          <Badge variant="accent">In-browser · privacy-first</Badge>
        </motion.div>

        <motion.h1
          id="hero-heading"
          variants={item}
          className="mt-6 text-balance bg-accent-gradient bg-clip-text text-4xl font-bold leading-tight text-transparent sm:text-6xl"
        >
          Read sign language in your browser
        </motion.h1>

        <motion.p
          variants={item}
          className="mt-6 max-w-2xl text-pretty text-lg text-fg-muted"
        >
          100% in-browser inference · webcam frames never leave your device · MobileNetV2,
          96.8% held-out test accuracy.
        </motion.p>

        <motion.div
          variants={item}
          className="mt-8 flex w-full flex-col items-center gap-3 sm:w-auto sm:flex-row"
        >
          <a href="#live" className="w-full sm:w-auto">
            <Button size="lg" className="w-full sm:w-auto">
              Try the live demo
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </Button>
          </a>
          <a href="#upload" className="w-full sm:w-auto">
            <Button variant="outline" size="lg" className="w-full sm:w-auto">
              <Upload className="h-4 w-4" aria-hidden="true" />
              Upload an image
            </Button>
          </a>
        </motion.div>

        <motion.p variants={item} className="mt-6 text-sm text-fg-subtle">
          Real-world webcam accuracy is lower than the benchmark —{" "}
          <a
            href="#how"
            className="rounded text-fg-muted underline underline-offset-4 transition-colors hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
          >
            see How it works
          </a>
          .
        </motion.p>
      </motion.div>
    </section>
  );
}
