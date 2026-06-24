interface Item {
  title: string;
  body: string;
}

const LIMITATIONS: ReadonlyArray<Item> = [
  {
    title: "Not an accessibility or medical tool",
    body: "This is an educational / portfolio demonstration of a computer-vision pipeline. It is not a certified accessibility, communication, medical, legal, or otherwise safety-critical product and must not be relied on as one.",
  },
  {
    title: "Alphabet only, static frames",
    body: "It classifies single still frames of the ASL alphabet. It does not recognize continuous signing, words, or grammar, and the motion-based letters (J, Z) are inherently ambiguous as single frames.",
  },
  {
    title: "Benchmark optimism",
    body: "The training data is fairly homogeneous, so a random split can leak near-duplicate frames across train and test. The benchmark accuracy is therefore optimistic; a group-aware split would lower it.",
  },
  {
    title: "Shared hand shapes get confused",
    body: "Letters that share a hand shape are the usual error sources — P/Q, V/W, and M/N/S are classic confusions.",
  },
];

const ETHICS: ReadonlyArray<Item> = [
  {
    title: "Demographic representation",
    body: "The training data appears to feature a narrow range of skin tones and a single signing environment, so the model will likely underperform for under-represented skin tones and settings. Any production or accessibility use would require a diverse, consented, group-split dataset and per-group fairness evaluation.",
  },
  {
    title: "Respect for the Deaf community",
    body: 'ASL is a complete language with its own grammar; alphabet fingerspelling is a small part of it. This project should not be presented as "translating ASL." Accessibility tools should be built with the Deaf and hard-of-hearing community, not merely for it.',
  },
  {
    title: "Honest reporting",
    body: "Benchmark accuracy is reported alongside its leakage caveat specifically to avoid overstating real-world capability.",
  },
];

function ItemList({ items }: { items: ReadonlyArray<Item> }) {
  return (
    <ul className="space-y-4">
      {items.map((item) => (
        <li key={item.title} className="border-l-2 border-border pl-4">
          <p className="font-medium text-fg">{item.title}</p>
          <p className="mt-1 text-sm">{item.body}</p>
        </li>
      ))}
    </ul>
  );
}

export function EthicsLimitations() {
  return (
    <div className="grid gap-8 lg:grid-cols-2">
      <div>
        <h3 className="text-lg font-semibold text-fg">Limitations</h3>
        <div className="mt-4">
          <ItemList items={LIMITATIONS} />
        </div>
      </div>
      <div>
        <h3 className="text-lg font-semibold text-fg">Ethical considerations</h3>
        <div className="mt-4">
          <ItemList items={ETHICS} />
        </div>
      </div>
    </div>
  );
}
