/**
 * Loads the Knowledge Hub content at build time.
 *
 * - Glossary entries live as .md files under `@/content/glossary/*.md` with a
 *   short YAML-like frontmatter (`---title:...---`). Vite's `import.meta.glob`
 *   eagerly imports them as raw strings — no runtime fetch, no extra round-trip.
 * - FAQ + templates are plain JSON.
 */
import faqData from "@/content/faq.json";
import templatesData from "@/content/templates.json";

export type GlossaryCategory =
  | "index"
  | "data-source"
  | "ml"
  | "anomaly"
  | "other";

export interface GlossaryEntry {
  id: string;
  title: string;
  short: string;
  category: GlossaryCategory;
  body: string;
}

export interface FaqEntry {
  id: string;
  q: string;
  a: string;
}

export interface TemplateEntry {
  id: string;
  label: string;
  category: string;
  prompt: string;
}

const rawGlossary = import.meta.glob("/src/content/glossary/*.md", {
  eager: true,
  query: "?raw",
  import: "default",
}) as Record<string, string>;

/**
 * Minimal YAML frontmatter parser — sufficient for our flat key:value style.
 *
 * Why not gray-matter or yaml: this is the only place we parse frontmatter,
 * and our format is fixed (no nested maps, no quotes, no list values). Saving
 * a ~30 KB dependency for one tiny use is wasteful.
 */
function parseFrontmatter(raw: string): { meta: Record<string, string>; body: string } {
  if (!raw.startsWith("---")) {
    return { meta: {}, body: raw };
  }
  const end = raw.indexOf("\n---", 3);
  if (end === -1) return { meta: {}, body: raw };

  const block = raw.slice(3, end).trim();
  const body = raw.slice(end + 4).replace(/^\n/, "");
  const meta: Record<string, string> = {};
  for (const line of block.split("\n")) {
    const idx = line.indexOf(":");
    if (idx === -1) continue;
    meta[line.slice(0, idx).trim()] = line.slice(idx + 1).trim();
  }
  return { meta, body };
}

function fileIdFromPath(path: string): string {
  const name = path.split("/").pop() ?? path;
  return name.replace(/\.md$/, "");
}

export const GLOSSARY: GlossaryEntry[] = Object.entries(rawGlossary)
  .map(([path, raw]) => {
    const { meta, body } = parseFrontmatter(raw);
    return {
      id: fileIdFromPath(path),
      title: meta.title || fileIdFromPath(path),
      short: meta.short || "",
      category: (meta.category as GlossaryCategory) || "other",
      body,
    };
  })
  .sort((a, b) => a.title.localeCompare(b.title, "uk"));

export const FAQ: FaqEntry[] = faqData as FaqEntry[];
export const TEMPLATES: TemplateEntry[] = templatesData as TemplateEntry[];

export const GLOSSARY_CATEGORY_LABELS: Record<GlossaryCategory, string> = {
  index: "Вегетаційні індекси",
  "data-source": "Джерела даних",
  ml: "Машинне навчання",
  anomaly: "Аномалії",
  other: "Інше",
};
