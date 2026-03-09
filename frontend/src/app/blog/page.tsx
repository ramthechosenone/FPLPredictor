import fs from "fs";
import path from "path";
import BlogContent from "../../components/BlogContent";

export const metadata = {
  title: "FPL Predictor — Build Journal",
  description:
    "A running log of decisions, discoveries, and learnings building an FPL points predictor from scratch.",
};

export default function BlogPage() {
  // Try local copy first (copied by build script), then parent dir
  const localPath = path.join(process.cwd(), "BLOG_JOURNAL.md");
  const parentPath = path.join(process.cwd(), "..", "BLOG_JOURNAL.md");
  let content = "";
  try {
    content = fs.existsSync(localPath)
      ? fs.readFileSync(localPath, "utf-8")
      : fs.readFileSync(parentPath, "utf-8");
  } catch {
    content = "# Blog Journal\n\nBlog content not found.";
  }

  return (
    <div className="max-w-3xl mx-auto px-4">
      <header className="text-center py-8">
        <a
          href="/fpl"
          className="text-sm text-[var(--color-text-muted)] hover:text-[var(--color-accent-maroon)] transition-colors"
        >
          &larr; Back to Predictions
        </a>
        <h1 className="font-[family-name:var(--font-playfair)] text-4xl md:text-5xl font-bold text-[var(--color-accent-maroon)] tracking-tight mt-4">
          Build Journal
        </h1>
        <p className="font-[family-name:var(--font-lora)] text-[var(--color-text-muted)] mt-2 text-lg italic">
          How we built the FPL Predictor, chapter by chapter
        </p>
        <hr className="double-rule max-w-md mx-auto" />
      </header>

      <BlogContent content={content} />

      <footer className="mt-12 pb-8">
        <hr className="double-rule max-w-lg mx-auto" />
        <p className="text-center text-sm text-[var(--color-text-muted)] font-[family-name:var(--font-lora)] italic">
          Built with curiosity, Python, and a lot of tea.
        </p>
      </footer>
    </div>
  );
}
