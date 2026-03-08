"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface BlogContentProps {
  content: string;
}

export default function BlogContent({ content }: BlogContentProps) {
  return (
    <article className="blog-content font-[family-name:var(--font-lora)]">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 className="font-[family-name:var(--font-playfair)] text-3xl font-bold text-[var(--color-accent-maroon)] mt-12 mb-4">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="font-[family-name:var(--font-playfair)] text-2xl font-bold text-[var(--color-text-primary)] mt-10 mb-3 border-b border-[var(--color-border)] pb-2">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="font-[family-name:var(--font-playfair)] text-xl font-semibold text-[var(--color-text-primary)] mt-6 mb-2">
              {children}
            </h3>
          ),
          p: ({ children }) => (
            <p className="text-[var(--color-text-primary)] leading-7 mb-4">
              {children}
            </p>
          ),
          ul: ({ children }) => (
            <ul className="list-disc list-outside ml-6 mb-4 space-y-1 text-[var(--color-text-primary)]">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-outside ml-6 mb-4 space-y-1 text-[var(--color-text-primary)]">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="leading-7">{children}</li>
          ),
          code: ({ className, children }) => {
            const isBlock = className?.includes("language-");
            if (isBlock) {
              return (
                <code className="block bg-[var(--color-text-primary)] text-[var(--color-cream)] p-4 rounded text-sm overflow-x-auto my-4 font-mono">
                  {children}
                </code>
              );
            }
            return (
              <code className="bg-[var(--color-cream-dark)] text-[var(--color-accent-maroon)] px-1.5 py-0.5 rounded text-sm font-mono">
                {children}
              </code>
            );
          },
          pre: ({ children }) => (
            <pre className="my-4">{children}</pre>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-[var(--color-accent-maroon)] pl-4 italic text-[var(--color-text-muted)] my-4">
              {children}
            </blockquote>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto my-4">
              <table className="w-full border-collapse text-sm">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-[var(--color-cream-dark)]">{children}</thead>
          ),
          th: ({ children }) => (
            <th className="table-header text-left px-3 py-2 border border-[var(--color-border)]">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-3 py-2 border border-[var(--color-border)]">
              {children}
            </td>
          ),
          hr: () => <hr className="double-rule max-w-xs mx-auto my-8" />,
          strong: ({ children }) => (
            <strong className="font-bold text-[var(--color-text-primary)]">
              {children}
            </strong>
          ),
          a: ({ href, children }) => (
            <a
              href={href}
              className="text-[var(--color-accent-maroon)] underline decoration-[var(--color-border)] hover:decoration-[var(--color-accent-maroon)] transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              {children}
            </a>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </article>
  );
}
