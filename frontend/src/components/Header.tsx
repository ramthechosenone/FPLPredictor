export default function Header() {
  return (
    <header className="text-center py-8 px-4">
      <h1 className="font-[family-name:var(--font-playfair)] text-4xl md:text-5xl font-bold text-[var(--color-accent-maroon)] tracking-tight">
        FPL Predictor
      </h1>
      <p className="font-[family-name:var(--font-lora)] text-[var(--color-text-muted)] mt-2 text-lg italic">
        Machine Learning Predictions for Fantasy Premier League
      </p>
      <hr className="double-rule max-w-md mx-auto" />
    </header>
  );
}
