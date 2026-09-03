import { useService } from "../state/ServiceProvider";

const LINKS = [
  ["#checks", "Input checks"],
  ["#shadow", "Shadow runs"],
  ["#swap", "Zero-drop swap"],
  ["#metrics", "Metrics"],
] as const;

export function Nav() {
  const { service, gen } = useService();
  const primary = service.registry.primary?.version ?? "none";
  const shadow = service.registry.shadow?.version ?? null;
  return (
    <header className="nav">
      <div className="container nav-inner">
        <a href="#top" className="brand" aria-label="ModelGate, back to top">
          <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden="true">
            <rect width="32" height="32" rx="7" fill="var(--bg-3)" />
            <path
              d="M8 22V10l8 7 8-7v12"
              fill="none"
              stroke="var(--orange)"
              strokeWidth="3"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          </svg>
          <span>ModelGate</span>
        </a>
        <nav aria-label="Sections" className="nav-links">
          {LINKS.map(([href, label]) => (
            <a key={href} href={href}>
              {label}
            </a>
          ))}
        </nav>
        <div className="nav-status mono" aria-live="polite">
          <span className="pill orange">
            <span className="dot" /> primary {primary}
          </span>
          <span className={`pill ${shadow ? "ice" : ""}`}>
            <span className="dot" /> shadow {shadow ?? "off"}
          </span>
          <span className={`pill ${gen.running ? "ok" : ""}`}>
            <span className={`dot ${gen.running ? "live" : ""}`} /> {gen.running ? `${gen.rps} rps` : "idle"}
          </span>
        </div>
      </div>
    </header>
  );
}
