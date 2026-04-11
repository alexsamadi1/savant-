"use client";
import Link from "next/link";

export default function Sidebar() {
  const navItems = [
    { href: "/query", label: "Query", icon: "search" },
    { href: "/knowledge", label: "Knowledge Map", icon: "map" },
    { href: "/documents", label: "Documents", icon: "folder" },
  ];

  return (
    <aside style={{
      width: 220,
      minHeight: "100vh",
      background: "#0a0f1a",
      borderRight: "0.5px solid rgba(0,201,167,0.12)",
      position: "fixed",
      top: 0,
      left: 0,
      display: "flex",
      flexDirection: "column",
      padding: "1.25rem 0.875rem",
      gap: "0.25rem",
      zIndex: 100,
    }}>
      <div style={{
        fontFamily: "var(--font-mono)",
        fontWeight: 500,
        fontSize: "0.78rem",
        letterSpacing: "0.18em",
        textTransform: "uppercase",
        color: "var(--teal)",
        padding: "0.25rem 0.625rem",
        marginBottom: "1.25rem",
      }}>
        Savant
      </div>

      {navItems.map(item => (
        <Link
          key={item.href}
          href={item.href}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.625rem",
            padding: "0.5rem 0.625rem",
            borderRadius: 8,
            color: "var(--text-secondary)",
            textDecoration: "none",
            fontSize: "0.875rem",
            fontWeight: 400,
            transition: "all 0.15s",
          }}
          onMouseOver={(e) => {
            (e.currentTarget as HTMLElement).style.backgroundColor = "rgba(0,201,167,0.07)";
            (e.currentTarget as HTMLElement).style.color = "var(--text-primary)";
          }}
          onMouseOut={(e) => {
            (e.currentTarget as HTMLElement).style.backgroundColor = "transparent";
            (e.currentTarget as HTMLElement).style.color = "var(--text-secondary)";
          }}
        >
          <NavIcon name={item.icon} />
          {item.label}
        </Link>
      ))}

      <div style={{ flex: 1 }} />

      <div style={{
        fontFamily: "var(--font-mono)",
        fontSize: "0.65rem",
        color: "var(--text-muted)",
        padding: "0.25rem 0.625rem",
        letterSpacing: "0.05em",
      }}>
        Savant v2.0
      </div>
    </aside>
  );
}

function NavIcon({ name }: { name: string }) {
  const size = 14;
  const s = { width: size, height: size, opacity: 0.6, flexShrink: 0 } as const;
  if (name === "search") return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={s}>
      <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
    </svg>
  );
  if (name === "map") return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={s}>
      <polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/>
      <line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/>
    </svg>
  );
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={s}>
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
    </svg>
  );
}
