"use client";

export default function Sidebar() {
  return (
    <aside style={{ width: 240, minHeight: "100vh", backgroundColor: "#111118", borderRight: "1px solid rgba(0,201,167,0.15)", padding: "1.5rem 1rem", display: "flex", flexDirection: "column", gap: "0.5rem", position: "fixed", top: 0, left: 0 }}>
      <div style={{ color: "#00C9A7", fontWeight: 700, fontSize: "1.25rem", marginBottom: "1rem", letterSpacing: "0.05em" }}>SAVANT</div>
      <SidebarLink href="/query" label="Query" icon="🔍" />
      <SidebarLink href="/knowledge" label="Knowledge Map" icon="🗺️" />
      <SidebarLink href="/documents" label="Documents" icon="📁" />
      <div style={{ marginTop: "auto", fontSize: "0.75rem", color: "#666" }}>Savant • v2.0</div>
    </aside>
  );
}

function SidebarLink({ href, label, icon }: { href: string; label: string; icon: string }) {
  return (
    <a href={href} style={{ display: "flex", alignItems: "center", gap: "0.5rem", padding: "0.6rem 0.75rem", borderRadius: 8, color: "#e0e0e0", textDecoration: "none", fontSize: "0.9rem" }}
       onMouseOver={e => { (e.currentTarget as HTMLElement).style.backgroundColor = "rgba(0,201,167,0.1)"; (e.currentTarget as HTMLElement).style.color = "#00C9A7"; }}
       onMouseOut={e => { (e.currentTarget as HTMLElement).style.backgroundColor = "transparent"; (e.currentTarget as HTMLElement).style.color = "#e0e0e0"; }}>
      <span>{icon}</span><span>{label}</span>
    </a>
  );
}
