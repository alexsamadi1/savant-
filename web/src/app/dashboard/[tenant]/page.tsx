"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* ------------------------------------------------------------------ */
/*  Types (mirrors api/models.py)                                     */
/* ------------------------------------------------------------------ */

interface MetricCard {
  id: string;
  label: string;
  value: string;
  change?: string | null;
  change_direction?: "up" | "down" | "neutral" | null;
  insight?: string | null;
}

interface ChartSpec {
  id: string;
  title: string;
  type: "bar" | "line" | "pie" | "scatter" | "area";
  data: Record<string, unknown>[];
  x_key: string;
  y_key: string;
  color?: string | null;
  insight: string;
}

interface Recommendation {
  priority: number;
  title: string;
  detail: string;
  evidence: string;
}

interface DashboardConfig {
  tenant: string;
  company_name: string;
  problem_statement: string;
  generated_at: string;
  executive_summary: string;
  metrics: MetricCard[];
  charts: ChartSpec[];
  recommendations: Recommendation[];
  data_sources_used: string[];
}

/* ------------------------------------------------------------------ */
/*  Palette for charts                                                */
/* ------------------------------------------------------------------ */

const CHART_COLORS = [
  "#00C9A7",
  "#6C63FF",
  "#FF6B6B",
  "#FFD93D",
  "#4ECDC4",
  "#45B7D1",
  "#96CEB4",
  "#FFEEAD",
];

/* ------------------------------------------------------------------ */
/*  Chart renderer                                                    */
/* ------------------------------------------------------------------ */

function DashboardChart({ spec }: { spec: ChartSpec }) {
  const fill = spec.color || CHART_COLORS[0];
  const stroke = spec.color || CHART_COLORS[0];

  const axisProps = {
    tick: { fill: "#7a8599", fontSize: 11 },
    axisLine: { stroke: "rgba(0,201,167,0.1)" },
    tickLine: false,
  };

  const grid = (
    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
  );

  const tooltip = (
    <Tooltip
      contentStyle={{
        background: "#161b27",
        border: "1px solid rgba(0,201,167,0.2)",
        borderRadius: 8,
        fontSize: "0.78rem",
        color: "#e8edf5",
      }}
    />
  );

  if (spec.data.length === 0) {
    return (
      <div
        style={{
          height: 300,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--text-muted)",
          fontSize: "0.85rem",
        }}
      >
        No chart data available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      {spec.type === "bar" ? (
        <BarChart data={spec.data}>
          {grid}
          <XAxis dataKey={spec.x_key} {...axisProps} />
          <YAxis {...axisProps} />
          {tooltip}
          <Legend />
          <Bar dataKey={spec.y_key} fill={fill} radius={[4, 4, 0, 0]} />
        </BarChart>
      ) : spec.type === "line" ? (
        <LineChart data={spec.data}>
          {grid}
          <XAxis dataKey={spec.x_key} {...axisProps} />
          <YAxis {...axisProps} />
          {tooltip}
          <Legend />
          <Line
            type="monotone"
            dataKey={spec.y_key}
            stroke={stroke}
            strokeWidth={2}
            dot={{ r: 3, fill: stroke }}
          />
        </LineChart>
      ) : spec.type === "area" ? (
        <AreaChart data={spec.data}>
          {grid}
          <XAxis dataKey={spec.x_key} {...axisProps} />
          <YAxis {...axisProps} />
          {tooltip}
          <Legend />
          <Area
            type="monotone"
            dataKey={spec.y_key}
            stroke={stroke}
            fill={`${stroke}33`}
            strokeWidth={2}
          />
        </AreaChart>
      ) : spec.type === "scatter" ? (
        <ScatterChart>
          {grid}
          <XAxis dataKey={spec.x_key} {...axisProps} name={spec.x_key} />
          <YAxis dataKey={spec.y_key} {...axisProps} name={spec.y_key} />
          {tooltip}
          <Scatter data={spec.data} fill={fill} />
        </ScatterChart>
      ) : (
        /* pie */
        <PieChart>
          {tooltip}
          <Legend />
          <Pie
            data={spec.data}
            dataKey={spec.y_key}
            nameKey={spec.x_key}
            cx="50%"
            cy="50%"
            outerRadius={110}
            strokeWidth={0}
          >
            {spec.data.map((_, i) => (
              <Cell
                key={i}
                fill={CHART_COLORS[i % CHART_COLORS.length]}
              />
            ))}
          </Pie>
        </PieChart>
      )}
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------------ */
/*  Change arrow                                                      */
/* ------------------------------------------------------------------ */

function ChangeIndicator({
  change,
  direction,
}: {
  change?: string | null;
  direction?: "up" | "down" | "neutral" | null;
}) {
  if (!change) return null;
  const color =
    direction === "up"
      ? "#00C9A7"
      : direction === "down"
        ? "#FF6B6B"
        : "var(--text-secondary)";
  const arrow =
    direction === "up" ? "\u2191" : direction === "down" ? "\u2193" : "\u2192";
  return (
    <span
      style={{
        fontSize: "0.78rem",
        fontFamily: "var(--font-mono)",
        color,
        marginLeft: "0.5rem",
      }}
    >
      {arrow} {change}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/*  Priority badge                                                    */
/* ------------------------------------------------------------------ */

function PriorityBadge({ priority }: { priority: number }) {
  const bg =
    priority === 1
      ? "rgba(0,201,167,0.15)"
      : priority === 2
        ? "rgba(255,217,61,0.12)"
        : "rgba(255,255,255,0.05)";
  const color =
    priority === 1
      ? "#00C9A7"
      : priority === 2
        ? "#FFD93D"
        : "var(--text-secondary)";
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: 28,
        height: 28,
        borderRadius: "50%",
        background: bg,
        color,
        fontFamily: "var(--font-mono)",
        fontWeight: 600,
        fontSize: "0.78rem",
        flexShrink: 0,
      }}
    >
      {priority}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/*  Main page                                                         */
/* ------------------------------------------------------------------ */

export default function DashboardPage() {
  const params = useParams();
  const tenant = params.tenant as string;

  const [dashboard, setDashboard] = useState<DashboardConfig | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${BASE}/dashboard/${tenant}`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data: DashboardConfig) => setDashboard(data))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [tenant]);

  /* ---- Card style ---- */

  const card: React.CSSProperties = {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: 12,
    padding: "1.25rem",
  };

  const sectionTitle: React.CSSProperties = {
    fontFamily: "var(--font-mono)",
    fontSize: "0.7rem",
    letterSpacing: "0.12em",
    textTransform: "uppercase",
    color: "var(--text-muted)",
    marginBottom: "1rem",
  };

  /* ---- Loading / Error ---- */

  if (loading) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          minHeight: "60vh",
          color: "var(--text-secondary)",
          fontFamily: "var(--font-mono)",
          fontSize: "0.85rem",
        }}
      >
        Loading dashboard...
      </div>
    );
  }

  if (error || !dashboard) {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          minHeight: "60vh",
          gap: "1rem",
        }}
      >
        <div style={{ color: "#FF6B6B", fontSize: "0.9rem" }}>
          {error === "404"
            ? "No dashboard found for this tenant."
            : `Failed to load dashboard (${error})`}
        </div>
        <Link
          href="/onboarding"
          style={{ color: "var(--teal)", fontSize: "0.85rem" }}
        >
          Start new engagement
        </Link>
      </div>
    );
  }

  const sortedRecs = [...dashboard.recommendations].sort(
    (a, b) => a.priority - b.priority,
  );

  return (
    <div
      style={{
        maxWidth: 960,
        margin: "0 auto",
        padding: "2rem 1.5rem 4rem",
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: "2rem" }}>
        <h1
          style={{
            fontFamily: "var(--font-answer)",
            fontWeight: 400,
            fontSize: "1.75rem",
            color: "var(--text-primary)",
            marginBottom: "0.35rem",
          }}
        >
          {dashboard.company_name}
        </h1>
        <p
          style={{
            color: "var(--text-secondary)",
            fontSize: "0.85rem",
            lineHeight: 1.5,
          }}
        >
          {dashboard.problem_statement}
        </p>
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.68rem",
            color: "var(--text-muted)",
            marginTop: "0.5rem",
          }}
        >
          Generated {new Date(dashboard.generated_at).toLocaleString()} | Data
          sources: {dashboard.data_sources_used.join(", ")}
        </div>
      </div>

      {/* Executive Summary */}
      <div style={sectionTitle}>Executive Summary</div>
      <div
        style={{
          ...card,
          borderLeft: "3px solid var(--teal)",
          marginBottom: "2rem",
        }}
      >
        <p
          style={{
            fontFamily: "var(--font-answer)",
            fontSize: "0.95rem",
            lineHeight: 1.75,
            color: "var(--text-primary)",
            whiteSpace: "pre-wrap",
          }}
        >
          {dashboard.executive_summary}
        </p>
      </div>

      {/* Metrics */}
      {dashboard.metrics.length > 0 && (
        <>
          <div style={sectionTitle}>Key Metrics</div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(2, 1fr)",
              gap: "0.75rem",
              marginBottom: "2rem",
            }}
          >
            {dashboard.metrics.map((m) => (
              <div key={m.id} style={card}>
                <div
                  style={{
                    fontSize: "0.75rem",
                    color: "var(--text-secondary)",
                    marginBottom: "0.4rem",
                    fontFamily: "var(--font-mono)",
                    letterSpacing: "0.02em",
                  }}
                >
                  {m.label}
                </div>
                <div
                  style={{
                    fontSize: "1.5rem",
                    fontWeight: 600,
                    color: "var(--text-primary)",
                    fontFamily: "var(--font-ui)",
                  }}
                >
                  {m.value}
                  <ChangeIndicator
                    change={m.change}
                    direction={m.change_direction}
                  />
                </div>
                {m.insight && (
                  <div
                    style={{
                      fontSize: "0.78rem",
                      color: "var(--text-secondary)",
                      marginTop: "0.5rem",
                      lineHeight: 1.5,
                    }}
                  >
                    {m.insight}
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {/* Charts */}
      {dashboard.charts.length > 0 && (
        <>
          <div style={sectionTitle}>Analysis</div>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "1rem",
              marginBottom: "2rem",
            }}
          >
            {dashboard.charts.map((spec) => (
              <div key={spec.id} style={card}>
                <div
                  style={{
                    fontWeight: 500,
                    fontSize: "0.92rem",
                    color: "var(--text-primary)",
                    marginBottom: "0.35rem",
                  }}
                >
                  {spec.title}
                </div>
                <div
                  style={{
                    fontSize: "0.78rem",
                    color: "var(--text-secondary)",
                    marginBottom: "1rem",
                  }}
                >
                  {spec.insight}
                </div>
                <DashboardChart spec={spec} />
              </div>
            ))}
          </div>
        </>
      )}

      {/* Recommendations */}
      {sortedRecs.length > 0 && (
        <>
          <div style={sectionTitle}>Recommendations</div>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "0.75rem",
              marginBottom: "2.5rem",
            }}
          >
            {sortedRecs.map((rec, i) => (
              <div
                key={i}
                style={{
                  ...card,
                  display: "flex",
                  gap: "1rem",
                  alignItems: "flex-start",
                  borderLeft:
                    rec.priority === 1
                      ? "3px solid var(--teal)"
                      : rec.priority === 2
                        ? "3px solid #FFD93D"
                        : "3px solid var(--border)",
                }}
              >
                <PriorityBadge priority={rec.priority} />
                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      fontWeight: 600,
                      fontSize: "0.92rem",
                      color: "var(--text-primary)",
                      marginBottom: "0.3rem",
                    }}
                  >
                    {rec.title}
                  </div>
                  <div
                    style={{
                      fontSize: "0.85rem",
                      color: "var(--text-secondary)",
                      lineHeight: 1.6,
                      marginBottom: "0.4rem",
                    }}
                  >
                    {rec.detail}
                  </div>
                  <div
                    style={{
                      fontSize: "0.75rem",
                      color: "var(--text-muted)",
                      fontStyle: "italic",
                    }}
                  >
                    Evidence: {rec.evidence}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* CTA */}
      <div style={{ textAlign: "center" }}>
        <Link
          href={`/chat/${tenant}`}
          style={{
            display: "inline-block",
            padding: "0.75rem 2rem",
            background: "var(--teal)",
            color: "var(--bg)",
            borderRadius: 8,
            fontFamily: "var(--font-ui)",
            fontWeight: 600,
            fontSize: "0.9rem",
            textDecoration: "none",
            transition: "opacity 0.15s",
          }}
        >
          Ask follow-up questions
        </Link>
      </div>
    </div>
  );
}
