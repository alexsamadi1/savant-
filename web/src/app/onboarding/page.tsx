"use client";

import { useState, useRef, useCallback, DragEvent, ChangeEvent } from "react";
import { useRouter } from "next/navigation";

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function toTenant(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-");
}

/* ------------------------------------------------------------------ */
/*  Drop zone                                                         */
/* ------------------------------------------------------------------ */

function DropZone({
  accept,
  multiple,
  files,
  onFiles,
  label,
  hint,
}: {
  accept: string;
  multiple: boolean;
  files: File[];
  onFiles: (f: File[]) => void;
  label: string;
  hint: string;
}) {
  const [over, setOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setOver(false);
      const dropped = Array.from(e.dataTransfer.files);
      onFiles(multiple ? [...files, ...dropped] : dropped.slice(0, 1));
    },
    [files, multiple, onFiles],
  );

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const picked = Array.from(e.target.files || []);
      onFiles(multiple ? [...files, ...picked] : picked.slice(0, 1));
      e.target.value = "";
    },
    [files, multiple, onFiles],
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      style={{
        border: `2px dashed ${over ? "var(--teal)" : "var(--border-bright)"}`,
        borderRadius: 12,
        padding: "2.5rem 1.5rem",
        textAlign: "center",
        cursor: "pointer",
        background: over ? "var(--teal-dim)" : "var(--surface)",
        transition: "all 0.2s",
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        onChange={onChange}
        style={{ display: "none" }}
      />
      <div style={{ fontSize: "2rem", marginBottom: "0.5rem", opacity: 0.5 }}>
        +
      </div>
      <div style={{ color: "var(--text-primary)", fontWeight: 500 }}>
        {label}
      </div>
      <div
        style={{
          color: "var(--text-secondary)",
          fontSize: "0.8rem",
          marginTop: "0.25rem",
        }}
      >
        {hint}
      </div>
      {files.length > 0 && (
        <div style={{ marginTop: "1rem" }}>
          {files.map((f, i) => (
            <div
              key={i}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "0.4rem",
                background: "var(--teal-dim)",
                border: "1px solid var(--border)",
                borderRadius: 6,
                padding: "0.3rem 0.65rem",
                margin: "0.2rem",
                fontSize: "0.78rem",
                color: "var(--teal)",
              }}
            >
              {f.name}
              <span
                onClick={(e) => {
                  e.stopPropagation();
                  onFiles(files.filter((_, j) => j !== i));
                }}
                style={{ cursor: "pointer", opacity: 0.6 }}
              >
                x
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Step indicator                                                    */
/* ------------------------------------------------------------------ */

function Steps({ current }: { current: number }) {
  const labels = ["Discovery", "Data Upload", "Documents", "Questions"];
  return (
    <div
      style={{
        display: "flex",
        gap: "0.25rem",
        marginBottom: "2rem",
      }}
    >
      {labels.map((label, i) => (
        <div key={i} style={{ flex: 1, textAlign: "center" }}>
          <div
            style={{
              height: 3,
              borderRadius: 2,
              background:
                i < current
                  ? "var(--teal)"
                  : i === current
                    ? "var(--border-bright)"
                    : "var(--border)",
              marginBottom: "0.5rem",
              transition: "background 0.3s",
            }}
          />
          <span
            style={{
              fontSize: "0.72rem",
              fontFamily: "var(--font-mono)",
              letterSpacing: "0.04em",
              color:
                i <= current ? "var(--text-primary)" : "var(--text-muted)",
            }}
          >
            {label}
          </span>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Progress overlay                                                  */
/* ------------------------------------------------------------------ */

function ProgressScreen({
  status,
  progress,
}: {
  status: string;
  progress: string;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "60vh",
        gap: "1.5rem",
      }}
    >
      <div
        style={{
          width: 48,
          height: 48,
          border: "3px solid var(--border)",
          borderTopColor: "var(--teal)",
          borderRadius: "50%",
          animation: "spin 1s linear infinite",
        }}
      />
      <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.85rem",
          color: "var(--teal)",
          letterSpacing: "0.04em",
        }}
      >
        {status === "pending" && "Queuing analysis..."}
        {status === "running" && "Agent analyzing your data..."}
        {status === "error" && "Analysis failed"}
      </div>
      {progress && (
        <div style={{ color: "var(--text-secondary)", fontSize: "0.8rem" }}>
          {progress}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main page                                                         */
/* ------------------------------------------------------------------ */

export default function OnboardingPage() {
  const router = useRouter();
  const [step, setStep] = useState(0);

  // Step 1
  const [companyName, setCompanyName] = useState("");
  const [industry, setIndustry] = useState("");
  const [problemStatement, setProblemStatement] = useState("");

  // Step 2
  const [dataFiles, setDataFiles] = useState<File[]>([]);

  // Step 3
  const [docFiles, setDocFiles] = useState<File[]>([]);

  // Step 4
  const [questions, setQuestions] = useState(["", "", ""]);

  // Submission state
  const [submitting, setSubmitting] = useState(false);
  const [pollStatus, setPollStatus] = useState("");
  const [pollProgress, setPollProgress] = useState("");
  const [error, setError] = useState("");

  const tenant = toTenant(companyName);

  const canAdvance = (): boolean => {
    if (step === 0) return companyName.trim() !== "" && problemStatement.trim() !== "";
    if (step === 1) return dataFiles.length > 0;
    // Step 2 (docs) is optional
    if (step === 3) return questions.filter((q) => q.trim()).length >= 1;
    return true;
  };

  const setQuestion = (i: number, val: string) => {
    const copy = [...questions];
    copy[i] = val;
    setQuestions(copy);
  };

  const addQuestion = () => {
    if (questions.length < 5) setQuestions([...questions, ""]);
  };

  const removeQuestion = (i: number) => {
    if (questions.length > 1) setQuestions(questions.filter((_, j) => j !== i));
  };

  /* ---- Submit ---- */

  const handleSubmit = async () => {
    setSubmitting(true);
    setError("");

    try {
      // 1. Ingest data files
      for (const file of dataFiles) {
        const form = new FormData();
        form.append("file", file);
        form.append("tenant", tenant);
        form.append("problem_statement", problemStatement);
        const r = await fetch(`${BASE}/ingest/data`, {
          method: "POST",
          body: form,
        });
        if (!r.ok) throw new Error(`Data ingest failed: ${r.status}`);
      }

      // 2. Ingest documents
      if (docFiles.length > 0) {
        const form = new FormData();
        for (const file of docFiles) form.append("files", file);
        form.append("tenant", tenant);
        form.append("problem_statement", problemStatement);
        const r = await fetch(`${BASE}/ingest/documents`, {
          method: "POST",
          body: form,
        });
        if (!r.ok) throw new Error(`Document ingest failed: ${r.status}`);
      }

      // 3. Trigger analysis
      const keyQuestions = questions.filter((q) => q.trim());
      const body = {
        tenant,
        discovery: {
          tenant,
          company_name: companyName,
          industry,
          problem_statement: problemStatement,
          key_questions: keyQuestions,
          data_description: dataFiles.map((f) => f.name).join(", "),
        },
        model: "gpt-4o-mini",
      };
      const r = await fetch(`${BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`Analysis trigger failed: ${r.status}`);

      // 4. Poll
      setPollStatus("pending");
      const poll = async () => {
        while (true) {
          await new Promise((res) => setTimeout(res, 3000));
          const pr = await fetch(`${BASE}/analyze/status/${tenant}`);
          const data = await pr.json();
          setPollStatus(data.status);
          setPollProgress(data.progress || "");
          if (data.status === "complete") {
            router.push(`/dashboard/${tenant}`);
            return;
          }
          if (data.status === "error") {
            setError(data.error || "Analysis failed");
            setSubmitting(false);
            return;
          }
        }
      };
      poll();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      setSubmitting(false);
    }
  };

  /* ---- Shared styles ---- */

  const inputStyle: React.CSSProperties = {
    width: "100%",
    padding: "0.65rem 0.85rem",
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    color: "var(--text-primary)",
    fontFamily: "var(--font-ui)",
    fontSize: "0.9rem",
    outline: "none",
    transition: "border-color 0.15s",
  };

  const labelStyle: React.CSSProperties = {
    display: "block",
    marginBottom: "0.35rem",
    fontSize: "0.78rem",
    fontFamily: "var(--font-mono)",
    color: "var(--text-secondary)",
    letterSpacing: "0.04em",
  };

  const btnPrimary: React.CSSProperties = {
    padding: "0.65rem 1.75rem",
    background: "var(--teal)",
    color: "var(--bg)",
    border: "none",
    borderRadius: 8,
    fontFamily: "var(--font-ui)",
    fontWeight: 600,
    fontSize: "0.88rem",
    cursor: "pointer",
    transition: "opacity 0.15s",
  };

  const btnSecondary: React.CSSProperties = {
    padding: "0.65rem 1.75rem",
    background: "transparent",
    color: "var(--text-secondary)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    fontFamily: "var(--font-ui)",
    fontWeight: 500,
    fontSize: "0.88rem",
    cursor: "pointer",
  };

  /* ---- Render ---- */

  if (submitting) {
    return (
      <div style={{ maxWidth: 600, margin: "3rem auto", padding: "0 1.5rem" }}>
        <ProgressScreen status={pollStatus} progress={pollProgress} />
        {error && (
          <div
            style={{
              textAlign: "center",
              color: "#ff6b6b",
              marginTop: "1rem",
              fontSize: "0.85rem",
            }}
          >
            {error}
          </div>
        )}
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 600, margin: "3rem auto", padding: "0 1.5rem" }}>
      <h1
        style={{
          fontFamily: "var(--font-answer)",
          fontWeight: 400,
          fontSize: "1.75rem",
          color: "var(--text-primary)",
          marginBottom: "0.5rem",
        }}
      >
        New Engagement
      </h1>
      <p
        style={{
          color: "var(--text-secondary)",
          fontSize: "0.88rem",
          marginBottom: "2rem",
        }}
      >
        Tell us about your company and upload your data.
      </p>

      <Steps current={step} />

      {/* Step 1: Discovery */}
      {step === 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
          <div>
            <label style={labelStyle}>Company Name *</label>
            <input
              style={inputStyle}
              value={companyName}
              onChange={(e) => setCompanyName(e.target.value)}
              placeholder="Acme Corp"
              onFocus={(e) => (e.target.style.borderColor = "var(--border-bright)")}
              onBlur={(e) => (e.target.style.borderColor = "var(--border)")}
            />
            {companyName && (
              <div
                style={{
                  fontSize: "0.72rem",
                  color: "var(--text-muted)",
                  marginTop: "0.3rem",
                  fontFamily: "var(--font-mono)",
                }}
              >
                Tenant: {tenant}
              </div>
            )}
          </div>
          <div>
            <label style={labelStyle}>Industry</label>
            <input
              style={inputStyle}
              value={industry}
              onChange={(e) => setIndustry(e.target.value)}
              placeholder="e.g. Professional Services, Healthcare, GovCon"
              onFocus={(e) => (e.target.style.borderColor = "var(--border-bright)")}
              onBlur={(e) => (e.target.style.borderColor = "var(--border)")}
            />
          </div>
          <div>
            <label style={labelStyle}>Problem Statement *</label>
            <textarea
              style={{ ...inputStyle, minHeight: 100, resize: "vertical" }}
              value={problemStatement}
              onChange={(e) => setProblemStatement(e.target.value)}
              placeholder="What challenge are you trying to solve? What decisions do you need to make?"
              onFocus={(e) => (e.target.style.borderColor = "var(--border-bright)")}
              onBlur={(e) => (e.target.style.borderColor = "var(--border)")}
            />
          </div>
        </div>
      )}

      {/* Step 2: Data upload */}
      {step === 1 && (
        <div>
          <DropZone
            accept=".csv,.xlsx,.xls"
            multiple={true}
            files={dataFiles}
            onFiles={setDataFiles}
            label="Drop CSV or Excel files here"
            hint="Accepts .csv, .xlsx, .xls"
          />
        </div>
      )}

      {/* Step 3: Document upload */}
      {step === 2 && (
        <div>
          <DropZone
            accept=".pdf,.docx,.doc,.txt"
            multiple={true}
            files={docFiles}
            onFiles={setDocFiles}
            label="Drop PDF or DOCX files here (optional)"
            hint="Accepts .pdf, .docx, .txt — skip if you have no documents"
          />
        </div>
      )}

      {/* Step 4: Questions + Review */}
      {step === 3 && (
        <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
          <div>
            <label style={labelStyle}>Key Questions (1–5)</label>
            {questions.map((q, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  gap: "0.5rem",
                  marginBottom: "0.5rem",
                  alignItems: "center",
                }}
              >
                <input
                  style={{ ...inputStyle, flex: 1 }}
                  value={q}
                  onChange={(e) => setQuestion(i, e.target.value)}
                  placeholder={`Question ${i + 1}`}
                  onFocus={(e) =>
                    (e.target.style.borderColor = "var(--border-bright)")
                  }
                  onBlur={(e) =>
                    (e.target.style.borderColor = "var(--border)")
                  }
                />
                {questions.length > 1 && (
                  <button
                    onClick={() => removeQuestion(i)}
                    style={{
                      background: "none",
                      border: "none",
                      color: "var(--text-muted)",
                      cursor: "pointer",
                      fontSize: "1.1rem",
                      padding: "0.25rem",
                    }}
                  >
                    x
                  </button>
                )}
              </div>
            ))}
            {questions.length < 5 && (
              <button
                onClick={addQuestion}
                style={{
                  ...btnSecondary,
                  padding: "0.35rem 0.85rem",
                  fontSize: "0.78rem",
                }}
              >
                + Add question
              </button>
            )}
          </div>

          {/* Review summary */}
          <div
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 10,
              padding: "1.25rem",
              marginTop: "0.5rem",
            }}
          >
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.72rem",
                color: "var(--teal)",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                marginBottom: "0.75rem",
              }}
            >
              Review
            </div>
            <div style={{ fontSize: "0.85rem", lineHeight: 1.7, color: "var(--text-secondary)" }}>
              <div>
                <strong style={{ color: "var(--text-primary)" }}>Company:</strong>{" "}
                {companyName} ({industry || "—"})
              </div>
              <div>
                <strong style={{ color: "var(--text-primary)" }}>Problem:</strong>{" "}
                {problemStatement.slice(0, 120)}
                {problemStatement.length > 120 ? "..." : ""}
              </div>
              <div>
                <strong style={{ color: "var(--text-primary)" }}>Data files:</strong>{" "}
                {dataFiles.map((f) => f.name).join(", ")}
              </div>
              <div>
                <strong style={{ color: "var(--text-primary)" }}>Documents:</strong>{" "}
                {docFiles.length > 0
                  ? docFiles.map((f) => f.name).join(", ")
                  : "None"}
              </div>
              <div>
                <strong style={{ color: "var(--text-primary)" }}>Questions:</strong>{" "}
                {questions.filter((q) => q.trim()).length}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error message */}
      {error && (
        <div
          style={{
            color: "#ff6b6b",
            fontSize: "0.85rem",
            marginTop: "1rem",
            padding: "0.65rem 0.85rem",
            background: "rgba(255,107,107,0.08)",
            borderRadius: 8,
            border: "1px solid rgba(255,107,107,0.2)",
          }}
        >
          {error}
        </div>
      )}

      {/* Navigation buttons */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: "2rem",
          gap: "0.75rem",
        }}
      >
        {step > 0 ? (
          <button onClick={() => setStep(step - 1)} style={btnSecondary}>
            Back
          </button>
        ) : (
          <div />
        )}

        {step < 3 ? (
          <button
            onClick={() => setStep(step + 1)}
            disabled={!canAdvance()}
            style={{
              ...btnPrimary,
              opacity: canAdvance() ? 1 : 0.4,
              cursor: canAdvance() ? "pointer" : "not-allowed",
            }}
          >
            {step === 2 && docFiles.length === 0 ? "Skip" : "Continue"}
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            disabled={!canAdvance()}
            style={{
              ...btnPrimary,
              opacity: canAdvance() ? 1 : 0.4,
              cursor: canAdvance() ? "pointer" : "not-allowed",
            }}
          >
            Submit & Analyze
          </button>
        )}
      </div>
    </div>
  );
}
