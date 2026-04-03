"use client";

import { useEffect, useRef, useState } from "react";

const PRIMARY = "#185FA5";

function getPredictUrl(): string {
  const base = (
    process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
  ).replace(/\/+$/, "");
  return `${base}/predict`;
}

const RESULTS = {
  AD: {
    key: "AD" as const,
    label: "Alzheimer's Disease (AD)",
    shortLabel: "AD",
    predicted: "Predicted class",
    bg: "#FCEBEB",
    text: "#791F1F",
    bar: "#E24B4A",
    dot: "#E24B4A",
    metrics: { AD: 80, MCI: 8, CN: 12 },
    alert:
      "High-probability neurodegeneration detected. Referral for comprehensive clinical and cognitive evaluation is strongly recommended.",
  },
  MCI: {
    key: "MCI" as const,
    label: "Mild Cognitive Impairment (MCI)",
    shortLabel: "MCI",
    predicted: "Predicted class",
    bg: "#FAEEDA",
    text: "#633806",
    bar: "#EF9F27",
    dot: "#EF9F27",
    metrics: { AD: 22, MCI: 65, CN: 13 },
    alert:
      "Mild cognitive decline indicated. Longitudinal monitoring and neuropsychological assessment are advised.",
  },
  CN: {
    key: "CN" as const,
    label: "Cognitively Normal (CN)",
    shortLabel: "CN",
    predicted: "Predicted class",
    bg: "#EAF3DE",
    text: "#27500A",
    bar: "#639922",
    dot: "#639922",
    metrics: { AD: 4, MCI: 9, CN: 87 },
    alert:
      "No significant neurodegeneration detected. Routine follow-up as clinically appropriate.",
  },
};

type ResultKey = keyof typeof RESULTS;

function UploadIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      width="40"
      height="40"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

function BrainIllustration({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 120 120"
      fill="none"
      aria-hidden
    >
      <path
        d="M60 12c-8 0-15 4-19 10-4-2-9-3-14-3-12 0-22 10-22 22 0 5 2 10 5 14-3 5-5 11-5 17 0 16 13 29 29 29h4c5 8 14 13 24 13s19-5 24-13h4c16 0 29-13 29-29 0-6-2-12-5-17 3-4 5-9 5-14 0-12-10-22-22-22-5 0-10 1-14 3-4-6-11-10-19-10Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
        opacity="0.35"
      />
      <path
        d="M45 52c0-6 5-11 11-11s11 5 11 11M64 52c0-6 5-11 11-11"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        opacity="0.35"
      />
    </svg>
  );
}

function BrainPlayIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <path
        d="M12 3a4 4 0 0 0-4 4v2.5a4 4 0 0 0 4 4 4 4 0 0 0 4-4V7a4 4 0 0 0-4-4Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <path
        d="M8 14v3a4 4 0 0 0 8 0v-3"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <path
        d="M12 21v-2M10 5l-1.5-1.5M14 5l1.5-1.5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <path d="M10 12.5 14 10v5l-4-2.5Z" fill="currentColor" />
    </svg>
  );
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      className={`h-5 w-5 shrink-0 text-slate-500 transition-transform ${open ? "rotate-180" : ""}`}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      aria-hidden
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

export default function Home() {
  const scanInputRef = useRef<HTMLInputElement>(null);
  const [scanFile, setScanFile] = useState<File | null>(null);
  const [scanFileName, setScanFileName] = useState<string | null>(null);
  const [csvFileName, setCsvFileName] = useState<string | null>(null);
  const [metadataOpen, setMetadataOpen] = useState(false);
  const [screeningType, setScreeningType] = useState<"binary" | "multiclass">(
    "multiclass",
  );
  const [loading, setLoading] = useState(false);
  const [analysisDone, setAnalysisDone] = useState(false);
  const [activeTab, setActiveTab] = useState<ResultKey>("AD");
  const [barProgress, setBarProgress] = useState({ AD: 0, MCI: 0, CN: 0 });
  const [apiMetrics, setApiMetrics] = useState<{
    AD: number;
    MCI: number;
    CN: number;
  } | null>(null);

  const current = RESULTS[activeTab];

  useEffect(() => {
    if (!analysisDone) {
      setBarProgress({ AD: 0, MCI: 0, CN: 0 });
      return;
    }
    const m = apiMetrics || RESULTS[activeTab].metrics;
    setBarProgress({ AD: 0, MCI: 0, CN: 0 });
    const t = window.setTimeout(() => {
      setBarProgress({ AD: m.AD, MCI: m.MCI, CN: m.CN });
    }, 50);
    return () => window.clearTimeout(t);
  }, [analysisDone, activeTab, apiMetrics]);

  function handlePickScan() {
    scanInputRef.current?.click();
  }

  function handleScanFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) {
      setScanFile(file);
      setScanFileName(file.name);
    }
    e.target.value = "";
  }

  function handlePickCsv() {
    setCsvFileName("subject_metadata.csv");
  }

  async function runAnalysis() {
    if (!scanFile || loading) return;
    setAnalysisDone(false);
    setApiMetrics(null);
    try {
      setLoading(true);
      const formData = new FormData();
      formData.append("file", scanFile);
      formData.append("model_type", screeningType);
      const response = await fetch(getPredictUrl(), {
        method: "POST",
        body: formData,
      });
      const data = (await response.json()) as {
        prediction: ResultKey;
        confidence: { AD: number; MCI: number; CN: number };
        detail?: unknown;
        filename?: string;
        model_used?: string;
      };
      if (!response.ok) {
        const detail = data.detail;
        const msg =
          typeof detail === "string"
            ? detail
            : `Request failed (${response.status})`;
        throw new Error(msg);
      }
      setActiveTab(data.prediction);
      setApiMetrics(data.confidence);
      setAnalysisDone(true);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <input
        ref={scanInputRef}
        type="file"
        accept=".nii,.nii.gz"
        className="hidden"
        onChange={handleScanFileChange}
      />

      <header className="sticky top-0 z-10 border-b border-slate-200/80 bg-white/95 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4 sm:px-6">
          <div className="flex items-center gap-2.5">
            <div
              className="flex h-9 w-9 items-center justify-center rounded-lg text-[11px] font-bold text-white shadow-sm"
              style={{ backgroundColor: PRIMARY }}
              aria-hidden
            >
              AI
            </div>
            <span className="text-lg font-semibold tracking-tight text-slate-900">
              CerebroAI
            </span>
          </div>
          <div
            className="flex h-9 w-9 items-center justify-center rounded-full text-sm font-semibold text-white"
            style={{ backgroundColor: PRIMARY }}
            title="User"
          >
            JD
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:py-8">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 lg:gap-8">
          {/* Left: Input */}
          <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
            <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-500">
              Input &amp; Controls
            </h2>

            <button
              type="button"
              onClick={handlePickScan}
              className="group w-full rounded-xl border-2 border-dashed border-slate-300 bg-slate-50/50 px-4 py-10 text-center transition-colors hover:border-[#185FA5]/50 hover:bg-slate-50"
            >
              <UploadIcon className="mx-auto mb-3 text-[#185FA5] opacity-90" />
              <p className="text-sm font-medium text-slate-800">
                Drag &amp; Drop MRI scan or Click to Browse
              </p>
              <p className="mt-1 text-xs text-slate-500">
                Accepts .nii and .nii.gz formats (T1-weighted)
              </p>
            </button>

            {scanFileName && (
              <div className="mt-3 flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-left text-sm text-slate-800 shadow-sm">
                <span className="truncate font-medium">{scanFileName}</span>
                <span
                  className="ml-auto shrink-0 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-white"
                  style={{ backgroundColor: PRIMARY }}
                >
                  NII.GZ
                </span>
              </div>
            )}

            <div className="mt-5 border-t border-slate-100 pt-5">
              <button
                type="button"
                onClick={() => setMetadataOpen((o) => !o)}
                className="flex w-full items-center justify-between rounded-lg border border-slate-200 bg-slate-50/80 px-4 py-3 text-left text-sm font-medium text-slate-800 transition-colors hover:bg-slate-50"
              >
                <span>Optional subject metadata</span>
                <ChevronIcon open={metadataOpen} />
              </button>
              {metadataOpen && (
                <button
                  type="button"
                  onClick={handlePickCsv}
                  className="mt-3 w-full rounded-xl border-2 border-dashed border-slate-300 bg-white px-4 py-8 text-center text-sm text-slate-600 transition-colors hover:border-[#185FA5]/40"
                >
                  Drop CSV here or click to browse
                </button>
              )}
              {metadataOpen && csvFileName && (
                <p className="mt-2 text-xs text-slate-600">
                  Selected:{" "}
                  <span className="font-medium text-slate-800">{csvFileName}</span>
                </p>
              )}
            </div>

            <div className="mt-6">
              <p className="mb-2 text-xs font-medium text-slate-500">
                Screening type
              </p>
              <div className="flex flex-col gap-2 sm:flex-row sm:rounded-full sm:bg-slate-100 sm:p-1">
                <button
                  type="button"
                  onClick={() => setScreeningType("binary")}
                  className={`rounded-full px-4 py-2.5 text-center text-xs font-medium leading-snug transition-colors sm:flex-1 ${
                    screeningType === "binary"
                      ? "text-white shadow-sm"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                  style={
                    screeningType === "binary"
                      ? { backgroundColor: PRIMARY }
                      : undefined
                  }
                >
                  Binary Classification (Normal vs. Alzheimer&apos;s)
                </button>
                <button
                  type="button"
                  onClick={() => setScreeningType("multiclass")}
                  className={`rounded-full px-4 py-2.5 text-center text-xs font-medium leading-snug transition-colors sm:flex-1 ${
                    screeningType === "multiclass"
                      ? "text-white shadow-sm"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                  style={
                    screeningType === "multiclass"
                      ? { backgroundColor: PRIMARY }
                      : undefined
                  }
                >
                  Multi-Class Assessment (Includes Mild Cognitive Impairment)
                </button>
              </div>
            </div>

            <div className="mt-8">
              <button
                type="button"
                disabled={!scanFile || loading}
                onClick={() => void runAnalysis()}
                className="flex w-full items-center justify-center gap-2 rounded-xl px-4 py-3.5 text-sm font-semibold text-white shadow-md transition-opacity disabled:cursor-not-allowed disabled:opacity-50"
                style={{ backgroundColor: PRIMARY }}
              >
                {loading ? (
                  <>
                    <svg
                      className="h-5 w-5 animate-spin text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                      aria-hidden
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    <span>Executing Preprocessing Pipeline &amp; Analyzing...</span>
                  </>
                ) : (
                  <>
                    <BrainPlayIcon className="text-white" />
                    <span>Run AI Analysis</span>
                  </>
                )}
              </button>
            </div>
          </section>

          {/* Right: Results */}
          <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
            <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-500">
              Screening Results
            </h2>

            {!analysisDone ? (
              <div className="flex min-h-[420px] flex-col items-center justify-center px-4 py-12 text-center">
                <BrainIllustration className="mb-4 h-28 w-28 text-slate-400 opacity-40" />
                <p className="text-base font-medium text-slate-600">
                  Upload a scan to view results
                </p>
                <p className="mt-1 max-w-xs text-sm text-slate-500">
                  AI screening results will appear here
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                <div className="flex flex-wrap gap-1 rounded-lg bg-slate-100 p-1">
                  {(["AD", "MCI", "CN"] as const).map((tab) => (
                    <button
                      key={tab}
                      type="button"
                      onClick={() => setActiveTab(tab)}
                      className={`flex-1 rounded-md px-3 py-2 text-center text-xs font-semibold transition-colors sm:text-sm ${
                        activeTab === tab
                          ? "bg-white text-slate-900 shadow-sm"
                          : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      {tab} result
                    </button>
                  ))}
                </div>

                <div
                  className="rounded-xl border border-slate-100 p-5"
                  style={{ backgroundColor: current.bg }}
                >
                  <p
                    className="text-[10px] font-semibold uppercase tracking-wider opacity-70"
                    style={{ color: current.text }}
                  >
                    {current.predicted}
                  </p>
                  <div className="mt-2 flex items-start gap-3">
                    <span
                      className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full"
                      style={{ backgroundColor: current.dot }}
                      aria-hidden
                    />
                    <h3
                      className="text-xl font-bold leading-snug sm:text-2xl"
                      style={{ color: current.text }}
                    >
                      {current.label}
                    </h3>
                  </div>
                </div>

                <div>
                  <p className="mb-3 text-sm font-semibold text-slate-800">
                    AI Confidence Metrics
                  </p>
                  <div className="space-y-4">
                    {(
                      [
                        {
                          key: "AD" as const,
                          bar: "#E24B4A",
                          text: "#791F1F",
                        },
                        {
                          key: "MCI" as const,
                          bar: "#EF9F27",
                          text: "#633806",
                        },
                        {
                          key: "CN" as const,
                          bar: "#639922",
                          text: "#27500A",
                        },
                      ] as const
                    ).map(({ key, bar, text }) => {
                      const pct = barProgress[key];
                      return (
                        <div key={key}>
                          <div className="mb-1 flex items-center justify-between text-xs font-medium text-slate-600">
                            <span>{key}</span>
                            <span style={{ color: text }}>{pct}%</span>
                          </div>
                          <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-200">
                            <div
                              className="h-full rounded-full transition-[width] duration-700 ease-out"
                              style={{
                                width: `${pct}%`,
                                backgroundColor: bar,
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div
                  className="border-l-4 py-3 pl-4 pr-2 text-sm leading-relaxed text-slate-700"
                  style={{ borderColor: current.bar }}
                >
                  {current.alert}
                </div>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
