"use client";

import Image from "next/image";
import { useRef, useState } from "react";

import { BinaryScreeningResult } from "./components/BinaryScreeningResult";
import { MultiClassScreeningResult } from "./components/MultiClassScreeningResult";

const PRIMARY = "#185FA5";

function getAnalyzeUrl(): string {
  return "/api/analyze";
}

type AveragedBinary = {
  result: "ad" | "cn";
  confidence: { ad: number; cn: number };
} | null;

type AveragedMulticlass = {
  result: "ad" | "mci" | "cn";
  confidence: { ad: number; mci: number; cn: number };
} | null;

type ModelSlotData = {
  binary: AveragedBinary;
  multiclass: AveragedMulticlass;
};

type ModelSelectionKey =
  | "averaged"
  | "3d_pca_svm"
  | "2d_cnn_model"
  | "custom_cnn";

type AnalyzeResponse = {
  averaged: ModelSlotData;
  "3d_pca_svm": ModelSlotData;
  "2d_cnn_model": ModelSlotData;
  "custom_cnn": ModelSlotData;
  modelErrors?: string[];
  filename?: string;
};

const MODEL_OPTIONS: { value: ModelSelectionKey; label: string }[] = [
  { value: "averaged", label: "Averaged (All Models)" },
  { value: "3d_pca_svm", label: "3D PCA SVM" },
  { value: "2d_cnn_model", label: "2D CNN Model" },
  { value: "custom_cnn", label: "Custom CNN" },
];

const MODEL_SUBLABELS: Record<ModelSelectionKey, string> = {
  averaged: "Combined result from all models",
  "3d_pca_svm": "Statistical model",
  "2d_cnn_model": "Image recognition model",
  custom_cnn: "Advanced deep learning model",
};

const MODEL_RESULT_LABELS: Record<ModelSelectionKey, string> = {
  averaged: "Averaged (all models)",
  "3d_pca_svm": "3D PCA SVM",
  "2d_cnn_model": "2D CNN Model",
  custom_cnn: "Custom CNN",
};

function getModelSlot(
  payload: AnalyzeResponse,
  key: ModelSelectionKey,
): ModelSlotData {
  if (key === "averaged") return payload.averaged;
  return payload[key];
}

function scanFileKindLabel(name: string): string {
  const n = name.toLowerCase();
  if (n.endsWith(".npy")) return "NPY";
  if (n.endsWith(".nii.gz")) return "NII.GZ";
  if (n.endsWith(".nii")) return "NII";
  return "SCAN";
}

function SectionDivider({ label }: { label: string }) {
  return (
    <div
      className="my-5 flex min-h-[1rem] items-center gap-3"
      role="separator"
    >
      <div className="h-[0.5px] min-w-0 flex-1 bg-slate-200" />
      <span className="shrink-0 text-[12px] font-medium uppercase tracking-wide text-slate-400">
        {label}
      </span>
      <div className="h-[0.5px] min-w-0 flex-1 bg-slate-200" />
    </div>
  );
}

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

function CheckCircleIcon({ className }: { className?: string }) {
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
        d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10Z"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <path
        d="m8.5 12.5 2.5 2.5 5-5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
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

const SECTION_LABEL =
  "text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-400";

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
  const [analysisPayload, setAnalysisPayload] = useState<AnalyzeResponse | null>(
    null,
  );
  const [selectedModel, setSelectedModel] =
    useState<ModelSelectionKey>("averaged");

  const showRunPulse =
    Boolean(scanFile) && !analysisDone && !loading;

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
    setAnalysisPayload(null);
    try {
      setLoading(true);
      const formData = new FormData();
      formData.append("file", scanFile);
      const response = await fetch(getAnalyzeUrl(), {
        method: "POST",
        body: formData,
      });
      const data = (await response.json()) as AnalyzeResponse & {
        detail?: unknown;
      };
      if (!response.ok) {
        const detail = data.detail;
        const msg =
          typeof detail === "string"
            ? detail
            : `Request failed (${response.status})`;
        throw new Error(msg);
      }
      setAnalysisPayload(data as AnalyzeResponse);
      setSelectedModel("averaged");
      setAnalysisDone(true);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      <input
        ref={scanInputRef}
        type="file"
        accept=".nii,.nii.gz,.npy"
        className="hidden"
        onChange={handleScanFileChange}
      />

      <header className="sticky top-0 z-50 border-b border-slate-200 bg-white">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-3 px-4 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <Image
              src="/brain-placeholder.png"
              alt=""
              width={32}
              height={32}
              className="h-8 w-auto shrink-0 object-contain"
              priority
            />
            <span className="truncate text-base font-semibold tracking-tight text-slate-900">
              CerebroAI
            </span>
          </div>
          <div className="flex shrink-0 items-center gap-2 sm:gap-3">
            <span className="hidden text-right text-[11px] leading-snug text-slate-400 lg:inline lg:max-w-[200px] xl:max-w-none">
              Clinical Decision Support Tool
            </span>
            <span className="text-[10px] font-medium uppercase tracking-wide text-slate-400 lg:hidden">
              CDS
            </span>
            <div
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-slate-200 bg-slate-100 text-xs font-semibold text-slate-700"
              title="User"
            >
              JD
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl min-w-0 overflow-x-hidden px-4 py-6 sm:px-6 lg:py-8">
        <div className="grid min-w-0 grid-cols-1 gap-5 lg:grid-cols-2 lg:gap-6">
          {/* Left: Input */}
          <section className="min-w-0 rounded-xl border border-slate-200 bg-white p-6">
            <h2 className={`mb-5 ${SECTION_LABEL}`}>Input &amp; Controls</h2>

            <div className="space-y-5">
              <button
                type="button"
                onClick={handlePickScan}
                className="group w-full rounded-lg border-[1.5px] border-dashed border-slate-300 bg-slate-50/80 px-4 py-10 text-center transition-colors hover:border-slate-400 hover:bg-slate-50"
              >
                <UploadIcon className="mx-auto mb-3 text-[#185FA5] opacity-90" />
                <p className="text-sm font-medium text-slate-800">
                  Drag &amp; Drop MRI scan or Click to Browse
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  Accepts Brain MRI scan (.nii, .nii.gz, and .npy T1-weighted)
                </p>
              </button>

              {scanFileName ? (
                <div className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-left text-sm text-slate-800">
                  <CheckCircleIcon
                    className="shrink-0 text-emerald-600"
                    aria-hidden
                  />
                  <span className="min-w-0 flex-1 truncate font-medium">
                    {scanFileName}
                  </span>
                  <span
                    className="shrink-0 rounded-md border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-slate-600"
                  >
                    {scanFileKindLabel(scanFileName)}
                  </span>
                </div>
              ) : null}

              <SectionDivider label="Optional" />

              <div className="space-y-3">
                <button
                  type="button"
                  onClick={() => setMetadataOpen((o) => !o)}
                  className="flex w-full items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-left text-sm font-medium text-slate-800 transition-colors hover:bg-slate-100"
                >
                  <span>Optional subject metadata</span>
                  <ChevronIcon open={metadataOpen} />
                </button>
                {metadataOpen ? (
                  <button
                    type="button"
                    onClick={handlePickCsv}
                    className="w-full rounded-lg border-[1.5px] border-dashed border-slate-300 bg-white px-4 py-8 text-center text-sm text-slate-600 transition-colors hover:border-slate-400"
                  >
                    Drop CSV here or click to browse
                  </button>
                ) : null}
                {metadataOpen && csvFileName ? (
                  <p className="text-xs text-slate-500">
                    Selected:{" "}
                    <span className="font-medium text-slate-700">
                      {csvFileName}
                    </span>
                  </p>
                ) : null}
              </div>

              <SectionDivider label="Screening" />

              <div className="space-y-3">
                <p className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
                  Screening type
                </p>
                <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:rounded-lg sm:bg-slate-100 sm:p-1">
                  <button
                    type="button"
                    onClick={() => setScreeningType("binary")}
                    className={`rounded-lg px-4 py-2.5 text-center text-xs font-medium leading-snug transition-colors sm:min-w-0 sm:flex-1 ${
                      screeningType === "binary"
                        ? "text-white"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                    style={
                      screeningType === "binary"
                        ? { backgroundColor: PRIMARY }
                        : undefined
                    }
                  >
                    Simple Screening (Normal vs. Alzheimer&apos;s)
                  </button>
                  <button
                    type="button"
                    onClick={() => setScreeningType("multiclass")}
                    className={`rounded-lg px-4 py-2.5 text-center text-xs font-medium leading-snug transition-colors sm:min-w-0 sm:flex-1 ${
                      screeningType === "multiclass"
                        ? "text-white"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                    style={
                      screeningType === "multiclass"
                        ? { backgroundColor: PRIMARY }
                        : undefined
                    }
                  >
                    Detailed Screening (includes early-stage detection)
                  </button>
                </div>
              </div>

              <button
                type="button"
                disabled={!scanFile || loading}
                onClick={() => void runAnalysis()}
                className={`flex w-full items-center justify-center gap-2 rounded-lg px-4 py-3.5 text-sm font-semibold text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-50 ${
                  showRunPulse ? "animate-run-attention" : ""
                }`}
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
                    <span>Preparing Scanning... &amp; Analyzing...</span>
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
          <section className="min-w-0 rounded-xl border border-slate-200 bg-white p-6">
            <div className="mb-5 flex min-w-0 flex-col gap-4 sm:flex-row sm:items-start sm:justify-between lg:mb-6">
              <h2 className={`shrink-0 ${SECTION_LABEL}`}>Screening Results</h2>
              {analysisDone && analysisPayload && !loading ? (
                <div className="flex w-full min-w-0 max-w-full flex-col sm:max-w-[min(100%,280px)] sm:items-end">
                  <label className="sr-only" htmlFor="model-select">
                    Result source model
                  </label>
                  <div className="w-full min-w-0 max-w-full sm:max-w-none">
                    <select
                      id="model-select"
                      value={selectedModel}
                      onChange={(e) =>
                        setSelectedModel(e.target.value as ModelSelectionKey)
                      }
                      className="box-border w-full max-w-full min-w-0 cursor-pointer rounded-lg border border-slate-200 bg-slate-50 px-2 py-2 text-[11px] font-medium text-slate-700 outline-none ring-slate-300 transition-colors hover:border-slate-300 hover:bg-slate-100 focus:border-[#185FA5]/50 focus:ring-2 focus:ring-[#185FA5]/20 sm:px-2.5 sm:text-right sm:text-[12px]"
                    >
                      {MODEL_OPTIONS.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <p className="mt-2 break-words text-left text-[11px] leading-snug text-slate-500 sm:text-right">
                    {MODEL_SUBLABELS[selectedModel]}
                  </p>
                </div>
              ) : null}
            </div>

            {!analysisDone ? (
              <div className="flex min-h-[420px] flex-col items-center justify-center overflow-hidden px-2 py-12 text-center">
                <Image
                  src="/brain-placeholder.png"
                  alt=""
                  width={120}
                  height={120}
                  className="mb-5 h-[120px] w-[120px] object-contain opacity-25 grayscale"
                />
                <p className="text-base font-medium text-slate-600">
                  Upload a scan to view results
                </p>
                <p className="mt-2 max-w-xs text-sm text-slate-500">
                  AI screening results will appear here
                </p>
              </div>
            ) : analysisPayload ? (
              (() => {
                const slot = getModelSlot(analysisPayload, selectedModel);
                const modelTitle = MODEL_RESULT_LABELS[selectedModel];
                if (screeningType === "binary") {
                  if (slot.binary) {
                    return (
                      <BinaryScreeningResult
                        result={slot.binary.result}
                        confidence={slot.binary.confidence}
                        model={modelTitle}
                      />
                    );
                  }
                  return (
                    <p className="text-sm leading-relaxed text-slate-600">
                      Results unavailable for this model. Please try Averaged or
                      another model.
                    </p>
                  );
                }
                if (slot.multiclass) {
                  return (
                    <MultiClassScreeningResult
                      result={slot.multiclass.result}
                      confidence={slot.multiclass.confidence}
                      model={modelTitle}
                    />
                  );
                }
                return (
                  <p className="text-sm leading-relaxed text-slate-600">
                    Results unavailable for this model. Please try Averaged or
                    another model.
                  </p>
                );
              })()
            ) : null}
          </section>
        </div>
      </main>
    </div>
  );
}
