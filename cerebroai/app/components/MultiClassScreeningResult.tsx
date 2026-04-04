"use client";

import { useEffect, useState } from "react";

const AD_BG = "#FCEBEB";
const AD_TEXT = "#791F1F";
const AD_BAR = "#E24B4A";
const MCI_BG = "#FAEEDA";
const MCI_TEXT = "#633806";
const MCI_BAR = "#EF9F27";
const CN_BG = "#EAF3DE";
const CN_TEXT = "#27500A";
const CN_BAR = "#639922";

export type MultiClassScreeningResultProps = {
  result: "ad" | "mci" | "cn";
  confidence: { ad: number; mci: number; cn: number };
  model: string;
};

export function MultiClassScreeningResult({
  result,
  confidence,
  model,
}: MultiClassScreeningResultProps) {
  const sum = confidence.ad + confidence.mci + confidence.cn;
  const adPct = sum > 0 ? (confidence.ad / sum) * 100 : 100 / 3;
  const mciPct = sum > 0 ? (confidence.mci / sum) * 100 : 100 / 3;
  const cnPct = sum > 0 ? (confidence.cn / sum) * 100 : 100 / 3;

  const [bars, setBars] = useState({ ad: 0, mci: 0, cn: 0 });

  useEffect(() => {
    setBars({ ad: 0, mci: 0, cn: 0 });
    const t = window.setTimeout(() => {
      setBars({ ad: adPct, mci: mciPct, cn: cnPct });
    }, 50);
    return () => window.clearTimeout(t);
  }, [adPct, mciPct, cnPct]);

  const badge =
    result === "ad"
      ? {
          bg: AD_BG,
          text: AD_TEXT,
          bar: AD_BAR,
          dot: AD_BAR,
          title: "Alzheimer's Disease (AD)",
        }
      : result === "mci"
        ? {
            bg: MCI_BG,
            text: MCI_TEXT,
            bar: MCI_BAR,
            dot: MCI_BAR,
            title: "Mild Cognitive Impairment (MCI)",
          }
        : {
            bg: CN_BG,
            text: CN_TEXT,
            bar: CN_BAR,
            dot: CN_BAR,
            title: "Cognitively Normal (CN)",
          };

  const alertText =
    result === "ad"
      ? "High-probability Brain cell decline detected. Referral for comprehensive clinical and cognitive evaluation is strongly recommended."
      : result === "mci"
        ? "Mild cognitive decline indicated. Regular check-ups over time and Memory and thinking tests are advised."
        : "No significant Brain cell decline detected. Routine follow-up as clinically appropriate.";

  return (
    <div className="space-y-5">
      <p className="text-xs text-slate-500">
        Model: <span className="font-medium text-slate-700">{model}</span>
      </p>

      <div
        className="rounded-lg border border-slate-100 p-5"
        style={{ backgroundColor: badge.bg }}
      >
        <p
          className="text-[10px] font-semibold uppercase tracking-wider opacity-70"
          style={{ color: badge.text }}
        >
          Predicted class
        </p>
        <div className="mt-2 flex items-start gap-3">
          <span
            className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full"
            style={{ backgroundColor: badge.dot }}
            aria-hidden
          />
          <h3
            className="text-xl font-bold leading-snug sm:text-2xl"
            style={{ color: badge.text }}
          >
            {badge.title}
          </h3>
        </div>
      </div>

      <div>
        <p className="mb-3 text-sm font-semibold text-slate-800">
        How confident is the AI?
        </p>
        <div className="space-y-4">
          {(
            [
              { key: "ad" as const, label: "AD", pct: bars.ad, bar: AD_BAR, text: AD_TEXT },
              {
                key: "mci" as const,
                label: "MCI",
                pct: bars.mci,
                bar: MCI_BAR,
                text: MCI_TEXT,
              },
              { key: "cn" as const, label: "CN", pct: bars.cn, bar: CN_BAR, text: CN_TEXT },
            ] as const
          ).map(({ key, label, pct, bar, text }) => (
            <div key={key}>
              <div className="mb-1 flex items-center justify-between text-xs font-medium text-slate-600">
                <span>{label}</span>
                <span style={{ color: text }}>{pct.toFixed(1)}%</span>
              </div>
              <div className="h-[10px] w-full overflow-hidden rounded-full bg-slate-100">
                <div
                  className="h-full rounded-full transition-[width] duration-1000 ease"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: bar,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div
        className="border-l-4 py-3 pl-4 pr-2 text-sm leading-relaxed text-slate-700"
        style={{ borderColor: badge.bar }}
      >
        {alertText}
      </div>
    </div>
  );
}
