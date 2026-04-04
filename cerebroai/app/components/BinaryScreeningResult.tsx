"use client";

import { useEffect, useState } from "react";

const AD_BG = "#FCEBEB";
const AD_TEXT = "#791F1F";
const AD_BAR = "#E24B4A";
const CN_BG = "#EAF3DE";
const CN_TEXT = "#27500A";
const CN_BAR = "#639922";

export type BinaryScreeningResultProps = {
  result: "ad" | "cn";
  confidence: { ad: number; cn: number };
  model: string;
};

export function BinaryScreeningResult({
  result,
  confidence,
  model,
}: BinaryScreeningResultProps) {
  const total = confidence.ad + confidence.cn;
  const adPct = total > 0 ? (confidence.ad / total) * 100 : 50;
  const cnPct = total > 0 ? (confidence.cn / total) * 100 : 50;

  const [bars, setBars] = useState({ ad: 0, cn: 0 });

  useEffect(() => {
    setBars({ ad: 0, cn: 0 });
    const t = window.setTimeout(() => {
      setBars({ ad: adPct, cn: cnPct });
    }, 50);
    return () => window.clearTimeout(t);
  }, [adPct, cnPct]);

  const isAd = result === "ad";

  return (
    <div className="space-y-5">
      <p className="text-xs text-slate-500">
        Model: <span className="font-medium text-slate-700">{model}</span>
      </p>

      <div
        className="rounded-lg border border-slate-100 p-5"
        style={{ backgroundColor: isAd ? AD_BG : CN_BG }}
      >
        <p
          className="text-[10px] font-semibold uppercase tracking-wider opacity-70"
          style={{ color: isAd ? AD_TEXT : CN_TEXT }}
        >
          Predicted class
        </p>
        <div className="mt-2 flex items-start gap-3">
          <span
            className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full"
            style={{ backgroundColor: isAd ? AD_BAR : CN_BAR }}
            aria-hidden
          />
          <h3
            className="text-xl font-bold leading-snug sm:text-2xl"
            style={{ color: isAd ? AD_TEXT : CN_TEXT }}
          >
            {isAd
              ? "Alzheimer's Disease (AD)"
              : "Cognitively Normal (CN)"}
          </h3>
        </div>
      </div>

      <div>
        <p className="mb-3 text-sm font-semibold text-slate-800">
        How confident is the AI?
        </p>
        <div className="space-y-4">
          <div>
            <div className="mb-1 flex items-center justify-between text-xs font-medium text-slate-600">
              <span>AD</span>
              <span style={{ color: AD_TEXT }}>
                {bars.ad.toFixed(1)}%
              </span>
            </div>
            <div className="h-[10px] w-full overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full rounded-full transition-[width] duration-1000 ease"
                style={{
                  width: `${bars.ad}%`,
                  backgroundColor: AD_BAR,
                }}
              />
            </div>
          </div>
          <div>
            <div className="mb-1 flex items-center justify-between text-xs font-medium text-slate-600">
              <span>CN</span>
              <span style={{ color: CN_TEXT }}>
                {bars.cn.toFixed(1)}%
              </span>
            </div>
            <div className="h-[10px] w-full overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full rounded-full transition-[width] duration-1000 ease"
                style={{
                  width: `${bars.cn}%`,
                  backgroundColor: CN_BAR,
                }}
              />
            </div>
          </div>
        </div>
      </div>

      <div
        className="border-l-4 py-3 pl-4 pr-2 text-sm leading-relaxed text-slate-700"
        style={{ borderColor: isAd ? AD_BAR : CN_BAR }}
      >
        {isAd
          ? "High-probability Brain cell decline detected. Referral for comprehensive clinical and cognitive evaluation is strongly recommended."
          : "No significant Brain cell decline detected. Routine follow-up as clinically appropriate."}
      </div>
    </div>
  );
}
