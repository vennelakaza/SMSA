import { useEffect, useRef, useState } from "react";

const SAMPLES = [
  { label: "Happy post", text: "Just got my order and the quality is amazing, totally worth it!" },
  { label: "Angry post", text: "Worst customer service ever. Waited two hours and nobody helped." },
  { label: "Plain question", text: "What time does the store open on Sunday?" },
];

const kind = (label = "") => {
  const l = label.toLowerCase();
  if (l.startsWith("pos")) return "pos";
  if (l.startsWith("neg")) return "neg";
  return "neu";
};

const NAMES = { pos: "Positive", neg: "Negative", neu: "Neutral" };
const HEADLINE = {
  pos: "Good vibes. This post sounds positive.",
  neg: "Rough tone. This post sounds negative.",
  neu: "Flat tone. This post sounds neutral.",
};

/* smoothly moves a number toward its target, so the face morphs instead of snapping */
function useTween(target, ms = 600) {
  const [v, setV] = useState(target);
  const from = useRef(target);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setV(target);
      from.current = target;
      return;
    }
    const start = performance.now();
    const a = from.current;
    let raf;
    const tick = (now) => {
      const p = Math.min(1, (now - start) / ms);
      const e = 1 - Math.pow(1 - p, 3);
      const cur = a + (target - a) * e;
      from.current = cur;
      setV(cur);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

function Face({ score, busy, idle }) {
  const t = useTween(score);
  const brow = 68 - 6 * t;
  const innerLift = Math.max(0, -t) * 11;
  const mouth = `M 64 132 Q 100 ${132 + 44 * t} 136 132`;
  return (
    <svg className={`face ${busy ? "busy" : ""} ${idle ? "idle" : ""}`} viewBox="0 0 200 200" role="img"
         aria-label={idle ? "Waiting for text" : t > 0.15 ? "Smiling face" : t < -0.15 ? "Frowning face" : "Neutral face"}>
      <circle cx="100" cy="100" r="82" className="head" />
      <circle cx="62" cy="118" r="13" className="cheek" style={{ opacity: Math.max(0, t) * 0.55 }} />
      <circle cx="138" cy="118" r="13" className="cheek" style={{ opacity: Math.max(0, t) * 0.55 }} />
      <g className="brows">
        <line x1="54" y1={brow} x2="88" y2={brow - innerLift} />
        <line x1="146" y1={brow} x2="112" y2={brow - innerLift} />
      </g>
      <g className="eyes">
        <ellipse cx="74" cy="90" rx="7" ry={idle ? 2 : 8} />
        <ellipse cx="126" cy="90" rx="7" ry={idle ? 2 : 8} />
      </g>
      <path d={mouth} className="mouth" />
    </svg>
  );
}

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [live, setLive] = useState(true);
  const reqId = useRef(0);

  async function analyze(value = text) {
    const clean = value.trim();
    if (!clean) {
      setResult(null);
      setError("");
      return;
    }
    const id = ++reqId.current;
    setBusy(true);
    setError("");
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: clean }),
      });
      if (!res.ok) throw new Error();
      const data = await res.json();
      if (id === reqId.current) setResult(data);
    } catch {
      if (id === reqId.current) {
        setResult(null);
        setError("Can't reach the server. Make sure the backend is running on port 8000.");
      }
    } finally {
      if (id === reqId.current) setBusy(false);
    }
  }

  /* live mode: analyze shortly after the person stops typing */
  useEffect(() => {
    if (!live) return;
    if (!text.trim()) {
      reqId.current++;
      setResult(null);
      setBusy(false);
      return;
    }
    const h = setTimeout(() => analyze(text), 600);
    return () => clearTimeout(h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, live]);

  const mood = result ? kind(result.sentiment) : "idle";
  const rows = ["pos", "neu", "neg"].map((k) => {
    const hit = Object.entries(result?.scores || {}).find(([name]) => kind(name) === k);
    return { k, value: hit ? hit[1] : 0 };
  });
  const score = result ? (rows[0].value - rows[2].value) : 0;
  const confidence = result?.confidence != null ? Math.round(result.confidence * 100) : null;

  return (
    <div className="app" data-mood={mood}>
      <main className="page">
        <header className="top">
          <h1>Tone Reader</h1>
          <p>Type or paste a social media post. The page reacts to how it sounds.</p>
        </header>

        <div className="grid">
          <section className="panel input">
            <label htmlFor="post">Your post</label>
            <textarea
              id="post"
              rows={7}
              maxLength={2000}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && (e.ctrlKey || e.metaKey) && analyze()}
              placeholder="Write something like “Loving the new update!”"
            />
            <div className="meta">
              <span>{text.length}/2000</span>
              <label className="switch">
                <input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} />
                Analyze as I type
              </label>
            </div>

            <div className="samples" aria-label="Example posts">
              {SAMPLES.map((s) => (
                <button type="button" key={s.label} className="chip" onClick={() => setText(s.text)}>
                  {s.label}
                </button>
              ))}
              {text && (
                <button type="button" className="chip ghost" onClick={() => setText("")}>
                  Clear
                </button>
              )}
            </div>

            {!live && (
              <button className="go" onClick={() => analyze()} disabled={busy || !text.trim()}>
                {busy ? "Analyzing…" : "Analyze sentiment"}
              </button>
            )}
            {error && <p className="error" role="alert">{error}</p>}
          </section>

          <section className="panel output" aria-live="polite">
            <Face score={score} busy={busy} idle={!result} />
            <p className="verdict">
              {result
                ? HEADLINE[mood]
                : text.trim() && busy
                ? "Reading your post…"
                : "Waiting for a post to read."}
            </p>
            {result && confidence != null && (
              <p className="confidence">
                {confidence}% sure it's <strong>{NAMES[mood].toLowerCase()}</strong>
              </p>
            )}

            <div className="bars">
              {rows.map(({ k, value }) => (
                <div className={`bar ${k} ${result && k === mood ? "lead" : ""}`} key={k}>
                  <span className="name">{NAMES[k]}</span>
                  <div className="track">
                    <div className="fill" style={{ width: `${Math.round(value * 100)}%` }} />
                  </div>
                  <span className="pct">{Math.round(value * 100)}%</span>
                </div>
              ))}
            </div>
          </section>
        </div>

        <footer>TF-IDF features + Logistic Regression · FastAPI · React</footer>
      </main>
    </div>
  );
}
