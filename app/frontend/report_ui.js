// Phase 9 UI components (additive)
// Assuming React & Babel are already loaded via CDN in index.html

const { useState, useEffect, useRef } = React;

// Utility: smooth scroll to element
function scrollToId(id) {
  const el = document.getElementById(id);
  if (el) {
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

// TOC component
function TOC({ sections, activeId, onSelect }) {
  return (
    <div className="p9-toc">
      <ul>
        {sections.map((sec) => (
          <li
            key={sec.id}
            className={activeId === sec.id ? "active" : ""}
            onClick={() => onSelect(sec.id)}
          >
            {sec.title}
          </li>
        ))}
      </ul>
    </div>
  );
}

// Claim span with tooltip
function ClaimSpan({ claim, depth, onClick }) {
  const unsupported = claim.unsupported;
  const hide = depth === "brief" && unsupported;
  if (hide) return null;
  const className = `p9-claim ${unsupported ? "unsupported" : ""}`;
  return (
    <span
      className={className}
      data-id={claim.id}
      data-type={claim.type}
      data-unsupported={unsupported}
      onClick={() => onClick(claim)}
    >
      {claim.text}
      <div className="p9-tooltip">
        {claim.type.toUpperCase()}{unsupported ? " (unsupported)" : ""}
      </div>
    </span>
  );
}

// Evidence inspector drawer
function EvidenceInspector({ claim, onClose }) {
  const [tab, setTab] = useState("Numbers");
  if (!claim) return null;
  const tabs = ["Numbers", "Cells", "Guidelines", "Verification"];
  return (
    <div className="p9-inspector open">
      <div className="p9-inspector-header">
        <strong>{claim.text}</strong>
        <button onClick={onClose} aria-label="Close">&times;</button>
      </div>
      <div className="p9-tab-header">
        {tabs.map((t) => (
          <button
            key={t}
            className={tab === t ? "active" : ""}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </div>
      <div className="p9-inspector-body">
        {tab === "Numbers" && (
          <div>
            {claim.metric_refs && claim.metric_refs.length > 0 ? (
              <ul>
                {claim.metric_refs.map((m) => (
                  <li key={m}>{m}</li>
                ))}
              </ul>
            ) : (
              <p>N/A</p>
            )}
          </div>
        )}
        {tab === "Cells" && (
          <div>
            {claim.cell_ids && claim.cell_ids.length > 0 ? (
              <ul>
                {claim.cell_ids.map((id) => (
                  <li key={id}>Cell {id}</li>
                ))}
              </ul>
            ) : (
              <p>N/A</p>
            )}
          </div>
        )}
        {tab === "Guidelines" && (
          <div>
            {claim.chunk_ids && claim.chunk_ids.length > 0 ? (
              <ul>
                {claim.chunk_ids.map((c) => (
                  <li key={c}>Chunk {c}</li>
                ))}
              </ul>
            ) : (
              <p>N/A</p>
            )}
          </div>
        )}
        {tab === "Verification" && (
          <div>
            {claim.critic_status ? (
              <p>{claim.critic_status}</p>
            ) : (
              <p>N/A</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// Main wrapper that replaces the old report rendering
function ReportWrapper({ report }) {
  const [depth, setDepth] = useState(localStorage.getItem("p9Depth") || "standard");
  const [highlight, setHighlight] = useState(localStorage.getItem("p9Highlight") === "true");
  const [activeSection, setActiveSection] = useState(null);
  const [inspectorClaim, setInspectorClaim] = useState(null);

  useEffect(() => {
    localStorage.setItem("p9Depth", depth);
  }, [depth]);
  useEffect(() => {
    localStorage.setItem("p9Highlight", highlight);
  }, [highlight]);

  // IntersectionObserver to track visible section
  useEffect(() => {
    if (!report || !report.sections) return;
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            setActiveSection(e.target.id);
          }
        });
      },
      { root: null, threshold: 0.3 }
    );
    report.sections.forEach((sec) => {
      const el = document.getElementById(sec.id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, [report]);

  const handleDepth = (mode) => setDepth(mode);

  return (
    <div className={`p9-report-wrap ${highlight ? "p9-highlight" : ""}`}>
      <TOC
        sections={report.sections}
        activeId={activeSection}
        onSelect={(id) => scrollToId(id)}
      />
      <div className="p9-content" style={{ flex: 1 }}>
        <div className="p9-toolbar">
          <button
            className={`p9-depth-btn ${depth === "brief" ? "active" : ""}`}
            onClick={() => handleDepth("brief")}
          >
            Brief
          </button>
          <button
            className={`p9-depth-btn ${depth === "standard" ? "active" : ""}`}
            onClick={() => handleDepth("standard")}
          >
            Standard
          </button>
          <button
            className={`p9-depth-btn ${depth === "full" ? "active" : ""}`}
            onClick={() => handleDepth("full")}
          >
            Full
          </button>
          <label className="p9-toggle">
            <input
              type="checkbox"
              checked={highlight}
              onChange={(e) => setHighlight(e.target.checked)}
            />
            Reading mode
          </label>
        </div>
        {report.sections.map((sec) => (
          <section key={sec.id} id={sec.id} className="p9-section" style={{ marginBottom: "2rem" }}>
            <h2>{sec.title}</h2>
            <p>{sec.content}</p>
            {sec.claims && sec.claims.map((claim) => (
              <ClaimSpan
                key={claim.id}
                claim={claim}
                depth={depth}
                onClick={(c) => setInspectorClaim(c)}
              />
            ))}
          </section>
        ))}
      </div>
      {inspectorClaim && (
        <EvidenceInspector
          claim={inspectorClaim}
          onClose={() => setInspectorClaim(null)}
        />
      )}
    </div>
  );
}

// Export globally so existing code can render it
window.ReportWrapper = ReportWrapper;

