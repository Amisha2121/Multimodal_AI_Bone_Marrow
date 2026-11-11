import re

with open("app/static/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# 1. Add CSS for disease differential and multi-databank cards
css_marker = "/* ── History ── */"
new_styles = """/* ── Phase 10: Clinical Disease Detection & Multi-Databank Evidence ── */
        .med-diff-card{background:#ffffff;border:1px solid #cbd5e1;border-radius:10px;padding:1.25rem;margin-bottom:1.5rem;box-shadow:0 2px 6px rgba(0,0,0,.04);}
        .med-diff-title{font-size:1.05rem;font-weight:700;color:#0f172a;display:flex;align-items:center;justify-content:space-between;margin-bottom:.85rem;border-bottom:2px solid #e2e8f0;padding-bottom:.5rem;}
        .med-cand-box{background:#f8fafc;border-left:4px solid #3b82f6;border-radius:6px;padding:.85rem 1rem;margin-bottom:1rem;border-top:1px solid #e2e8f0;border-right:1px solid #e2e8f0;border-bottom:1px solid #e2e8f0;}
        .med-cand-box.high{border-left-color:#ef4444;background:#fffaf0;}
        .med-cand-box.moderate{border-left-color:#f59e0b;background:#fefce8;}
        .med-cand-box.skew{border-left-color:#8b5cf6;background:#faf5ff;}
        .med-cand-box.baseline{border-left-color:#10b981;background:#f0fdf4;}
        .med-cand-header{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.4rem;margin-bottom:.4rem;}
        .med-cand-name{font-weight:700;font-size:.95rem;color:#1e293b;}
        .med-badge{font-size:.7rem;font-weight:700;padding:2px 8px;border-radius:12px;text-transform:uppercase;letter-spacing:.4px;}
        .med-badge.high{background:#fee2e2;color:#991b1b;border:1px solid #f87171;}
        .med-badge.moderate{background:#fef3c7;color:#92400e;border:1px solid #fcd34d;}
        .med-badge.skew{background:#ede9fe;color:#5b21b6;border:1px solid #c4b5fd;}
        .med-badge.baseline{background:#dcfce7;color:#166534;border:1px solid #86efac;}
        .med-cand-criteria{margin-top:.4rem;font-size:.825rem;color:#334155;line-height:1.5;}
        .med-cand-criteria ul{margin-left:1.2rem;margin-top:.2rem;}
        .med-biomarkers{display:flex;flex-wrap:wrap;gap:4px;margin-top:.5rem;align-items:center;}
        .med-bio-tag{font-size:.7rem;font-family:monospace;background:#e0f2fe;color:#0369a1;padding:2px 7px;border-radius:4px;font-weight:600;}
        .med-tests-box{margin-top:.6rem;background:#ffffff;border:1px dashed #cbd5e1;padding:.6rem .8rem;border-radius:6px;font-size:.78rem;color:#475569;}
        .med-databanks-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:.85rem;margin-top:1rem;}
        .med-bank-card{background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:.85rem;box-shadow:0 1px 3px rgba(0,0,0,.03);}
        .med-bank-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:.5rem;font-size:.8rem;font-weight:700;color:#1e293b;}
        .med-bank-item{padding:.5rem;background:#f8fafc;border-radius:5px;margin-bottom:.4rem;font-size:.76rem;border-left:3px solid #3b82f6;}
        .med-bank-link{color:#2563eb;text-decoration:none;font-weight:600;display:inline-flex;align-items:center;gap:3px;margin-top:3px;}
        .med-bank-link:hover{text-decoration:underline;}

        """

if css_marker in html:
    html = html.replace(css_marker, new_styles + css_marker, 1)
    print("CSS for disease differential and databanks added!")
else:
    print("Warning: css_marker not found")

# 2. Extract properties in ReportTab
target_props = "const plausibilityFlags = report.plausibility_flags || [];"
new_props = """const plausibilityFlags = report.plausibility_flags || [];
    const diseaseDiff = report.disease_differential || [];
    const multibankEv = report.multibank_evidence || {};"""

if target_props in html:
    html = html.replace(target_props, new_props, 1)
    print("ReportTab unpack updated with diseaseDiff and multibankEv!")

# 3. Add Medical Disease Detection & Differential Diagnostic Section in the Report
target_loc = """                        {/* Figure 1: Annotated Scan Field */}
                        <div className="p10-fig-container">"""

medical_section = """                        {/* ── CLINICAL DISEASE DETECTION & DIFFERENTIAL SPECTRUM ── */}
                        <div className="med-diff-card">
                            <div className="med-diff-title">
                                <span><i className="fas fa-stethoscope" style={{color:'#2563eb',marginRight:'8px'}}></i>Differential Diagnostic Spectrum &amp; Disease Matching</span>
                                <span style={{fontSize:'.75rem',fontWeight:600,color:'#64748b',backgroundColor:'#f1f5f9',padding:'3px 8px',borderRadius:'12px'}}>
                                    WHO 5th Ed. / ICC 2022 Guidelines Grounded
                                </span>
                            </div>
                            <p style={{fontSize:'.84rem',color:'#475569',marginBottom:'1rem',lineHeight:1.55}}>
                                Automated screening evaluates morphological criteria, cellular distribution fractions, and blast percentages to generate an evidence-grounded differential diagnosis spectrum. Definitive clinical diagnosis requires integration with ancillary cytogenetic, flow cytometric, and molecular studies.
                            </p>

                            {diseaseDiff.length === 0 ? (
                                <div style={{padding:'1rem',textAlign:'center',color:'#94a3b8',fontSize:'.82rem'}}>
                                    Analyzing clinical diagnostic spectrum against WHO/ICC criteria...
                                </div>
                            ) : (
                                diseaseDiff.map((d, i) => {
                                    const lvl = (d.match_level || '').toLowerCase();
                                    const cls = lvl.includes('high') ? 'high' : (lvl.includes('moderate') ? 'moderate' : (lvl.includes('skew') ? 'skew' : 'baseline'));
                                    return (
                                        <div key={i} className={`med-cand-box ${cls}`}>
                                            <div className="med-cand-header">
                                                <span className="med-cand-name">{d.disease_name}</span>
                                                <span className={`med-badge ${cls}`}>{d.match_level}</span>
                                            </div>
                                            <div className="med-cand-criteria">
                                                <strong>Observed Cytomorphologic Findings:</strong>
                                                <ul>
                                                    {(d.matched_criteria || []).map((c, ci) => <li key={ci}>{c}</li>)}
                                                </ul>
                                            </div>
                                            {d.relevant_biomarkers && d.relevant_biomarkers.length > 0 && (
                                                <div className="med-biomarkers">
                                                    <span style={{fontSize:'.72rem',fontWeight:700,color:'#475569',marginRight:'4px'}}>Driver Biomarkers &amp; Target Genes:</span>
                                                    {d.relevant_biomarkers.map((b, bi) => (
                                                        <span key={bi} className="med-bio-tag">{b}</span>
                                                    ))}
                                                </div>
                                            )}
                                            {d.confirmatory_tests && d.confirmatory_tests.length > 0 && (
                                                <div className="med-tests-box">
                                                    <strong style={{color:'#1e293b'}}><i className="fas fa-vial" style={{marginRight:'5px',color:'#2563eb'}}></i>Required Confirmatory Workup:</strong>
                                                    <ul style={{margin:'4px 0 0 1.2rem',padding:0}}>
                                                        {d.confirmatory_tests.map((t, ti) => <li key={ti}>{t}</li>)}
                                                    </ul>
                                                </div>
                                            )}
                                        </div>
                                    );
                                })
                            )}

                            {/* ── MULTI-DATABANK MEDICAL EVIDENCE INTEGRATION ── */}
                            <div style={{marginTop:'1.25rem',borderTop:'1px solid #e2e8f0',paddingTop:'1rem'}}>
                                <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:'.5rem'}}>
                                    <strong style={{fontSize:'.9rem',color:'#1e293b'}}>
                                        <i className="fas fa-database" style={{color:'#0284c7',marginRight:'6px'}}></i>Authoritative Medical Data Banks Evidence
                                    </strong>
                                    <span style={{fontSize:'.7rem',color:'#64748b'}}>Federated Queries: PubMed Central · Europe PMC · ClinVar · ClinicalTrials.gov</span>
                                </div>

                                <div className="med-databanks-grid">
                                    {/* NCBI PubMed Central */}
                                    <div className="med-bank-card">
                                        <div className="med-bank-header">
                                            <span><i className="fas fa-book-medical" style={{color:'#2563eb',marginRight:'5px'}}></i>NCBI PubMed Central</span>
                                            <span style={{fontSize:'.68rem',color:'#64748b'}}>{(multibankEv.pmc_guidelines || []).length} Records</span>
                                        </div>
                                        {(multibankEv.pmc_guidelines || []).map((p, pi) => (
                                            <div key={pi} className="med-bank-item">
                                                <div style={{fontWeight:600,color:'#1e293b'}}>{p.title}</div>
                                                <div style={{color:'#64748b',fontSize:'.7rem'}}>{p.source} ({p.pubdate})</div>
                                                <a href={p.url} target="_blank" rel="noopener noreferrer" className="med-bank-link">
                                                    Open Article <i className="fas fa-external-link-alt" style={{fontSize:'.65rem'}}></i>
                                                </a>
                                            </div>
                                        ))}
                                    </div>

                                    {/* Europe PMC */}
                                    <div className="med-bank-card">
                                        <div className="med-bank-header">
                                            <span><i className="fas fa-microscope" style={{color:'#7c3aed',marginRight:'5px'}}></i>Europe PMC Guidelines</span>
                                            <span style={{fontSize:'.68rem',color:'#64748b'}}>{(multibankEv.europe_pmc_papers || []).length} Papers</span>
                                        </div>
                                        {(multibankEv.europe_pmc_papers || []).map((e, ei) => (
                                            <div key={ei} className="med-bank-item" style={{borderLeftColor:'#7c3aed'}}>
                                                <div style={{fontWeight:600,color:'#1e293b'}}>{e.title}</div>
                                                <div style={{color:'#64748b',fontSize:'.7rem'}}>{e.source} ({e.pubdate})</div>
                                                <a href={e.url} target="_blank" rel="noopener noreferrer" className="med-bank-link" style={{color:'#7c3aed'}}>
                                                    Open Europe PMC <i className="fas fa-external-link-alt" style={{fontSize:'.65rem'}}></i>
                                                </a>
                                            </div>
                                        ))}
                                    </div>

                                    {/* NCBI ClinVar */}
                                    <div className="med-bank-card">
                                        <div className="med-bank-header">
                                            <span><i className="fas fa-dna" style={{color:'#059669',marginRight:'5px'}}></i>NCBI ClinVar Genetics</span>
                                            <span style={{fontSize:'.68rem',color:'#64748b'}}>{(multibankEv.clinvar_variants || []).length} Genes</span>
                                        </div>
                                        {(multibankEv.clinvar_variants || []).map((v, vi) => (
                                            <div key={vi} className="med-bank-item" style={{borderLeftColor:'#059669'}}>
                                                <div style={{fontWeight:700,color:'#0f766e'}}>Gene: {v.gene} &bull; {v.bank_id}</div>
                                                <div style={{color:'#334155'}}>{v.title}</div>
                                                <a href={v.url} target="_blank" rel="noopener noreferrer" className="med-bank-link" style={{color:'#059669'}}>
                                                    Inspect ClinVar Variant <i className="fas fa-external-link-alt" style={{fontSize:'.65rem'}}></i>
                                                </a>
                                            </div>
                                        ))}
                                    </div>

                                    {/* ClinicalTrials.gov */}
                                    <div className="med-bank-card">
                                        <div className="med-bank-header">
                                            <span><i className="fas fa-clipboard-check" style={{color:'#d97706',marginRight:'5px'}}></i>ClinicalTrials.gov Protocols</span>
                                            <span style={{fontSize:'.68rem',color:'#64748b'}}>{(multibankEv.clinical_trials || []).length} Protocols</span>
                                        </div>
                                        {(multibankEv.clinical_trials || []).map((t, ti) => (
                                            <div key={ti} className="med-bank-item" style={{borderLeftColor:'#d97706'}}>
                                                <div style={{fontWeight:700,color:'#b45309'}}>{t.bank_id} &bull; Status: {t.status}</div>
                                                <div style={{color:'#334155'}}>{t.title}</div>
                                                <a href={t.url} target="_blank" rel="noopener noreferrer" className="med-bank-link" style={{color:'#d97706'}}>
                                                    View Clinical Protocol <i className="fas fa-external-link-alt" style={{fontSize:'.65rem'}}></i>
                                                </a>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>

"""

if target_loc in html:
    html = html.replace(target_loc, medical_section + target_loc, 1)
    print("Medical Disease Detection and Multi-Databank section injected successfully!")
else:
    print("Warning: target_loc for medical section not found")

with open("app/static/index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Updated index.html saved!")

