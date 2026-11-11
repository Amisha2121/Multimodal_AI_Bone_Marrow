import re

with open('app/static/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Add CSS for Lightbox, Print/PDF, Diagram Cards, and Literature Chips
extra_css = '''
        /* ── Phase 10 Step 9 – Polish, Diagrams, Figures, Lightbox & PDF ── */
        .p10-diagram-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.5rem;}
        @media (max-width:900px){.p10-diagram-grid{grid-template-columns:1fr;}}
        .p10-diagram-card{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:1rem;box-shadow:0 1px 3px rgba(0,0,0,.04);}
        .p10-diagram-head{font-size:.82rem;font-weight:700;color:#1e293b;margin-bottom:.5rem;display:flex;justify-content:space-between;align-items:center;}
        .p10-diagram-img{width:100%;height:auto;border-radius:4px;cursor:pointer;transition:transform .15s;}
        .p10-diagram-img:hover{transform:scale(1.01);}

        /* Literature Reference Chips */
        .p10-lit-chip{display:inline-flex;align-items:center;background:#f8fafc;border:1px solid #cbd5e1;color:#475569;border-radius:12px;padding:1px 7px;font-size:.68rem;font-weight:600;margin-left:4px;cursor:pointer;vertical-align:middle;transition:all .15s;}
        .p10-lit-chip:hover{background:#ede9fe;border-color:#a78bfa;color:#6b21a8;}
        
        /* Lightbox modal */
        .p10-lightbox{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(15,23,42,.85);z-index:9999;display:flex;align-items:center;justify-content:center;padding:2rem;}
        .p10-lightbox-content{background:#fff;border-radius:8px;max-width:90vw;max-height:90vh;overflow:auto;padding:1.5rem;position:relative;box-shadow:0 20px 40px rgba(0,0,0,.35);}
        .p10-lightbox-img{max-width:100%;max-height:75vh;object-fit:contain;display:block;margin:0 auto;border-radius:4px;}
        .p10-lightbox-close{position:absolute;top:1rem;right:1rem;background:#f1f5f9;border:none;border-radius:50%;width:32px;height:32px;font-size:1.1rem;cursor:pointer;display:flex;align-items:center;justify-content:center;color:#475569;}
        .p10-lightbox-close:hover{background:#e2e8f0;color:#0f172a;}

        /* Literature popover */
        .p10-lit-popover{position:fixed;z-index:9100;background:#1e293b;color:#f8fafc;border-radius:8px;padding:.85rem 1rem;font-size:.78rem;max-width:340px;box-shadow:0 10px 25px rgba(0,0,0,.3);line-height:1.5;}
        .p10-lit-pop-head{font-size:.7rem;font-weight:700;color:#c084fc;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.3rem;}
        .p10-lit-pop-src{font-size:.72rem;color:#94a3b8;margin-bottom:.4rem;font-style:italic;}

        /* Print / PDF Styles */
        .p10-print-btn{padding:.4rem .85rem;background:#fff;border:1px solid #cbd5e0;border-radius:6px;font-size:.78rem;font-weight:600;color:#334155;cursor:pointer;display:inline-flex;align-items:center;gap:.35rem;transition:all .15s;}
        .p10-print-btn:hover{background:#f1f5f9;border-color:#94a3b8;}
        .p10-disclaimer-footer{background:#f8fafc;border-top:1px solid #e2e8f0;padding:1.25rem;margin-top:2rem;font-size:.72rem;color:#64748b;line-height:1.6;border-radius:6px;}

        @media print{
          .sidebar, .topbar, .content-sidebar, .p9-toolbar, .p9-toc, .regen-btn, .p10-print-btn, .p10-signoff-btn{display:none !important;}
          .app{display:block !important;height:auto !important;}
          .main-content, .content-area, .content-main{overflow:visible !important;padding:0 !important;display:block !important;}
          .p9-report-wrap{display:block !important;}
          .p9-sections{width:100% !important;}
          .p10-diagram-grid{grid-template-columns:1fr 1fr !important;page-break-inside:avoid;}
          .p10-fig-container{page-break-inside:avoid;}
          .p10-chapter{page-break-inside:avoid;}
        }
'''

if '/* ── Phase 10 Step 9 – Polish' not in html:
    end_style = html.find('</style>')
    html = html[:end_style] + extra_css + html[end_style:]
    with open('app/static/index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print('Injected Phase 10 Polish CSS successfully!')
else:
    print('CSS already present.')

