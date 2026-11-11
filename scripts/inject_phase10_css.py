import re

with open('app/static/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Add CSS for depth tiers and cards if not already present
p10_css = '''
        /* ── Phase 10 – Depth Tier Blocks ── */
        .p10-reading-time{font-size:.73rem;color:#64748b;display:inline-flex;align-items:center;gap:.3rem;margin-left:.5rem;background:#f1f5f9;padding:.2rem .55rem;border-radius:12px;}
        .p10-kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.75rem;margin-bottom:1.25rem;}
        .p10-kpi-card{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:.85rem;box-shadow:0 1px 3px rgba(0,0,0,.04);}
        .p10-kpi-label{font-size:.72rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.05em;}
        .p10-kpi-val{font-size:1.4rem;font-weight:700;color:#1e293b;margin-top:.2rem;}
        .p10-kpi-sub{font-size:.72rem;color:#94a3b8;margin-top:.15rem;}

        .p10-flags-card{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:1rem;margin-bottom:1.25rem;}
        .p10-flag-item{display:flex;gap:.6rem;align-items:flex-start;margin-top:.5rem;font-size:.82rem;color:#78350f;}
        .p10-flag-icon{color:#d97706;font-size:.9rem;margin-top:.15rem;}

        .p10-evidence-badge{display:inline-block;padding:.25rem .65rem;border-radius:20px;font-size:.72rem;font-weight:600;margin-left:.6rem;}
        .p10-evidence-badge.low{background:#fee2e2;color:#991b1b;border:1px solid #fca5a5;}
        .p10-evidence-badge.adequate{background:#dcfce7;color:#166534;border:1px solid #86efac;}

        .p10-signoff-bar{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:1rem;margin-top:1.5rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;}
        .p10-signoff-btn{padding:.5rem 1.1rem;background:#2563eb;color:#fff;border:none;border-radius:6px;font-weight:600;font-size:.82rem;cursor:pointer;display:inline-flex;align-items:center;gap:.4rem;}
        .p10-signoff-btn:hover{background:#1d4ed8;}

        .p10-fig-container{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:1rem;margin-bottom:1.25rem;}
        .p10-fig-img{width:100%;max-height:420px;object-fit:contain;border-radius:6px;background:#0f172a;}
        .p10-fig-caption{font-size:.76rem;color:#64748b;margin-top:.5rem;font-style:italic;}

        .p10-class-card{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:1rem;margin-bottom:1rem;transition:box-shadow .15s;}
        .p10-class-card:hover{box-shadow:0 2px 6px rgba(0,0,0,.06);}
        .p10-class-strip{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.6rem;}
        .p10-class-crop{width:64px;height:64px;object-fit:cover;border-radius:4px;border:1px solid #e2e8f0;}

        .p10-chapter{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:1.25rem;margin-bottom:1.25rem;box-shadow:0 1px 3px rgba(0,0,0,.04);}
        .p10-ch-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:.85rem;}
        .p10-ch-box{background:#f8fafc;border:1px solid #f1f5f9;border-radius:6px;padding:.75rem;}
        .p10-ch-title{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#64748b;margin-bottom:.3rem;}
        .p10-ch-text{font-size:.8rem;color:#334155;line-height:1.5;}
'''

if '/* ── Phase 10 – Depth Tier Blocks ── */' not in html:
    # insert before </style>
    style_end = html.find('</style>')
    html = html[:style_end] + p10_css + html[style_end:]
    print('Inserted Phase 10 CSS!')

with open('app/static/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Static index.html saved with Phase 10 styles.')

