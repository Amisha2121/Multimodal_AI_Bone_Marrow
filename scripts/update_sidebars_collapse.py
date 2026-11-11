import re

with open("app/static/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# 1. Update CSS
old_css = """        /* ── Sidebar ── */
        .sidebar{width:200px;min-width:200px;background:#1a202c;color:#fff;display:flex;flex-direction:column;flex-shrink:0;}
        .sidebar-header{padding:1.25rem 1rem;border-bottom:1px solid rgba(255,255,255,.1);}
        .sidebar-header h1{font-size:1rem;font-weight:700;margin-bottom:.2rem;}
        .sidebar-header p{font-size:.75rem;color:#a0aec0;}
        .sidebar-nav{flex:1;padding:.5rem 0;}
        .nav-item{padding:.65rem 1rem;cursor:pointer;display:flex;align-items:center;gap:.6rem;font-size:.85rem;border-left:3px solid transparent;transition:all .15s;}
        .nav-item:hover{background:rgba(255,255,255,.06);}
        .nav-item.active{background:rgba(59,130,246,.18);border-left-color:#3b82f6;color:#fff;}
        .nav-item i{width:16px;text-align:center;font-size:.85rem;}

        /* ── Main ── */
        .main-content{flex:1;display:flex;flex-direction:column;overflow:hidden;}
        .topbar{background:#fff;border-bottom:1px solid #e2e8f0;padding:.85rem 1.5rem;}
        .topbar h2{font-size:1.3rem;font-weight:600;color:#1a202c;}
        .content-area{flex:1;display:flex;overflow:hidden;}
        .content-main{flex:1;padding:1.5rem;overflow-y:auto;}
        .content-main.chat-view{padding:0;}
        .content-sidebar{width:280px;min-width:280px;background:#fff;border-left:1px solid #e2e8f0;overflow-y:auto;}"""

new_css = """        /* ── Sidebar ── */
        .sidebar{width:200px;min-width:200px;background:#1a202c;color:#fff;display:flex;flex-direction:column;flex-shrink:0;transition:width .22s cubic-bezier(0.4,0,0.2,1),min-width .22s cubic-bezier(0.4,0,0.2,1),opacity .18s ease;position:relative;overflow-x:hidden;}
        .sidebar.collapsed{width:0 !important;min-width:0 !important;overflow:hidden !important;opacity:0 !important;pointer-events:none;}
        .sidebar-header{padding:1.25rem 1rem;border-bottom:1px solid rgba(255,255,255,.1);min-width:200px;}
        .sidebar-header h1{font-size:1rem;font-weight:700;margin-bottom:.2rem;}
        .sidebar-header p{font-size:.75rem;color:#a0aec0;}
        .sidebar-nav{flex:1;padding:.5rem 0;min-width:200px;}
        .nav-item{padding:.65rem 1rem;cursor:pointer;display:flex;align-items:center;gap:.6rem;font-size:.85rem;border-left:3px solid transparent;transition:all .15s;white-space:nowrap;}
        .nav-item:hover{background:rgba(255,255,255,.06);}
        .nav-item.active{background:rgba(59,130,246,.18);border-left-color:#3b82f6;color:#fff;}
        .nav-item i{width:16px;text-align:center;font-size:.85rem;}

        /* ── Main ── */
        .main-content{flex:1;display:flex;flex-direction:column;overflow:hidden;}
        .topbar{background:#fff;border-bottom:1px solid #e2e8f0;padding:.85rem 1.5rem;}
        .topbar h2{font-size:1.3rem;font-weight:600;color:#1a202c;}
        .content-area{flex:1;display:flex;overflow:hidden;position:relative;}
        .content-main{flex:1;padding:1.5rem;overflow-y:auto;}
        .content-main.chat-view{padding:0;}
        .content-sidebar{width:280px;min-width:280px;background:#fff;border-left:1px solid #e2e8f0;overflow-y:auto;transition:width .22s cubic-bezier(0.4,0,0.2,1),min-width .22s cubic-bezier(0.4,0,0.2,1),border .18s ease;position:relative;}
        .content-sidebar.collapsed{width:0 !important;min-width:0 !important;border-left:none !important;overflow:hidden !important;pointer-events:none;}

        .sidebar-arrow-btn:hover{background:rgba(255,255,255,.2) !important;color:#fff !important;}
        .topbar-sidebar-toggle-btn:hover{background:#e2e8f0 !important;color:#1e293b !important;}
        .history-arrow-btn:hover{background:#e2e8f0 !important;color:#1e293b !important;}
        .sidebar-edge-tab{transition:all .18s ease;user-select:none;}
        .sidebar-edge-tab:hover{filter:brightness(1.08);transform:translateY(-50%) scale(1.05) !important;}"""

if old_css in html:
    html = html.replace(old_css, new_css, 1)
    print("CSS replaced successfully!")
else:
    print("ERROR: old_css not found in html!")

# 2. Update App, Sidebar, Topbar components
old_app_block = """/* ═══════════════════════════════════════════════════════════ App */
function App() {
    const [tab, setTab]           = useState('upload');
    const [analyses, setAnalyses] = useState([]);
    const [selId, setSelId]       = useState(null);
    const [uploading, setUploading] = useState(false);
    const [showAgentFeed, setShowAgentFeed] = useState(false);
    const [agentEventCount, setAgentEventCount] = useState(0);

    useEffect(() => { fetchList(); }, []);

    async function fetchList() {
        try {
            let d;
            try {
                d = await apiFetch('/api/history');
            } catch (err) {
                d = await apiFetch('/api/analyses');
            }
            const list = d.analyses || d.cases || [];
            setAnalyses(list);
            // auto-select most-recent
            if (list.length > 0 && !selId) setSelId(list[list.length - 1].id);
        } catch {}
    }

    async function handleUpload(files) {
        setUploading(true);
        const fd = new FormData();
        Array.from(files).forEach(f => fd.append('files', f));
        try {
            const d = await apiFetch('/api/upload', { method: 'POST', body: fd });
            if (d.success) {
                await fetchList();
                setSelId(d.analysis_id);
                setTab('upload');
            }
        } catch (e) {
            alert('Upload failed: ' + e.message);
        } finally {
            setUploading(false);
        }
    }

    async function handleDelete(id, e) {
        e.stopPropagation();
        if (!confirm('Delete this analysis?')) return;
        try {
            await apiFetch(`/api/analysis/${id}`, { method: 'DELETE' });
            await fetchList();
            if (selId === id) setSelId(null);
        } catch {}
    }

    return (
        <div className="app">
            <Sidebar tab={tab} setTab={setTab} />
            <div className="main-content">
                <Topbar tab={tab} showAgentFeed={showAgentFeed} setShowAgentFeed={setShowAgentFeed} agentEventCount={agentEventCount} />
                <div className="content-area">
                    <div className={`content-main ${tab === 'chat' ? 'chat-view' : ''}`}>
                        {tab === 'upload'  && <UploadTab  selId={selId} onUpload={handleUpload} uploading={uploading} />}
                        {tab === 'chat'    && <ChatTab    selId={selId} />}
                        {tab === 'report'  && <ReportTab  selId={selId} />}
                        {tab === 'cells'   && <CellsTab   selId={selId} onRefreshHistory={fetchList} />}
                    </div>
                    <div className="content-sidebar">
                        <History analyses={analyses} selId={selId} onSelect={setSelId} onDelete={handleDelete} />
                    </div>
                </div>
                <AgentActivityPanel selId={selId} open={showAgentFeed} onClose={() => setShowAgentFeed(false)} onCountUpdate={setAgentEventCount} />
            </div>
        </div>
    );
}

/* ── Sidebar ── */
function Sidebar({ tab, setTab }) {
    const items = [
        { id: 'upload', icon: 'fa-upload',      label: 'Upload & Analyze' },
        { id: 'chat',   icon: 'fa-comments',     label: 'Ask Questions'   },
        { id: 'report', icon: 'fa-file-medical', label: 'RAG Report'      },
        { id: 'cells',  icon: 'fa-microscope',   label: 'Cell Explorer'   },
    ];
    return (
        <div className="sidebar">
            <div className="sidebar-header">
                <h1>🔬 Bone Marrow Analysis</h1>
                <p>Cytomorphology Assistant</p>
            </div>
            <div className="sidebar-nav">
                {items.map(it => (
                    <div key={it.id} className={`nav-item ${tab === it.id ? 'active' : ''}`} onClick={() => setTab(it.id)}>
                        <i className={`fas ${it.icon}`}></i> {it.label}
                    </div>
                ))}
            </div>
        </div>
    );
}

/* ── Topbar ── */
function Topbar({ tab, showAgentFeed, setShowAgentFeed, agentEventCount }) {
    const titles = { upload: 'Upload & Analyze Images', chat: 'Ask Questions', report: 'Comprehensive RAG Report', cells: 'Cell Explorer' };
    return (
        <div className="topbar" style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
            <h2>{titles[tab]}</h2>
            <button
                type="button"
                onClick={() => setShowAgentFeed(prev => !prev)}
                style={{
                    display:'inline-flex',
                    alignItems:'center',
                    gap:'6px',
                    backgroundColor: showAgentFeed ? '#eff6ff' : '#ffffff',
                    border: '1px solid ' + (showAgentFeed ? '#3b82f6' : '#cbd5e1'),
                    color: showAgentFeed ? '#2563eb' : '#475569',
                    padding: '.35rem .75rem',
                    borderRadius: '6px',
                    fontSize: '.82rem',
                    fontWeight: 600,
                    cursor: 'pointer',
                    boxShadow: '0 1px 2px rgba(0,0,0,.05)',
                }}
                title="Toggle Agent Activity Live Feed"
            >
                <i className="fas fa-robot" style={{color: showAgentFeed ? '#2563eb' : '#3b82f6'}}></i>
                <span>Agent Activity</span>
                {agentEventCount > 0 && (
                    <span style={{
                        backgroundColor: '#3b82f6',
                        color: '#fff',
                        borderRadius: '10px',
                        fontSize: '.68rem',
                        padding: '1px 6px',
                        marginLeft: '3px',
                        fontWeight: 700,
                    }}>
                        {agentEventCount}
                    </span>
                )}
            </button>
        </div>
    );
}"""

new_app_block = """/* ═══════════════════════════════════════════════════════════ App */
function App() {
    const [tab, setTab]           = useState('upload');
    const [analyses, setAnalyses] = useState([]);
    const [selId, setSelId]       = useState(null);
    const [uploading, setUploading] = useState(false);
    const [showAgentFeed, setShowAgentFeed] = useState(false);
    const [agentEventCount, setAgentEventCount] = useState(0);

    // Collapsible sidebars state with persistence
    const [leftSidebarOpen, setLeftSidebarOpen] = useState(() => {
        try { return localStorage.getItem('bm_left_sidebar') !== 'false'; } catch (e) { return true; }
    });
    const [rightSidebarOpen, setRightSidebarOpen] = useState(() => {
        try { return localStorage.getItem('bm_right_sidebar') !== 'false'; } catch (e) { return true; }
    });

    useEffect(() => {
        try { localStorage.setItem('bm_left_sidebar', leftSidebarOpen); } catch (e) {}
    }, [leftSidebarOpen]);

    useEffect(() => {
        try { localStorage.setItem('bm_right_sidebar', rightSidebarOpen); } catch (e) {}
    }, [rightSidebarOpen]);

    useEffect(() => { fetchList(); }, []);

    async function fetchList() {
        try {
            let d;
            try {
                d = await apiFetch('/api/history');
            } catch (err) {
                d = await apiFetch('/api/analyses');
            }
            const list = d.analyses || d.cases || [];
            setAnalyses(list);
            // auto-select most-recent
            if (list.length > 0 && !selId) setSelId(list[list.length - 1].id);
        } catch {}
    }

    async function handleUpload(files) {
        setUploading(true);
        const fd = new FormData();
        Array.from(files).forEach(f => fd.append('files', f));
        try {
            const d = await apiFetch('/api/upload', { method: 'POST', body: fd });
            if (d.success) {
                await fetchList();
                setSelId(d.analysis_id);
                setTab('upload');
            }
        } catch (e) {
            alert('Upload failed: ' + e.message);
        } finally {
            setUploading(false);
        }
    }

    async function handleDelete(id, e) {
        e.stopPropagation();
        if (!confirm('Delete this analysis?')) return;
        try {
            await apiFetch(`/api/analysis/${id}`, { method: 'DELETE' });
            await fetchList();
            if (selId === id) setSelId(null);
        } catch {}
    }

    return (
        <div className="app">
            <Sidebar
                tab={tab}
                setTab={setTab}
                collapsed={!leftSidebarOpen}
                onToggle={() => setLeftSidebarOpen(prev => !prev)}
            />
            <div className="main-content">
                <Topbar
                    tab={tab}
                    showAgentFeed={showAgentFeed}
                    setShowAgentFeed={setShowAgentFeed}
                    agentEventCount={agentEventCount}
                    leftSidebarOpen={leftSidebarOpen}
                    setLeftSidebarOpen={setLeftSidebarOpen}
                    rightSidebarOpen={rightSidebarOpen}
                    setRightSidebarOpen={setRightSidebarOpen}
                />
                <div className="content-area">
                    {/* Left edge tab to reopen left sidebar when closed */}
                    {!leftSidebarOpen && (
                        <button
                            className="sidebar-edge-tab left"
                            onClick={() => setLeftSidebarOpen(true)}
                            title="Open navigation sidebar (click to expand)"
                            style={{
                                position: 'absolute',
                                left: 0,
                                top: '50%',
                                transform: 'translateY(-50%)',
                                zIndex: 30,
                                background: '#1a202c',
                                border: '1px solid rgba(255,255,255,0.25)',
                                borderLeft: 'none',
                                borderRadius: '0 8px 8px 0',
                                padding: '12px 6px',
                                cursor: 'pointer',
                                boxShadow: '2px 0 8px rgba(0,0,0,0.2)',
                                color: '#ffffff',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                gap: '5px'
                            }}
                        >
                            <i className="fas fa-chevron-right" style={{fontSize:'.82rem'}}></i>
                            <span style={{writingMode:'vertical-rl',fontSize:'.7rem',fontWeight:600,letterSpacing:'.5px',color:'#cbd5e1'}}>Nav</span>
                        </button>
                    )}

                    <div className={`content-main ${tab === 'chat' ? 'chat-view' : ''}`}>
                        {tab === 'upload'  && <UploadTab  selId={selId} onUpload={handleUpload} uploading={uploading} />}
                        {tab === 'chat'    && <ChatTab    selId={selId} />}
                        {tab === 'report'  && <ReportTab  selId={selId} />}
                        {tab === 'cells'   && <CellsTab   selId={selId} onRefreshHistory={fetchList} />}
                    </div>

                    <div className={`content-sidebar ${!rightSidebarOpen ? 'collapsed' : ''}`}>
                        <History
                            analyses={analyses}
                            selId={selId}
                            onSelect={setSelId}
                            onDelete={handleDelete}
                            onToggle={() => setRightSidebarOpen(false)}
                        />
                    </div>

                    {/* Right edge tab to reopen right sidebar when closed */}
                    {!rightSidebarOpen && (
                        <button
                            className="sidebar-edge-tab right"
                            onClick={() => setRightSidebarOpen(true)}
                            title="Open Analysis History (click to expand)"
                            style={{
                                position: 'absolute',
                                right: 0,
                                top: '50%',
                                transform: 'translateY(-50%)',
                                zIndex: 30,
                                background: '#ffffff',
                                border: '1px solid #cbd5e1',
                                borderRight: 'none',
                                borderRadius: '8px 0 0 8px',
                                padding: '12px 6px',
                                cursor: 'pointer',
                                boxShadow: '-2px 0 8px rgba(0,0,0,0.1)',
                                color: '#2563eb',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                gap: '5px'
                            }}
                        >
                            <i className="fas fa-chevron-left" style={{fontSize:'.82rem'}}></i>
                            <span style={{writingMode:'vertical-rl',fontSize:'.7rem',fontWeight:600,letterSpacing:'.5px',color:'#475569'}}>History</span>
                        </button>
                    )}
                </div>
                <AgentActivityPanel selId={selId} open={showAgentFeed} onClose={() => setShowAgentFeed(false)} onCountUpdate={setAgentEventCount} />
            </div>
        </div>
    );
}

/* ── Sidebar ── */
function Sidebar({ tab, setTab, collapsed, onToggle }) {
    const items = [
        { id: 'upload', icon: 'fa-upload',      label: 'Upload & Analyze' },
        { id: 'chat',   icon: 'fa-comments',     label: 'Ask Questions'   },
        { id: 'report', icon: 'fa-file-medical', label: 'RAG Report'      },
        { id: 'cells',  icon: 'fa-microscope',   label: 'Cell Explorer'   },
    ];
    return (
        <div className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
            <div className="sidebar-header" style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start'}}>
                <div style={{overflow:'hidden'}}>
                    <h1>🔬 Bone Marrow Analysis</h1>
                    <p>Cytomorphology Assistant</p>
                </div>
                {onToggle && (
                    <button
                        type="button"
                        onClick={onToggle}
                        className="sidebar-arrow-btn"
                        title="Close navigation sidebar"
                        style={{
                            background: 'rgba(255,255,255,0.08)',
                            border: '1px solid rgba(255,255,255,0.15)',
                            borderRadius: '6px',
                            color: '#cbd5e1',
                            width: '26px',
                            height: '26px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer',
                            padding: 0,
                            marginLeft: '4px',
                            flexShrink: 0,
                            transition: 'all .15s'
                        }}
                    >
                        <i className="fas fa-chevron-left" style={{fontSize:'.75rem'}}></i>
                    </button>
                )}
            </div>
            <div className="sidebar-nav">
                {items.map(it => (
                    <div key={it.id} className={`nav-item ${tab === it.id ? 'active' : ''}`} onClick={() => setTab(it.id)}>
                        <i className={`fas ${it.icon}`}></i> {it.label}
                    </div>
                ))}
            </div>
        </div>
    );
}

/* ── Topbar ── */
function Topbar({
    tab,
    showAgentFeed,
    setShowAgentFeed,
    agentEventCount,
    leftSidebarOpen,
    setLeftSidebarOpen,
    rightSidebarOpen,
    setRightSidebarOpen
}) {
    const titles = { upload: 'Upload & Analyze Images', chat: 'Ask Questions', report: 'Comprehensive RAG Report', cells: 'Cell Explorer' };
    return (
        <div className="topbar" style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
            <div style={{display:'flex',alignItems:'center',gap:'.75rem'}}>
                <button
                    type="button"
                    onClick={() => setLeftSidebarOpen(prev => !prev)}
                    className="topbar-sidebar-toggle-btn"
                    title={leftSidebarOpen ? "Close navigation sidebar" : "Open navigation sidebar"}
                    style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        width: '32px',
                        height: '32px',
                        backgroundColor: leftSidebarOpen ? '#ffffff' : '#eff6ff',
                        border: '1px solid ' + (leftSidebarOpen ? '#cbd5e1' : '#3b82f6'),
                        color: leftSidebarOpen ? '#475569' : '#2563eb',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        boxShadow: '0 1px 2px rgba(0,0,0,.04)',
                        transition: 'all .15s'
                    }}
                >
                    <i className={`fas ${leftSidebarOpen ? 'fa-chevron-left' : 'fa-chevron-right'}`} style={{fontSize:'.82rem'}}></i>
                </button>
                <h2 style={{margin:0}}>{titles[tab]}</h2>
            </div>

            <div style={{display:'flex',alignItems:'center',gap:'.5rem'}}>
                <button
                    type="button"
                    onClick={() => setRightSidebarOpen(prev => !prev)}
                    style={{
                        display:'inline-flex',
                        alignItems:'center',
                        gap:'6px',
                        backgroundColor: rightSidebarOpen ? '#ffffff' : '#eff6ff',
                        border: '1px solid ' + (rightSidebarOpen ? '#cbd5e1' : '#3b82f6'),
                        color: rightSidebarOpen ? '#475569' : '#2563eb',
                        padding: '.35rem .75rem',
                        borderRadius: '6px',
                        fontSize: '.82rem',
                        fontWeight: 600,
                        cursor: 'pointer',
                        boxShadow: '0 1px 2px rgba(0,0,0,.05)',
                        transition: 'all .15s'
                    }}
                    title={rightSidebarOpen ? "Close Analysis History" : "Open Analysis History"}
                >
                    <i className={`fas ${rightSidebarOpen ? 'fa-chevron-right' : 'fa-chevron-left'}`} style={{fontSize:'.8rem'}}></i>
                    <span>{rightSidebarOpen ? 'Close History' : 'Open History'}</span>
                </button>

                <button
                    type="button"
                    onClick={() => setShowAgentFeed(prev => !prev)}
                    style={{
                        display:'inline-flex',
                        alignItems:'center',
                        gap:'6px',
                        backgroundColor: showAgentFeed ? '#eff6ff' : '#ffffff',
                        border: '1px solid ' + (showAgentFeed ? '#3b82f6' : '#cbd5e1'),
                        color: showAgentFeed ? '#2563eb' : '#475569',
                        padding: '.35rem .75rem',
                        borderRadius: '6px',
                        fontSize: '.82rem',
                        fontWeight: 600,
                        cursor: 'pointer',
                        boxShadow: '0 1px 2px rgba(0,0,0,.05)',
                    }}
                    title="Toggle Agent Activity Live Feed"
                >
                    <i className="fas fa-robot" style={{color: showAgentFeed ? '#2563eb' : '#3b82f6'}}></i>
                    <span>Agent Activity</span>
                    {agentEventCount > 0 && (
                        <span style={{
                            backgroundColor: '#3b82f6',
                            color: '#fff',
                            borderRadius: '10px',
                            fontSize: '.68rem',
                            padding: '1px 6px',
                            marginLeft: '3px',
                            fontWeight: 700,
                        }}>
                            {agentEventCount}
                        </span>
                    )}
                </button>
            </div>
        </div>
    );
}"""

if old_app_block in html:
    html = html.replace(old_app_block, new_app_block, 1)
    print("App, Sidebar, Topbar replaced successfully!")
else:
    print("ERROR: old_app_block not found in html!")

# 3. Update History component
old_history = """function History({ analyses, selId, onSelect, onDelete }) {
    // newest first
    const sorted = [...analyses].reverse();
    return (
        <div>
            <div style={{padding:'.85rem 1rem',fontWeight:600,fontSize:'.9rem',borderBottom:'1px solid #e2e8f0'}}>Analysis History</div>
            {sorted.length === 0"""

new_history = """function History({ analyses, selId, onSelect, onDelete, onToggle }) {
    // newest first
    const sorted = [...analyses].reverse();
    return (
        <div>
            <div style={{
                padding:'.85rem 1rem',
                fontWeight:600,
                fontSize:'.9rem',
                borderBottom:'1px solid #e2e8f0',
                display:'flex',
                justifyContent:'space-between',
                alignItems:'center'
            }}>
                <span>Analysis History</span>
                {onToggle && (
                    <button
                        type="button"
                        onClick={onToggle}
                        className="history-arrow-btn"
                        title="Close Analysis History"
                        style={{
                            background: '#f8fafc',
                            border: '1px solid #cbd5e1',
                            borderRadius: '6px',
                            color: '#475569',
                            width: '26px',
                            height: '26px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer',
                            padding: 0,
                            transition: 'all .15s'
                        }}
                    >
                        <i className="fas fa-chevron-right" style={{fontSize:'.75rem'}}></i>
                    </button>
                )}
            </div>
            {sorted.length === 0"""

if old_history in html:
    html = html.replace(old_history, new_history, 1)
    print("History component replaced successfully!")
else:
    print("ERROR: old_history not found in html!")

with open("app/static/index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Saved updated index.html!")

