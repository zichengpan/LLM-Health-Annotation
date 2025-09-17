import React, { useEffect, useMemo, useRef, useState } from 'react'
import { addEntity, addRelation, createDocument, deleteEntity, deleteRelation, exportCurrentDocument, exportJSONArr, getDocument, getLabelset, listDocuments, runSuggestions } from './api'
import type { Document, DocumentDetail, Entity, Relation } from './types'

const DEFAULT_ENTITY_TYPES = ['Disease','Medication','Symptom','Procedure']
const DEFAULT_REL_RULES = [
  {type:'treats', source:['Medication','Procedure'], target:['Disease']},
  {type:'has_symptom', source:['Disease'], target:['Symptom']},
  {type:'causes', source:['Disease','Medication'], target:['Disease','Symptom']},
  {type:'worsens', source:['Disease','Medication','Symptom'], target:['Disease','Symptom']},
  {type:'indicates', source:['Symptom'], target:['Disease']}
]

const COLORS: Record<string,string> = {
  Disease: 'etype-Disease',
  Medication: 'etype-Medication',
  Symptom: 'etype-Symptom',
  Procedure: 'etype-Procedure',
  Test: 'etype-Test',
  Patient: 'etype-Patient',
}

export default function App() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [doc, setDoc] = useState<DocumentDetail | null>(null)

  const [newTitle, setNewTitle] = useState('Untitled Note')
  const [newDocText, setNewDocText] = useState('')
  const [annotator, setAnnotator] = useState<string>('annotator_1')

  const [selectedType, setSelectedType] = useState(DEFAULT_ENTITY_TYPES[0])
  const [selStart, setSelStart] = useState(0)
  const [selEnd, setSelEnd] = useState(0)

  const [relType, setRelType] = useState('treats')
  const [sourceEnt, setSourceEnt] = useState<string>('')
  const [targetEnt, setTargetEnt] = useState<string>('')
  const [relWarning, setRelWarning] = useState<string>('')

  const [suggestions, setSuggestions] = useState<any>(null)
  const [suggestionCache, setSuggestionCache] = useState<Record<string, any>>({})

  const [labelset, setLabelset] = useState<any>(null)
  const [exportText, setExportText] = useState('')

  const docPanelRef = useRef<HTMLDivElement>(null)
  const textAreaRef = useRef<HTMLTextAreaElement>(null)

  const [autoLabelMatches, setAutoLabelMatches] = useState(false)
  const [panelHeight, setPanelHeight] = useState<number | null>(null)
  const [genBusy, setGenBusy] = useState(false)
  const [genError, setGenError] = useState<string>('')
  const [genInfo, setGenInfo] = useState<string>('')

  useEffect(() => {
    (async () => {
      setLabelset(await getLabelset().catch(()=>null))
      const docs = await listDocuments()
      setDocuments(docs)
      if (docs.length) setDoc(await getDocument(docs[0].id))
    })()
  }, [])

  useEffect(() => {
    setGenInfo('')
    setGenError('')
    if (!doc?.id) {
      setSuggestions(null)
      return
    }
    const cached = suggestionCache[doc.id]
    setSuggestions(cached || null)
  }, [doc?.id])

  useEffect(() => {
    const node = docPanelRef.current
    if (!node) return

    const updateHeight = () => setPanelHeight(node.offsetHeight)
    updateHeight()

    let observer: ResizeObserver | null = null
    if (typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(() => updateHeight())
      observer.observe(node)
    }

    window.addEventListener('resize', updateHeight)

    return () => {
      window.removeEventListener('resize', updateHeight)
      if (observer) observer.disconnect()
    }
  }, [])

  const onTextMouseUp = () => {
    const ta = textAreaRef.current
    if (!ta) return
    setSelStart(ta.selectionStart || 0)
    setSelEnd(ta.selectionEnd || 0)
  }

  const createNewDoc = async () => {
    if (!newDocText.trim()) return
    try {
      const created = await createDocument(newTitle.trim() || 'Untitled Note', newDocText.trim(), {source:'manual', language:'en'})
      setNewDocText('')
      setNewTitle('Untitled Note')
      clearDocumentState();
      const docs = await listDocuments()
      setDocuments(docs)
      if (created && created.id) {
        setDoc(await getDocument(created.id))
      }
    } catch (error) {
      console.error('Error creating document:', error)
    }
  }

  const addNewEntity = async () => {
    if (!doc) return
    if (selEnd <= selStart) return
    const selectedText = doc.raw_text.slice(selStart, selEnd)
    if (!selectedText.trim()) return

    if (autoLabelMatches) {
      const matches = findMatchingSpans(selectedText)
      const spans = matches.length ? matches : [{ start: selStart, end: selEnd }]
      const added = await addEntitiesForSpans(spans, selectedType)
      if (!added) {
        await addEntity({
          document_id: doc.id,
          start_offset: selStart,
          end_offset: selEnd,
          type: selectedType,
          annotator_id: annotator
        })
        setDoc(await getDocument(doc.id))
      }
    } else {
      await addEntity({ document_id: doc.id, start_offset: selStart, end_offset: selEnd, type: selectedType, annotator_id: annotator })
      setDoc(await getDocument(doc.id))
    }
  }

  const removeEntity = async (id: string) => {
    await deleteEntity(id)
    if (doc) setDoc(await getDocument(doc.id))
  }

  // De-dupe keys
  const entityKey = (e:any) => {
    const start = e.start_offset ?? e.start;
    const end = e.end_offset ?? e.end;
    const text = e.text || (doc ? doc.raw_text.slice(start, end) : '');
    return `${e.type}:${text.toLowerCase()}:${start}:${end}`;
  };

  const relationKey = (r:any) => {
    let sourceText = r.source_text || '';
    let targetText = r.target_text || '';

    // If relation uses entity IDs, convert to text
    if (!sourceText && r.source_entity_id && doc) {
      const sourceEntity = doc.entities.find(e => e.id === r.source_entity_id);
      sourceText = sourceEntity ? doc.raw_text.slice(sourceEntity.start_offset, sourceEntity.end_offset) : '';
    }

    if (!targetText && r.target_entity_id && doc) {
      const targetEntity = doc.entities.find(e => e.id === r.target_entity_id);
      targetText = targetEntity ? doc.raw_text.slice(targetEntity.start_offset, targetEntity.end_offset) : '';
    }

    return `${r.type}:${sourceText.toLowerCase()}=>${targetText.toLowerCase()}`;
  };

  const entityExists = (cand:any) =>
    (doc?.entities || []).some((e:any) => entityKey(e) === entityKey(cand));

  const relationExists = (cand:any) =>
    (doc?.relations || []).some((r:any) => relationKey(r) === relationKey(cand));

  const getEntityText = (entityId: string) => {
    if (!doc) return entityId;
    const entity = doc.entities.find(e => e.id === entityId);
    return entity ? doc.raw_text.slice(entity.start_offset, entity.end_offset) : entityId;
  };

  const getEntitySnippet = (start:number, end:number) => {
    if (!doc) return '';
    const raw = doc.raw_text.slice(start, end);
    if (raw.length <= 10) return raw;
    return `${raw.slice(0, 10)}…`;
  };

  const spansOverlap = (aStart:number, aEnd:number, bStart:number, bEnd:number) =>
    Math.max(aStart, bStart) < Math.min(aEnd, bEnd);

  const overlapsExistingEntity = (start:number, end:number) =>
    (doc?.entities || []).some(e => spansOverlap(start, end, e.start_offset, e.end_offset));

  const findMatchingSpans = (term: string) => {
    if (!doc) return [] as { start: number; end: number }[];
    const clean = term.trim();
    if (!clean) return [];

    const text = doc.raw_text;
    const haystack = text.toLowerCase();
    const needle = clean.toLowerCase();
    const enforceWordBoundary = !clean.includes(' ');

    const matches: { start: number; end: number }[] = [];
    let index = 0;
    while (index <= haystack.length) {
      const found = haystack.indexOf(needle, index);
      if (found === -1) break;
      const start = found;
      const end = found + clean.length;

      if (enforceWordBoundary) {
        const before = start === 0 ? '' : text[start - 1];
        const after = end >= text.length ? '' : text[end];
        const beforeOk = !before || !/\w/.test(before);
        const afterOk = !after || !/\w/.test(after);
        if (!beforeOk || !afterOk) {
          index = found + clean.length;
          continue;
        }
      }

      matches.push({ start, end });
      index = found + clean.length;
    }
    return matches;
  };

  const buildCandidate = (start:number, end:number, type:string) => ({
    type,
    start,
    end,
    text: doc ? doc.raw_text.slice(start, end) : '',
  });

  const addEntitiesForSpans = async (spans: { start: number; end: number }[], type: string) => {
    if (!doc || !spans.length) return 0;
    const unique = spans.filter(({ start, end }) => {
      if (overlapsExistingEntity(start, end)) return false;
      return !entityExists(buildCandidate(start, end, type));
    });

    if (!unique.length) return 0;

    let added = 0;
    for (const { start, end } of unique) {
      await addEntity({
        document_id: doc.id,
        start_offset: start,
        end_offset: end,
        type,
        annotator_id: annotator
      });
      added += 1;
    }

    if (added) {
      setDoc(await getDocument(doc.id));
    }

    return added;
  };

  const clearDocumentState = () => {
    setSuggestions(null);
    setExportText('');
    setGenError('');
    setGenInfo('');
    setRelWarning('');
    setSourceEnt('');
    setTargetEnt('');
  };


  const acceptSuggestedEntity = async (cand:any) => {
    if (!doc) return;

    const baseStart = typeof cand.start === 'number' ? cand.start : null;
    const baseEnd = typeof cand.end === 'number' ? cand.end : null;
    const candTextRaw = cand.text || (baseStart !== null && baseEnd !== null ? doc.raw_text.slice(baseStart, baseEnd) : '');
    const candText = (candTextRaw || '').trim();
    if (!candText) return;

    if (autoLabelMatches) {
      let spans = findMatchingSpans(candText);
      if (!spans.length && baseStart !== null && baseEnd !== null) {
        spans = [{ start: baseStart, end: baseEnd }];
      }

      const added = await addEntitiesForSpans(spans, cand.type);
      if (!added && baseStart !== null && baseEnd !== null && !entityExists(cand)) {
        await addEntity({
          document_id: doc.id,
          start_offset: baseStart,
          end_offset: baseEnd,
          type: cand.type,
          annotator_id: annotator
        });
        setDoc(await getDocument(doc.id));
      }
      return;
    }

    if (entityExists(cand)) return;                     // prevent duplicates
    if (baseStart === null || baseEnd === null) return;

    await addEntity({
      document_id: doc.id,
      start_offset: baseStart,
      end_offset: baseEnd,
      type: cand.type,
      annotator_id: annotator
    });
    const fresh = await getDocument(doc.id);
    setDoc(fresh);
  };

  const validateRelation = (): string | '' => {
    if (!doc) return ''
    if (!sourceEnt || !targetEnt) return 'Pick source and target entities.'
    if (sourceEnt === targetEnt) return 'Cannot create a relation to the same entity.'
    const src = doc.entities.find(e=>e.id===sourceEnt)
    const tgt = doc.entities.find(e=>e.id===targetEnt)
    if (!src || !tgt) return 'Pick source and target entities from this document.'
    const rules = (labelset?.relations || DEFAULT_REL_RULES)
    const ok = rules.some((r:any)=> r.type===relType && r.source.includes(src.type) && r.target.includes(tgt.type))
    return ok ? '' : `Invalid: ${src.type} --${relType}--> ${tgt.type}`
  }

  useEffect(()=>{
    setRelWarning(validateRelation())
  }, [relType, sourceEnt, targetEnt, doc])

  useEffect(() => {
    setSourceEnt('')
    setTargetEnt('')
  }, [doc?.id])

  const addNewRelation = async () => {
    const warn = validateRelation()
    if (warn) { setRelWarning(warn); return }
    if (!doc) return
    await addRelation({
      document_id: doc.id,
      source_entity_id: sourceEnt,
      target_entity_id: targetEnt,
      type: relType,
      annotator_id: annotator
    });
    setDoc(await getDocument(doc.id))
  }

  const removeRelation = async (id: string) => {
    await deleteRelation(id)
    if (doc) setDoc(await getDocument(doc.id))
  }

  const doExportAllDocuments = async () => {
    try {
      const arr = await exportJSONArr();
      const text = JSON.stringify(arr, null, 2);
      setExportText(text);
      const blob = new Blob([text], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'all_documents_export.json';
      document.body.appendChild(a);
      a.click();
      URL.revokeObjectURL(a.href);
      a.remove();
    } catch (e: any) {
      setExportText(`ERROR: ${e?.message || 'export failed'}`);
    }
  };

  const doExportCurrentDocument = async () => {
    if (!doc) return;
    try {
      const docData = await exportCurrentDocument(doc);
      const text = JSON.stringify(docData, null, 2);
      setExportText(text);
      const blob = new Blob([text], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `document_${doc.id}.json`;
      document.body.appendChild(a);
      a.click();
      URL.revokeObjectURL(a.href);
      a.remove();
    } catch (e: any) {
      setExportText(`ERROR: ${e?.message || 'export failed'}`);
    }
  };


  const generateSuggestions = async (force = false) => {
    if (!doc) return

    const cacheKey = doc.id
    if (!force && suggestionCache[cacheKey]) {
      setGenInfo('Loaded cached suggestions. Use Refresh to regenerate.')
      setSuggestions(suggestionCache[cacheKey])
      return
    }

    setGenBusy(true); setGenError(''); setGenInfo('')
    try {
      const data = await runSuggestions(doc.id, 'hf_local')
      if (data.ok === false) {
        setGenError(data.error || 'Backend returned no suggestions.')
      }
      const result = data.result || { entities: [], relations: [] }
      setSuggestions(result)
      setSuggestionCache(prev => ({ ...prev, [cacheKey]: result }))
      if (!result.entities?.length && !result.relations?.length) {
        setGenError(prev => prev || 'Model returned no suggestions.')
      } else {
        setGenInfo('Suggestions updated from model run.')
      }
    } catch (e: any) {
      const msg = e?.response?.data?.error || e?.message || 'Failed to generate suggestions.'
      setGenError(msg)
    } finally {
      setGenBusy(false)
    }
  }

  const acceptSuggestion = async (s: any) => {
    await acceptSuggestedEntity(s);
  };

  const acceptRelationSuggestion = async (s: any) => {
    if (!doc || relationExists(s)) return;
    // find source/target by text; require entities to be accepted first
    const src = (doc.entities || []).find(e =>
      doc.raw_text.slice(e.start_offset, e.end_offset).toLowerCase() === (s.source_text || '').toLowerCase());
    const tgt = (doc.entities || []).find(e =>
      doc.raw_text.slice(e.start_offset, e.end_offset).toLowerCase() === (s.target_text || '').toLowerCase());
    if (!src || !tgt) { alert('Accept the related entities first.'); return; }

    await addRelation({
      document_id: doc.id,
      type: s.type,
      // use whichever keys your backend expects
      source_entity_id: src.id,
      target_entity_id: tgt.id,
      // or: source_id: src.id, target_id: tgt.id,
      annotator_id: annotator
    });
    setDoc(await getDocument(doc.id));
  };

  const prettySegments = useMemo(() => {
    if (!doc) return []
    const ents = [...doc.entities].sort((a,b)=>a.start_offset - b.start_offset)
    const segs: { text: string; type?: string }[] = []
    let idx = 0
    for (const e of ents) {
      if (e.start_offset > idx) segs.push({ text: doc.raw_text.slice(idx, e.start_offset) })
      segs.push({ text: doc.raw_text.slice(e.start_offset, e.end_offset), type: e.type })
      idx = e.end_offset
    }
    if (idx < doc.raw_text.length) segs.push({ text: doc.raw_text.slice(idx) })
    return segs
  }, [doc])

  const entityTypes = labelset?.entities || DEFAULT_ENTITY_TYPES
  const relTypes = (labelset?.relations?.map((r:any)=>r.type) || DEFAULT_REL_RULES.map(r=>r.type))
  const fixedCardStyle = useMemo<React.CSSProperties>(() => (
    panelHeight ? { height: panelHeight } : {}
  ), [panelHeight])

  return (
    <div className="container">
      <h2>Medical Entity & Relation Annotation</h2>

      <div className="grid">
        <div className="card" ref={docPanelRef}>
          <h3>Create / Select Document</h3>
          <div className="row">
            <label>Document:</label>
            <input type="text" value={newTitle} onChange={e=>setNewTitle(e.target.value)} placeholder="Document title"/>
          </div>
          <div className="row">
            <label>Annotator:</label>
            <input type="text" value={annotator} onChange={e=>setAnnotator(e.target.value)} placeholder="your_name"/>
          </div>
          <textarea placeholder="Paste clinical text here..." value={newDocText} onChange={e=>setNewDocText(e.target.value)} />
          <div className="row"><button onClick={createNewDoc}>Create Document</button></div>
          <hr/>
          <div className="row">
            <label>Select:</label>
            <select
              value={doc?.id || ''}
              onChange={async e => {
                if (e.target.value) {
                  clearDocumentState();
                  setDoc(await getDocument(e.target.value));
                } else {
                  clearDocumentState();
                  setDoc(null);
                }
              }}
            >
              <option value="">-- Select Document --</option>
              {documents.map(d => (
                <option key={d.id} value={d.id}>
                  {d.title || d.id}
                </option>
              ))}
            </select>
            {doc && <span className="small" style={{ marginLeft: 8 }}>ID: {doc.id}</span>}
            {doc && (
              <button style={{ marginLeft: 12 }} onClick={doExportCurrentDocument}>
                Export Current Document
              </button>
            )}
            <button style={{ marginLeft: doc ? 8 : 12 }} onClick={doExportAllDocuments}>
              Export All Documents
            </button>
          </div>

          {exportText && (
            <div style={{ marginTop: 16 }}>
              <h4>JSON Export Preview</h4>
              <textarea
                value={exportText}
                readOnly
                style={{
                  width: '100%',
                  height: '200px',
                  fontFamily: 'monospace',
                  fontSize: '12px',
                  border: '1px solid #ccc',
                  padding: '8px'
                }}
              />
            </div>
          )}
        </div>

        <div className="card">
          <h3>Annotate Entities</h3>
          {doc ? (
            <>
              <div className="toolbar">
                <span className="toolbar-instruction">Select text below, then add an entity.</span>
                <div className="toolbar-controls">
                  <label className="toolbar-label">Type:</label>
                  <select value={selectedType} onChange={e=>setSelectedType(e.target.value)}>
                    {entityTypes.map((t: string) => <option key={t} value={t}>{t}</option>)}
                  </select>
                  <button onClick={addNewEntity}>Add Entity</button>
                  <label className="toggle-label">
                    <input
                      type="checkbox"
                      checked={autoLabelMatches}
                      onChange={e => setAutoLabelMatches(e.target.checked)}
                    />
                    Auto-label repeats
                  </label>
                </div>
              </div>
              <textarea
                ref={textAreaRef}
                value={doc.raw_text}
                readOnly
                onMouseUp={onTextMouseUp}
                onKeyUp={onTextMouseUp}
              />
              <div className="selection-meta">Selection: [{selStart}, {selEnd})</div>


              <h4>Preview with Highlights</h4>
              <pre className="code preview-scroll">
                {prettySegments.map((s, i) => s.type
                  ? <span key={i} className={`hl ${COLORS[s.type] || ''}`}>{s.text} <strong>{s.type}</strong></span>
                  : <span key={i}>{s.text}</span>
                )}
              </pre>

              <h4>Entities</h4>
              {(doc.entities?.length ?? 0) > 0 ? (
                <div style={{
                  maxHeight: '200px',
                  overflowY: 'auto',
                  border: '1px solid #ddd',
                  padding: '8px',
                  borderRadius: '4px'
                }}>
                  {doc.entities.map(e => (
                    <div key={e.id} className="entity-row">
                      <span className="badge">{e.type}</span>
                      <div className="entity-meta">
                        <span className={`entity-text ${COLORS[e.type] || ''}`}>"{getEntitySnippet(e.start_offset, e.end_offset)}"</span>
                        <code className="entity-range">[{e.start_offset}, {e.end_offset})</code>
                        {e.codes && <span className="small">codes: {Object.entries(e.codes).map(([k,v])=>`${k}=${v}`).join(', ')}</span>}
                        {e.annotator_id && <span className="small">by {e.annotator_id}</span>}
                        {e.created_at && <span className="small">@ {new Date(e.created_at).toLocaleString('en-AU', {timeZone: 'Australia/Sydney'})}</span>}
                      </div>
                      <button onClick={()=>removeEntity(e.id)}>Delete</button>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="muted">No entities yet.</p>
              )}
            </>
          ) : <p>No document selected.</p>}
        </div>
      </div>

      <div className="grid" style={{marginTop: 16}}>
        <div className="card card--fixed" style={fixedCardStyle}>
          <h3>Create Relations</h3>
          {doc ? (
            <div className="card-body">
              <div className="row">
                <label>Type:</label>
                <select value={relType} onChange={e=>setRelType(e.target.value)}>
                  {relTypes.map((t: string) => <option key={t} value={t}>{t}</option>)}
                </select>
                <label>Source:</label>
                <select value={sourceEnt} onChange={e=>setSourceEnt(e.target.value)}>
                  <option value="">--</option>
                  {doc.entities.map(e => <option key={e.id} value={e.id}>{`${e.type}: "${doc.raw_text.slice(e.start_offset, e.end_offset)}"`}</option>)}
                </select>
                <label>Target:</label>
                <select value={targetEnt} onChange={e=>setTargetEnt(e.target.value)}>
                  <option value="">--</option>
                  {doc.entities.map(e => <option key={e.id} value={e.id}>{`${e.type}: "${doc.raw_text.slice(e.start_offset, e.end_offset)}"`}</option>)}
                </select>
                <button onClick={addNewRelation}>Add Relation</button>
              </div>
              {relWarning && <div className="warn">{relWarning}</div>}

              <h4>Relations</h4>
              {(doc.relations?.length ?? 0) > 0 ? (
                <div style={{
                  maxHeight: '200px',
                  overflowY: 'auto',
                  border: '1px solid #ddd',
                  padding: '8px',
                  borderRadius: '4px'
                }}>
                  {(doc.relations ?? []).map(r => (
                    <div key={r.id} className="relation-row">
                      <span className="badge">{r.type}</span>
                      <code>{`"${getEntityText(r.source_entity_id)}" ➜ "${getEntityText(r.target_entity_id)}"`}</code>
                      {r.annotator_id && <span className="small"> by {r.annotator_id}</span>}
                      {r.created_at && <span className="small"> @ {new Date(r.created_at).toLocaleString('en-AU', {timeZone: 'Australia/Sydney'})}</span>}
                      <button onClick={()=>removeRelation(r.id)}>Delete</button>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="muted">No relations yet.</p>
              )}
            </div>
          ) : (
            <div className="card-body">
              <p className="muted">Select a document to create relations.</p>
            </div>
          )}
        </div>

        <div className="card card--fixed" style={fixedCardStyle}>
          <h3>Auto Suggestions (Local LLM)</h3>
          <div className="card-body">
            <div className="row" style={{ alignItems: 'center' }}>
              <button onClick={()=>generateSuggestions()} disabled={!doc || genBusy}>
                {genBusy ? 'Generating…' : 'Generate Suggestions (Local)'}
              </button>
              {doc?.id && suggestionCache[doc.id] && (
                <button
                  onClick={()=>generateSuggestions(true)}
                  disabled={genBusy}
                  style={{ marginLeft: 6 }}
                >
                  Refresh
                </button>
              )}
              {genError && <span className="warn" style={{ marginLeft: 8 }}>{genError}</span>}
              {!genError && genInfo && <span className="small" style={{ marginLeft: 8 }}>{genInfo}</span>}
            </div>

            {!doc && <p className="muted">Select a document to generate suggestions.</p>}
            {doc && !suggestions && !genBusy && <p className="muted">Run the model to see suggested entities and relations.</p>}

            {suggestions && (
              <div>
                <h4>Suggested Entities</h4>
                {(suggestions?.entities?.length ?? 0) > 0 ? (
                  <div style={{
                    maxHeight: '200px',
                    overflowY: 'auto',
                    border: '1px solid #ddd',
                    padding: '8px',
                    borderRadius: '4px'
                  }}>
                    {suggestions?.entities?.map((e:any, idx:number) => (
                      <div key={`sent-${idx}`} className="chip">
                        <span className="badge">{e.type}</span>
                        <span className="chip-text">
                          "{e.text}" [{e.start}, {e.end}]
                        </span>
                        <button
                          onClick={() => acceptSuggestedEntity(e)}
                          disabled={entityExists(e)}
                        >
                          {entityExists(e) ? 'Accepted' : 'Accept'}
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="muted">No entity suggestions yet.</p>
                )}
                <h4>Suggested Relations</h4>
                {(suggestions?.relations?.length ?? 0) > 0 ? (
                  <div style={{
                    maxHeight: '200px',
                    overflowY: 'auto',
                    border: '1px solid #ddd',
                    padding: '8px',
                    borderRadius: '4px'
                  }}>
                    {suggestions?.relations?.map((s:any, i:number) => (
                      <div key={`srel-${i}`} className="relation-row">
                        <span className="badge">{s.type}</span>
                        <code>{`${s.source_text} ➜ ${s.target_text}`}</code>
                        <button
                          onClick={()=>acceptRelationSuggestion(s)}
                          disabled={relationExists(s)}
                        >
                          {relationExists(s) ? 'Accepted' : 'Accept'}
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="muted">No relation suggestions yet.</p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
