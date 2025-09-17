import axios from 'axios'
import type { Document, DocumentDetail, Entity, Relation } from './types'

const API = axios.create({ baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000' })

export const listDocuments = async (): Promise<Document[]> => {
  const { data } = await API.get(`/projects/prj_default/documents`)
  return data
}
export const createDocument = async (title: string, raw_text: string, meta?: any): Promise<Document> => {
  const { data } = await API.post('/documents', { project_id: 'prj_default', title, raw_text, status: 'in_progress', meta })
  return data
}
export const getDocument = async (docId: string): Promise<DocumentDetail> => {
  const { data } = await API.get(`/documents/${docId}`)
  return data
}

export const addEntity = async (payload: Omit<Entity, 'id' | 'created_at'> & { id?: string }): Promise<Entity> => {
  const { data } = await API.post('/entities', payload)
  return data
}
export const deleteEntity = async (id: string): Promise<void> => { await API.delete(`/entities/${id}`) }
export async function addRelation(payload: {
  document_id: string;
  type: string;
  source_entity_id: string;
  target_entity_id: string;
  annotator_id?: string;
}) {
  const { data } = await API.post('/relations', payload);
  return data;
}

export const deleteRelation = async (id: string): Promise<void> => { await API.delete(`/relations/${id}`) }

export async function exportJSONArr(): Promise<any[]> {
  try {
    const { data } = await API.get('/export/json');
    return data;
  } catch {
    const { data } = await API.get('/projects/prj_default/export/json');
    return data;
  }
}

export async function exportCurrentDocument(doc: any): Promise<any> {
  // Create a single document export in the same format as the backend
  return {
    document_id: doc.id,
    title: doc.title,
    text: doc.raw_text,
    entities: doc.entities.map((e: any) => ({
      id: e.id,
      start: e.start_offset,
      end: e.end_offset,
      type: e.type,
      codes: e.codes,
      annotator_id: e.annotator_id,
      created_at: e.created_at
    })),
    relations: doc.relations.map((r: any) => ({
      id: r.id,
      source: r.source_entity_id,
      target: r.target_entity_id,
      type: r.type,
      direction: "source_to_target",
      annotator_id: r.annotator_id,
      created_at: r.created_at
    })),
    meta: {
      project_id: doc.project_id || "prj_default"
    }
  };
}

export const getLabelset = async (): Promise<any> => (await API.get('/labelset')).data
export const runSuggestions = async (document_id: string, provider = 'llama_cpp'): Promise<any> => (await API.post('/suggestions/run', null, { params: { document_id, provider } })).data
export const suggestRelations = async (document_id: string): Promise<any> => (await API.post('/suggestions/relations', null, { params: { document_id } })).data
