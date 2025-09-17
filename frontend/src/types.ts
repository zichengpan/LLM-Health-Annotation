export type Project = { id: string; name: string; description?: string };
export type Document = { id: string; project_id: string; title?: string | null; raw_text: string; external_doc_id?: string | null; status?: string; meta?: any };
export type Entity = { id: string; document_id: string; start_offset: number; end_offset: number; type: string; codes?: Record<string, string> | null; annotator_id?: string | null; created_at?: string | null };
export type Relation = { id: string; document_id: string; source_entity_id: string; target_entity_id: string; type: string; annotator_id?: string | null; created_at?: string | null };
export type DocumentDetail = Document & { entities: Entity[]; relations: Relation[] };
